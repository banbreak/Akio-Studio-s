"""Remote (cloud GPU) video rendering stage for Akio Studio.

The video stage is the one pipeline stage that cannot run on the target
machine. A WAN-class 14B checkpoint is ~28 GiB in fp16 — larger than the whole
24 GiB unified pool — and even a quantized fit would evict every other stage
and crawl on MPS. Moving it to a rented GPU changes three things at once:

* **Model class** — a 14B-class checkpoint becomes usable at all, instead of
  the ~1.3B local fallback that had to fit alongside macOS.
* **Local pressure** — the stage keeps nothing on-device but an HTTP client
  (~0.2 GiB), so the pool coordinator no longer has to evict the LLM or the
  image stack to make room for frames.
* **Concurrency** — remote jobs run in parallel with local work *and* with
  each other; ``max_concurrent_jobs`` is remote capacity, not local memory.

Operational care taken here, because a cloud GPU bills by the second:

* Submissions carry an ``Idempotency-Key`` derived from the request contents,
  so a retried POST after a dropped connection cannot start (and bill for) a
  second identical render.
* Only idempotent reads are retried freely; a submission is retried solely
  when no job id was ever obtained.
* A job that outlives its wall-clock budget, exceeds its cost ceiling, or
  whose awaiting task is cancelled is **cancelled remotely**, never abandoned.
* The bearer token is read from the environment and never logged, persisted,
  or written into asset metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from akio_studio._io import atomic_write_bytes
from akio_studio.config import CloudVideoConfig, StudioConfig
from akio_studio.exceptions import (
    CloudAuthError,
    CloudBudgetExceededError,
    CloudJobFailedError,
    CloudRenderError,
    CloudTimeoutError,
)

__all__ = [
    "CloudTransport",
    "CloudVideoRenderer",
    "HttpCloudTransport",
    "JobStatus",
    "MockCloudTransport",
    "VideoRenderRequest",
    "VideoRenderResult",
]

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """Lifecycle state of a remote render job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def is_terminal(self) -> bool:
        """True when the job will not change state again."""
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED)

    @classmethod
    def parse(cls, raw: Any) -> JobStatus:
        """Map a provider status string onto the canonical lifecycle.

        Unknown values are treated as ``RUNNING`` rather than an error: a
        provider inventing a new in-flight state (``"starting"``,
        ``"provisioning"``) must not abort a job that is merely progressing.
        """
        text = str(raw or "").strip().lower()
        direct = {s.value: s for s in cls}
        if text in direct:
            return direct[text]
        aliases = {
            "success": cls.SUCCEEDED,
            "succeeded": cls.SUCCEEDED,
            "completed": cls.SUCCEEDED,
            "complete": cls.SUCCEEDED,
            "done": cls.SUCCEEDED,
            "error": cls.FAILED,
            "failure": cls.FAILED,
            "failed": cls.FAILED,
            "cancelled": cls.CANCELED,
            "aborted": cls.CANCELED,
            "pending": cls.QUEUED,
            "in_queue": cls.QUEUED,
            "processing": cls.RUNNING,
            "in_progress": cls.RUNNING,
        }
        if text in aliases:
            return aliases[text]
        logger.debug("unknown provider status %r; treating as running", raw)
        return cls.RUNNING


@dataclass(frozen=True)
class VideoRenderRequest:
    """One shot to render remotely.

    ``shot_id`` and ``prompt_hash`` tie the result back to the DPO registry so
    a retention drop-off can still be attributed to the render that caused it.
    """

    shot_id: str
    prompt: str
    seed: int
    prompt_hash: str = ""
    negative_prompt: str = ""
    num_frames: int = 81
    width: int = 1280
    height: int = 720
    denoise: float = 0.30
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject requests that would waste a paid render."""
        if not self.shot_id:
            raise CloudRenderError("shot_id must not be empty")
        if not self.prompt.strip():
            raise CloudRenderError(f"shot {self.shot_id!r} has an empty prompt")
        if self.num_frames <= 0:
            raise CloudRenderError(f"shot {self.shot_id!r} has num_frames <= 0")
        if self.width <= 0 or self.height <= 0:
            raise CloudRenderError(f"shot {self.shot_id!r} has non-positive dimensions")

    def to_payload(self, model: str) -> dict[str, Any]:
        """Provider-agnostic job submission body."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "denoise": self.denoise,
        }
        payload.update(self.extra)
        return payload

    def idempotency_key(self, model: str) -> str:
        """Stable key over the exact render this request describes.

        A retried submission after a dropped connection reuses this key, so
        the provider can collapse the duplicate instead of billing twice.
        """
        canonical = json.dumps(
            {"shot_id": self.shot_id, **self.to_payload(model)},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VideoRenderResult:
    """Outcome of a completed remote render."""

    shot_id: str
    job_id: str
    status: JobStatus
    output_path: Path | None
    sha256: str | None
    cost_usd: float | None
    wall_clock_s: float
    model: str
    seed: int
    prompt_hash: str

    def to_metadata(self) -> dict[str, Any]:
        """Asset-metadata view. Carries no credential material."""
        return {
            "render_backend": "cloud",
            "job_id": self.job_id,
            "status": self.status.value,
            "model": self.model,
            "seed": self.seed,
            "prompt_hash": self.prompt_hash,
            "sha256": self.sha256,
            "cost_usd": self.cost_usd,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "output": self.output_path.name if self.output_path else None,
        }


class CloudTransport(Protocol):
    """Pluggable HTTP layer, so providers and tests can swap the wire."""

    async def request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, dict[str, Any]]:
        """Perform a JSON request; return ``(status_code, parsed_body)``."""
        ...

    async def download(self, url: str, timeout: float) -> bytes:
        """Fetch raw bytes for a completed job's output."""
        ...


class HttpCloudTransport:
    """Default transport: stdlib ``urllib`` executed off the event loop.

    Kept dependency-free on purpose — the orchestrator process stays small
    (the audit's whole point about not importing heavyweight libraries into
    the machine whose memory is the bottleneck).
    """

    async def request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, dict[str, Any]]:
        """Perform a JSON request in a worker thread."""
        return await asyncio.to_thread(
            self._request_blocking, method, url, payload, headers, timeout
        )

    @staticmethod
    def _request_blocking(
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, dict[str, Any]]:
        """Blocking JSON request. HTTP error statuses are returned, not raised."""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        all_headers = dict(headers)
        if data is not None:
            all_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(url, data=data, method=method, headers=all_headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                body = response.read()
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            body = exc.read()
        if not body:
            return status, {}
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return status, {}
        return status, parsed if isinstance(parsed, dict) else {"data": parsed}

    async def download(self, url: str, timeout: float) -> bytes:
        """Fetch the rendered artifact in a worker thread."""
        return await asyncio.to_thread(self._download_blocking, url, timeout)

    @staticmethod
    def _download_blocking(url: str, timeout: float) -> bytes:
        """Blocking artifact fetch."""
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return bytes(response.read())


class MockCloudTransport:
    """In-process stand-in for a render provider.

    Lets the demo and the test-suite exercise the full submit -> poll ->
    download -> verify path with no endpoint, no credential, and no spend.
    Each job reports ``QUEUED`` then ``RUNNING`` for ``running_polls`` polls
    before succeeding.
    """

    def __init__(
        self,
        running_polls: int = 1,
        fail_job: bool = False,
        cost_usd: float = 0.42,
        payload_bytes: bytes = b"AKIO-MOCK-MP4\x00",
    ) -> None:
        """Configure the simulated provider's behaviour."""
        self.running_polls = running_polls
        self.fail_job = fail_job
        self.cost_usd = cost_usd
        self.payload_bytes = payload_bytes
        self.submissions: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self._polls: dict[str, int] = {}
        self._by_key: dict[str, str] = {}

    async def request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, dict[str, Any]]:
        """Simulate the provider's three endpoints."""
        path = urllib.parse.urlparse(url).path
        if method == "POST" and path.endswith("/jobs"):
            key = headers.get("Idempotency-Key", "")
            if key in self._by_key:  # duplicate submission collapses, as a real provider would
                return 200, {"job_id": self._by_key[key], "status": "queued"}
            job_id = f"mock-job-{len(self.submissions) + 1}"
            self._by_key[key] = job_id
            self.submissions.append({"job_id": job_id, "payload": payload or {}})
            self._polls[job_id] = 0
            return 201, {"job_id": job_id, "status": "queued"}
        if method == "POST" and path.endswith("/cancel"):
            job_id = path.rsplit("/", 2)[-2]
            self.cancelled.append(job_id)
            return 200, {"job_id": job_id, "status": "canceled"}
        if method == "GET":
            job_id = path.rsplit("/", 1)[-1]
            seen = self._polls.get(job_id, 0)
            self._polls[job_id] = seen + 1
            if self.fail_job:
                return 200, {"status": "failed", "error": "simulated provider failure"}
            if seen < self.running_polls:
                return 200, {"status": "running", "progress": 0.5}
            digest = hashlib.sha256(self.payload_bytes).hexdigest()
            return 200, {
                "status": "succeeded",
                "progress": 1.0,
                "output_url": f"https://mock.invalid/{job_id}.mp4",
                "sha256": digest,
                "cost_usd": self.cost_usd,
            }
        return 404, {"error": f"unmocked {method} {path}"}

    async def download(self, url: str, timeout: float) -> bytes:
        """Return the simulated artifact bytes."""
        return self.payload_bytes


class CloudVideoRenderer:
    """Submits shots to a remote GPU, waits, verifies, and stores the result.

    Construct with ``transport=MockCloudTransport()`` to exercise the whole
    path offline; the default :class:`HttpCloudTransport` talks to
    :class:`~akio_studio.config.CloudVideoConfig`'s endpoint.
    """

    def __init__(
        self,
        config: StudioConfig | None = None,
        transport: CloudTransport | None = None,
    ) -> None:
        """Create a renderer bound to ``config``'s cloud settings."""
        self._config = config if config is not None else StudioConfig()
        self._cloud: CloudVideoConfig = self._config.cloud_video
        self._transport: CloudTransport = transport or HttpCloudTransport()
        self._is_mock = isinstance(self._transport, MockCloudTransport)

    @property
    def model(self) -> str:
        """Remote checkpoint this renderer submits to."""
        return self._cloud.model

    # ------------------------------------------------------------- plumbing

    def _base_url(self) -> str:
        """Validated endpoint base, or raise :class:`CloudAuthError`.

        HTTPS is mandatory: the token travels in a header, and a plaintext
        endpoint would leak it. Loopback is exempt so a local emulator works.
        """
        if self._is_mock:
            return (self._cloud.resolve_endpoint() or "https://mock.invalid/v1").rstrip("/")
        endpoint = self._cloud.resolve_endpoint()
        if not endpoint:
            raise CloudAuthError(
                "no cloud video endpoint configured; set "
                f"${self._cloud.endpoint_env_var} or CloudVideoConfig.endpoint"
            )
        parsed = urllib.parse.urlparse(endpoint)
        is_loopback = parsed.hostname in ("127.0.0.1", "localhost", "::1")
        if parsed.scheme != "https" and not is_loopback:
            raise CloudAuthError(
                f"cloud video endpoint must use https (got {parsed.scheme!r}); "
                "the bearer token would otherwise cross the network in plaintext"
            )
        return endpoint.rstrip("/")

    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        """Auth + tracing headers. The token is never logged."""
        headers = {"Accept": "application/json"}
        token = self._cloud.resolve_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif not self._is_mock:
            raise CloudAuthError(
                f"no cloud credential; set ${self._cloud.token_env_var}"
            )
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
        what: str,
    ) -> dict[str, Any]:
        """Issue a request, retrying transient failures with jittered backoff.

        Retried on connection errors and 5xx/429 only. 401/403 fail fast —
        a bad credential will not fix itself, and retrying wastes the budget.
        """
        last_error = ""
        for attempt in range(1, self._cloud.max_attempts + 1):
            try:
                status, body = await self._transport.request(
                    method, url, payload, headers, timeout
                )
            except OSError as exc:
                last_error = f"transport error: {exc}"
                status, body = 0, {}
            else:
                if 200 <= status < 300:
                    return body
                if status in (401, 403):
                    raise CloudAuthError(
                        f"{what} rejected with HTTP {status}: check "
                        f"${self._cloud.token_env_var}"
                    )
                last_error = f"HTTP {status}: {str(body)[:200]}"
                if status < 500 and status != 429:
                    raise CloudRenderError(f"{what} failed — {last_error}")
            if attempt < self._cloud.max_attempts:
                backoff = min(2.0 ** (attempt - 1), 15.0) * (1.0 + random.random() * 0.25)
                logger.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    what,
                    attempt,
                    self._cloud.max_attempts,
                    last_error,
                    backoff,
                )
                await asyncio.sleep(backoff)
        raise CloudRenderError(
            f"{what} failed after {self._cloud.max_attempts} attempts — {last_error}"
        )

    # ---------------------------------------------------------- job control

    async def submit(self, request: VideoRenderRequest) -> str:
        """Submit one render job; returns the provider job id.

        The ``Idempotency-Key`` makes a retried POST safe: a provider that
        honours it collapses the duplicate instead of starting a second
        billable render.
        """
        base = self._base_url()
        key = request.idempotency_key(self.model)
        body = await self._request_with_retry(
            "POST",
            f"{base}/jobs",
            request.to_payload(self.model),
            self._headers(idempotency_key=key),
            self._cloud.submit_timeout_s,
            f"submit of shot {request.shot_id!r}",
        )
        job_id = body.get("job_id") or body.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise CloudRenderError(
                f"submit of shot {request.shot_id!r} returned no job id: {body!r:.200}"
            )
        logger.info("shot %s submitted as cloud job %s", request.shot_id, job_id)
        return job_id

    async def poll(self, job_id: str) -> dict[str, Any]:
        """Fetch one status snapshot for ``job_id``."""
        base = self._base_url()
        return await self._request_with_retry(
            "GET",
            f"{base}/jobs/{urllib.parse.quote(job_id)}",
            None,
            self._headers(),
            self._cloud.poll_timeout_s,
            f"status poll of job {job_id!r}",
        )

    async def cancel(self, job_id: str) -> bool:
        """Best-effort remote cancellation; never raises.

        Used on timeout, budget breach, and task cancellation — an abandoned
        GPU job keeps billing until it finishes on its own.
        """
        try:
            base = self._base_url()
            await self._transport.request(
                "POST",
                f"{base}/jobs/{urllib.parse.quote(job_id)}/cancel",
                None,
                self._headers(),
                self._cloud.poll_timeout_s,
            )
        except (OSError, CloudRenderError) as exc:
            logger.error(
                "could not cancel cloud job %s (%s) — verify it in the "
                "provider console so it does not keep billing",
                job_id,
                exc,
            )
            return False
        logger.info("cancelled cloud job %s", job_id)
        return True

    async def wait_for(self, job_id: str) -> dict[str, Any]:
        """Poll until the job is terminal, the budget trips, or time runs out.

        Backs off from ``poll_interval_s`` to ``poll_interval_max_s``. On
        timeout, cost breach, or cancellation the remote job is cancelled
        before the exception propagates.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._cloud.max_wait_s
        interval = self._cloud.poll_interval_s
        try:
            while True:
                snapshot = await self.poll(job_id)
                status = JobStatus.parse(snapshot.get("status"))
                cost = snapshot.get("cost_usd")
                ceiling = self._cloud.max_cost_usd
                if (
                    ceiling is not None
                    and isinstance(cost, int | float)
                    and float(cost) > ceiling
                ):
                    await self.cancel(job_id)
                    raise CloudBudgetExceededError(
                        f"cloud job {job_id} reported ${float(cost):.2f}, over the "
                        f"${ceiling:.2f} ceiling; job cancelled"
                    )
                if status.is_terminal:
                    return {**snapshot, "status": status.value}
                if loop.time() >= deadline:
                    await self.cancel(job_id)
                    raise CloudTimeoutError(
                        f"cloud job {job_id} exceeded its {self._cloud.max_wait_s}s "
                        "budget; job cancelled"
                    )
                remaining = max(0.0, deadline - loop.time())
                await asyncio.sleep(min(interval, remaining) or 0.01)
                interval = min(interval * 1.5, self._cloud.poll_interval_max_s)
        except asyncio.CancelledError:
            # The awaiting task went away; the GPU would otherwise keep running.
            await self.cancel(job_id)
            raise

    # ------------------------------------------------------------ rendering

    async def render_shot(
        self,
        request: VideoRenderRequest,
        dest_dir: Path,
    ) -> VideoRenderResult:
        """Render one shot end to end: submit, wait, download, verify, store.

        The artifact is written atomically into ``dest_dir`` and its digest is
        checked against the provider's when one is supplied — a truncated
        download must never be mistaken for a finished shot.
        """
        gate = self._config.quality_gate
        gate.validate_wan_params(request.denoise)  # fail before paying for the render

        loop = asyncio.get_running_loop()
        started = loop.time()
        job_id = await self.submit(request)
        snapshot = await self.wait_for(job_id)
        status = JobStatus.parse(snapshot.get("status"))

        if status is not JobStatus.SUCCEEDED:
            raise CloudJobFailedError(
                f"cloud job {job_id} for shot {request.shot_id!r} ended "
                f"{status.value}: {snapshot.get('error') or 'no error detail'}"
            )

        output_url = snapshot.get("output_url")
        if not isinstance(output_url, str) or not output_url:
            raise CloudJobFailedError(
                f"cloud job {job_id} succeeded without an output_url"
            )
        try:
            payload = await self._transport.download(
                output_url, self._cloud.download_timeout_s
            )
        except OSError as exc:
            raise CloudRenderError(
                f"downloading output of job {job_id} failed: {exc}"
            ) from exc

        digest = hashlib.sha256(payload).hexdigest()
        expected = snapshot.get("sha256")
        if isinstance(expected, str) and expected and expected.lower() != digest:
            raise CloudRenderError(
                f"checksum mismatch for job {job_id}: provider reported "
                f"{expected}, downloaded bytes hash to {digest}"
            )

        dest_dir = Path(dest_dir)
        output_path = atomic_write_bytes(dest_dir / request.shot_id, payload)
        cost = snapshot.get("cost_usd")
        result = VideoRenderResult(
            shot_id=request.shot_id,
            job_id=job_id,
            status=status,
            output_path=output_path,
            sha256=digest,
            cost_usd=float(cost) if isinstance(cost, int | float) else None,
            wall_clock_s=loop.time() - started,
            model=self.model,
            seed=request.seed,
            prompt_hash=request.prompt_hash,
        )
        logger.info(
            "shot %s rendered remotely in %.1fs (%d bytes, cost %s)",
            request.shot_id,
            result.wall_clock_s,
            len(payload),
            f"${result.cost_usd:.2f}" if result.cost_usd is not None else "n/a",
        )
        return result

    async def render_batch(
        self,
        requests: list[VideoRenderRequest],
        dest_dir: Path,
    ) -> list[VideoRenderResult | BaseException]:
        """Render many shots concurrently, bounded by ``max_concurrent_jobs``.

        Concurrency here is *remote* capacity — it costs the local pool
        nothing, so seed variants for a shot render side by side instead of
        one after another. Per-shot failures are returned in place rather than
        cancelling siblings that may already have incurred cost.
        """
        semaphore = asyncio.Semaphore(max(1, self._cloud.max_concurrent_jobs))

        async def _one(req: VideoRenderRequest) -> VideoRenderResult:
            async with semaphore:
                return await self.render_shot(req, dest_dir)

        results = await asyncio.gather(
            *(_one(req) for req in requests), return_exceptions=True
        )
        failures = sum(1 for r in results if isinstance(r, BaseException))
        logger.info(
            "cloud batch complete: %d succeeded, %d failed",
            len(results) - failures,
            failures,
        )
        return list(results)

    def render_manifest(self, results: list[VideoRenderResult]) -> dict[str, Any]:
        """Summarize a batch for the production metadata tree."""
        total = sum(r.cost_usd or 0.0 for r in results)
        return {
            "backend": "cloud",
            "model": self.model,
            "rendered_at": datetime.now(UTC).isoformat(),
            "shots": [r.to_metadata() for r in results],
            "total_cost_usd": round(total, 4),
        }
