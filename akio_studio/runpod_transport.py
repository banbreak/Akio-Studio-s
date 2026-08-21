"""RunPod Serverless adapter for the cloud video stage.

:mod:`akio_studio.cloud_video` speaks one canonical job contract
(``POST /jobs``, ``GET /jobs/{id}``, ``POST /jobs/{id}/cancel``). RunPod's
Serverless v2 API is shaped differently, so this module is the adapter the
architecture anticipated — the pipeline never learns that RunPod exists.

Mapping (RunPod Serverless v2, base ``https://api.runpod.ai/v2/{endpoint_id}``):

===========================  =======================================
canonical                    RunPod
===========================  =======================================
``POST /jobs``               ``POST /run``   body ``{"input": {...}}``
``GET  /jobs/{id}``          ``GET  /status/{id}``
``POST /jobs/{id}/cancel``   ``POST /cancel/{id}``
job id field ``job_id``      ``id``
===========================  =======================================

Status vocabulary: ``IN_QUEUE``, ``IN_PROGRESS``, ``COMPLETED``, ``FAILED``,
``CANCELLED``, ``TIMED_OUT``.

Two RunPod realities this adapter has to paper over:

* **No idempotency keys.** RunPod does not honour an ``Idempotency-Key``
  header, so a blindly retried ``/run`` after a dropped connection could start
  a second billable job. Submissions are therefore attempted exactly once
  (:attr:`RunPodTransport.retries_submits` is ``False``) and every attempt is
  appended to a local journal so an ambiguous submit can be reconciled in the
  RunPod console instead of silently double-charging.
* **No cost field.** RunPod reports ``executionTime``/``delayTime`` in
  milliseconds, not dollars. Cost is derived from those against the endpoint's
  configured GPU rate so the renderer's ``max_cost_usd`` ceiling is a real
  guard rather than a no-op; while a job is still in flight the cost is
  *estimated* from elapsed wall-clock, so a runaway job still trips the
  ceiling.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any

from akio_studio._io import atomic_write_json
from akio_studio.cloud_video import HttpCloudTransport
from akio_studio.config import CloudVideoConfig
from akio_studio.exceptions import (
    CloudAuthError,
    CloudRenderError,
    CloudSubmitAmbiguousError,
)

__all__ = ["RUNPOD_API_ROOT", "RunPodTransport"]

logger = logging.getLogger(__name__)

#: Base of the RunPod Serverless v2 API.
RUNPOD_API_ROOT = "https://api.runpod.ai/v2"

#: Keys a worker handler may use for a finished video, in preference order.
_URL_KEYS = ("video_url", "output_url", "url", "mp4_url", "video")
_B64_KEYS = ("video_base64", "video_b64", "base64", "b64", "data")

#: Synthetic scheme for inline (base64) results held in memory by the adapter.
_INLINE_SCHEME = "runpod-inline"


class RunPodTransport:
    """Speaks the canonical job contract; talks RunPod Serverless underneath.

    Args:
        config: cloud settings; supplies the endpoint id, credential env var
            and GPU rate.
        journal_path: where ambiguous submissions are recorded. Defaults to
            ``~/Library/Application Support/AkioStudio/runpod_submits.jsonl``
            on macOS and ``~/.akio_studio/runpod_submits.jsonl`` elsewhere.
        http: injectable inner transport (tests supply a fake).
    """

    #: The renderer consults this before retrying a failed submission.
    retries_submits = False
    #: This adapter resolves its own credential (``$RUNPOD_API_KEY`` as well
    #: as the configured var), so the renderer must not demand one itself.
    handles_auth = True

    def __init__(
        self,
        config: CloudVideoConfig | None = None,
        journal_path: Path | None = None,
        http: HttpCloudTransport | None = None,
    ) -> None:
        """Bind the adapter to an endpoint and its credential."""
        self._config = config if config is not None else CloudVideoConfig()
        self._http = http if http is not None else HttpCloudTransport()
        self._journal_path = Path(journal_path) if journal_path else _default_journal()
        #: job_id -> monotonic timestamp of first sighting, for cost estimates.
        self._first_seen: dict[str, float] = {}
        self._inline: dict[str, bytes] = {}

    # ----------------------------------------------------------- addressing

    @property
    def endpoint_id(self) -> str:
        """Resolved RunPod endpoint id, or raise :class:`CloudAuthError`."""
        endpoint_id = (
            os.environ.get(self._config.runpod_endpoint_id_env_var, "")
            or self._config.runpod_endpoint_id
        )
        if not endpoint_id:
            raise CloudAuthError(
                "no RunPod endpoint id; set "
                f"${self._config.runpod_endpoint_id_env_var} (or "
                "CloudVideoConfig.runpod_endpoint_id) to the id shown on the "
                "endpoint's page in the RunPod console"
            )
        if "/" in endpoint_id or not endpoint_id.strip():
            raise CloudAuthError(f"malformed RunPod endpoint id {endpoint_id!r}")
        return endpoint_id.strip()

    def _api_base(self) -> str:
        """Full RunPod API base for this endpoint."""
        return f"{RUNPOD_API_ROOT}/{urllib.parse.quote(self.endpoint_id)}"

    @property
    def canonical_base(self) -> str:
        """Base the renderer builds canonical paths against.

        Read lazily by the renderer, so a missing endpoint id surfaces as a
        clear error on first use rather than at construction time.
        """
        return self._api_base()

    def _token(self) -> str:
        """RunPod API key from the environment. Never logged."""
        token = self._config.resolve_token() or os.environ.get("RUNPOD_API_KEY")
        if not token:
            raise CloudAuthError(
                "no RunPod API key; set "
                f"${self._config.token_env_var} or $RUNPOD_API_KEY "
                "(create one under Settings -> API Keys in the RunPod console)"
            )
        return token

    def _auth_headers(self, incoming: dict[str, str]) -> dict[str, str]:
        """Authorization header for RunPod; drops inapplicable ones.

        The canonical layer sends ``Idempotency-Key``; RunPod ignores it, and
        forwarding it would imply a protection that is not there.
        """
        headers = {
            k: v
            for k, v in incoming.items()
            if k.lower() not in ("authorization", "idempotency-key")
        }
        headers["Authorization"] = f"Bearer {self._token()}"
        return headers

    # ------------------------------------------------------------- journal

    def _record_submission(self, phase: str, detail: dict[str, Any]) -> None:
        """Append a submission record so ambiguous submits are reconcilable."""
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "endpoint_id": detail.get("endpoint_id", ""),
            "phase": phase,
            **{k: v for k, v in detail.items() if k != "endpoint_id"},
        }
        try:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._journal_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
        except OSError as exc:  # journaling must never break a render
            logger.warning("could not write RunPod submit journal: %s", exc)

    # ------------------------------------------------------------ transport

    async def request(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, dict[str, Any]]:
        """Translate one canonical request into its RunPod equivalent."""
        path = urllib.parse.urlparse(url).path.rstrip("/")
        base = self._api_base()
        auth = self._auth_headers(headers)

        if method == "POST" and path.endswith("/jobs"):
            return await self._submit(base, payload or {}, auth, timeout)
        if method == "POST" and path.endswith("/cancel"):
            job_id = path.rsplit("/", 2)[-2]
            status, body = await self._http.request(
                "POST", f"{base}/cancel/{urllib.parse.quote(job_id)}", None, auth, timeout
            )
            self._first_seen.pop(job_id, None)
            return status, body
        if method == "GET" and "/jobs/" in f"{path}/":
            job_id = path.rsplit("/", 1)[-1]
            return await self._status(base, job_id, auth, timeout)
        raise CloudRenderError(f"RunPod adapter cannot map {method} {path}")

    async def _submit(
        self,
        base: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, dict[str, Any]]:
        """``POST /run`` with the canonical payload wrapped as ``input``."""
        endpoint_id = self.endpoint_id
        shot = str(payload.get("shot_id") or payload.get("prompt", ""))[:80]
        self._record_submission(
            "attempt", {"endpoint_id": endpoint_id, "shot": shot, "seed": payload.get("seed")}
        )
        try:
            status, body = await self._http.request(
                "POST", f"{base}/run", {"input": payload}, headers, timeout
            )
        except OSError as exc:
            # The request may or may not have reached RunPod. Retrying could
            # start a second billable job, so surface the ambiguity instead.
            self._record_submission(
                "ambiguous",
                {"endpoint_id": endpoint_id, "shot": shot, "error": str(exc)},
            )
            logger.error(
                "RunPod submit for %r failed in flight (%s). It may still have "
                "started — check the endpoint's Requests tab; the attempt is "
                "journaled at %s",
                shot,
                exc,
                self._journal_path,
            )
            raise CloudSubmitAmbiguousError(
                f"RunPod submit for shot {shot!r} failed in flight ({exc}); the "
                "job may have started and may be billing. Reconcile against "
                f"the endpoint's Requests tab — attempt journaled at "
                f"{self._journal_path}"
            ) from exc
        job_id = body.get("id") or body.get("job_id")
        if 200 <= status < 300 and isinstance(job_id, str) and job_id:
            self._first_seen[job_id] = time.monotonic()
            self._record_submission(
                "accepted", {"endpoint_id": endpoint_id, "shot": shot, "job_id": job_id}
            )
            return status, {"job_id": job_id, "status": _map_status(body.get("status"))}
        return status, body

    async def _status(
        self,
        base: str,
        job_id: str,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, dict[str, Any]]:
        """``GET /status/{id}``, normalized to the canonical snapshot shape."""
        status, body = await self._http.request(
            "GET", f"{base}/status/{urllib.parse.quote(job_id)}", None, headers, timeout
        )
        if not 200 <= status < 300:
            return status, body
        self._first_seen.setdefault(job_id, time.monotonic())
        snapshot: dict[str, Any] = {
            "status": _map_status(body.get("status")),
            "cost_usd": self._derive_cost(job_id, body),
        }
        if body.get("error"):
            snapshot["error"] = str(body["error"])

        output = body.get("output")
        if isinstance(output, list) and output:
            output = output[0]
        if isinstance(output, dict):
            if output.get("error") and "error" not in snapshot:
                snapshot["error"] = str(output["error"])
            if isinstance(output.get("sha256"), str):
                snapshot["sha256"] = output["sha256"]
            url = _first_str(output, _URL_KEYS)
            if url and url.startswith(("http://", "https://")):
                snapshot["output_url"] = url
            else:
                inline = _first_str(output, _B64_KEYS) or (
                    url if url and not url.startswith("http") else None
                )
                if inline:
                    snapshot["output_url"] = self._stash_inline(job_id, inline)
        elif isinstance(output, str) and output:
            if output.startswith(("http://", "https://")):
                snapshot["output_url"] = output
            else:
                snapshot["output_url"] = self._stash_inline(job_id, output)
        return status, snapshot

    def _stash_inline(self, job_id: str, encoded: str) -> str:
        """Decode a base64 result and hand back a fetchable synthetic URL."""
        payload = encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded
        try:
            self._inline[job_id] = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise CloudRenderError(
                f"RunPod job {job_id} returned an output that is neither a URL "
                f"nor valid base64: {exc}"
            ) from exc
        return f"{_INLINE_SCHEME}:{job_id}"

    def _derive_cost(self, job_id: str, body: dict[str, Any]) -> float | None:
        """Dollars from RunPod's millisecond timings (estimated while running).

        RunPod bills GPU-seconds and reports no cost field, so without this the
        renderer's ``max_cost_usd`` ceiling could never trip on RunPod.
        """
        rate_per_hour = self._config.resolve_gpu_cost_per_hour()
        if not rate_per_hour:
            return None
        execution_ms = body.get("executionTime")
        delay_ms = body.get("delayTime") or 0
        if isinstance(execution_ms, int | float):
            billable_s = (float(execution_ms) + float(delay_ms)) / 1000.0
        else:  # still in flight — estimate from elapsed wall-clock
            started = self._first_seen.get(job_id)
            if started is None:
                return None
            billable_s = max(0.0, time.monotonic() - started)
        return round(billable_s * (rate_per_hour / 3600.0), 6)

    async def download(self, url: str, timeout: float) -> bytes:
        """Fetch a result URL, or return bytes already held inline."""
        if url.startswith(f"{_INLINE_SCHEME}:"):
            job_id = url.split(":", 1)[1]
            payload = self._inline.pop(job_id, None)
            if payload is None:
                raise CloudRenderError(
                    f"inline RunPod result for job {job_id} is no longer held"
                )
            return payload
        return await self._http.download(url, timeout)

    # --------------------------------------------------------------- health

    async def health(self, timeout: float = 15.0) -> dict[str, Any]:
        """``GET /health`` — worker and queue counts for the endpoint."""
        status, body = await self._http.request(
            "GET", f"{self._api_base()}/health", None, self._auth_headers({}), timeout
        )
        if not 200 <= status < 300:
            raise CloudRenderError(f"RunPod /health returned HTTP {status}: {body!r:.200}")
        return body


def _default_journal() -> Path:
    """Per-user submit journal path, outside any signed app bundle."""
    if os.uname().sysname == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "AkioStudio"
            / "runpod_submits.jsonl"
        )
    return Path.home() / ".akio_studio" / "runpod_submits.jsonl"


def _first_str(mapping: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """First non-empty string value among ``keys``."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _map_status(raw: Any) -> str:
    """RunPod status -> canonical status string."""
    text = str(raw or "").strip().upper()
    return {
        "IN_QUEUE": "queued",
        "IN_PROGRESS": "running",
        "COMPLETED": "succeeded",
        "FAILED": "failed",
        "CANCELLED": "canceled",
        "TIMED_OUT": "failed",
    }.get(text, "running")


def write_journal_summary(journal_path: Path, out_path: Path) -> Path:
    """Summarize the submit journal — used when reconciling an ambiguous run."""
    counts: dict[str, int] = {}
    ambiguous: list[dict[str, Any]] = []
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase = str(record.get("phase", "unknown"))
            counts[phase] = counts.get(phase, 0) + 1
            if phase == "ambiguous":
                ambiguous.append(record)
    return atomic_write_json(out_path, {"counts": counts, "ambiguous": ambiguous})
