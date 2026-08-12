"""Local unified-memory pool coordination for Akio Studio.

This module is the stage ledger for a 24 GiB Apple Silicon machine on which
the LLM (Ollama), image diffusion, video diffusion (ComfyUI) and edit stages
can never co-reside (audit A1/M2): :class:`LocalPoolCoordinator` enforces
one-resident-stage-at-a-time and gates every load against the usable-memory
budget from :mod:`akio_studio.config`.

Corrected semantics baked in from the architecture audit:

* **B3** — the purge talks to the processes that actually hold the memory
  (Ollama ``keep_alive: 0``, ComfyUI ``POST /free``); the orchestrator only
  runs ``gc.collect()``, touches ``torch.mps.empty_cache()`` strictly when
  torch is *already* imported, and never shells out to ``sync``.
* **B6** — the Ollama CLI fallback resolves absolute candidate paths and the
  ``AKIO_OLLAMA_BIN`` environment variable because Dock-launched apps do not
  inherit the shell ``PATH``.
* **M1** — the Ollama **HTTP API** is the primary interface (system turn,
  ``keep_alive`` and ``num_ctx`` control); the CLI is a reachability fallback
  only, fed via stdin with ``communicate()`` to avoid pipe-buffer deadlock.
* **M3** — retention timestamps are mapped to frame indices with exact
  ``Fraction(24000, 1001)`` arithmetic, never the rounded float 23.976.
* **A4** — DPO pairs are only formed between renders sharing an identical
  ``prompt_hash`` (identical conditioning); cross-prompt pairs would teach a
  confounded preference and are never emitted.
"""

from __future__ import annotations

import asyncio
import enum
import gc
import json
import logging
import os
import shutil
import sys
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from akio_studio._io import _flush_to_stable_storage
from akio_studio.config import MASTER_FPS, MODEL_FOOTPRINTS_GIB, StudioConfig
from akio_studio.exceptions import (
    MemoryBudgetExceededError,
    OllamaUnavailableError,
    PoolCoordinatorError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["DPOFeedbackLogger", "LocalPoolCoordinator", "Stage"]

logger = logging.getLogger(__name__)

#: Environment variable overriding the Ollama CLI location (audit B6: apps
#: launched from the Dock/Finder inherit a minimal PATH).
_OLLAMA_BIN_ENV = "AKIO_OLLAMA_BIN"

#: Short timeout for ``/api/ps`` liveness/verification polls — the configured
#: generation timeout (minutes) would stall the purge path.
_PS_TIMEOUT_S = 5.0

#: Unload verification: the audit requires polling, never assuming — Ollama's
#: unload is asynchronous.
_PURGE_VERIFY_ATTEMPTS = 5
_PURGE_VERIFY_SLEEP_S = 0.5


class Stage(enum.Enum):
    """A mutually exclusive residency stage of the local model pool.

    Footprints come from :data:`akio_studio.config.MODEL_FOOTPRINTS_GIB`;
    on the target machine no two heavy stages fit together, so the coordinator
    treats residency as exclusive (audit A1/M2).
    """

    LLM = "llm"
    IMAGE_DIFFUSION = "image_diffusion"
    VIDEO_DIFFUSION = "video_diffusion"
    EDIT = "edit"

    @property
    def footprint_gib(self) -> float:
        """Estimated resident footprint of this stage from the config ledger."""
        return MODEL_FOOTPRINTS_GIB[self.value]


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string for JSONL records."""
    return datetime.now(UTC).isoformat()


def _seconds_to_frame(seconds: float) -> int:
    """Map a timeline second to an exact master-timeline frame index.

    Audit M3: 23.976 fps is exactly ``24000/1001``; the conversion uses pure
    rational arithmetic (``Fraction(seconds)`` is the exact value of the
    float) so no accumulated rounding can mis-attribute a frame.
    """
    return round(Fraction(seconds) * MASTER_FPS)


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, dict[str, Any]]:
    """Blocking JSON-over-HTTP helper shared by every HTTP path.

    Callers on the event loop must wrap this in :func:`asyncio.to_thread`.
    Returns ``(status_code, parsed_body)``. An HTTP *error status* is returned
    rather than raised so callers can distinguish "reachable but unhappy"
    (no CLI fallback — it round-trips through the same server) from
    "unreachable" (``URLError``/``OSError`` propagate to the caller).
    """
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
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
        logger.debug("non-JSON response body from %s %s", method, url)
        return status, {}
    return status, parsed if isinstance(parsed, dict) else {"data": parsed}


@dataclass
class _ShotRender:
    """In-memory registry entry for one rendered shot variant."""

    asset_id: str
    prompt_hash: str
    seed: int
    start_second: float
    end_second: float
    start_frame: int
    end_frame: int
    comfy_params: dict[str, Any]
    mean_retention: float | None = None
    rejected: bool = False


class DPOFeedbackLogger:
    """Registry + append-only JSONL log turning retention data into DPO pairs.

    Every mutation is recorded twice: in an in-memory registry (for pairing
    lookups) and as a single JSONL line appended, flushed and ``fsync``-ed to
    ``log_path`` — the log is training data and must survive a crash.

    Audit A4: a DPO pair is only ever formed between two renders of the same
    shot — identical ``prompt_hash`` (identical conditioning), different seed.
    Pairing across prompts would teach content preference, not visual quality.
    Audit M3: seconds are converted to frame indices with exact
    ``Fraction(24000, 1001)`` arithmetic; both representations are persisted.

    Local scope is *pair logging only* — on-device DPO training does not fit
    the 24 GiB pool (audit A4); the JSONL feeds an offline/cloud stage.
    """

    def __init__(self, log_path: Path) -> None:
        """Create the logger, ensuring ``log_path``'s parent directory exists."""
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: dict[tuple[str, str, int], _ShotRender] = {}

    @property
    def log_path(self) -> Path:
        """Path of the append-only JSONL feedback log."""
        return self._log_path

    def _append(self, record: dict[str, Any]) -> None:
        """Append one JSONL line: open in append mode, write, flush, fsync.

        A single sub-pipe-buffer ``write()`` in append mode is atomic for a
        single-writer log; the whole-file atomic writers in ``_io`` are not
        used because they would rewrite the entire log per line.
        """
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with open(self._log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            # Training data must survive a crash: on Darwin plain fsync stops
            # at the drive's write cache, so use the shared F_FULLFSYNC helper.
            _flush_to_stable_storage(handle.fileno())

    def register_shot_render(
        self,
        asset_id: str,
        prompt_hash: str,
        seed: int,
        start_second: float,
        end_second: float,
        comfy_params: dict[str, Any],
    ) -> None:
        """Record a rendered shot variant in the registry and the JSONL log.

        ``start_second``/``end_second`` are stored as given *and* as exact
        frame indices (``round(seconds * 24000/1001)`` in rational arithmetic)
        so a retention drop-off second can later be mapped to the responsible
        shot without float drift (audit M3). ``comfy_params`` must be
        JSON-serializable.
        """
        if end_second <= start_second:
            raise PoolCoordinatorError(
                f"shot for asset {asset_id!r} has end_second {end_second} <= "
                f"start_second {start_second}"
            )
        render = _ShotRender(
            asset_id=asset_id,
            prompt_hash=prompt_hash,
            seed=seed,
            start_second=start_second,
            end_second=end_second,
            start_frame=_seconds_to_frame(start_second),
            end_frame=_seconds_to_frame(end_second),
            comfy_params=dict(comfy_params),
        )
        key = (asset_id, prompt_hash, seed)
        previous = self._registry.get(key)
        if previous is not None:
            # Same conditioning + same seed is the same deterministic output:
            # carry recorded retention/rejection forward instead of silently
            # discarding an observation that already cost a platform upload.
            render.mean_retention = previous.mean_retention
            render.rejected = previous.rejected
            logger.warning(
                "re-registering shot render %s; timing/params refreshed, "
                "recorded retention preserved",
                key,
            )
        self._registry[key] = render
        self._append(
            {
                "type": "shot_render",
                "asset_id": asset_id,
                "prompt_hash": prompt_hash,
                "seed": seed,
                "start_second": start_second,
                "end_second": end_second,
                "start_frame": render.start_frame,
                "end_frame": render.end_frame,
                "comfy_params": render.comfy_params,
                "ts": _utc_now_iso(),
            }
        )

    def record_retention_result(
        self,
        asset_id: str,
        prompt_hash: str,
        seed: int,
        mean_retention: float,
    ) -> None:
        """Attach an observed mean retention to a registered render.

        Raises :class:`PoolCoordinatorError` when no render was registered
        under ``(asset_id, prompt_hash, seed)`` — retention for an unknown
        render cannot participate in pairing and indicates a pipeline bug.
        """
        key = (asset_id, prompt_hash, seed)
        render = self._registry.get(key)
        if render is None:
            raise PoolCoordinatorError(
                f"retention result for unregistered render {key}"
            )
        render.mean_retention = mean_retention
        self._append(
            {
                "type": "retention_result",
                "asset_id": asset_id,
                "prompt_hash": prompt_hash,
                "seed": seed,
                "mean_retention": mean_retention,
                "ts": _utc_now_iso(),
            }
        )

    def _shot_covering_frame(self, asset_id: str, frame: int) -> _ShotRender:
        """Return the registered shot of ``asset_id`` covering ``frame``.

        Coverage is half-open ``[start_frame, end_frame)``; a drop-off landing
        exactly on a shot's final boundary frame (e.g. the asset's last frame)
        falls back to inclusive-end matching. Among multiple covering variants
        the latest-starting one is chosen deterministically.
        """
        candidates = [r for r in self._registry.values() if r.asset_id == asset_id]
        matches = [r for r in candidates if r.start_frame <= frame < r.end_frame]
        if not matches:
            matches = [r for r in candidates if frame == r.end_frame]
        if not matches:
            raise PoolCoordinatorError(
                f"drop-off frame {frame} maps to no registered shot of asset "
                f"{asset_id!r}"
            )
        if len(matches) > 1:
            logger.warning(
                "%d registered shots of asset %r cover frame %d; using the "
                "latest-starting one",
                len(matches),
                asset_id,
                frame,
            )
        return max(matches, key=lambda r: r.start_frame)

    def log_dpo_latent_feedback(
        self,
        asset_id: str,
        retention_dropoff_second: float,
        comfy_params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Turn a retention drop-off into a DPO pair, or an unpaired record.

        The drop-off second is converted to an exact frame index (audit M3)
        and mapped to the registered shot of ``asset_id`` covering that frame;
        that render's ``(prompt_hash, seed)`` is marked REJECTED. The registry
        is then searched for the best CHOSEN counterpart: **same**
        ``prompt_hash``, different seed, highest recorded mean retention.
        Audit A4: identical conditioning is mandatory — a pair is *never*
        formed across different prompt hashes, because it would confound
        content preference with visual quality.

        Returns the appended ``{"type": "dpo_pair", ...}`` record when a valid
        counterpart exists. Otherwise appends a ``{"type": "dpo_unpaired",
        ...}`` record — still useful later for KTO-style *unpaired* preference
        objectives, which consume lone negatives — and returns ``None``.

        Raises :class:`PoolCoordinatorError` when the drop-off frame maps to
        no registered shot of the asset.
        """
        dropoff_frame = _seconds_to_frame(retention_dropoff_second)
        rejected = self._shot_covering_frame(asset_id, dropoff_frame)
        rejected.rejected = True

        chosen: _ShotRender | None = None
        for render in self._registry.values():
            if render.prompt_hash != rejected.prompt_hash:
                continue  # audit A4: never pair across conditioning
            if render.seed == rejected.seed or render.rejected:
                continue
            if render.mean_retention is None:
                continue
            # A "chosen" that retained *worse* than the rejected render would
            # invert the preference signal — require strictly better retention
            # whenever the rejected render's own retention is known.
            if (
                rejected.mean_retention is not None
                and render.mean_retention <= rejected.mean_retention
            ):
                continue
            if chosen is None or render.mean_retention > (chosen.mean_retention or 0.0):
                chosen = render

        rejected_payload: dict[str, Any] = {
            "seed": rejected.seed,
            "params": dict(comfy_params),
            "dropoff_frame": dropoff_frame,
            "dropoff_second": retention_dropoff_second,
        }
        if chosen is not None:
            record: dict[str, Any] = {
                "type": "dpo_pair",
                "asset_id": asset_id,
                "prompt_hash": rejected.prompt_hash,
                "chosen": {
                    "asset_id": chosen.asset_id,
                    "seed": chosen.seed,
                    "params": dict(chosen.comfy_params),
                    "mean_retention": chosen.mean_retention,
                },
                "rejected": rejected_payload,
                "ts": _utc_now_iso(),
            }
            self._append(record)
            logger.info(
                "DPO pair logged for prompt %s: chosen seed %d vs rejected seed %d",
                rejected.prompt_hash,
                chosen.seed,
                rejected.seed,
            )
            return record

        self._append(
            {
                "type": "dpo_unpaired",
                "asset_id": asset_id,
                "prompt_hash": rejected.prompt_hash,
                "rejected": rejected_payload,
                "reason": "no chosen counterpart with identical conditioning",
                "ts": _utc_now_iso(),
            }
        )
        logger.info(
            "no identically-conditioned counterpart for prompt %s; logged "
            "unpaired rejection (KTO-usable)",
            rejected.prompt_hash,
        )
        return None


class LocalPoolCoordinator:
    """Stage ledger and per-engine purge orchestrator for the unified pool.

    Mutual exclusion is *the* invariant (audit A1/M2): at most one
    :class:`Stage` is resident at a time, and every acquisition is gated
    against ``StudioConfig.usable_memory_gib``. Purges act on the processes
    that actually hold the memory — the Ollama and ComfyUI servers — never on
    the orchestrator's own heap alone (audit B3), and Ollama unloads are
    *verified* by polling ``/api/ps`` rather than assumed.
    """

    def __init__(
        self,
        config: StudioConfig | None = None,
        auto_evict: bool = True,
        dpo_log_path: Path | None = None,
    ) -> None:
        """Create the coordinator.

        ``auto_evict=True`` makes :meth:`acquire_stage` purge a conflicting
        resident stage instead of raising. When ``dpo_log_path`` is given a
        :class:`DPOFeedbackLogger` is exposed as ``.dpo``.
        """
        self._config = config if config is not None else StudioConfig()
        self._auto_evict = auto_evict
        self._resident: Stage | None = None
        self._lock = asyncio.Lock()
        self.dpo: DPOFeedbackLogger | None = (
            DPOFeedbackLogger(dpo_log_path) if dpo_log_path is not None else None
        )

    @property
    def resident_stage(self) -> Stage | None:
        """The currently resident stage, or ``None`` when the pool is idle."""
        return self._resident

    async def acquire_stage(self, stage: Stage) -> None:
        """Make ``stage`` the resident stage, enforcing the memory budget.

        Raises :class:`MemoryBudgetExceededError` when the stage's footprint
        exceeds the usable pool, or when another stage is resident and
        ``auto_evict`` is disabled. With ``auto_evict`` the conflicting stage
        is released (its engine purged and verified) first. Re-acquiring the
        already-resident stage is a no-op.
        """
        async with self._lock:
            await self._acquire_locked(stage)

    async def _acquire_locked(self, stage: Stage) -> None:
        """Budget-gate and make ``stage`` resident; caller holds the lock."""
        footprint = stage.footprint_gib
        usable = self._config.usable_memory_gib
        if footprint > usable:
            raise MemoryBudgetExceededError(
                f"stage {stage.value!r} needs ~{footprint} GiB but only "
                f"{usable} GiB of the unified pool is usable"
            )
        if self._resident is not None and self._resident is not stage:
            if not self._auto_evict:
                raise MemoryBudgetExceededError(
                    f"cannot acquire stage {stage.value!r}: stage "
                    f"{self._resident.value!r} is resident and auto_evict "
                    "is disabled"
                )
            logger.info(
                "auto-evicting resident stage %r for %r",
                self._resident.value,
                stage.value,
            )
            evicted = self._resident
            released_ok = await self._release_locked(evicted)
            if not released_ok:
                # Acquiring anyway would overlap two heavy stages in the pool
                # (audit A1) — refuse rather than proceed into swap thrash.
                raise MemoryBudgetExceededError(
                    f"cannot acquire stage {stage.value!r}: eviction of "
                    f"{evicted.value!r} could not be verified (its engine may "
                    "still hold the model)"
                )
        self._resident = stage
        logger.info("stage %r resident (~%s GiB)", stage.value, footprint)

    @asynccontextmanager
    async def stage(self, stage: Stage) -> AsyncIterator[None]:
        """``async with coordinator.stage(Stage.LLM):`` acquire/release scope.

        The coordinator lock is held for the *entire* scope, so a concurrent
        task cannot auto-evict this stage's engine mid-generation; concurrent
        ``stage()`` users serialize instead.
        """
        async with self._lock:
            await self._acquire_locked(stage)
            try:
                yield
            finally:
                await self._release_locked(stage)

    async def release_stage(self, stage: Stage) -> None:
        """Release ``stage``: purge its engine, clear residency, collect."""
        async with self._lock:
            await self._release_locked(stage)

    async def _release_locked(self, stage: Stage) -> bool:
        """Purge ``stage``'s engine and clear residency; caller holds the lock.

        Returns ``False`` when the engine was reachable but its unload could
        not be verified — i.e. the memory may genuinely still be occupied.
        An unreachable engine holds nothing and counts as released.
        """
        released_ok = True
        if stage is Stage.LLM:
            status, verified = await self._purge_ollama()
            if status == "unloaded" and not verified:
                released_ok = False
                logger.warning(
                    "ollama unload not verified (status=%s); the model may "
                    "still be resident",
                    status,
                )
        elif stage in (Stage.IMAGE_DIFFUSION, Stage.VIDEO_DIFFUSION):
            comfy_status = await self._purge_comfyui()
            logger.info("comfyui purge on release of %r: %s", stage.value, comfy_status)
        else:  # Stage.EDIT holds no external server-side model
            logger.debug("stage %r has no external engine to purge", stage.value)
        if self._resident is stage or self._resident is None:
            self._resident = None
        else:
            logger.warning(
                "released stage %r while %r was resident; residency unchanged",
                stage.value,
                self._resident.value,
            )
        collected = gc.collect()
        logger.debug("released stage %r; gc collected %d objects", stage.value, collected)
        return released_ok

    # ------------------------------------------------------------------ Ollama

    def _locate_ollama_cli(self) -> str | None:
        """Resolve the Ollama CLI binary (audit B6: minimal Dock PATH).

        Checks ``AKIO_OLLAMA_BIN`` first, then the configured candidates —
        absolute paths by existence, bare names via :func:`shutil.which`.
        """
        env_bin = os.environ.get(_OLLAMA_BIN_ENV)
        candidates: tuple[str, ...] = self._config.ollama.cli_candidates
        if env_bin:
            candidates = (env_bin, *candidates)
        for candidate in candidates:
            path = Path(candidate)
            if path.is_absolute():
                if path.exists() and os.access(path, os.X_OK):
                    return str(path)
                continue
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None

    async def _ollama_ps(self) -> list[str] | None:
        """Names of models resident in the Ollama server; ``None`` if unreachable."""
        cfg = self._config.ollama
        try:
            status, body = await asyncio.to_thread(
                _http_json, "GET", f"{cfg.host}/api/ps", None, _PS_TIMEOUT_S
            )
        except OSError as exc:  # URLError is an OSError subclass
            logger.debug("ollama /api/ps unreachable: %s", exc)
            return None
        if status != 200:
            logger.debug("ollama /api/ps returned HTTP %s", status)
            return None
        models = body.get("models", [])
        if not isinstance(models, list):
            return []
        return [m["name"] for m in models if isinstance(m, dict) and "name" in m]

    async def ollama_loaded_models(self) -> list[str]:
        """Models currently loaded in the Ollama server; ``[]`` if unreachable."""
        return await self._ollama_ps() or []

    async def run_ollama_qwen(self, prompt: str, system_prompt: str | None = None) -> str:
        """Run one generation on the configured Ollama model.

        Primary path (audit M1): ``POST /api/chat`` with an explicit
        ``keep_alive`` and ``num_ctx``, run in a worker thread. The CLI
        fallback is used only when the server is *unreachable*
        (``URLError``/``OSError``) — an HTTP error status means the server is
        up and the CLI would round-trip into the same failure.

        Raises :class:`OllamaUnavailableError` when both paths fail.
        """
        cfg = self._config.ollama
        if self._resident is not Stage.LLM:
            logger.warning(
                "run_ollama_qwen called while resident stage is %s; acquire "
                "Stage.LLM first to keep the ledger honest",
                self._resident.value if self._resident else "<idle>",
            )
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "stream": False,
            "keep_alive": cfg.keep_alive_active,
            "options": {"num_ctx": cfg.num_ctx},
        }
        try:
            status, body = await asyncio.to_thread(
                _http_json, "POST", f"{cfg.host}/api/chat", payload, cfg.request_timeout_s
            )
        except OSError as exc:  # URLError is an OSError subclass (audit B6/M1)
            # A read timeout means the server is up but the generation is
            # slow — falling back to the CLI would spawn a DUPLICATE
            # generation against the same busy server. Only a genuine
            # connection failure warrants the fallback.
            reason = getattr(exc, "reason", None)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                raise OllamaUnavailableError(
                    f"ollama /api/chat timed out after {cfg.request_timeout_s}s "
                    "(server reachable — not retrying via CLI)"
                ) from exc
            logger.warning("ollama HTTP API unreachable (%s); falling back to CLI", exc)
            return await self._run_ollama_cli(prompt, system_prompt)
        if status != 200:
            raise OllamaUnavailableError(
                f"ollama /api/chat returned HTTP {status}: {body!r:.300}"
            )
        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaUnavailableError(
                "ollama /api/chat response is missing message.content"
            )
        return content

    async def _run_ollama_cli(self, prompt: str, system_prompt: str | None) -> str:
        """CLI fallback: ``ollama run <model>`` with the prompt fed via stdin.

        Stdin + ``communicate()`` avoids both ARG_MAX/quoting limits of argv
        and the 64 KiB pipe-buffer deadlock of ``proc.wait()`` with attached
        pipes (audit M1). The CLI cannot send a true system turn, so a given
        ``system_prompt`` is prepended with a delimiter (and a warning).
        """
        cfg = self._config.ollama
        binary = self._locate_ollama_cli()
        if binary is None:
            raise OllamaUnavailableError(
                "ollama HTTP API unreachable and no CLI binary found via "
                f"${_OLLAMA_BIN_ENV} or candidates {cfg.cli_candidates}"
            )
        if system_prompt is not None:
            logger.warning(
                "ollama CLI cannot send a true system turn; prepending the "
                "system prompt to the user prompt with a delimiter"
            )
            prompt = (
                "[SYSTEM INSTRUCTIONS]\n"
                f"{system_prompt}\n"
                "[/SYSTEM INSTRUCTIONS]\n\n"
                f"{prompt}"
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                "run",
                cfg.model,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise OllamaUnavailableError(f"failed to spawn {binary!r}: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=cfg.request_timeout_s,
            )
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise OllamaUnavailableError(
                f"ollama CLI timed out after {cfg.request_timeout_s}s"
            ) from exc
        except asyncio.CancelledError:
            # The awaiting task was cancelled: without this, the spawned
            # `ollama run` (holding the ~10 GiB model) would keep running.
            proc.kill()
            await proc.wait()
            raise
        if proc.returncode != 0:
            raise OllamaUnavailableError(
                f"ollama CLI exited {proc.returncode}: "
                f"{stderr.decode('utf-8', errors='replace')[:500]}"
            )
        return stdout.decode("utf-8", errors="replace")

    async def _ollama_cli_stop(self) -> bool:
        """Best-effort ``ollama stop <model>``; ``True`` on exit code 0."""
        binary = self._locate_ollama_cli()
        if binary is None:
            logger.warning("no ollama CLI available for 'stop' fallback")
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                "stop",
                self._config.ollama.model,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=30.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return False
        except OSError as exc:
            logger.warning("ollama CLI stop failed to spawn: %s", exc)
            return False
        return proc.returncode == 0

    async def _purge_ollama(self) -> tuple[str, bool]:
        """Unload the LLM from the Ollama server process and verify.

        Returns ``(status, verified)`` where status is ``"unloaded"``,
        ``"already-idle"`` or ``"unreachable"``. Verification polls
        ``/api/ps`` until the model list is empty (audit A1: unload is
        asynchronous — never assume).
        """
        cfg = self._config.ollama
        loaded = await self._ollama_ps()
        if loaded is None:
            logger.info("ollama unreachable during purge; nothing to unload")
            return "unreachable", False
        if not loaded:
            return "already-idle", True
        # Unload every model the server reports, not just the configured one —
        # a stray second model would otherwise survive the purge while the
        # report claimed the pool was drained (audit B3/A1).
        for model_name in loaded:
            try:
                status, _ = await asyncio.to_thread(
                    _http_json,
                    "POST",
                    f"{cfg.host}/api/generate",
                    {"model": model_name, "keep_alive": 0},
                    cfg.request_timeout_s,
                )
                if status != 200:
                    logger.warning(
                        "ollama unload of %r returned HTTP %s", model_name, status
                    )
            except OSError as exc:
                logger.warning(
                    "ollama HTTP unload of %r failed (%s); trying CLI stop",
                    model_name,
                    exc,
                )
                await self._ollama_cli_stop()
                break
        verified = False
        for attempt in range(_PURGE_VERIFY_ATTEMPTS):
            loaded = await self._ollama_ps()
            if loaded == []:
                verified = True
                break
            if attempt < _PURGE_VERIFY_ATTEMPTS - 1:
                await asyncio.sleep(_PURGE_VERIFY_SLEEP_S)
        return "unloaded", verified

    # ----------------------------------------------------------------- ComfyUI

    async def _purge_comfyui(self) -> str:
        """Free ComfyUI's models and memory via ``POST /free``.

        Returns ``"freed"`` or ``"unreachable"``; an offline ComfyUI is
        logged, never raised — it holds no memory when it is not running.
        """
        cfg = self._config.comfyui
        try:
            status, _ = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{cfg.host}/free",
                {"unload_models": True, "free_memory": True},
                cfg.request_timeout_s,
            )
        except OSError as exc:
            logger.info("comfyui unreachable during purge: %s", exc)
            return "unreachable"
        if status != 200:
            logger.warning("comfyui /free returned HTTP %s", status)
            return "unreachable"
        return "freed"

    # ------------------------------------------------------------- Full purge

    def _purge_mps_cache(self) -> str:
        """Clear the MPS allocator cache only when torch is already resident.

        Audit B3: importing torch just to free memory would *cost* ~0.5+ GiB;
        when ``"torch" not in sys.modules`` this is a guaranteed no-op.
        """
        if "torch" not in sys.modules:
            return "not-loaded"
        import torch  # already imported — this is a sys.modules lookup, not a load

        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                return "cleared"
        except Exception:
            logger.exception("torch.mps.empty_cache() failed")
        return "not-loaded"

    async def purge_unified_memory_pool(self) -> dict[str, Any]:
        """Staged purge of every engine holding unified memory.

        The spec called this a purge of "the unified memory pool" from inside
        the orchestrator; the name is kept for spec compatibility, but the
        corrected semantics (audit B3) purge *where the memory actually is*:
        the Ollama server (``keep_alive: 0``, verified via ``/api/ps``), the
        ComfyUI server (``POST /free``), and only then the orchestrator's own
        heap (``gc.collect()`` plus ``torch.mps.empty_cache()`` strictly when
        torch was already imported). ``sync`` is deliberately absent — it
        flushes disk write buffers and frees zero RAM.

        Each step is failure-isolated; the returned report has the shape
        ``{"ollama": "unloaded|already-idle|unreachable", "ollama_verified":
        bool, "comfyui": "freed|unreachable", "orchestrator_gc": int,
        "mps_cache": "cleared|not-loaded"}``. Residency is cleared.
        """
        async with self._lock:
            report: dict[str, Any] = {}
            ollama_status, ollama_verified = await self._purge_ollama()
            report["ollama"] = ollama_status
            report["ollama_verified"] = ollama_verified
            report["comfyui"] = await self._purge_comfyui()
            report["orchestrator_gc"] = gc.collect()
            report["mps_cache"] = self._purge_mps_cache()
            if self._resident is not None:
                logger.info(
                    "full purge clears resident stage %r", self._resident.value
                )
                self._resident = None
            logger.info("unified pool purge report: %s", report)
            return report
