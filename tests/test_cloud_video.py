"""Tests for the remote (cloud GPU) video stage."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest

from akio_studio.cloud_video import (
    CloudVideoRenderer,
    HttpCloudTransport,
    JobStatus,
    MockCloudTransport,
    VideoRenderRequest,
    build_transport,
)
from akio_studio.config import CloudVideoConfig, StudioConfig
from akio_studio.exceptions import (
    CloudAuthError,
    CloudBudgetExceededError,
    CloudJobFailedError,
    CloudRenderError,
    CloudTimeoutError,
)
from akio_studio.pool_coordinator import LocalPoolCoordinator, Stage


def _config(**cloud_kwargs: Any) -> StudioConfig:
    """StudioConfig with fast, test-sized cloud timings."""
    defaults: dict[str, Any] = {
        "poll_interval_s": 0.001,
        "poll_interval_max_s": 0.002,
        "max_wait_s": 5.0,
        "max_attempts": 2,
    }
    defaults.update(cloud_kwargs)
    return StudioConfig(cloud_video=CloudVideoConfig(**defaults))


def _request(shot_id: str = "shot001.mp4", seed: int = 7) -> VideoRenderRequest:
    return VideoRenderRequest(
        shot_id=shot_id, prompt="a ronin draws a blade", seed=seed, prompt_hash="abc123"
    )


# ------------------------------------------------------------------- status


def test_job_status_parses_provider_dialects() -> None:
    assert JobStatus.parse("COMPLETED") is JobStatus.SUCCEEDED
    assert JobStatus.parse("in_progress") is JobStatus.RUNNING
    assert JobStatus.parse("cancelled") is JobStatus.CANCELED
    # An unknown in-flight state must not abort a job that is merely running.
    assert JobStatus.parse("provisioning") is JobStatus.RUNNING
    assert JobStatus.SUCCEEDED.is_terminal and not JobStatus.QUEUED.is_terminal


# ------------------------------------------------------------------ request


def test_request_rejects_waste_before_paying() -> None:
    with pytest.raises(CloudRenderError):
        VideoRenderRequest(shot_id="", prompt="x", seed=1)
    with pytest.raises(CloudRenderError):
        VideoRenderRequest(shot_id="s.mp4", prompt="   ", seed=1)
    with pytest.raises(CloudRenderError):
        VideoRenderRequest(shot_id="s.mp4", prompt="x", seed=1, num_frames=0)


def test_idempotency_key_is_stable_and_content_sensitive() -> None:
    a = _request()
    b = _request()
    assert a.idempotency_key("m") == b.idempotency_key("m")
    assert a.idempotency_key("m") != _request(seed=8).idempotency_key("m")
    assert a.idempotency_key("m") != a.idempotency_key("other-model")


async def test_retried_submission_does_not_double_bill() -> None:
    """The idempotency key must collapse a duplicate submit into one job."""
    transport = MockCloudTransport()
    renderer = CloudVideoRenderer(_config(), transport=transport)
    first = await renderer.submit(_request())
    second = await renderer.submit(_request())  # identical -> same job
    assert first == second
    assert len(transport.submissions) == 1


# ----------------------------------------------------------------- rendering


async def test_render_shot_round_trip(tmp_path) -> None:
    payload = b"MP4-BYTES-XYZ"
    transport = MockCloudTransport(payload_bytes=payload, cost_usd=1.25)
    renderer = CloudVideoRenderer(_config(), transport=transport)

    result = await renderer.render_shot(_request(), tmp_path)

    assert result.status is JobStatus.SUCCEEDED
    assert result.output_path is not None
    assert result.output_path.read_bytes() == payload
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.cost_usd == 1.25
    # Metadata must never carry credential material.
    assert "token" not in str(result.to_metadata()).lower()


async def test_checksum_mismatch_is_rejected(tmp_path) -> None:
    class LyingTransport(MockCloudTransport):
        async def download(self, url: str, timeout: float) -> bytes:
            return b"TRUNCATED"  # does not match the advertised digest

    renderer = CloudVideoRenderer(_config(), transport=LyingTransport())
    with pytest.raises(CloudRenderError, match="checksum mismatch"):
        await renderer.render_shot(_request(), tmp_path)


async def test_failed_job_raises(tmp_path) -> None:
    renderer = CloudVideoRenderer(_config(), transport=MockCloudTransport(fail_job=True))
    with pytest.raises(CloudJobFailedError):
        await renderer.render_shot(_request(), tmp_path)


async def test_out_of_gate_denoise_never_reaches_the_gpu(tmp_path) -> None:
    transport = MockCloudTransport()
    renderer = CloudVideoRenderer(_config(), transport=transport)
    bad = VideoRenderRequest(shot_id="s.mp4", prompt="x", seed=1, denoise=0.9)
    with pytest.raises(ValueError):
        await renderer.render_shot(bad, tmp_path)
    assert transport.submissions == []  # nothing was submitted, nothing billed


async def test_batch_renders_concurrently_and_isolates_failures(tmp_path) -> None:
    class OneBadShot(MockCloudTransport):
        async def request(self, method, url, payload, headers, timeout):
            status, body = await super().request(method, url, payload, headers, timeout)
            if method == "GET" and body.get("status") == "succeeded":
                job = url.rsplit("/", 1)[-1]
                if job.endswith("2"):
                    return status, {"status": "failed", "error": "boom"}
            return status, body

    renderer = CloudVideoRenderer(_config(), transport=OneBadShot())
    results = await renderer.render_batch(
        [_request("a.mp4", 1), _request("b.mp4", 2)], tmp_path
    )
    assert len(results) == 2
    # A sibling failure must not cancel the shot that already succeeded.
    assert sum(1 for r in results if isinstance(r, BaseException)) == 1
    assert sum(1 for r in results if not isinstance(r, BaseException)) == 1


# ------------------------------------------------------- spend/abort control


async def test_cost_ceiling_cancels_the_job(tmp_path) -> None:
    transport = MockCloudTransport(cost_usd=99.0)
    renderer = CloudVideoRenderer(_config(max_cost_usd=1.0), transport=transport)
    with pytest.raises(CloudBudgetExceededError):
        await renderer.render_shot(_request(), tmp_path)
    assert transport.cancelled, "an over-budget job must be cancelled remotely"


async def test_timeout_cancels_the_job(tmp_path) -> None:
    transport = MockCloudTransport(running_polls=10_000)  # never finishes
    renderer = CloudVideoRenderer(_config(max_wait_s=0.05), transport=transport)
    with pytest.raises(CloudTimeoutError):
        await renderer.render_shot(_request(), tmp_path)
    assert transport.cancelled, "a timed-out job must be cancelled remotely"


async def test_task_cancellation_cancels_the_job(tmp_path) -> None:
    """An abandoned GPU job keeps billing — cancellation must propagate."""
    transport = MockCloudTransport(running_polls=10_000)
    renderer = CloudVideoRenderer(_config(max_wait_s=30.0), transport=transport)
    task = asyncio.create_task(renderer.render_shot(_request(), tmp_path))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert transport.cancelled


# ----------------------------------------------------------------- security


async def test_plaintext_endpoint_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("AKIO_CLOUD_VIDEO_ENDPOINT", "http://gpu.example.com/v1")
    monkeypatch.setenv("AKIO_CLOUD_VIDEO_TOKEN", "secret")
    renderer = CloudVideoRenderer(_config(provider="generic"))  # real transport
    with pytest.raises(CloudAuthError, match="https"):
        await renderer.submit(_request())


async def test_missing_credentials_raise_before_any_request(monkeypatch) -> None:
    monkeypatch.delenv("AKIO_CLOUD_VIDEO_TOKEN", raising=False)
    monkeypatch.setenv("AKIO_CLOUD_VIDEO_ENDPOINT", "https://gpu.example.com/v1")
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    # Explicit real transport: the factory would otherwise hand back the mock,
    # which is the documented offline fallback (covered separately below).
    renderer = CloudVideoRenderer(
        _config(provider="generic"), transport=HttpCloudTransport()
    )
    with pytest.raises(CloudAuthError):
        await renderer.submit(_request())


def test_unconfigured_provider_falls_back_to_mock(monkeypatch) -> None:
    """No credentials must mean 'render nothing, bill nothing' — not a crash."""
    for var in (
        "AKIO_CLOUD_VIDEO_TOKEN",
        "AKIO_CLOUD_VIDEO_ENDPOINT",
        "AKIO_RUNPOD_ENDPOINT_ID",
        "RUNPOD_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(build_transport(_config()), MockCloudTransport)


def test_configured_runpod_selects_the_runpod_adapter(monkeypatch) -> None:
    from akio_studio.runpod_transport import RunPodTransport

    monkeypatch.setenv("AKIO_RUNPOD_ENDPOINT_ID", "abc123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    transport = build_transport(_config(provider="runpod"))
    assert isinstance(transport, RunPodTransport)
    # RunPod cannot deduplicate submissions, so they must not be retried.
    assert transport.retries_submits is False


async def test_auth_failure_is_not_retried() -> None:
    calls = {"n": 0}

    class Denying(MockCloudTransport):
        async def request(self, method, url, payload, headers, timeout):
            calls["n"] += 1
            return 401, {"error": "bad token"}

    renderer = CloudVideoRenderer(_config(max_attempts=4), transport=Denying())
    with pytest.raises(CloudAuthError):
        await renderer.submit(_request())
    assert calls["n"] == 1, "a bad credential must fail fast, not burn retries"


# --------------------------------------------------- coordinator integration


async def test_remote_video_stage_does_not_evict_local_stages() -> None:
    """The point of moving video off-device: local work is left alone."""
    coordinator = LocalPoolCoordinator(_config(), auto_evict=False)
    await coordinator.acquire_stage(Stage.LLM)
    # With a local video backend this would raise (budget/exclusivity)...
    await coordinator.acquire_stage(Stage.VIDEO_DIFFUSION)
    assert coordinator.resident_stage is Stage.LLM
    await coordinator.release_stage(Stage.VIDEO_DIFFUSION)
    assert coordinator.resident_stage is Stage.LLM


async def test_local_video_backend_still_enforces_exclusivity() -> None:
    config = StudioConfig(video_backend="local")
    coordinator = LocalPoolCoordinator(config, auto_evict=False)
    await coordinator.acquire_stage(Stage.LLM)
    from akio_studio.exceptions import MemoryBudgetExceededError

    with pytest.raises(MemoryBudgetExceededError):
        await coordinator.acquire_stage(Stage.VIDEO_DIFFUSION)


def test_backend_selector_is_validated() -> None:
    with pytest.raises(ValueError):
        StudioConfig(video_backend="gpu-farm")
    assert StudioConfig().video_model_id == CloudVideoConfig().model
    assert StudioConfig(video_backend="local").video_model_id == "wan2.1-t2v-1.3b"
