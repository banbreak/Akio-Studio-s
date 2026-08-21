"""Tests for the RunPod Serverless adapter."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import pytest

from akio_studio.cloud_video import CloudVideoRenderer, VideoRenderRequest
from akio_studio.config import CloudVideoConfig, StudioConfig
from akio_studio.exceptions import (
    CloudAuthError,
    CloudRenderError,
    CloudSubmitAmbiguousError,
)
from akio_studio.runpod_transport import RunPodTransport


class FakeRunPodAPI:
    """Simulates RunPod Serverless v2: /run, /status/{id}, /cancel/{id}."""

    def __init__(
        self,
        output: Any = None,
        status_sequence: list[str] | None = None,
        execution_ms: int = 30_000,
        fail_submit: bool = False,
    ) -> None:
        self.output = output
        self.status_sequence = status_sequence or ["IN_QUEUE", "COMPLETED"]
        self.execution_ms = execution_ms
        self.fail_submit = fail_submit
        self.calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str]]] = []
        self._polls = 0

    async def request(self, method, url, payload, headers, timeout):
        self.calls.append((method, url, payload, headers))
        if self.fail_submit and url.endswith("/run"):
            raise OSError("connection reset by peer")
        if url.endswith("/run"):
            return 200, {"id": "rp-job-1", "status": "IN_QUEUE"}
        if "/cancel/" in url:
            return 200, {"id": "rp-job-1", "status": "CANCELLED"}
        if "/status/" in url:
            index = min(self._polls, len(self.status_sequence) - 1)
            state = self.status_sequence[index]
            self._polls += 1
            body: dict[str, Any] = {"id": "rp-job-1", "status": state}
            if state == "COMPLETED":
                body["output"] = self.output
                body["executionTime"] = self.execution_ms
                body["delayTime"] = 500
            if state == "FAILED":
                body["error"] = "worker raised"
            return 200, body
        if url.endswith("/health"):
            return 200, {"workers": {"ready": 1}, "jobs": {"inQueue": 0}}
        return 404, {}

    async def download(self, url, timeout):
        return b"HTTP-FETCHED-VIDEO"


def _cfg(**kw: Any) -> CloudVideoConfig:
    base: dict[str, Any] = {
        "provider": "runpod",
        "runpod_endpoint_id": "ep-test",
        "poll_interval_s": 0.001,
        "poll_interval_max_s": 0.002,
        "max_wait_s": 5.0,
        "runpod_gpu_cost_per_hour": 3.6,  # $0.001/second — round numbers
    }
    base.update(kw)
    return CloudVideoConfig(**base)


def _renderer(api: FakeRunPodAPI, tmp_path, **kw: Any) -> CloudVideoRenderer:
    cloud = _cfg(**kw)
    transport = RunPodTransport(cloud, journal_path=tmp_path / "submits.jsonl", http=api)
    return CloudVideoRenderer(StudioConfig(cloud_video=cloud), transport=transport)


def _request() -> VideoRenderRequest:
    return VideoRenderRequest(
        shot_id="s1.mp4", prompt="ronin draws a blade", seed=41, prompt_hash="h1"
    )


# ------------------------------------------------------------------ mapping


async def test_canonical_calls_map_onto_runpod_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    payload = b"MP4"
    api = FakeRunPodAPI(
        output={
            "video_base64": base64.b64encode(payload).decode(),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    result = await _renderer(api, tmp_path).render_shot(_request(), tmp_path)

    urls = [url for _, url, _, _ in api.calls]
    assert any(u.endswith("/v2/ep-test/run") for u in urls)
    assert any("/v2/ep-test/status/rp-job-1" in u for u in urls)
    # The payload must be wrapped in RunPod's "input" envelope.
    submit = next(p for m, u, p, _ in api.calls if u.endswith("/run"))
    assert set(submit) == {"input"} and submit["input"]["seed"] == 41
    assert result.output_path is not None
    assert result.output_path.read_bytes() == payload


async def test_runpod_statuses_map_to_the_canonical_vocabulary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    api = FakeRunPodAPI(status_sequence=["IN_QUEUE", "IN_PROGRESS", "FAILED"])
    from akio_studio.exceptions import CloudJobFailedError

    with pytest.raises(CloudJobFailedError, match="worker raised"):
        await _renderer(api, tmp_path).render_shot(_request(), tmp_path)


async def test_http_url_output_is_downloaded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    api = FakeRunPodAPI(output={"video_url": "https://cdn.example/out.mp4"})
    result = await _renderer(api, tmp_path).render_shot(_request(), tmp_path)
    assert result.output_path is not None
    assert result.output_path.read_bytes() == b"HTTP-FETCHED-VIDEO"


async def test_malformed_inline_output_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    api = FakeRunPodAPI(output={"video_base64": "!!!not base64!!!"})
    with pytest.raises(CloudRenderError, match="base64"):
        await _renderer(api, tmp_path).render_shot(_request(), tmp_path)


# --------------------------------------------------------------- economics


async def test_cost_is_derived_from_runpod_milliseconds(tmp_path, monkeypatch) -> None:
    """RunPod reports no dollars; without derivation the ceiling is a no-op."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    payload = b"MP4"
    api = FakeRunPodAPI(
        output={"video_base64": base64.b64encode(payload).decode()},
        execution_ms=30_000,  # 30s + 0.5s delay at $0.001/s => ~$0.0305
    )
    result = await _renderer(api, tmp_path).render_shot(_request(), tmp_path)
    assert result.cost_usd == pytest.approx(0.0305, abs=1e-4)


async def test_no_rate_configured_reports_no_cost(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    api = FakeRunPodAPI(output={"video_base64": base64.b64encode(b"x").decode()})
    result = await _renderer(
        api, tmp_path, runpod_gpu_cost_per_hour=0.0
    ).render_shot(_request(), tmp_path)
    assert result.cost_usd is None


# ------------------------------------------------- double-billing avoidance


async def test_submissions_are_never_retried(tmp_path, monkeypatch) -> None:
    """RunPod ignores Idempotency-Key, so a retry could bill twice."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    api = FakeRunPodAPI(fail_submit=True)
    renderer = _renderer(api, tmp_path, max_attempts=5)
    with pytest.raises(CloudSubmitAmbiguousError):
        await renderer.submit(_request())
    assert sum(1 for _, u, _, _ in api.calls if u.endswith("/run")) == 1


async def test_ambiguous_submit_is_journaled_for_reconciliation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    api = FakeRunPodAPI(fail_submit=True)
    journal = tmp_path / "submits.jsonl"
    transport = RunPodTransport(_cfg(), journal_path=journal, http=api)
    renderer = CloudVideoRenderer(
        StudioConfig(cloud_video=_cfg()), transport=transport
    )
    with pytest.raises(CloudSubmitAmbiguousError, match="may be billing"):
        await renderer.submit(_request())
    phases = [json.loads(line)["phase"] for line in journal.read_text().splitlines()]
    assert "ambiguous" in phases


async def test_idempotency_header_is_not_forwarded(tmp_path, monkeypatch) -> None:
    """Sending it would imply a protection RunPod does not provide."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    api = FakeRunPodAPI(output={"video_base64": base64.b64encode(b"x").decode()})
    await _renderer(api, tmp_path).render_shot(_request(), tmp_path)
    submit_headers = next(h for _, u, _, h in api.calls if u.endswith("/run"))
    assert "Idempotency-Key" not in submit_headers
    assert submit_headers["Authorization"] == "Bearer rp-secret"


# ------------------------------------------------------------ configuration


async def test_missing_endpoint_id_is_a_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    monkeypatch.delenv("AKIO_RUNPOD_ENDPOINT_ID", raising=False)
    transport = RunPodTransport(
        _cfg(runpod_endpoint_id=""), journal_path=tmp_path / "j.jsonl", http=FakeRunPodAPI()
    )
    with pytest.raises(CloudAuthError, match="endpoint id"):
        _ = transport.endpoint_id


async def test_missing_api_key_is_a_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("AKIO_CLOUD_VIDEO_TOKEN", raising=False)
    transport = RunPodTransport(
        _cfg(), journal_path=tmp_path / "j.jsonl", http=FakeRunPodAPI()
    )
    with pytest.raises(CloudAuthError, match="API key"):
        await transport.request("GET", "https://x/jobs/abc", None, {}, 5.0)


async def test_health_check(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    transport = RunPodTransport(
        _cfg(), journal_path=tmp_path / "j.jsonl", http=FakeRunPodAPI()
    )
    assert (await transport.health())["workers"]["ready"] == 1


# ------------------------------------------------------------- gpu rate


def test_gpu_rate_has_no_fabricated_default() -> None:
    """A plausible-but-wrong default would guard at the wrong threshold."""
    assert CloudVideoConfig().resolve_gpu_cost_per_hour() == 0.0


def test_gpu_rate_env_override(monkeypatch) -> None:
    cfg = CloudVideoConfig(runpod_gpu_cost_per_hour=0.5)
    monkeypatch.setenv("AKIO_RUNPOD_GPU_RATE_USD_PER_HOUR", "1.19")
    assert cfg.resolve_gpu_cost_per_hour() == 1.19


def test_malformed_gpu_rate_falls_back_never_guesses(monkeypatch) -> None:
    cfg = CloudVideoConfig(runpod_gpu_cost_per_hour=0.5)
    monkeypatch.setenv("AKIO_RUNPOD_GPU_RATE_USD_PER_HOUR", "not-a-number")
    assert cfg.resolve_gpu_cost_per_hour() == 0.5
    monkeypatch.setenv("AKIO_RUNPOD_GPU_RATE_USD_PER_HOUR", "-3")
    assert cfg.resolve_gpu_cost_per_hour() == 0.5


async def test_env_rate_drives_derived_cost(tmp_path, monkeypatch) -> None:
    """The rate the operator sets is the rate the ceiling actually uses."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-secret")
    monkeypatch.setenv("AKIO_RUNPOD_GPU_RATE_USD_PER_HOUR", "3.6")  # $0.001/s
    api = FakeRunPodAPI(
        output={"video_base64": base64.b64encode(b"MP4").decode()},
        execution_ms=10_000,
    )
    result = await _renderer(
        api, tmp_path, runpod_gpu_cost_per_hour=0.0
    ).render_shot(_request(), tmp_path)
    assert result.cost_usd == pytest.approx(0.0105, abs=1e-4)
