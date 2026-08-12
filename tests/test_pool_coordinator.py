"""Unit tests for :mod:`akio_studio.pool_coordinator` — HTTP is monkeypatched."""

from __future__ import annotations

import json
import urllib.error
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

import akio_studio.pool_coordinator as pool_coordinator
from akio_studio.config import MASTER_FPS
from akio_studio.exceptions import (
    MemoryBudgetExceededError,
    OllamaUnavailableError,
    PoolCoordinatorError,
)
from akio_studio.pool_coordinator import DPOFeedbackLogger, LocalPoolCoordinator, Stage


@pytest.fixture
def unreachable_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every HTTP call behaves as if neither Ollama nor ComfyUI is running."""

    def _refuse(
        method: str, url: str, payload: dict[str, Any] | None, timeout: float
    ) -> tuple[int, dict[str, Any]]:
        raise urllib.error.URLError("connection refused (test)")

    monkeypatch.setattr(pool_coordinator, "_http_json", _refuse)


def _records(log_path: Path) -> list[dict[str, Any]]:
    """Parse every JSONL record from a DPO feedback log."""
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


class TestStageLedger:
    """Mutual exclusion is *the* invariant (audit A1/M2)."""

    async def test_acquire_conflicting_stage_without_auto_evict_raises(self) -> None:
        coordinator = LocalPoolCoordinator(auto_evict=False)
        await coordinator.acquire_stage(Stage.LLM)
        with pytest.raises(MemoryBudgetExceededError):
            await coordinator.acquire_stage(Stage.IMAGE_DIFFUSION)
        assert coordinator.resident_stage is Stage.LLM

    async def test_auto_evict_purges_then_swaps(self, unreachable_http: None) -> None:
        coordinator = LocalPoolCoordinator(auto_evict=True)
        await coordinator.acquire_stage(Stage.LLM)
        await coordinator.acquire_stage(Stage.IMAGE_DIFFUSION)
        assert coordinator.resident_stage is Stage.IMAGE_DIFFUSION

    async def test_reacquiring_resident_stage_is_noop(self) -> None:
        coordinator = LocalPoolCoordinator(auto_evict=False)
        await coordinator.acquire_stage(Stage.LLM)
        await coordinator.acquire_stage(Stage.LLM)
        assert coordinator.resident_stage is Stage.LLM

    async def test_stage_context_manager_releases(self, unreachable_http: None) -> None:
        coordinator = LocalPoolCoordinator()
        async with coordinator.stage(Stage.LLM):
            assert coordinator.resident_stage is Stage.LLM
        assert coordinator.resident_stage is None

    async def test_purge_report_shape_with_servers_unreachable(
        self, unreachable_http: None
    ) -> None:
        coordinator = LocalPoolCoordinator()
        await coordinator.acquire_stage(Stage.LLM)
        report = await coordinator.purge_unified_memory_pool()
        assert report["ollama"] == "unreachable"
        assert report["ollama_verified"] is False
        assert report["comfyui"] == "unreachable"
        assert isinstance(report["orchestrator_gc"], int)
        assert report["mps_cache"] == "not-loaded"  # torch is never imported
        assert coordinator.resident_stage is None


class TestOllamaPaths:
    """HTTP-primary / CLI-fallback interface (audit M1/B6)."""

    async def test_http_api_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def _ok(
            method: str, url: str, payload: dict[str, Any] | None, timeout: float
        ) -> tuple[int, dict[str, Any]]:
            captured["url"] = url
            captured["payload"] = payload
            return 200, {"message": {"content": "pong"}}

        monkeypatch.setattr(pool_coordinator, "_http_json", _ok)
        coordinator = LocalPoolCoordinator()
        await coordinator.acquire_stage(Stage.LLM)
        result = await coordinator.run_ollama_qwen("ping", "sys")
        assert result == "pong"
        assert captured["url"].endswith("/api/chat")
        payload = captured["payload"]
        assert payload["keep_alive"] == "10m"
        assert payload["options"]["num_ctx"] == 8192
        assert payload["messages"][0] == {"role": "system", "content": "sys"}

    async def test_http_error_status_raises_without_cli_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _error(
            method: str, url: str, payload: dict[str, Any] | None, timeout: float
        ) -> tuple[int, dict[str, Any]]:
            return 500, {"error": "boom"}

        monkeypatch.setattr(pool_coordinator, "_http_json", _error)
        coordinator = LocalPoolCoordinator()

        async def _forbidden(prompt: str, system_prompt: str | None) -> str:
            raise AssertionError("CLI fallback must not run when the server responded")

        monkeypatch.setattr(coordinator, "_run_ollama_cli", _forbidden)
        with pytest.raises(OllamaUnavailableError):
            await coordinator.run_ollama_qwen("ping")

    async def test_cli_fallback_feeds_prompt_via_stdin(
        self, unreachable_http: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        script = tmp_path / "fake_ollama"
        script.write_text("#!/bin/sh\ncat -\n", encoding="utf-8")
        script.chmod(0o755)
        coordinator = LocalPoolCoordinator()
        monkeypatch.setattr(coordinator, "_locate_ollama_cli", lambda: str(script))
        result = await coordinator.run_ollama_qwen("ping prompt", "sys inst")
        assert "ping prompt" in result
        assert "[SYSTEM INSTRUCTIONS]" in result  # CLI has no true system turn

    async def test_both_paths_failing_raises(
        self, unreachable_http: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        coordinator = LocalPoolCoordinator()
        monkeypatch.setattr(coordinator, "_locate_ollama_cli", lambda: None)
        with pytest.raises(OllamaUnavailableError):
            await coordinator.run_ollama_qwen("ping")


class TestDPOFeedback:
    """Exact frame math (M3) and identical-conditioning pairing (A4)."""

    def test_exact_frame_math(self, tmp_path: Path) -> None:
        log_path = tmp_path / "dpo.jsonl"
        dpo = DPOFeedbackLogger(log_path)
        dpo.register_shot_render("a", "hash", 1, 2.0, 6.5, {})
        record = _records(log_path)[0]
        assert record["start_frame"] == round(Fraction(2.0) * MASTER_FPS) == 48
        assert record["end_frame"] == round(Fraction(6.5) * MASTER_FPS) == 156
        for seconds in (0.0, 0.041, 41.7, 1000.0):
            assert pool_coordinator._seconds_to_frame(seconds) == round(
                Fraction(seconds) * Fraction(24000, 1001)
            )

    def test_same_prompt_hash_produces_pair(self, tmp_path: Path) -> None:
        dpo = DPOFeedbackLogger(tmp_path / "dpo.jsonl")
        dpo.register_shot_render("shot_s41", "h1", 41, 2.0, 6.5, {"denoise": 0.3})
        dpo.register_shot_render("shot_s77", "h1", 77, 2.0, 6.5, {"denoise": 0.3})
        dpo.record_retention_result("shot_s41", "h1", 41, 0.83)
        dpo.record_retention_result("shot_s77", "h1", 77, 0.41)
        pair = dpo.log_dpo_latent_feedback("shot_s77", 4.2, {"denoise": 0.3})
        assert pair is not None
        assert pair["type"] == "dpo_pair"
        assert pair["prompt_hash"] == "h1"
        assert pair["chosen"]["seed"] == 41
        assert pair["chosen"]["mean_retention"] == 0.83
        assert pair["rejected"]["seed"] == 77
        assert pair["rejected"]["dropoff_frame"] == round(Fraction(4.2) * MASTER_FPS)

    def test_different_prompt_hash_never_pairs(self, tmp_path: Path) -> None:
        """A4 regression: cross-conditioning pairs teach content preference."""
        log_path = tmp_path / "dpo.jsonl"
        dpo = DPOFeedbackLogger(log_path)
        dpo.register_shot_render("shot_a", "hash_a", 1, 0.0, 5.0, {})
        dpo.register_shot_render("shot_b", "hash_b", 2, 5.0, 10.0, {})
        dpo.record_retention_result("shot_a", "hash_a", 1, 0.30)
        dpo.record_retention_result("shot_b", "hash_b", 2, 0.90)
        pair = dpo.log_dpo_latent_feedback("shot_a", 1.0, {})
        assert pair is None
        last = _records(log_path)[-1]
        assert last["type"] == "dpo_unpaired"
        assert last["prompt_hash"] == "hash_a"

    def test_unpaired_rejection_logged_and_returns_none(self, tmp_path: Path) -> None:
        log_path = tmp_path / "dpo.jsonl"
        dpo = DPOFeedbackLogger(log_path)
        dpo.register_shot_render("only", "h", 7, 0.0, 3.0, {})
        assert dpo.log_dpo_latent_feedback("only", 1.5, {}) is None
        assert _records(log_path)[-1]["type"] == "dpo_unpaired"

    def test_dropoff_outside_any_shot_raises(self, tmp_path: Path) -> None:
        dpo = DPOFeedbackLogger(tmp_path / "dpo.jsonl")
        dpo.register_shot_render("shot", "h", 1, 0.0, 5.0, {})
        with pytest.raises(PoolCoordinatorError):
            dpo.log_dpo_latent_feedback("shot", 30.0, {})

    def test_retention_for_unregistered_render_raises(self, tmp_path: Path) -> None:
        dpo = DPOFeedbackLogger(tmp_path / "dpo.jsonl")
        with pytest.raises(PoolCoordinatorError):
            dpo.record_retention_result("ghost", "h", 1, 0.5)

    def test_invalid_shot_interval_raises(self, tmp_path: Path) -> None:
        dpo = DPOFeedbackLogger(tmp_path / "dpo.jsonl")
        with pytest.raises(PoolCoordinatorError):
            dpo.register_shot_render("shot", "h", 1, 5.0, 5.0, {})
