"""Regression tests for the post-build adversarial-review fixes."""

from __future__ import annotations

import pytest

from akio_studio.config import ComfyQualityGate
from akio_studio.exceptions import LoreGraphError
from akio_studio.lore_graph_agent import LoreGraphManager
from akio_studio.pool_coordinator import DPOFeedbackLogger


def test_quality_gate_rejects_out_of_range_params() -> None:
    with pytest.raises(ValueError):
        ComfyQualityGate(ip_adapter_faceid_weight=0.9)
    with pytest.raises(ValueError):
        ComfyQualityGate(character_lora_weight=0.5)
    with pytest.raises(ValueError):
        ComfyQualityGate(wan_denoise_min=0.5, wan_denoise_max=0.3)


def test_dpo_never_emits_inverted_pair(tmp_path) -> None:
    """A 'chosen' render must retain strictly better than the rejected one."""
    dpo = DPOFeedbackLogger(tmp_path / "dpo.jsonl")
    dpo.register_shot_render("ep1", "hash-a", seed=1, start_second=0.0, end_second=5.0,
                             comfy_params={})
    dpo.register_shot_render("ep1", "hash-a", seed=2, start_second=0.0, end_second=5.0,
                             comfy_params={})
    # The would-be counterpart retained WORSE than the rejected render.
    dpo.record_retention_result("ep1", "hash-a", seed=1, mean_retention=0.60)
    dpo.record_retention_result("ep1", "hash-a", seed=2, mean_retention=0.40)
    assert dpo.log_dpo_latent_feedback("ep1", 2.0, {}) is None


def test_dpo_reregistration_preserves_retention(tmp_path) -> None:
    dpo = DPOFeedbackLogger(tmp_path / "dpo.jsonl")
    dpo.register_shot_render("ep1", "hash-a", seed=1, start_second=0.0, end_second=5.0,
                             comfy_params={})
    dpo.record_retention_result("ep1", "hash-a", seed=1, mean_retention=0.9)
    dpo.register_shot_render("ep1", "hash-a", seed=1, start_second=0.0, end_second=6.0,
                             comfy_params={})
    render = dpo._registry[("ep1", "hash-a", 1)]
    assert render.mean_retention == 0.9
    assert render.end_second == 6.0


def test_lore_graph_rejects_reserved_attribute_names() -> None:
    graph = LoreGraphManager()
    with pytest.raises(LoreGraphError):
        graph.add_lore_entity("kael", {"entity_type": "character", "id": "x"})
    graph.add_lore_entity("kael", {"entity_type": "character"})
    graph.add_lore_entity("blade", {"entity_type": "artifact"})
    with pytest.raises(LoreGraphError):
        graph.add_lore_relation("kael", "blade", "WIELDS", {"key": "boom"})
