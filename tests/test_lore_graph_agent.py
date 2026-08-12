"""Unit tests for :mod:`akio_studio.lore_graph_agent`."""

from __future__ import annotations

from pathlib import Path

import pytest

from akio_studio.exceptions import EntityNotFoundError, LoreGraphError
from akio_studio.lore_graph_agent import PERSONAS, LoreGraphManager, WritersRoom


def _small_canon() -> LoreGraphManager:
    """Two characters, an artifact, and parallel relations between one pair."""
    lore = LoreGraphManager()
    lore.add_lore_entity("kael", {"entity_type": "character", "power_level": 6})
    lore.add_lore_entity("mentor", {"entity_type": "character", "power_level": 9})
    lore.add_lore_entity("blade", {"entity_type": "artifact"})
    return lore


class TestMultiDiGraphRelations:
    """THE B2 regression: parallel relations must both stay visible."""

    def test_parallel_relations_between_same_pair_coexist(self) -> None:
        lore = _small_canon()
        # A plain DiGraph would silently overwrite the first edge here.
        lore.add_lore_relation("kael", "mentor", "ALLIED_WITH")
        lore.add_lore_relation("kael", "mentor", "WIELDS")
        relations = lore.relations_between("kael", "mentor")
        types = {rel["relation_type"] for rel in relations}
        assert types == {"ALLIED_WITH", "WIELDS"}
        assert len(relations) == 2

    def test_symmetric_relation_visible_from_both_sides(self) -> None:
        lore = _small_canon()
        lore.add_lore_relation("kael", "mentor", "ALLIED_WITH")
        lore.add_lore_relation("kael", "blade", "WIELDS")
        # Stored kael -> mentor, but ALLIED_WITH is symmetric: visible reversed.
        reverse = lore.relations_between("mentor", "kael")
        assert [rel["relation_type"] for rel in reverse] == ["ALLIED_WITH"]
        # WIELDS is directional: never mirrored.
        assert lore.relations_between("blade", "kael") == []

    def test_readding_same_relation_type_updates_not_duplicates(self) -> None:
        lore = _small_canon()
        lore.add_lore_relation("kael", "mentor", "ALLIED_WITH", {"since": "ep-0"})
        lore.add_lore_relation("kael", "mentor", "ALLIED_WITH", {"since": "ep-3"})
        relations = lore.relations_between("kael", "mentor")
        assert len(relations) == 1
        assert relations[0]["attributes"] == {"since": "ep-3"}
        assert lore.relation_count == 1

    def test_relation_to_unknown_entity_raises(self) -> None:
        lore = _small_canon()
        with pytest.raises(EntityNotFoundError):
            lore.add_lore_relation("kael", "ghost", "ALLIED_WITH")

    def test_invalid_relation_type_raises(self) -> None:
        lore = _small_canon()
        with pytest.raises(LoreGraphError):
            lore.add_lore_relation("kael", "mentor", "BEFRIENDS")

    def test_invalid_entity_type_raises(self) -> None:
        lore = LoreGraphManager()
        with pytest.raises(LoreGraphError):
            lore.add_lore_entity("x", {"entity_type": "starship"})
        with pytest.raises(LoreGraphError):
            lore.add_lore_entity("y", {})


class TestCanonContext:
    """Budgeted canon brief (audit A3)."""

    def test_truncation_at_max_chars(self) -> None:
        lore = _small_canon()
        lore.add_lore_entity(
            "shrine",
            {"entity_type": "location", "notes": "a very long descriptive attribute " * 10},
        )
        lore.add_lore_relation("kael", "mentor", "ALLIED_WITH")
        lore.add_lore_relation("kael", "blade", "WIELDS")
        for max_chars in (60, 120, 400):
            brief = lore.query_canon_context("kael", max_chars=max_chars)
            assert len(brief) <= max_chars
        truncated = lore.query_canon_context("kael", max_chars=120)
        assert truncated.endswith("…[canon truncated]")

    def test_bfs_scope_excludes_disconnected_entities(self) -> None:
        lore = _small_canon()
        lore.add_lore_relation("kael", "mentor", "ALLIED_WITH")
        lore.add_lore_entity("island", {"entity_type": "location"})  # disconnected
        brief = lore.query_canon_context("kael", max_chars=4000)
        assert "mentor" in brief
        assert "island" not in brief

    def test_unknown_concept_falls_back_to_full_canon(self) -> None:
        lore = _small_canon()
        brief = lore.query_canon_context("nope", max_chars=4000)
        assert "full canon" in brief
        assert "kael" in brief

    def test_inject_audience_theory_constraint_appends_block(self) -> None:
        lore = _small_canon()
        combined = lore.inject_audience_theory_constraint("CANON", "the mentor is a fox")
        assert combined.startswith("CANON")
        assert "MANDATORY AUDIENCE CANON CONSTRAINT" in combined
        assert "the mentor is a fox" in combined


class TestPersistence:
    """Atomic node-link JSON save/load (audit M5)."""

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "canon.json"
        lore = LoreGraphManager(persist_path=path)
        lore.add_lore_entity("kael", {"entity_type": "character", "power_level": 6})
        lore.add_lore_entity("mentor", {"entity_type": "character"})
        lore.add_lore_relation("kael", "mentor", "ALLIED_WITH", {"since": "ep-0"})
        lore.add_lore_relation("kael", "mentor", "RIVAL_OF")
        assert lore.save() == path

        restored = LoreGraphManager(persist_path=path)
        restored.load()
        assert restored.entity_count == 2
        assert restored.relation_count == 2
        assert restored.relations_between("kael", "mentor") == lore.relations_between(
            "kael", "mentor"
        )

    def test_load_missing_file_starts_empty(self, tmp_path: Path) -> None:
        lore = LoreGraphManager(persist_path=tmp_path / "absent.json")
        lore.load()
        assert lore.entity_count == 0

    def test_load_corrupt_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "canon.json"
        path.write_text("{not json", encoding="utf-8")
        lore = LoreGraphManager(persist_path=path)
        with pytest.raises(LoreGraphError):
            lore.load()

    def test_save_without_persist_path_raises(self) -> None:
        with pytest.raises(LoreGraphError):
            LoreGraphManager().save()


class TestWritersRoom:
    """The persona chain over one injected async LLM callable (audit M1)."""

    async def test_chain_order_and_theory_injection(self) -> None:
        calls: list[tuple[str, str | None]] = []

        async def fake_llm(prompt: str, system_prompt: str | None) -> str:
            calls.append((prompt, system_prompt))
            return f"OUT-{len(calls)}"

        room = WritersRoom(llm=fake_llm)
        result = await room.draft_episode_script(
            "premise", "CANON CONTEXT LINE", ["the mentor is a fox spirit"]
        )
        assert result["personas_run"] == [
            "Director",
            "Canon Keeper",
            "Director",
            "Visual Style",
            "Analytics Feedback",
        ]
        assert set(result["passes"]) == {
            "Director",
            "Canon Keeper",
            "Director (revision)",
            "Visual Style",
            "Analytics Feedback",
        }
        assert result["script"] == "OUT-5"
        # The audience theory is injected as a mandatory constraint block.
        first_prompt, first_system = calls[0]
        assert "MANDATORY AUDIENCE CANON CONSTRAINT" in first_prompt
        assert "the mentor is a fox spirit" in first_prompt
        assert first_system == PERSONAS["Director"].system_prompt

    async def test_unexpected_llm_failure_wrapped(self) -> None:
        async def broken_llm(prompt: str, system_prompt: str | None) -> str:
            raise RuntimeError("boom")

        room = WritersRoom(llm=broken_llm)
        with pytest.raises(LoreGraphError):
            await room.draft_episode_script("p", "c", [])
