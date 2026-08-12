"""Lore knowledge graph and the writers' room persona chain.

The lore graph is the studio's canon of record. Per the architecture audit it
is backed by a :class:`networkx.MultiDiGraph` keyed by relation type (finding
B2: a plain ``DiGraph`` holds one edge per ordered pair and silently
overwrites parallel relations), persisted as atomic node-link JSON instead of
a Neo4j server (finding M5), and queried through a character-budgeted canon
brief rather than a full graph dump (finding A3). Symmetric relations such as
``ALLIED_WITH`` are stored once and normalized at query time so canon is
visible from both endpoints.

The writers' room implements the audit's M1 correction: the four "agents" are
prompt personas serialized onto the single resident LLM, not four model
instances. :class:`WritersRoom` therefore takes an *injected* async callable
``(prompt, system_prompt) -> response`` and never imports the pool
coordinator.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from akio_studio._io import atomic_write_json
from akio_studio.config import GreenlightThresholds
from akio_studio.exceptions import AkioStudioError, EntityNotFoundError, LoreGraphError

logger = logging.getLogger(__name__)

#: Every relation type the canon graph accepts.
RELATION_TYPES: frozenset[str] = frozenset(
    {"ALLIED_WITH", "ENEMY_OF", "WIELDS", "LOCATED_AT", "MEMBER_OF", "RIVAL_OF"}
)

#: Relations that hold in both directions; stored once, mirrored at query time.
SYMMETRIC_RELATIONS: frozenset[str] = frozenset({"ALLIED_WITH", "ENEMY_OF", "RIVAL_OF"})

#: Attribute names that collide with networkx node-link serialization fields
#: ("id" for nodes; "source"/"target"/"key" for edges) — they would be
#: silently destroyed by a save/load round trip, and "key" additionally
#: collides with the ``add_edge`` keyword. Rejected up front.
_RESERVED_NODE_ATTRS: frozenset[str] = frozenset({"id"})
_RESERVED_EDGE_ATTRS: frozenset[str] = frozenset({"source", "target", "key", "id"})


def _reject_reserved_attributes(
    subject: str, attributes: dict, reserved: frozenset[str]
) -> None:
    """Raise :class:`LoreGraphError` when ``attributes`` uses reserved names."""
    clashes = reserved.intersection(attributes)
    if clashes:
        raise LoreGraphError(
            f"{subject!r} uses reserved attribute name(s) {sorted(clashes)}; "
            "these collide with the graph's persistence format"
        )

#: Every entity type the canon graph accepts.
ENTITY_TYPES: frozenset[str] = frozenset(
    {"character", "faction", "artifact", "location", "concept"}
)

#: Marker appended when a canon brief is cut to fit its character budget.
_TRUNCATION_MARKER = "…[canon truncated]"


def _audience_theory_block(audience_theory: str) -> str:
    """Return the delimited mandatory-constraint block for one fan theory."""
    return (
        "==== MANDATORY AUDIENCE CANON CONSTRAINT ====\n"
        "The following audience theory has been promoted to canon direction.\n"
        "You MUST weave it into the episode organically, and you MUST NOT\n"
        "contradict any entity, attribute, or relation in the canon context\n"
        "above. Where the theory and canon appear to conflict, reinterpret the\n"
        "theory until it fits canon — never the reverse.\n"
        f"THEORY: {audience_theory}\n"
        "==== END AUDIENCE CANON CONSTRAINT ===="
    )


class LoreGraphManager:
    """Canon knowledge graph with typed entities and typed, keyed relations.

    Backed by :class:`networkx.MultiDiGraph` with the relation type as the
    edge key, so ``ALLIED_WITH`` and ``WIELDS`` coexist between the same pair
    of nodes and re-adding an existing relation type *updates* that edge
    instead of duplicating it (audit finding B2). Persistence is atomic
    node-link JSON (audit finding M5); pass ``persist_path`` to enable
    :meth:`save` / :meth:`load`.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        """Create an empty graph, optionally bound to a JSON persist path."""
        self.persist_path: Path | None = Path(persist_path) if persist_path else None
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

    @property
    def entity_count(self) -> int:
        """Number of entities (nodes) currently in the canon graph."""
        return self._graph.number_of_nodes()

    @property
    def relation_count(self) -> int:
        """Number of stored relations (edges); symmetric ones count once."""
        return self._graph.number_of_edges()

    def add_lore_entity(self, entity_id: str, attributes: dict) -> None:
        """Add or update an entity node.

        ``attributes`` must include an ``entity_type`` drawn from
        :data:`ENTITY_TYPES`. Re-adding an existing entity merges the new
        attributes over the old ones (idempotent update).

        Raises:
            LoreGraphError: when ``entity_type`` is missing or unknown.
        """
        entity_type = attributes.get("entity_type")
        if entity_type not in ENTITY_TYPES:
            raise LoreGraphError(
                f"Entity {entity_id!r} has invalid entity_type {entity_type!r}; "
                f"expected one of {sorted(ENTITY_TYPES)}"
            )
        _reject_reserved_attributes(entity_id, attributes, _RESERVED_NODE_ATTRS)
        existed = self._graph.has_node(entity_id)
        self._graph.add_node(entity_id, **attributes)
        logger.debug(
            "%s lore entity %r (type=%s)",
            "Updated" if existed else "Added",
            entity_id,
            entity_type,
        )

    def add_lore_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        attributes: dict | None = None,
    ) -> None:
        """Add or update a typed relation between two existing entities.

        The relation type is the MultiDiGraph edge *key*: parallel relations
        of different types coexist, while re-adding the same type updates
        that edge's attributes in place.

        Raises:
            EntityNotFoundError: when either endpoint was never added.
            LoreGraphError: when ``relation_type`` is not in
                :data:`RELATION_TYPES`.
        """
        if relation_type not in RELATION_TYPES:
            raise LoreGraphError(
                f"Unknown relation_type {relation_type!r}; "
                f"expected one of {sorted(RELATION_TYPES)}"
            )
        for node in (source_id, target_id):
            if not self._graph.has_node(node):
                raise EntityNotFoundError(
                    f"Relation {source_id!r} -{relation_type}-> {target_id!r} "
                    f"references unknown entity {node!r}; add it first"
                )
        attrs = attributes or {}
        _reject_reserved_attributes(
            f"{source_id}-{relation_type}->{target_id}", attrs, _RESERVED_EDGE_ATTRS
        )
        self._graph.add_edge(source_id, target_id, key=relation_type, **attrs)
        logger.debug("Related %r -%s-> %r", source_id, relation_type, target_id)

    def relations_between(self, a: str, b: str) -> list[dict]:
        """Return every typed relation from ``a`` to ``b``.

        Includes all edges stored ``a -> b`` plus, for
        :data:`SYMMETRIC_RELATIONS`, edges stored ``b -> a`` — symmetric
        relations are stored once but must be visible from both sides. Each
        item is ``{"source", "target", "relation_type", "attributes"}`` with
        ``source``/``target`` reflecting the stored direction. Deterministic
        (sorted) ordering.
        """
        found: list[dict] = []
        if self._graph.has_edge(a, b):
            for relation_type, attrs in self._graph[a][b].items():
                found.append(
                    {
                        "source": a,
                        "target": b,
                        "relation_type": relation_type,
                        "attributes": dict(attrs),
                    }
                )
        if a != b and self._graph.has_edge(b, a):
            for relation_type, attrs in self._graph[b][a].items():
                if relation_type in SYMMETRIC_RELATIONS:
                    found.append(
                        {
                            "source": b,
                            "target": a,
                            "relation_type": relation_type,
                            "attributes": dict(attrs),
                        }
                    )
        found.sort(key=lambda rel: (rel["relation_type"], rel["source"], rel["target"]))
        return found

    def query_canon_context(self, concept_id: str, max_chars: int = 4000) -> str:
        """Build a character-budgeted canon brief around ``concept_id``.

        Starts from the concept node when present (else covers all entities),
        walks its BFS neighborhood over an undirected view, and emits sorted,
        grouped lines: entities by type with their key attributes, then
        relations (``A —REL→ B``; symmetric ones once as ``A ⟷REL⟷ B``).
        Output never exceeds ``max_chars``; when over budget it is cut at a
        line boundary and ends with ``…[canon truncated]`` (audit finding A3:
        token budget — never dump the whole graph).
        """
        if self._graph.has_node(concept_id):
            undirected = self._graph.to_undirected(as_view=True)
            scope = set(nx.bfs_tree(undirected, concept_id).nodes)
            header = f"== CANON BRIEF: {concept_id} =="
        else:
            scope = set(self._graph.nodes)
            header = f"== CANON BRIEF: full canon (no entity {concept_id!r}) =="

        lines: list[str] = [header]

        by_type: dict[str, list[str]] = {}
        for node in sorted(scope):
            attrs = self._graph.nodes[node]
            entity_type = attrs.get("entity_type", "concept")
            extras = ", ".join(
                f"{key}={attrs[key]}" for key in sorted(attrs) if key != "entity_type"
            )
            by_type.setdefault(entity_type, []).append(
                f"- {node} ({extras})" if extras else f"- {node}"
            )
        for entity_type in sorted(by_type):
            lines.append("")
            lines.append(f"[{entity_type}]")
            lines.extend(by_type[entity_type])

        relation_lines: set[str] = set()
        for src, dst, relation_type in self._graph.edges(keys=True):
            if src not in scope or dst not in scope:
                continue
            if relation_type in SYMMETRIC_RELATIONS:
                left, right = sorted((src, dst))
                relation_lines.add(f"- {left} ⟷{relation_type}⟷ {right}")
            else:
                relation_lines.add(f"- {src} —{relation_type}→ {dst}")
        if relation_lines:
            lines.append("")
            lines.append("[relations]")
            lines.extend(sorted(relation_lines))

        brief = "\n".join(lines)
        if len(brief) <= max_chars:
            return brief
        logger.debug(
            "Canon brief for %r over budget (%d > %d chars); truncating",
            concept_id,
            len(brief),
            max_chars,
        )
        while lines and len("\n".join(lines)) + 1 + len(_TRUNCATION_MARKER) > max_chars:
            lines.pop()
        if not lines:
            return _TRUNCATION_MARKER[:max_chars]
        return "\n".join([*lines, _TRUNCATION_MARKER])

    def inject_audience_theory_constraint(
        self, lore_context: str, audience_theory: str
    ) -> str:
        """Append a mandatory audience-canon constraint block to a context.

        The block instructs the writer that the promoted fan theory must be
        woven into the episode *without* contradicting the canon above.
        Returns the combined prompt context.
        """
        return f"{lore_context}\n\n{_audience_theory_block(audience_theory)}"

    def save(self) -> Path:
        """Persist the graph as node-link JSON via an atomic write.

        Raises:
            LoreGraphError: when no ``persist_path`` was configured.
        """
        if self.persist_path is None:
            raise LoreGraphError("Cannot save: LoreGraphManager has no persist_path")
        payload = nx.node_link_data(self._graph, edges="links")
        atomic_write_json(self.persist_path, payload)
        logger.info(
            "Saved lore graph (%d entities, %d relations) to %s",
            self.entity_count,
            self.relation_count,
            self.persist_path,
        )
        return self.persist_path

    def load(self) -> None:
        """Load the graph from ``persist_path``; a missing file means empty.

        Raises:
            LoreGraphError: when no ``persist_path`` was configured, or the
                file exists but cannot be parsed as a node-link graph.
        """
        if self.persist_path is None:
            raise LoreGraphError("Cannot load: LoreGraphManager has no persist_path")
        if not self.persist_path.exists():
            logger.info("No lore graph at %s; starting empty", self.persist_path)
            self._graph = nx.MultiDiGraph()
            return
        try:
            payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
            self._graph = nx.node_link_graph(
                payload, directed=True, multigraph=True, edges="links"
            )
        except AkioStudioError:
            raise
        except Exception as exc:
            raise LoreGraphError(
                f"Corrupt lore graph file at {self.persist_path}: {exc}"
            ) from exc
        logger.info(
            "Loaded lore graph (%d entities, %d relations) from %s",
            self.entity_count,
            self.relation_count,
            self.persist_path,
        )


@dataclass(frozen=True)
class WriterRoomPersona:
    """One prompt persona in the writers' room chain.

    Per audit finding M1 the four personas are serialized onto the single
    resident LLM — a persona is a system prompt, not a model instance.
    """

    name: str
    role: str
    system_prompt: str


_THRESHOLDS = GreenlightThresholds()

#: The four writers'-room personas, in canonical chain order by name.
PERSONAS: dict[str, WriterRoomPersona] = {
    "Director": WriterRoomPersona(
        name="Director",
        role="Narrative drive and episode beat structure",
        system_prompt=(
            "You are the Director of a data-driven anime studio's writers' room. "
            "You own narrative drive and episode beat structure: cold open, "
            "inciting beat, escalation, midpoint reversal, climax, and a "
            "next-episode pull. Every episode must open on one of the five hook "
            "archetypes: (1) Instant Power Subversion — the strong one loses or "
            "the weak one wins within seconds; (2) Visual Sakuga Momentum — open "
            "mid-motion inside a high-effort animation burst; (3) Impact Frame "
            "contrast — a stark stylized impact frame smash-cut against calm; "
            "(4) Antagonist Symmetry — mirror the hero and villain in one "
            "composition or line; (5) Horizon Mystery — a lore hook glimpsed at "
            "the edge of the world that demands explanation. Honor every canon "
            "fact and every MANDATORY AUDIENCE CANON CONSTRAINT you are given; "
            "canon always wins over invention. Write tight, producible scripts "
            "with numbered scenes and clear beats."
        ),
    ),
    "Canon Keeper": WriterRoomPersona(
        name="Canon Keeper",
        role="Strict canon-consistency review",
        system_prompt=(
            "You are the Canon Keeper. Check the draft strictly against the "
            "canon context you are given — entities, attributes (power_level, "
            "world_rules, ...), and relations. List every violation as a "
            "numbered item citing the conflicting canon line, then propose the "
            "MINIMAL correction for each — the smallest rewrite that restores "
            "canon while preserving the draft's intent. Do not restyle, expand, "
            "or improve anything that is not a violation. If there are no "
            "violations, state 'NO VIOLATIONS' and nothing else."
        ),
    ),
    "Visual Style": WriterRoomPersona(
        name="Visual Style",
        role="Sakuga timing and Shinkai-style lighting annotation",
        system_prompt=(
            "You are the Visual Style lead. Annotate the script scene by scene "
            "with production notes, changing no dialogue or story beats. Sakuga "
            "timing: mark where to hold stills on twos versus snapping to "
            "animation on ones for burst impact (snappy stills-vs-bursts "
            "contrast), where to drop stylized impact frames, and where shots "
            "need overflow margins painted beyond frame for camera moves. "
            "Lighting is Shinkai-style: triadic color palettes, blue-tinted "
            "shadows, and volumetric god-ray light. Keep notes terse and tagged "
            "per scene, e.g. [SAKUGA], [IMPACT FRAME], [OVERFLOW], [LIGHT]."
        ),
    ),
    "Analytics Feedback": WriterRoomPersona(
        name="Analytics Feedback",
        role="Retention-driven cold-open tightening",
        system_prompt=(
            "You are the Analytics Feedback editor. Our greenlight gate "
            f"requires {_THRESHOLDS.hook_retention_3s:.0%} retention at 3 "
            f"seconds, {_THRESHOLDS.midpoint_retention:.0%} at the midpoint, "
            f"and a {_THRESHOLDS.completion_rate:.0%} completion rate. Rewrite "
            "the cold open so the FIRST 2 SECONDS hook: open mid-action or "
            "mid-question, no logos, no establishing dead air, no dialogue "
            "preamble — the first cut must pose a question only watching "
            "answers. Touch only the cold open and keep every visual-style "
            "annotation intact; return the full script with the tightened "
            "opening in place."
        ),
    ),
}


class WritersRoom:
    """Sequential persona chain over one injected LLM callable.

    ``llm`` is an async callable ``(prompt, system_prompt) -> response``
    provided by the composition layer (audit M1: one loaded model, four
    prompt personas — this class never imports the pool coordinator).
    """

    def __init__(self, llm: Callable[[str, str | None], Awaitable[str]]) -> None:
        """Store the injected async LLM callable."""
        self._llm = llm

    async def _run_persona(self, persona: WriterRoomPersona, prompt: str) -> str:
        """Run one persona pass, wrapping non-studio failures.

        :class:`AkioStudioError` subclasses propagate untouched; any other
        exception is wrapped in :class:`LoreGraphError` with persona context.
        """
        logger.info("Writers' room pass: %s", persona.name)
        try:
            return await self._llm(prompt, persona.system_prompt)
        except AkioStudioError:
            raise
        except Exception as exc:
            raise LoreGraphError(
                f"Writers' room persona {persona.name!r} failed: {exc}"
            ) from exc

    async def draft_episode_script(
        self, premise: str, canon_context: str, audience_theories: list[str]
    ) -> dict:
        """Draft an episode script through the four-persona chain.

        Sequence: Director drafts from the premise plus the canon context with
        audience theories injected as mandatory constraints; Canon Keeper
        reviews the draft against canon and lists minimal corrections; the
        Director revises with those corrections; Visual Style annotates; and
        Analytics Feedback tightens the cold open. Returns ``{"script":
        final_script, "passes": {pass_label: raw_output}, "personas_run":
        [persona_names_in_order]}`` (the Director's second pass is labeled
        ``"Director (revision)"`` in ``passes`` so the draft is not lost).
        """
        context = canon_context
        for theory in audience_theories:
            context = f"{context}\n\n{_audience_theory_block(theory)}"

        passes: dict[str, str] = {}
        personas_run: list[str] = []

        director = PERSONAS["Director"]
        draft = await self._run_persona(
            director,
            f"EPISODE PREMISE:\n{premise}\n\nCANON CONTEXT:\n{context}\n\n"
            "Write the full episode script, opening on one of the five hook "
            "archetypes.",
        )
        passes[director.name] = draft
        personas_run.append(director.name)

        keeper = PERSONAS["Canon Keeper"]
        corrections = await self._run_persona(
            keeper,
            f"CANON CONTEXT:\n{context}\n\nDRAFT SCRIPT:\n{draft}\n\n"
            "List canon violations and minimal corrections.",
        )
        passes[keeper.name] = corrections
        personas_run.append(keeper.name)

        revised = await self._run_persona(
            director,
            f"CANON CONTEXT:\n{context}\n\nYOUR DRAFT:\n{draft}\n\n"
            f"CANON KEEPER CORRECTIONS:\n{corrections}\n\n"
            "Apply every correction with the smallest possible rewrite and "
            "return the full revised script.",
        )
        passes[f"{director.name} (revision)"] = revised
        personas_run.append(director.name)

        style = PERSONAS["Visual Style"]
        annotated = await self._run_persona(
            style,
            f"REVISED SCRIPT:\n{revised}\n\n"
            "Annotate every scene with sakuga timing and lighting notes; "
            "return the full annotated script.",
        )
        passes[style.name] = annotated
        personas_run.append(style.name)

        analytics = PERSONAS["Analytics Feedback"]
        final = await self._run_persona(
            analytics,
            f"ANNOTATED SCRIPT:\n{annotated}\n\n"
            "Tighten the cold open per the retention thresholds and return "
            "the full script.",
        )
        passes[analytics.name] = final
        personas_run.append(analytics.name)

        logger.info(
            "Writers' room chain complete: %d passes, final script %d chars",
            len(personas_run),
            len(final),
        )
        return {"script": final, "passes": passes, "personas_run": personas_run}
