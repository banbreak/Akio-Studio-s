"""Akio Studio — end-to-end demonstration pipeline (composition layer).

The six feature modules never import each other; this file is the *only*
place they are wired together (metrics engine -> lore graph -> writers' room
-> pool coordinator -> DPO logger -> file tree -> actor exporter). It runs a
complete concept-to-bundle demo that is **green with no Ollama and no ComfyUI
running**: the LLM stage falls back to a deterministic offline stub script,
and the purge step reports both servers as unreachable instead of failing.

Flags:

* ``--base-dir PATH``   production tree root (default ``~/AkioStudio/productions``)
* ``--webhook-url URL`` optional greenlight webhook; skipped when absent
* ``--dashboard`` / ``--daemon``  accepted aliases that currently run this
  same demo pipeline — documented placeholders for the macOS launcher built
  by ``build_mac_app.sh``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from akio_studio._io import atomic_write_bytes, atomic_write_json
from akio_studio.actor_sdk_exporter import SyntheticActorSDKExporter
from akio_studio.config import StudioConfig
from akio_studio.exceptions import OllamaUnavailableError, WebhookDispatchError
from akio_studio.file_tree_manager import ProductionFileTreeManager
from akio_studio.lore_graph_agent import LoreGraphManager, WritersRoom
from akio_studio.metrics_engine import MetricsIngestionEngine
from akio_studio.pool_coordinator import LocalPoolCoordinator, Stage

logger = logging.getLogger(__name__)

#: Concept exercised by the demo pipeline.
CONCEPT_ID = "ronin-arc-001"

#: Episode premise fed to the writers' room (and the offline stub).
PREMISE = (
    "Kael's first duel with the Masked Mentor ends with a shattered blade "
    "and a lantern that should not still be burning."
)


def offline_stub_script(premise: str, top_theory: str) -> str:
    """Deterministic 3-beat cold-open script used when Ollama is unreachable.

    Keeps the demo runnable on any machine: the beats honor the injected
    audience theory and the five-hook-archetype rule the Director persona
    would otherwise enforce.
    """
    return (
        'COLD OPEN — "The Lantern That Would Not Burn"  [offline stub draft]\n'
        f"PREMISE: {premise}\n"
        f"AUDIENCE CANON CONSTRAINT HONORED: {top_theory}\n"
        "\n"
        "BEAT 1 (0.0s-2.0s)  [HOOK: Instant Power Subversion] [IMPACT FRAME]\n"
        "  Mid-swing: Kael's Duskfang Blade SHATTERS against an unarmed palm.\n"
        "  The Masked Mentor has not moved from the shrine's threshold.\n"
        "\n"
        "BEAT 2 (2.0s-6.5s)  [SAKUGA] [LIGHT: blue shadows, god-rays]\n"
        "  Kael skids back through the lantern rows; every lantern he passes\n"
        "  gutters out — except one, burning fox-orange, leaning toward the\n"
        "  Mentor like a compass needle.\n"
        "\n"
        "BEAT 3 (6.5s-10.0s) [HORIZON MYSTERY]\n"
        "  The Mentor finally steps forward — and the sacred ground takes no\n"
        "  footprint. Kael sees it. So does the fox shape in the smoke.\n"
        "  SMASH CUT TO TITLE.\n"
    )


async def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full demo pipeline; returns the final summary dictionary."""
    base_dir = Path(args.base_dir).resolve()
    config = StudioConfig(base_dir=base_dir)

    # ------------------------------------------------------------ a. file tree
    logger.info("[1/9] Initializing production file tree under %s", base_dir)
    ftm = ProductionFileTreeManager(base_dir=base_dir)
    tree = ftm.initialize_concept_tree(CONCEPT_ID)
    logger.info("concept tree ready: %s", tree["root"])

    # --------------------------------------------------------------- b. metrics
    logger.info("[2/9] Ingesting mock platform metrics for 3 posts")
    engine = MetricsIngestionEngine(config)
    posts = [
        engine.fetch_mock_platform_data(f"{CONCEPT_ID}-post-{index:02d}", CONCEPT_ID)
        for index in (1, 2, 3)
    ]
    for post in posts:
        report = engine.calculate_retention_to_engagement(post)
        logger.info(
            "post %s: evaluable=%s passes=%s r2e=%s",
            report["post_id"],
            report["evaluable"],
            report["passes"],
            f"{report['r2e_score']:.3f}" if report["r2e_score"] is not None else "n/a",
        )
    validated = engine.evaluate_concept_validation(CONCEPT_ID, posts)
    pooled_comments = [comment for post in posts for comment in post.comments_text_list]
    theories = engine.mine_audience_lore_theories(pooled_comments, top_n=3)
    logger.info("concept validated: %s; %d audience theories mined", validated, len(theories))
    for rank, theory in enumerate(theories, start=1):
        logger.info("  theory #%d: %s", rank, theory)
    top_theory = theories[0] if theories else "the masked mentor is a fox spirit"

    # ------------------------------------------------------------ c. lore graph
    logger.info("[3/9] Building canon lore graph (persisted to lore_graph/)")
    lore = LoreGraphManager(persist_path=tree["lore_graph"] / "canon.json")
    lore.add_lore_entity(
        CONCEPT_ID,
        {"entity_type": "concept", "logline": "a disgraced ronin and a mentor who casts no steps"},
    )
    lore.add_lore_entity(
        "kael", {"entity_type": "character", "power_level": 6, "signature": "Duskfang Blade"}
    )
    lore.add_lore_entity(
        "masked-mentor",
        {"entity_type": "character", "power_level": 9, "secret": "never steps on sacred ground"},
    )
    lore.add_lore_entity("ember-shrine-order", {"entity_type": "faction", "alignment": "guardian"})
    lore.add_lore_entity("duskfang-blade", {"entity_type": "artifact", "state": "cracked"})
    lore.add_lore_entity("lantern-festival-grounds", {"entity_type": "location", "mood": "dusk"})
    lore.add_lore_relation("kael", "masked-mentor", "ALLIED_WITH", {"since": "episode-0"})
    lore.add_lore_relation("kael", "duskfang-blade", "WIELDS")
    lore.add_lore_relation("kael", "lantern-festival-grounds", "LOCATED_AT")
    lore.add_lore_relation("masked-mentor", "ember-shrine-order", "MEMBER_OF")
    lore.add_lore_relation("kael", CONCEPT_ID, "MEMBER_OF")
    lore.save()
    canon = lore.query_canon_context(CONCEPT_ID, max_chars=2500)
    constrained_context = lore.inject_audience_theory_constraint(canon, top_theory)
    logger.info(
        "canon brief: %d chars (%d entities, %d relations); top theory injected as constraint",
        len(canon),
        lore.entity_count,
        lore.relation_count,
    )

    # ------------------------------------------------- d. LLM stage + writers' room
    logger.info("[4/9] Acquiring LLM stage and drafting the episode script")
    coordinator = LocalPoolCoordinator(
        config=config, auto_evict=True, dpo_log_path=tree["logs"] / "dpo_feedback.jsonl"
    )
    script_source = "ollama"
    async with coordinator.stage(Stage.LLM):
        room = WritersRoom(llm=coordinator.run_ollama_qwen)
        try:
            draft = await room.draft_episode_script(PREMISE, constrained_context, [])
            script = str(draft["script"])
        except OllamaUnavailableError as exc:
            logger.warning("Ollama unreachable — using offline stub (%s)", exc)
            script = offline_stub_script(PREMISE, top_theory)
            script_source = "offline-stub"
    logger.info("script drafted via %s (%d chars)", script_source, len(script))

    # --------------------------------------------------------- e. DPO feedback
    logger.info("[5/9] Registering shot renders and logging DPO latent feedback")
    dpo = coordinator.dpo
    assert dpo is not None  # dpo_log_path was provided above
    quality_gate = config.quality_gate
    char_weight, style_weight = quality_gate.effective_lora_weights()
    comfy_params: dict[str, Any] = {
        "ip_adapter_faceid_weight": quality_gate.ip_adapter_faceid_weight,
        "pulid_weight": quality_gate.pulid_weight,
        "controlnet_pose_strength": quality_gate.controlnet_pose_strength,
        "controlnet_pose_end_step": quality_gate.controlnet_pose_end_step,
        "character_lora_weight": char_weight,
        "style_lora_weight": style_weight,
        "wan_denoise": 0.30,
    }
    quality_gate.validate_wan_params(comfy_params["wan_denoise"])
    prompt_hash = hashlib.sha256(f"{CONCEPT_ID}/shot012/{PREMISE}".encode()).hexdigest()[:16]
    dpo.register_shot_render("shot012_seed41.mp4", prompt_hash, 41, 2.0, 6.5, comfy_params)
    dpo.register_shot_render("shot012_seed77.mp4", prompt_hash, 77, 2.0, 6.5, comfy_params)
    dpo.record_retention_result("shot012_seed41.mp4", prompt_hash, 41, 0.83)
    dpo.record_retention_result("shot012_seed77.mp4", prompt_hash, 77, 0.41)
    pair = dpo.log_dpo_latent_feedback("shot012_seed77.mp4", 4.2, comfy_params)
    if pair is not None:
        logger.info("DPO pair record:")
        print(json.dumps(pair, indent=2, sort_keys=True))
    else:
        logger.warning("no DPO pair formed (unpaired rejection logged)")

    # ------------------------------------------------------- f. asset metadata
    logger.info("[6/9] Writing asset metadata with quality-gate parameters")
    for asset_id, seed, retention in (
        ("shot012_seed41.mp4", 41, 0.83),
        ("shot012_seed77.mp4", 77, 0.41),
    ):
        path = ftm.write_asset_metadata(
            CONCEPT_ID,
            asset_id,
            {
                "prompt_hash": prompt_hash,
                "seed": seed,
                "mean_retention": retention,
                "quality_gate": comfy_params,
                "image_model": config.image_model_id,
                "video_model": config.video_model_id,
            },
        )
        logger.info("metadata written: %s", path)

    # --------------------------------------------------------- g. actor bundle
    logger.info("[7/9] Exporting the ronin_kael .synthetic_actor bundle")
    staging = tree["actors"] / "_staging"
    lora_path = atomic_write_bytes(
        staging / "ronin_kael_identity.safetensors", b"AKIO-PLACEHOLDER-TENSOR\x00" * 256
    )
    pose_path = atomic_write_json(
        staging / "pose_default.json", {"rig": "openpose-25", "keyframes": 12}
    )
    voice_path = atomic_write_json(
        staging / "voice_config.json",
        {"engine": "gpt-sovits", "language": "ja", "reference": "kael_ref.wav"},
    )
    exporter = SyntheticActorSDKExporter(tree["actors"])
    bundle_path = exporter.export_synthetic_actor(
        "ronin_kael",
        {"lora_weights": lora_path, "pose_rigs": pose_path, "voice_model": voice_path},
        manifest_extra={
            "danbooru_tag_anchors": ["1boy", "katana", "red_scarf", "topknot"],
            "power_level": 6,
            "model_specs": {
                "image_base": config.image_model_id,
                "video_base": config.video_model_id,
            },
        },
    )
    manifest = exporter.inspect_bundle(bundle_path)  # checksum round-trip verification
    logger.info(
        "bundle verified: %s (%d files, %d bytes)",
        bundle_path,
        len(manifest["files"]),
        bundle_path.stat().st_size,
    )

    # ----------------------------------------------------------- h. full purge
    logger.info("[8/9] Purging the unified memory pool (servers may be absent)")
    purge_report = await coordinator.purge_unified_memory_pool()

    # -------------------------------------------------------------- i. webhook
    webhook_status = "skipped (no --webhook-url)"
    if args.webhook_url:
        logger.info("[9/9] Dispatching greenlight webhook to %s", args.webhook_url)
        summary_payload: dict[str, object] = {
            "validated": validated,
            "posts_evaluated": len(posts),
            "top_theory": top_theory,
            "script_source": script_source,
        }
        try:
            await engine.dispatch_greenlight_webhook(
                CONCEPT_ID, summary_payload, args.webhook_url
            )
            webhook_status = "delivered"
        except WebhookDispatchError as exc:
            webhook_status = f"failed: {exc}"
            logger.warning("webhook dispatch failed: %s", exc)
    else:
        logger.info("[9/9] No --webhook-url given — skipping greenlight webhook dispatch")

    summary: dict[str, Any] = {
        "concept_id": CONCEPT_ID,
        "validated": validated,
        "theories": theories,
        "script_source": script_source,
        "script_preview": script[:200],
        "bundle_path": str(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "dpo_pair_logged": pair is not None,
        "dpo_log_path": str(dpo.log_path),
        "purge_report": purge_report,
        "webhook": webhook_status,
    }
    _print_summary(summary)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    """Print the human-readable final summary block."""
    line = "=" * 72
    print(f"\n{line}")
    print(f"AKIO STUDIO DEMO SUMMARY — concept {summary['concept_id']}")
    print(line)
    print(f"concept validated (greenlight): {summary['validated']}")
    print(f"audience theories mined ({len(summary['theories'])}):")
    for rank, theory in enumerate(summary["theories"], start=1):
        print(f"  {rank}. {theory}")
    print(f"script source: {summary['script_source']}")
    print(f"script preview (200 chars): {summary['script_preview']!r}")
    print(f"actor bundle: {summary['bundle_path']} ({summary['bundle_bytes']} bytes)")
    print(f"dpo pair logged: {summary['dpo_pair_logged']} -> {summary['dpo_log_path']}")
    print(f"purge report: {summary['purge_report']}")
    print(f"webhook: {summary['webhook']}")
    print(line)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line flags for the demo pipeline."""
    parser = argparse.ArgumentParser(
        prog="akio-studio",
        description="Akio Studio end-to-end demo pipeline (runs offline).",
    )
    # Default must be user-writable and cwd-independent: Dock/launchd start
    # processes with cwd '/', where a cwd-relative default would try to mkdir
    # on macOS's sealed read-only system volume and crash every app launch.
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=StudioConfig().base_dir,
        help="production tree root (default: ~/AkioStudio/productions)",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="optional https greenlight webhook URL; skipped when absent",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="placeholder for the macOS launcher dashboard; runs the demo pipeline",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="placeholder for the macOS launcher daemon mode; runs the demo pipeline",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.dashboard or args.daemon:
        logger.info(
            "--dashboard/--daemon are placeholders for the macOS launcher; "
            "running the demo pipeline"
        )
    asyncio.run(run_demo(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
