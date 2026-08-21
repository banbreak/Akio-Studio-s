#!/usr/bin/env python3
"""Configure and verify the RunPod endpoint for Akio Studio's video stage.

Usage::

    python scripts/runpod_setup.py --set-gpu-rate 0.89   # your real USD/hour
    python scripts/runpod_setup.py --check          # config + /health probe
    python scripts/runpod_setup.py --smoke          # + one real 33-frame render
    python scripts/runpod_setup.py --write-env-file # write cloud.env (chmod 600)
    python scripts/runpod_setup.py --reconcile      # list ambiguous submits

``--smoke`` starts a real job and therefore **spends money** (typically a few
cents); it asks for confirmation unless ``--yes`` is passed.

Credentials are read from the environment and never printed, and this script
never writes them into the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akio_studio.cloud_video import (  # noqa: E402
    CloudVideoRenderer,
    VideoRenderRequest,
)
from akio_studio.config import StudioConfig  # noqa: E402
from akio_studio.exceptions import AkioStudioError  # noqa: E402
from akio_studio.runpod_transport import (  # noqa: E402
    RunPodTransport,
    write_journal_summary,
)

OK = "  ok  "
BAD = " FAIL "
WARN = " warn "


def _mask(value: str) -> str:
    """Show only enough of a secret to confirm which one is loaded."""
    return f"{value[:4]}…{value[-2:]} ({len(value)} chars)" if len(value) > 8 else "set"


def check_config(config: StudioConfig) -> bool:
    """Report which pieces of configuration are present."""
    cloud = config.cloud_video
    print("Configuration")
    ok = True

    endpoint_id = cloud.resolve_runpod_endpoint_id()
    if endpoint_id:
        print(f"[{OK}] endpoint id      {endpoint_id}")
    else:
        ok = False
        print(f"[{BAD}] endpoint id      unset — export ${cloud.runpod_endpoint_id_env_var}")

    token = cloud.resolve_token() or os.environ.get("RUNPOD_API_KEY", "")
    if token:
        source = cloud.token_env_var if cloud.resolve_token() else "RUNPOD_API_KEY"
        print(f"[{OK}] api key          {_mask(token)} from ${source}")
    else:
        ok = False
        print(f"[{BAD}] api key          unset — export $RUNPOD_API_KEY")

    print(f"[{OK}] provider         {cloud.provider}")
    print(f"[{OK}] model            {cloud.model}")
    rate = cloud.resolve_gpu_cost_per_hour()
    ceiling = cloud.max_cost_usd
    if rate:
        source = (
            f"${cloud.runpod_gpu_rate_env_var}"
            if os.environ.get(cloud.runpod_gpu_rate_env_var)
            else "config"
        )
        print(
            f"[{OK}] gpu rate         ${rate:.4f}/hr (${rate / 60:.4f}/min) "
            f"from {source}"
        )
        if ceiling:
            minutes = ceiling / (rate / 60)
            print(
                f"[{OK}] cost ceiling     ${ceiling:.2f} per job "
                f"≈ {minutes:.0f} GPU-minutes before a job is cancelled"
            )
    else:
        ok = False
        print(f"[{BAD}] gpu rate         UNSET — cost tracking and the ceiling are inert")
        print("       RunPod reports milliseconds, not dollars, so without your")
        print("       actual rate a runaway job cannot be cancelled on cost.")
        print("       Fix: python scripts/runpod_setup.py --set-gpu-rate <USD/hr>")
        print("       Find it on the endpoint's page in the RunPod console")
        print("       (flex and active workers bill at different rates).")
        if ceiling:
            print(f"[{WARN}] cost ceiling    ${ceiling:.2f} configured but UNENFORCEABLE")
    print(f"[{OK}] wall-clock cap   {cloud.max_wait_s:.0f}s per job")
    return ok


async def check_health(config: StudioConfig) -> bool:
    """Probe ``GET /health`` on the configured endpoint."""
    print("\nEndpoint health")
    transport = RunPodTransport(config.cloud_video)
    try:
        health = await transport.health()
    except AkioStudioError as exc:
        print(f"[{BAD}] {exc}")
        return False
    except OSError as exc:
        print(f"[{BAD}] cannot reach RunPod: {exc}")
        return False

    workers = health.get("workers", {})
    jobs = health.get("jobs", {})
    print(f"[{OK}] reachable        workers={workers} jobs={jobs}")
    if not any(int(workers.get(k, 0) or 0) for k in ("ready", "idle", "running")):
        print(
            f"[{WARN}] no warm workers — first render pays a cold start "
            "(weights load from the network volume)"
        )
    return True


async def smoke_test(config: StudioConfig, assume_yes: bool) -> bool:
    """Submit one small real render to prove the whole path works."""
    print("\nSmoke test")
    cloud = config.cloud_video
    rate = cloud.resolve_gpu_cost_per_hour()
    estimate = f"${rate * (90 / 3600):.3f}" if rate else "unknown (no gpu rate set)"
    print(f"  This starts a REAL render (~33 frames). Rough cost: {estimate}.")
    if not assume_yes:
        reply = input("  Proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("  skipped")
            return True

    renderer = CloudVideoRenderer(config, transport=RunPodTransport(cloud))
    request = VideoRenderRequest(
        shot_id="akio_smoke_test.mp4",
        prompt=(
            "a lone ronin draws a katana in falling snow, anime key animation, "
            "crisp heavy linework, blue-tinted shadows"
        ),
        seed=1,
        prompt_hash="smoketest",
        num_frames=33,
        width=832,
        height=480,
        denoise=0.30,
    )
    dest = Path(os.environ.get("AKIO_SMOKE_DIR", "./runpod_smoke"))
    dest.mkdir(parents=True, exist_ok=True)
    try:
        result = await renderer.render_shot(request, dest)
    except AkioStudioError as exc:
        print(f"[{BAD}] {type(exc).__name__}: {exc}")
        return False

    size = result.output_path.stat().st_size if result.output_path else 0
    print(f"[{OK}] job {result.job_id} -> {result.output_path} ({size} bytes)")
    print(f"[{OK}] sha256 verified  {result.sha256}")
    print(
        f"[{OK}] wall clock       {result.wall_clock_s:.1f}s"
        + (f", cost ${result.cost_usd:.4f}" if result.cost_usd is not None else "")
    )
    return True


def _env_file_path() -> Path:
    """Where the launcher looks for credentials and per-machine settings."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "AkioStudio" / "cloud.env"
    return Path.home() / ".akio_studio" / "cloud.env"


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse an existing cloud.env into a dict (comments and blanks ignored)."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    """Write cloud.env with 0600 permissions from the moment of creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("# Akio Studio cloud video settings — keep chmod 600.\n")
        handle.write("# Sourced by the macOS launcher; outside the signed .app bundle.\n")
        for key in sorted(values):
            handle.write(f"{key}={values[key]}\n")


def set_gpu_rate(config: StudioConfig, rate: float) -> bool:
    """Persist the endpoint's real USD/hour rate to cloud.env."""
    if rate <= 0:
        print(f"[{BAD}] rate must be greater than 0 (got {rate})")
        return False
    cloud = config.cloud_video
    path = _env_file_path()
    values = _read_env_file(path)
    values[cloud.runpod_gpu_rate_env_var] = f"{rate:g}"
    _write_env_file(path, values)

    print(f"[{OK}] gpu rate set to ${rate:.4f}/hr (${rate / 60:.4f}/min)")
    print(f"[{OK}] written to {path} (mode {oct(path.stat().st_mode & 0o777)})")
    ceiling = cloud.max_cost_usd
    if ceiling:
        print(
            f"[{OK}] the ${ceiling:.2f} ceiling now cancels a job after about "
            f"{ceiling / (rate / 60):.0f} GPU-minutes"
        )
    print("\n  Active in this shell:")
    print(f"    export {cloud.runpod_gpu_rate_env_var}={rate:g}")
    return True


def write_env_file(config: StudioConfig) -> bool:
    """Write credentials (and any known rate) to the launcher's cloud.env."""
    cloud = config.cloud_video
    endpoint_id = cloud.resolve_runpod_endpoint_id()
    token = cloud.resolve_token() or os.environ.get("RUNPOD_API_KEY", "")
    if not (endpoint_id and token):
        print(f"[{BAD}] both endpoint id and api key must be set in the environment first")
        return False

    path = _env_file_path()
    values = _read_env_file(path)
    values["AKIO_RUNPOD_ENDPOINT_ID"] = endpoint_id
    values["RUNPOD_API_KEY"] = token
    rate = cloud.resolve_gpu_cost_per_hour()
    if rate:
        values[cloud.runpod_gpu_rate_env_var] = f"{rate:g}"
    _write_env_file(path, values)

    print(f"[{OK}] wrote {path} (mode {oct(path.stat().st_mode & 0o777)})")
    if not rate:
        print(f"[{WARN}] no gpu rate stored — run --set-gpu-rate <USD/hr>")
    print("      The macOS launcher sources this on start; it is outside the")
    print("      signed .app bundle, so the credential is never distributed.")
    return True


def reconcile(config: StudioConfig) -> bool:
    """Summarize journaled submissions whose fate is unknown."""
    print("\nSubmission journal")
    transport = RunPodTransport(config.cloud_video)
    journal = transport._journal_path  # noqa: SLF001 — operator tooling
    if not journal.exists():
        print(f"[{OK}] no journal yet at {journal}")
        return True
    out = journal.with_name("runpod_submits_summary.json")
    write_journal_summary(journal, out)
    import json

    summary = json.loads(out.read_text())
    print(f"[{OK}] {journal}")
    print(f"       phases: {summary['counts']}")
    if summary["ambiguous"]:
        print(f"[{WARN}] {len(summary['ambiguous'])} ambiguous submit(s) — these may")
        print("       have started and billed. Check the endpoint's Requests tab:")
        for record in summary["ambiguous"][-5:]:
            print(f"         {record.get('ts')} shot={record.get('shot')!r}")
    else:
        print(f"[{OK}] no ambiguous submissions")
    return True


async def main_async(args: argparse.Namespace) -> int:
    """Run the requested checks; returns a process exit code."""
    config = StudioConfig()
    print("Akio Studio — RunPod endpoint setup\n" + "=" * 44)

    if args.set_gpu_rate is not None:
        return 0 if set_gpu_rate(config, args.set_gpu_rate) else 1
    if args.write_env_file:
        return 0 if write_env_file(config) else 1
    if args.reconcile:
        return 0 if reconcile(config) else 1

    ok = check_config(config)
    if not ok:
        print("\nConfiguration incomplete — see README 'RunPod setup'.")
        return 1
    ok = await check_health(config) and ok
    if ok and args.smoke:
        ok = await smoke_test(config, args.yes) and ok

    print("\n" + ("All checks passed." if ok else "Some checks failed."))
    return 0 if ok else 1


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="config + health probe (default)")
    parser.add_argument(
        "--smoke", action="store_true", help="also run one real render (costs money)"
    )
    parser.add_argument("--yes", action="store_true", help="skip the smoke-test confirmation")
    parser.add_argument(
        "--set-gpu-rate",
        type=float,
        metavar="USD_PER_HOUR",
        help="persist the endpoint's real hourly rate (enables the cost ceiling)",
    )
    parser.add_argument(
        "--write-env-file", action="store_true", help="write cloud.env (chmod 600)"
    )
    parser.add_argument("--reconcile", action="store_true", help="summarize ambiguous submissions")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
