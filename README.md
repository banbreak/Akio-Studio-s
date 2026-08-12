# Akio Studio

A data-driven, audience-reinforced **local** anime studio engine. Short-form
posts are metric-gated (retention + engagement velocity) before a concept earns
a 16:9 episode greenlight; audience fan theories are mined from comments and
promoted into a canon lore graph that constrains a four-persona writers' room;
rendering runs as strictly sequential model stages inside a 24 GiB Apple
Silicon unified-memory budget, with retention drop-offs logged as DPO
preference pairs and validated characters exported as portable
`.synthetic_actor` bundles.

The orchestrator itself is deliberately lightweight: **stdlib + networkx
only** — no torch, no HTTP client libraries, no embedding models. Heavy lifting
happens in the external Ollama / ComfyUI server processes, which the
coordinator gates and purges over their HTTP APIs.

## Module map

| Module | Role |
| --- | --- |
| `akio_studio/config.py` | All thresholds, model IDs, memory budgets, `MASTER_FPS = 24000/1001` |
| `akio_studio/exceptions.py` | `AkioStudioError` hierarchy |
| `akio_studio/_io.py` | Atomic write helpers (`.tmp` + `os.replace`, `F_FULLFSYNC` on Darwin) |
| `akio_studio/metrics_engine.py` | Metrics ingestion, R2E scoring, greenlight validation, fan-theory mining, greenlight webhook |
| `akio_studio/lore_graph_agent.py` | Canon lore graph (`MultiDiGraph` + atomic JSON) and the writers'-room persona chain |
| `akio_studio/pool_coordinator.py` | Stage ledger for the unified-memory pool, Ollama/ComfyUI purge, DPO feedback logger |
| `akio_studio/actor_sdk_exporter.py` | Portable `.synthetic_actor` bundles with streamed SHA-256 manifests |
| `akio_studio/file_tree_manager.py` | Production directory layout + atomic asset metadata |
| `main.py` | Composition layer — the only place the six modules meet; runnable demo |
| `build_mac_app.sh` | Native macOS app bundler (signed `.app`, launcher with explicit PATH) |

The feature modules never import each other; `main.py` composes them, and the
writers' room receives its LLM as an injected async callable.

## Quickstart

Linux / development:

```sh
pip install -e '.[dev]'
pytest              # unit tests, fully offline
python main.py      # end-to-end demo — green with NO Ollama / ComfyUI running
python main.py --base-dir /tmp/akio_demo --webhook-url https://discord.com/api/webhooks/...
```

macOS (Apple Silicon):

```sh
./build_mac_app.sh          # build, ad-hoc sign, install AkioStudio.app
./build_mac_app.sh --lint   # CI-safe: validate launcher + Info.plist only
```

`--dashboard` / `--daemon` are accepted by `main.py` as placeholders for the
macOS launcher and currently run the same demo pipeline.

## Memory stages

The whole stack can never co-reside in the 24 GiB pool, so residency is a
strictly sequential, verified rotation:

```
LLM (~10 GiB) -> purge Ollama (keep_alive:0, verified via /api/ps)
             -> IMAGE (~10 GiB) -> purge ComfyUI (POST /free)
             -> VIDEO (~14 GiB) -> purge ComfyUI (POST /free)
             -> EDIT (~8 GiB working set)
```

`LocalPoolCoordinator` enforces one-resident-stage-at-a-time and refuses (or
auto-evicts before) any load that would exceed the usable budget.

## What changed vs. the PDF spec

The original master-architecture PDF was audited before implementation; seven
would-not-work defects (B1–B7) and a dozen major/minor issues were corrected.
Highlights:

- **Purge where the memory lives**: Ollama/ComfyUI server APIs, not
  `gc.collect()`/`sync` inside the orchestrator (B3/A1).
- **`MultiDiGraph` canon**: parallel typed relations no longer overwrite each
  other (B2); Neo4j dropped for atomic node-link JSON (M5).
- **Exact 24000/1001 frame math** for DPO drop-off attribution (M3), and DPO
  pairs only between renders with identical conditioning (A4).
- **`PostMetrics` gained midpoint counts + timestamps** so all four greenlight
  gates are actually computable and "3 consecutive" is chronological (B7).
- **Real macOS packaging**: signed bundle, launcher-safe PATH, mutable state
  outside the app (B4–B6, A7, A8).

The full findings list, with rationale for every deviation, is in
[ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md).
