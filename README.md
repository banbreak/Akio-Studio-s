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
| `akio_studio/cloud_video.py` | Remote GPU video stage — submit/poll/download with idempotent submits, cost ceiling, and cancel-on-abort |
| `akio_studio/runpod_transport.py` | RunPod Serverless adapter (canonical contract → `/run`, `/status`, `/cancel`) |
| `runpod_worker/` | Deployable worker image: `handler.py`, `Dockerfile`, `requirements.txt` |
| `scripts/runpod_setup.py` | Endpoint config check, health probe, paid smoke test, credential file, reconciliation |
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

The local stack can never co-reside in the 24 GiB pool, so *local* residency
is a strictly sequential, verified rotation — while the video stage runs
off-device and therefore overlaps it:

```
local pool (exclusive, ~18 GiB usable):
  LLM (~10 GiB) -> purge Ollama (keep_alive:0, verified via /api/ps)
               -> IMAGE (~10 GiB) -> purge ComfyUI (POST /free)
               -> EDIT (~8 GiB working set)

cloud GPU (concurrent, ~0.2 GiB local for the HTTP client):
  VIDEO -> submit N seed variants -> poll -> download + verify -> store
```

`LocalPoolCoordinator` enforces one-resident-stage-at-a-time for local stages
and refuses (or auto-evicts before) any load that would exceed the usable
budget. `Stage.VIDEO_DIFFUSION` is exempt while `video_backend == "cloud"`:
it holds no local weights, so it neither evicts nor waits for local work.

### RunPod setup

The default provider is RunPod Serverless. Standing up an endpoint is four
steps; only the first needs the RunPod console.

**1. Build and push the worker image**

```bash
docker build -t <registry>/akio-video-worker:1.0.0 runpod_worker/
docker push <registry>/akio-video-worker:1.0.0
```

Weights are deliberately *not* baked into the image — a 14B checkpoint would
make a ~60 GB image and a punishing cold start.

**2. Create the endpoint (RunPod console)**

* **Serverless → New Endpoint**, container image = the tag you just pushed.
* GPU: 48 GB class (A40 / L40S) for `Wan2.2-T2V-A14B`; 24 GB (A5000 / 4090)
  is enough for the 5B/1.3B variants.
* **Attach a network volume** and set `AKIO_MODEL_CACHE=/runpod-volume/models`
  so weights download once, not per worker.
* Container disk ≥ 20 GB; set **idle timeout** low (5–10 s) — idle workers
  bill.
* Optional: set `AKIO_S3_BUCKET` (+ `AKIO_S3_ENDPOINT`, `AKIO_S3_REGION`) to
  return presigned URLs instead of inline base64. Recommended for anything
  longer than a few seconds — base64 inflates the payload ~33% and RunPod caps
  response size.

**3. Point the studio at it**

```bash
export AKIO_RUNPOD_ENDPOINT_ID="<id from the endpoint page>"
export RUNPOD_API_KEY="<Settings → API Keys>"
```

Set `runpod_gpu_cost_per_hour` in `CloudVideoConfig` to the rate you are
actually paying — RunPod reports milliseconds, not dollars, so the cost
ceiling is derived from that rate and cannot protect you if it is wrong.

**4. Verify**

```bash
python scripts/runpod_setup.py --check   # config + /health, spends nothing
python scripts/runpod_setup.py --smoke   # one real 33-frame render (~$0.02)
python scripts/runpod_setup.py --write-env-file   # cloud.env, chmod 600, for the .app
python scripts/runpod_setup.py --reconcile        # list ambiguous submissions
```

#### RunPod-specific caveats

* **No idempotency keys.** RunPod ignores `Idempotency-Key`, so submissions
  are attempted **exactly once** and never retried — a retry could start a
  second billable job. A submit that fails mid-flight raises
  `CloudSubmitAmbiguousError` (distinct from a plain failure: the job may be
  running *right now*) and is journaled to
  `~/Library/Application Support/AkioStudio/runpod_submits.jsonl`. Run
  `--reconcile`, then check the endpoint's Requests tab.
* **No cost field.** Cost is derived from `executionTime` + `delayTime`
  against your configured rate; while a job is in flight it is *estimated*
  from elapsed wall-clock so a runaway job still trips the ceiling.
* **Cold starts.** The first render after scale-to-zero loads the checkpoint
  from the network volume. Keep one worker warm if latency matters; it bills.

### Cloud video stage

A WAN-class 14B checkpoint is ~28 GiB in fp16 — larger than the entire
unified pool — so video renders remotely by default. Configure it with two
environment variables (never committed, never baked into the `.app`):

```bash
export AKIO_CLOUD_VIDEO_ENDPOINT="https://your-gpu-provider.example/v1"
export AKIO_CLOUD_VIDEO_TOKEN="..."      # bearer token, HTTPS enforced
```

On macOS the launcher also sources `~/Library/Application Support/AkioStudio/cloud.env`
(keep it `chmod 600`). With neither set, `main.py` renders through an
in-process mock transport so the demo runs offline and bills nothing.

The endpoint contract is provider-agnostic REST — `POST /jobs`,
`GET /jobs/{id}`, `POST /jobs/{id}/cancel`. For a provider that speaks a
different dialect, pass a custom `transport` to `CloudVideoRenderer` rather
than changing the pipeline.

Spend controls, because a rented GPU bills by the second:

* submissions carry an `Idempotency-Key` over the request contents, so a
  retried POST after a dropped connection cannot start a second billable
  render;
* `max_cost_usd` cancels a job that reports a cost above the ceiling;
* `max_wait_s` cancels a job that outlives its budget — never abandons it;
* task cancellation propagates to a remote cancel;
* auth failures (401/403) fail fast instead of burning retries;
* out-of-gate parameters are rejected client-side, before anything is billed;
* downloads are checksum-verified, so a truncated fetch is never mistaken
  for a finished shot.

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
