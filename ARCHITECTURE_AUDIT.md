# Akio Studio — Architecture Audit

Audit of `Akio_Studio_Master_Architecture.pdf` performed before implementation.
Every finding below is either **fixed in this codebase** or **documented as a
deliberate design change**. Findings are grouped by severity: *blocker* (would
not work as specified), *major* (would work badly — wasted memory, wrong
results in edge cases), *minor* (spec inconsistencies).

---

## Blockers (would not work as written)

### B1. The After Effects re-framing expression is mathematically wrong
The spec's expression calls `targetNull.toComp([0,0,0])`, which resolves the
null's position into the **16:9 master comp's** coordinate space — but then
uses that value directly as if it were a coordinate in the **9:16 vertical
comp**. The two spaces differ by the nested layer's scale (~177.8% for a 1080p
master filling a 1920 px-tall canvas), so the computed offset is wrong by that
factor. Its clamp bounds `[compWidth − scaledWidth, 0]` are additionally only
valid for a top-left anchor point; After Effects layers default to a **center**
anchor, so the clamp would let the frame slide off the canvas.

**Fix:** corrected expression in
`resources/after_effects/vertical_reframe_expression.jsx` — the focal offset is
scaled into vertical-comp pixels and the clamp bounds are derived from the
actual anchor point, so they hold for any anchor and any master resolution.

### B2. `networkx.DiGraph` silently loses lore relations
A `DiGraph` holds at most **one** edge per ordered node pair. Two entities
routinely need multiple relations (`RIVAL_OF` *and* `ALLIED_WITH` across arcs,
or a character who both `WIELDS` an artifact and is `LOCATED_AT` the same
place's node) — each `add_edge` would silently overwrite the previous relation,
corrupting canon. **Fix:** `lore_graph_agent.py` uses `networkx.MultiDiGraph`
keyed by relation type, so parallel relations coexist and re-adding the same
relation type updates rather than duplicates.

### B3. The "3-tier memory purge" mostly frees memory in the wrong process
On a 24 GB unified-memory machine the big allocations live in the **Ollama
server process** (~10 GiB for qwen2.5-coder:14b) and the **ComfyUI server
process** (10–14 GiB for image/video models). The spec's purge runs
`gc.collect()` and `torch.mps.empty_cache()` **inside the Python
orchestrator**, which holds almost nothing, and `sync`, which flushes *disk
write buffers* and does nothing for RAM at all. Worse, importing `torch` into
the orchestrator just to call `empty_cache()` would itself consume ~1–2 GiB of
the memory being "purged."

**Fix:** `pool_coordinator.py` purges where the memory actually is:
1. Ollama: `keep_alive: 0` via the HTTP API (with `ollama stop` CLI fallback)
   → unloads the LLM from the server process.
2. ComfyUI: `POST /free {"unload_models": true, "free_memory": true}` → unloads
   diffusion models from the ComfyUI process.
3. Orchestrator: `gc.collect()`, and `torch.mps.empty_cache()` **only if torch
   is already in `sys.modules`** (never imported for the purpose).
`sync` is dropped entirely.

### B4. There is no Info.plist "permission" for Metal/MPS
The spec requires "macOS permissions for Metal / MPS hardware access" in
`Info.plist`. No such key or entitlement exists — Metal requires no permission
for a normal app. What an Apple Silicon app **does** need (and the spec omits)
is **code signing**: unsigned ad-hoc bundles are the minimum for arm64
binaries, and Gatekeeper/quarantine will block a downloaded unsigned app.
**Fix:** `build_mac_app.sh` drops the fictional permission keys, sets real ones
(`LSMinimumSystemVersion`, `NSHighResolutionCapable`, `LSApplicationCategoryType`),
and ad-hoc signs the bundle (`codesign --force --deep -s -`).

### B5. Building directly into `/Applications` is fragile
Writing the bundle straight into `/Applications` can require admin rights and
leaves a half-built app behind on any failure. **Fix:** the bundler builds into
`./dist/AkioStudio.app`, verifies the bundle, then installs with
`ditto dist/AkioStudio.app /Applications/AkioStudio.app` as a final atomic-ish
step (with a clear fallback to `~/Applications` when `/Applications` is not
writable).

### B6. Dock-launched apps don't inherit the shell PATH
The launcher script as specified calls `ollama`, `python3`, etc. by bare name.
Apps launched from the Dock/Finder get a minimal PATH (`/usr/bin:/bin:...`) —
Homebrew's `/opt/homebrew/bin` is absent, so every external tool lookup fails.
**Fix:** the generated launcher resolves absolute paths
(`/opt/homebrew/bin`, `/usr/local/bin` fallbacks) and exports an explicit PATH
before booting the dashboard. Model/CLI paths in `config.py` carry the same
absolute-path candidates.

### B7. `PostMetrics` cannot express two of the four thresholds
The greenlight gate requires **midpoint retention ≥ 45%** and the
**3-consecutive-uploads** rule, but the spec's `PostMetrics` has *no midpoint
view counter and no timestamp*. Midpoint retention is uncomputable, and
"consecutive" is undefined without ordering. **Fix:** `PostMetrics` gains
`midpoint_view_count` and `posted_at`; validation sorts by `posted_at` and
evaluates the **most recent** N uploads (any-3-of-N would be a different, laxer
rule than the spec's intent).

## Major (works, but badly)

### M1. `ollama run` subprocess-per-call is the wrong interface
The CLI round-trips through the local HTTP server anyway, provides no
`system` prompt parameter, no `keep_alive` control (so model residency can't
be managed — the whole point of Pillar 5), and burns a process spawn per
generation. **Fix:** `pool_coordinator.py` talks to the Ollama **HTTP API**
(`/api/chat`) with explicit `keep_alive`, falling back to the CLI (via stdin,
with `communicate()` to avoid pipe-buffer deadlock) only when the API is
unreachable.

### M2. Nothing in the spec prevents co-loading models that overflow 24 GB
qwen2.5-coder:14b (~10 GiB) + an SDXL-class anime checkpoint stack (~10 GiB)
+ a WAN-class video model (~14 GiB) cannot co-reside in a 24 GB pool that
also runs macOS (~6 GiB). The spec gestures at purging but never *gates
loads*. **Fix:** the coordinator keeps a stage ledger
(`MODEL_FOOTPRINTS_GIB` in `config.py`) and `acquire_stage()` refuses — or
auto-purges, in `auto_evict` mode — before a load that would exceed the
usable budget. Writing, image, and video become strictly sequential stages,
which on this hardware they must be.

### M3. 23.976 fps is not 23.976
It is exactly **24000/1001**. Using the rounded float to map a retention
drop-off timestamp to a frame index drifts by a frame roughly every ~41
seconds of accumulated rounding across a timeline, mis-attributing DPO
"rejected" frames. **Fix:** `config.MASTER_FPS = Fraction(24000, 1001)`; the
DPO logger converts seconds → frame index with exact rational arithmetic.

### M4. sentence-transformers is an unnecessary third model
Loading an embedding model to cluster a few hundred comments adds memory
pressure to a pool that is the system's scarcest resource. **Fix:**
`mine_audience_lore_theories` uses lightweight lexical clustering (normalized
token-shingle similarity — pure stdlib) which is ample for "top fan theories
by volume," with an optional LLM-assisted refinement that runs **only while
the LLM stage is already resident**.

### M5. Neo4j is the wrong default for a single-user local app
Running a JVM database server to store a few thousand lore nodes on a laptop
whose memory is the bottleneck is pure overhead. **Fix:** NetworkX
`MultiDiGraph` + atomic JSON (node-link format) persistence is the default;
the storage layer is isolated behind `LoreGraphManager` so a Neo4j adapter
can be swapped in when a franchise actually outgrows in-memory scale.

### M6. Metric denominators must not mix
Retention family (R₃ₛ, R₅₀, CR) is per-**view**; Engagement Velocity is
per-**impression** (that's how platforms report them). The spec states the EV
formula correctly but never pins the retention denominators, and none of the
formulas guard division by zero. **Fix:** explicit denominators and zero-safe
math in `metrics_engine.py`, with validation errors on inconsistent counters
(e.g. `views > impressions`).

## Minor

- **"3-tier" purge lists four steps.** Renamed to what it is: a staged purge.
- **"WAN 2.7" / "Anima-Aesthetic 2B"** are unverifiable version pins;
  model identifiers are config values, not hard-coded strings.
- **PyInstaller/Platypus are unnecessary** for a launcher whose job is
  "activate venv, exec python": a plain shell `CFBundleExecutable` is more
  robust (no onefile unpack latency, no third-party GUI tool dependency) —
  `build_mac_app.sh` generates the bundle structure directly.
- **Atomic writes**: the `.tmp` file must be created in the destination
  directory (same filesystem — `os.replace` is only atomic within one) and
  fsynced before rename. Centralized in `akio_studio/_io.py`.
- **Webhook payload shape** differs per service (Discord `{"content": ...}`,
  Slack `{"text": ...}`); the dispatcher formats both and retries with
  exponential backoff. Standard-library HTTP in a worker thread — no aiohttp
  dependency for one POST.
- **Impact-frame/quality-gate numbers** (IP-Adapter 0.65–0.75, PuLID 0.60,
  LoRA sum ≤ 1.15, motion bucket 127–150, WAN denoise 0.25–0.35) are kept, but
  as a **validated** `ComfyQualityGate` config object so out-of-range params
  raise instead of silently drifting.

---

## Panel addenda (three-lens adversarial audit)

A parallel audit panel (memory feasibility / correctness / macOS packaging)
confirmed all findings above and added the following, all incorporated into the
implementation:

### A1. The full stack cannot co-reside — mutual exclusion is *the* invariant
Arithmetic over real footprints: macOS baseline 4–5 GiB; qwen2.5-coder:14b
Q4_K_M ≈ 9 GiB weights + KV cache (~10.5 GiB process total); an SDXL-class
quality-gate graph (UNet + dual CLIP encoders + ControlNet-DWPose + IP-Adapter
FaceID + PuLID + VAE + LoRAs + sampling activations) peaks at 11–14 GiB; and
macOS caps the *Metal working set* at roughly 75% of unified memory (~18 GiB)
shared across **all** GPU processes. Any two heavy stages overlap → compressor/
swap thrash or MPS allocation failure. Implemented: the coordinator's stage
ledger enforces one-resident-stage-at-a-time, the launcher sets
`OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_NUM_PARALLEL=1`, and purges are
**verified** (poll Ollama `/api/ps` until empty; ComfyUI `POST /free`) rather
than assumed.

### A2. "WAN 2.7" does not exist, and its quality-gate knob belongs to SVD
Open-weight WAN releases are 2.1/2.2; the 14B variants (~28 GiB fp16) can never
fit this machine. Worse, `motion_bucket_id 127–150` is **Stable Video
Diffusion** conditioning — WAN has no such parameter, so the spec's video gate
is unenforceable as written. Implemented: model IDs are config values pinned to
real, budgetable checkpoints (e.g. `Wan2.1-T2V-1.3B` / `Wan2.2-TI2V-5B` class),
and the quality gate keeps `motion_bucket` explicitly marked as SVD-only
alongside real WAN knobs (denoise 0.25–0.35, flow shift, frame count).

### A3. KV-cache/context budget was unaccounted
Qwen2.5-14B costs ≈192 KB/token of fp16 KV cache (~3 GiB at 16k context) on
top of weights; Ollama's default 4k `num_ctx` would silently truncate long lore
contexts. Implemented: `num_ctx` pinned explicitly (8k), a token budget in
`query_canon_context` (truncate the graph neighborhood, don't dump the whole
graph), and `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` in the
launcher.

### A4. DPO pairs must share identical conditioning
Pairing a well-retained asset from prompt A ("chosen") against a poorly
retained asset from prompt B ("rejected") teaches a confounded preference —
content differences, not visual quality. Implemented: `log_dpo_latent_feedback`
only forms pairs among renders of the **same shot** (same `prompt_hash`),
differing in seed/params, ranked by that shot's retention; shot timing metadata
(`start_second`, `end_second`, `prompt_hash`, `seed`) is persisted so a
drop-off second maps to the responsible shot. Frame indices are stored as exact
ints via `Fraction(24000, 1001)`. Local scope is **pair logging only** —
on-device DPO *training* of a 14B LLM is impossible in 24 GiB, and SDXL-scale
DPO on MPS (no bitsandbytes/xformers) is a multi-hour exclusive stage that
belongs offline/cloud.

### A5. The LoRA ranges contradict the LoRA cap
Spec allows char 0.70–0.80 + style 0.30–0.40, but 0.80 + 0.40 = 1.20 > the
stated 1.15 cap. Implemented: the cap is authoritative —
`ComfyQualityGate.validate()` scales the style weight down when the sum
exceeds 1.15.

### A6. One webhook payload cannot serve Discord and Slack
Discord requires `{"content": ...}` (204 on success), Slack requires
`{"text": ...}`; each 400s on the other's shape. Implemented: the dispatcher
branches on the webhook host, treats 204 as success, and honors `Retry-After`
on 429.

### A7. Unsigned arm64 binaries are SIGKILLed
On Apple Silicon the kernel kills unsigned native binaries ("Killed: 9") — the
spec never signs anything. Implemented in `build_mac_app.sh`: ad-hoc
`codesign --force --deep -s -` as the **final** build step (sufficient for a
locally built app; a Developer ID + notarization would be needed only for
distribution), `plutil -lint` validation, `lsregister -f` after install, and
`ditto` (not `cp`) for the install copy.

### A8. Mutable state must live outside the bundle
A venv created *inside* `/Applications/AkioStudio.app` would break the code
signature and may not be writable. Implemented: the launcher bootstraps the
venv under `~/Library/Application Support/AkioStudio/venv` on first run; logs
and the production tree default under user-writable locations; the bundle
stays immutable after signing.

### A9. Misc panel corrections
- `fsync` is not a stable-storage barrier on Darwin — `fcntl(F_FULLFSYNC)` is
  (implemented in `_io.py`, with fallback for filesystems that reject it).
- 4K 16-bit masters for 1080p deliverables store upscaler interpolation, not
  information (~66 MB/frame): documented recommendation is 1080p ProRes
  intermediates, 16-bit float reserved for genuine HDR grading shots.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is a CPU-fallback escape hatch, not an "MPS
  enabler" — kept in the launcher, documented as such.
- Symmetric lore relations (`ALLIED_WITH`, `ENEMY_OF`) are normalized at query
  time so canon queries from either side see them.
- Zero-view/zero-impression posts return "not evaluable" rather than crashing
  (`ZeroDivisionError`) or silently counting as threshold failures.
- Neo4j would be the only always-on multi-GB process outside the stage
  rotation — NetworkX + atomic JSON stays the default backend.

---

## Verdict

The architecture's *product* design (metrics-gated greenlighting, lore-graph
constrained writing, portable actor bundles, staged local pipeline) is sound
and was implemented as specified. Its *systems* design contained seven
would-not-work defects (B1–B7) — most stemming from one root error: treating
cross-process unified memory as if it were manageable from inside a single
Python process. The implemented system replaces that with per-engine APIs and
a verified stage ledger, which is both correct and *more* efficient: models
load exactly once per stage instead of once per call, and the orchestrator
itself stays torch-free and lightweight (~50 MB).

