# Design Decisions

**This is the design log for the SYS-304 House Prices project.**

Whenever we make a meaningful design choice (architecture, stack, API shape,
UX, testing, deploy, tradeoffs), record it **here** — not only in chat or
commit messages. Prefer appending a dated section under the relevant milestone
(or a new milestone heading) rather than rewriting history silently.

### How to add an entry

1. Add or extend a section under the current milestone (or create `## Milestone N`).
2. State **what** was chosen and **why** (one short paragraph or bullets).
3. If something was rejected, note the tradeoff in the milestone’s tradeoffs table
   or in a short “Alternatives considered” bullet.
4. Keep the [README.md](README.md) for *how to run* and file maps;
   keep this file for *why we built it this way*.

---

## Milestone 1 — Problem scoping & prototype training

Source of truth: [`model_training/prototype.ipynb`](model_training/prototype.ipynb)
(later mirrored by [`model_training/train_baseline.py`](model_training/train_baseline.py)
so CI/Docker can retrain without Jupyter).

### Goal

Build a **naive but complete** tabular regression prototype: load Ames housing
data → light EDA → minimal preprocessing → baseline booster → holdout metrics →
serialize the full pipeline for later serving.

### Data

- Load `dataset/train.csv` (Kaggle House Prices schema): **1,460 rows × 81
  columns** (79 features + `Id` + `SalePrice`).
- Drop `Id` from features; target is `SalePrice`.
- 80/20 train/test split with `random_state=42` (~1,168 / 292 rows).

### EDA (what informed the baseline)

- **Missingness audit** — many columns are mostly null (`PoolQC`, `MiscFeature`,
  `Alley`, `Fence`, …). For the prototype these are handled by simple
  imputation rather than domain-specific “None means no pool” encoding (that
  richer treatment appears in the experimental notebook, not the shipped
  baseline).
- **Target** — plot `SalePrice` and `log1p(SalePrice)` to see skew (competition
  metric is log-space RMSLE).
- **Correlations / key plots** — numeric correlation heatmap plus focused plots
  for strong predictors (`GrLivArea`, `OverallQual`, `YearBuilt`) to confirm
  the problem has learnable signal before modeling.

### Preprocessing design

Intentionally **minimal** so the prototype stays easy to ship and debug:

| Feature type | Handling |
|--------------|----------|
| Numeric | `SimpleImputer(strategy="median")` |
| Categorical | most-frequent impute → `OneHotEncoder(handle_unknown="ignore")` |

Wired with `ColumnTransformer` inside a single sklearn `Pipeline` so fit/transform
stays consistent between training and later inference.

### Model

- **`XGBRegressor`** as the baseline regressor (strong default for mixed
  tabular data; no heavy tuning required for a Milestone 1 proof-of-concept).
- Hyperparameters chosen as light / “default-ish”, **not** grid-searched:
  - `n_estimators=300`
  - `max_depth=4`
  - `learning_rate=0.05`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `random_state=42`, `n_jobs=-1`
- Full object trained: `Pipeline([preprocessor, regressor])`.

### Evaluation

Holdout metrics printed in the notebook (same split as above):

| Metric | Notebook result | Why it matters |
|--------|-----------------|----------------|
| MAE | ~$15,490 | Interpretable dollar error |
| RMSE | ~$24,669 | Penalizes large misses |
| R² | ~0.921 | Variance explained on holdout |
| RMSLE | ~0.132 | Kaggle leaderboard-style metric (`log1p` of clipped preds) |

Also: predicted-vs-actual scatter and top-15 XGBoost feature importances (after
expanding one-hot names) for a quick sanity check that quality/size signals
dominate.

### Serialization & smoke test

- Save with `joblib` to `models/baseline_xgb_pipeline.pkl` (preprocessing +
  model together).
- Reload from disk and score a few holdout rows to prove inference works
  without the training notebook state.

### Explicit non-goals (called out in the notebook)

Left for later on purpose: hyperparameter search, hand-crafted features
(total SF, age, interactions), other model families (LightGBM / NN), modeling
`log1p(SalePrice)` directly, and cross-validation instead of a single split.

---

## Milestone 2 — Deploying the model service

Wrapping the Phase 1 house-price model in an API, UI, containers, tests, and CI.

### Overall approach

Treat Milestone 1 as the source of truth for the model, and Milestone 2 as a thin,
reproducible serving layer around it. Prefer simple, course-friendly defaults
(FastAPI, static HTML, Docker Compose, GitHub Actions) over extra frameworks or
cloud deploy.

### Model & training

- **Same pipeline as the notebook** — median impute + most-frequent/one-hot +
  untuned `XGBRegressor`, so the API doesn’t invent a different model.
- **Separate `train_baseline.py`** — CI/Docker can train without Jupyter;
  notebooks stay for EDA/demo.
- **Save full sklearn `Pipeline`** — preprocessing stays glued to the model; no
  duplicated feature logic in the API.
- **Also write `feature_schema.json`** — feature name lists + a sample row for
  `/schema` and the UI “Load sample” button.
- **Partial inputs allowed** — missing keys become `NaN` and the imputer fills
  them, so the form doesn’t need all ~79 fields.

### Backend

- **FastAPI** — clear OpenAPI docs at `/docs`, easy `TestClient` tests, fits the
  assignment.
- **Endpoints** — `/predict` (core), `/health` (compose/CI), `/schema` (UI sample).
- **`GET /` redirects to `/docs`** — visiting the API root in a browser used to
  404 because only `/health`, `/schema`, and `/predict` existed; redirect avoids
  that footgun in uvicorn dev mode.
- **Request shape `{ "features": { ... } }`** — one nested object, easier to
  extend than a giant flat pydantic model.
- **CORS open (`*`)** — local UI on another origin/port without friction for the
  course demo.
- **Mutate `feature_schema` in place** — avoids the import/reference bug when
  tests reload artifacts.

### Frontend

- **Static HTML/JS/CSS** — no React/Node build; easy Docker (nginx) and demos.
- **Key features only** — quality, area, year, baths, neighborhood, etc.; rest
  imputed.
- **`config.js` for API base URL** — local static server and Docker both often
  use host port 3000, so port-based `/api` detection was wrong: requests hit the
  Python `http.server` (GET → 404, POST → 501) instead of uvicorn. Default
  `config.js` points at `http://localhost:8000`; the Docker image copies
  `config.docker.js` as `config.js` with `API_BASE = "/api"` for the nginx proxy.

### Containers & deploy

- **Separate Dockerfiles** — matches the “frontend + backend” requirement.
- **Compose + `deploy.sh`** — one-command local CD; train model if the pickle is
  missing, then `up --build`.
- **Mount `models/` into backend** — swap/retrain without rebuilding the image
  every time.
- **Healthcheck + `depends_on`** — frontend waits until the API is actually up.

### Tests & CI

- **Unit tests** — feature-frame / missing-value behavior (pure logic).
- **Integration tests** — `/predict` → 200 + price; `/health`; UI contains form
  and calls `/predict` (assignment’s “≥2 integration” bar without heavy browser
  automation).
- **Ruff + pytest on `main`** — lint then tests; train in CI only if the pickle
  isn’t present so the job stays self-contained.

### Repo & docs

- **Layout** — `backend/`, `frontend/`, `models/`, `tests/`, `dataset/`,
  `model_training/` instead of the old README’s `data/` + root notebook story.
- **`README.md`** — runbook, file map, and how to run the stack without
  duplicating design rationale (that lives here). Merged former
  `MILESTONE2_GUIDE.md` into the README and dropped the “what Milestone 2 does”
  narrative so it isn’t duplicated with this file.
- **`sys-304` conda env** — isolate course deps from system Python.

### Tradeoffs accepted on purpose

| Choice | Tradeoff |
|--------|----------|
| Untuned XGB baseline | Not SOTA; good enough for a working service |
| Partial feature form | Less control than full 79-field UI; much more usable |
| Static UI | Less polish than React; zero frontend toolchain |
| Local Docker CD only | No required cloud deploy (assignment marks cloud optional) |
| UI “integration” via file/API checks | Not full Selenium/Playwright; still meets the spirit of the rubric with less flake |

---

## Milestone 3: Architectural scaling and optimization

**Current design (2026-09-27).** The service keeps the fitted Phase 2 model and feature schema unchanged. This milestone changes model serialization and inference execution; it does not train a smaller model.

### Model Optimizations

1. **Native XGBoost format (UBJ).** The frozen booster is exported to XGBoost's native UBJ format and loaded through its native prediction API.
2. **Fused preprocessing.** The serving code applies the fitted imputation and category mappings, builds sparse CSR input, and sends it directly to the booster. This removes pandas, scikit-learn, and joblib from the inference path.

These techniques target inference time and runtime dependencies. They do not change the learned weights. The project did not use quantization, distillation, or pruning. Export checks prediction parity with the frozen pipeline across 2,919 dataset rows and edge cases; the maximum prediction difference was zero.

| Model measurement | Phase 2 pipeline | Native inference |
|---|---:|---:|
| One-row p50 latency | 22.13 ms | 0.53 ms |
| 64-row p50 latency | 23.82 ms | 4.50 ms |

The native archive is 95,850 B. The original uncompressed Phase 2 file is 492,927 B, but an equally compressed baseline is 102,904 B. The fair size difference is only 7,054 B (about 6.9 KiB), so this is not a meaningful model-weight compression result. Full accuracy, memory, and benchmark details are in [report/phase3_report.md](report/phase3_report.md).

### System Optimizations

1. **Exact-input TTL/LRU cache with in-flight coalescing.** Enabled by default. It avoids repeat inference and shares work for simultaneous identical requests. Cache state is local to one API process.
2. **Bounded dynamic batching.** Implemented as an optional queue, with a maximum batch of 16 rows and capacity of 256. It is disabled by default because the measured ablation reduced throughput; it is not claimed as a default performance improvement.

Compose uses one Uvicorn worker so the process-local cache is shared. The selected default is native inference plus caching, with batching off. The HTTP measurements are local comparisons, not production capacity guarantees.

| Setting | Default |
|---|---:|
| Cache enabled | `PREDICTION_CACHE_ENABLED=1` |
| Cache capacity / lifetime | 1,024 entries / 300 seconds |
| Dynamic batching | `DYNAMIC_BATCHING_ENABLED=0` |
| Optional batch size / queue capacity | 16 rows / 256 requests |
| Compose Uvicorn workers | 1 |

### Superseded approach

An earlier Milestone 3 attempt trained `models/optimized_xgb_pipeline.pkl` with domain feature engineering and a `log1p(SalePrice)` target, then served it through a scikit-learn pipeline. That artifact is retained only because `model_training/benchmark_models.py` compares it with the frozen Phase 2 pipeline and the current native export; the saved comparison is in `benchmarks/model_benchmark.json`. It is not loaded by the API, is not rebuilt by deployment, and is not the current default. Its slower inference and the added preprocessing work motivated the native runtime redesign. The old training, model-family, notebook, and PDF-builder files are not part of the current workflow.

### Deployment limits

The amd64 Linux image uses CPU-only XGBoost and runtime-only dependencies. Compose configuration was checked, but a Docker engine was unavailable, so image build and container execution remain unverified.

---
## Related docs

- [README.md](README.md) — project overview, file map, and how to run

---

## Milestone 4 — Audited continual learning and guarded serving (2026-10-05)

The user authorized implementation of the continual-learning plan. This milestone adds durable request events, a static operations dashboard, verified delayed labels, scheduled champion/challenger evaluation, immutable model versions, and backend-owned activation. It protects the frozen baseline archive and feature schema; a candidate that passes the gate may change the active booster version.

### Data and evaluation boundaries

The source training CSV is pinned by SHA-256 1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41. The deterministic split manifest identifies 1,168 original fit rows, 146 adaptation rows, 73 promotion-gate rows, and 73 final-audit rows. The reference histogram is built only from the 1,168 original fit rows. The native baseline archive and schema remain frozen and hash-checked.

The split is a provenance and eligibility boundary, not a claim of untouched validation data. The 73 gate rows and the earlier 292-row holdout were already used in prior project evaluation. The gate is reused only to apply a fixed paired policy. The final audit was evaluated once after selection; it did not affect model choice and is descriptive rather than fresh external accuracy evidence.

Only an unchanged adaptation example can become training data. A private trainer-side importer checks the source dataset hash before reading the target, split membership, exact canonical feature fingerprint, successful persisted prediction, request completion, and configured label delay. Synthetic drift, changed feature records, repeated observations, previous fit rows, gate rows, audit rows, failed requests, and predictions themselves are never ground truth.

### Request path and monitoring

~~~mermaid
flowchart LR
    Browser[Browser or HTTP client] --> Form[Static prediction form]
    Form -->|POST /predict| API[FastAPI, one Uvicorn worker]
    API --> Admit[Durable request admission]
    Admit --> DB[(SQLite WAL request and label store)]
    API --> Snapshot[Capture model and schema version]
    Snapshot --> Cache[Version-qualified TTL/LRU cache]
    Cache -->|hit or coalesced result| Reply[Prepared response]
    Cache -->|miss| Native[Native preprocessing and UBJ prediction]
    Native --> Reply
    Reply --> Complete[Durable completion audit]
    Complete --> Browser
    Dashboard[Operations dashboard] -->|summary window| API
~~~

Every HTTP route is audited with a server-generated UUID. Prediction bodies are bounded to 65,536 bytes; malformed and oversized inputs retain bounded parse context. Admission and completion writes are acknowledged. When logging cannot be acknowledged, the API returns a bounded 503 and marks the outcome as logging-failed where persistence is possible. A request row records the prepared HTTP status, not proof that the client received the response. A crash may leave an admitted request pending.

The event writer is bounded (256 queued commands, 250 ms acknowledgement budget) and uses the backend-owned SQLite schema. Completed-response error rates include logged 4xx/5xx outcomes; pending and aborted arrivals are separate. The monitoring summary covers only prediction requests, so health checks, the dashboard's own polling, and admin calls do not inflate prediction volume. Handler latency is the server-side measurement with its own audit timing boundary. It is distinct from end-to-end client latency, which includes HTTP, network, and durable audit acknowledgements.

The dashboard polls every five seconds and supports 15-minute, 60-minute, and 24-hour UTC windows. It distinguishes absent fields, explicit null values, empty traffic, and insufficient drift samples. It shows request counts, latency, completed-response errors, cache outcomes, logging failures, active version, delayed labels, latest training outcome, drift, and per-feature missing/unknown counts. Stale responses are discarded after window changes; the latest requested refresh is queued.

The immutable drift reference uses base-2 Jensen-Shannon distance, the square root of Jensen-Shannon divergence. Numeric and categorical buckets include missing values and out-of-range or unknown values. A score is advisory only: at least 100 valid prediction observations are required and the default threshold is 0.10. Drift is an input-distribution signal, not proof of target error or model degradation.


### Delayed labels, retraining, and serving versions

~~~mermaid
flowchart LR
    Workload[Adaptation workload over HTTP] --> Audit[Completed request and source provenance]
    Audit --> Delay[Private truth file waits for configured delay]
    Delay --> Import[Trainer verifies hash, row, features, completion and delay]
    Import --> Labels[(Verified labels in SQLite)]
    Labels --> Cutoff[Timed trainer snapshots eligible rows at fixed cutoff]
    Cutoff --> Lease[Owned trainer lease and immutable snapshot hash]
    Lease --> Candidate[Continue private booster copy]
    Candidate --> Gate{Paired 73-row gate}
    Gate -->|Reject, skip, or fail| Run[Record terminal run verdict]
    Gate -->|At least 1% RMSLE gain and RMSE no worse| Registry[Publish immutable version and manifest]
    Registry --> API[Token-protected activation with parent CAS and lease fence]
    API --> Pointer[Backend atomically publishes current.json]
    Pointer --> Active[New requests capture the new serving snapshot]
    Run --> Dashboard[Dashboard shows outcome and active version]
    Active --> Dashboard
~~~

The scheduler is time-triggered. It needs at least 32 new verified observations and a cooldown after a rejected snapshot. The trainer fixes a label-availability cutoff, deduplicates observations, records the eligible and new IDs with the snapshot hash, and trains a private booster copy. It keeps the original preprocessing coordinates and model objective, adds 25 rounds at learning rate 0.01, and respects a cumulative tree-growth cap of 600 by default.

A candidate must produce finite nonnegative predictions, improve RMSLE by at least 1% relative to the champion on the same 73 gate rows, and have RMSE no worse than the champion. Export, load, schema/preprocessing, and representative-edge checks also apply. A candidate can therefore be rejected even when labels are present. The evaluation threshold is a fixed operating policy on a small reused gate, not statistical proof. The system never degrades a real incumbent to create a promotion.

Training owns immutable registry version directories and manifests. The API alone owns the serving pointer. Candidate activation requires an expected-parent compare-and-swap, matching candidate manifest, and a live trainer-run lease fence; the API checks the lease while the publication transaction is protected. Pointer replacement is atomic and reconciled against activation intent after restart. An activation whose outcome remains unknown blocks new trainer runs until reconciliation. Lease heartbeats check current ownership using a fresh clock value while holding the database update lock. Repeating an already committed activation returns that activation's historical result even if a later activation has changed the current pointer; the current model must be read separately from health or the pointer. Rollback targets the previous served version or the frozen bootstrap, using the same checked API path. Each in-flight prediction retains the model/schema snapshot it captured; version-qualified cache keys prevent results crossing activations.

### Deployment choices and remaining bottlenecks

Compose uses a one-shot trainer-image bootstrap, the API, a scheduled trainer, and nginx. Named volumes persist request data, immutable artifacts, and backend serving state. The API mounts artifacts read-only; the trainer writes immutable versions and reads serving state but does not publish current.json. The shared ADMIN_TOKEN is required by trusted activation calls, and admin paths are not exposed through nginx. The API is one Uvicorn worker, the exact-input cache is on, and dynamic batching remains off. Local runtime state defaults outside the synced repository.

SQLite is appropriate for this one-host demonstration but has one serialized writer and a shared local file. The API is a single process because cache/coalescing state is process-local. A growing event history, database-writer pressure, prediction concurrency, training CPU contention, label quality, and dashboard aggregation are the main constraints.

For roughly 100 times the request volume, first establish an SLO and repeat the workload on the intended hardware. Then move event persistence to PostgreSQL and a durable ingestion/outbox path, run API replicas behind a load balancer, and move training to isolated workers with explicit CPU/memory limits. Publish immutable artifacts to remote object storage and coordinate activations through a versioned control plane, retaining parent checks and per-request snapshots. Either use a shared cache keyed by model version or document the consistency/cost of replica-local caches. Pre-aggregate monitoring windows, set explicit retention, and validate with fresh temporal labels before rollout. These are scaling proposals; the current implementation is not a 100x capacity test.

| Current constraint | Implemented protection | 100x direction |
|---|---|---|
| SQLite write serialization and history growth | Bounded acknowledged writer, health visibility, explicit archive/checkpoint helpers | PostgreSQL plus durable event ingestion and retention |
| Single API process and process-local cache | One captured model version per request; bounded cache and queue behavior | Multiple API replicas with coordinated version rollout and measured shared/local cache |
| Training competes for CPU and shared database | Separate trainer container, singleton lease, fixed snapshot cutoff and tree cap | Isolated training workers, resource quotas, queued jobs and object-backed immutable registry |
| Sparse, delayed, or incorrect labels | Hash-pinned source rows, private truth import, delay/cutoff and minimum count | Fresh temporal labels, label-quality monitoring, stronger source integration |
| Small, historically evaluated promotion gate | Fixed paired RMSLE/RMSE policy and one reserved descriptive audit | Fresh temporal or external holdout, canary/shadow evaluation |
| Lost activation acknowledgements or pointer/database split | Idempotency ID, parent CAS, atomic API-owned pointer and restart reconciliation | Replicated control plane, canary rollout and automated rollback |
| Repeated dashboard queries on event history | Indexed bounded windows and small static dashboard | Materialized summaries and separate monitoring storage |


### October 5 measurement and verification record

The matched logging measurement is stored in [benchmarks/logging_overhead.json](benchmarks/logging_overhead.json). It contains 24 fresh-process cases: two workloads (unique and repeated), two concurrency levels (1 and 8), three trials, and logging on/off. Each case requested 160 predictions, for 3,840 calls total and zero errors. Every case used the same dataset and native model, one Uvicorn worker, cache enabled, and dynamic batching disabled.

The client figures are medians across the three trials. Values are logging on / logging off:

| Workload | Concurrency | Client p50 (ms) | Client p95 (ms) | Successful requests/s |
|---|---:|---:|---:|---:|
| Unique | 1 | 4.24 / 2.15 | 7.66 / 3.25 | 209.17 / 436.00 |
| Unique | 8 | 15.91 / 10.23 | 24.95 / 16.89 | 455.27 / 729.17 |
| Repeated | 1 | 6.51 / 1.09 | 9.71 / 2.42 | 140.57 / 754.02 |
| Repeated | 8 | 36.85 / 5.01 | 46.45 / 22.61 | 208.32 / 848.63 |

The cost is visible in this workload, particularly on repeated requests where cached inference is short but both durable audit acknowledgements remain. These are client round-trip measurements, including HTTP and audit writes. The benchmark does not contain per-request handler percentiles; dashboard handler latency has a different server-side boundary. Do not compare the two directly or treat these results as a capacity guarantee.

The benchmark environment was Windows 11 build 26300 with Python 3.12.14. It used dataset SHA-256 1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41 and native archive SHA-256 4e605ea450b01531142398e273729f326df625ae21b8f0a5235f030382721db4. The Docker stack was paused during the measurement; these values are not container/Linux latency.

The protected exporter produced exact parity on 1,460 train rows, 1,459 test rows, and 3 edge records with a maximum absolute difference of zero; the frozen archive was preserved. The strict split-manifest check and root global Ruff check passed in the recorded verification run. Linux images built and started under an isolated Compose project. Real HTTP baseline and shifted traffic plus browser dashboard states were checked: the shifted run raised Neighborhood distance from 0.4502 to 0.5155 and exposed 40 missing and 40 unknown values, with no labels created. The dashboard also rendered empty traffic, insufficient samples, a selected 60-minute window, populated drift, and monitoring unavailability while the stack was paused.

The final combined suite passed 100 tests with six deprecation warnings in 39.16 seconds, and global Ruff passed. The protected exporter matched all 1,460 train rows, 1,459 test rows, and three edge cases exactly while preserving the frozen source archive. A fresh isolated Linux Compose project completed the real HTTP smoke: 40 delayed labels were imported, the real candidate was correctly rejected by the 1% RMSLE gate, the explicitly tagged same-model fixture committed activation, and exact request UUID, imported label IDs, model version, and pointer state survived backend restart. A separate rollback check returned the serving pointer to the frozen bootstrap; repeating the same activation ID returned its committed result, and a stale expected parent received HTTP 409. The final audit ran once after selection and was not used for model selection. In the browser, API unavailability showed a stale-data banner that cleared after recovery; the recovered dashboard showed the bootstrap and all 40 imported labels. The form loaded the schema sample with `Sawyer` selected and successfully displayed a $143,950 prediction. Remote GitHub Actions was not run.

A private retrospective feasibility check used the first 32 adaptation labels with the 1,168 fit rows and the fixed 73-row gate. RMSLE changed from 0.1527863 to 0.1509785 (1.183% relative gain), and RMSE changed from $24,598.10 to $24,510.42. This did not use live HTTP-delayed truth, publish a candidate, or activate a model; the reused small gate is not independent evidence of generalization. In the live run, the champion/challenger RMSLE values were 0.1527862786/0.1517689782, a 0.6658% relative gain below policy; RMSE improved from $24,598.10 to $24,548.88. The live candidate was not promoted, and the frozen bootstrap remains selected after the rollback check. The final audit then recorded RMSLE 0.1385911, MAE $16,402.36, and RMSE $24,117.92 on 73 rows, with no role in selection; these rows were part of the historically evaluated holdout.

The proposed 100x architecture is a roadmap, not a measured result. The Compose demonstration flow was exercised; no presentation is claimed here. Full evidence is indexed in [report/evidence](report/evidence/README.md).

