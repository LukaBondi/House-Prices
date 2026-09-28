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
