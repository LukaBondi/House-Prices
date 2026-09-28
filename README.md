# House Prices — SYS-304

FastAPI service that predicts Ames home sale prices from the Kaggle [House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The current serving model is a native XGBoost export of the frozen Phase 2 pipeline. Design rationale and superseded experiments are recorded in [DESIGN.md](DESIGN.md); measured Phase 3 results are in [report/phase3_report.md](report/phase3_report.md).

## Run locally

Use the project environment, or another Python environment with `backend/requirements.txt` installed:

```powershell
conda activate sys-304
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Open `http://localhost:8000/docs`. The health check is `http://localhost:8000/health`, and `GET /schema` returns the feature schema and example input. To run the static form, open a second terminal and run `python -m http.server 3000 --directory frontend`; visit `http://localhost:3000`.

## Run with Docker Compose

```powershell
docker compose up --build
```

The UI is at `http://localhost:3000`, and the API is at `http://localhost:8000`. Stop the stack with `docker compose down`. On amd64 Linux, the backend image uses the CPU-only XGBoost package and the smaller `backend/requirements-runtime.txt`; other platforms use the regular XGBoost package. Local development and benchmarks use the full requirements file. Container execution requires Docker Desktop's Linux engine to be running.

## Predict

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"features":{"OverallQual":7,"GrLivArea":1710,"YearBuilt":2003,"Neighborhood":"CollgCr"}}'
```

Features can be omitted and will use the baseline pipeline's fitted imputation behavior. Unknown categories map to the same all-zero one-hot block used by the original encoder. The response contains `sale_price`.

## Current serving design

The service loads `models/house_prices_native.zip`, containing fitted preprocessing metadata and the XGBoost UBJ booster. Its inference path builds sparse CSR rows directly and does not require pandas, scikit-learn, or joblib. The export is generated from the frozen `models/baseline_xgb_pipeline.pkl`; it does not retrain or change the baseline model or feature schema.

The default request path uses a bounded exact-input TTL/LRU cache with in-flight coalescing, then predicts cache misses directly with the native model. Dynamic batching is available as an opt-in path (`DYNAMIC_BATCHING_ENABLED=1`); it groups queued rows into batches of up to 16, with a queue capped at 256 and one inference worker. The benchmark did not show a consistent batching benefit for this workload, so batching is disabled by default. Compose uses one Uvicorn worker so process-local state is shared. Environment settings are documented in [DESIGN.md](DESIGN.md) and declared in `backend/app.py`.

## Export and benchmark

Rebuild the native archive and verify prediction parity against the frozen baseline:

```powershell
python -m model_training.export_native
```

Run the fresh-process model benchmark, then reproduce the selected cache-only default HTTP run:

```powershell
python model_training/benchmark_models.py
python model_training/benchmark_api.py --configs native_cache --workloads unique repeated --concurrency 1 --requests 160 --trials 3 --output benchmarks/api_selected_default.json
```

The model benchmark compares the baseline pickle, historical Phase 3 pickle, and native runtime using the same raw feature records. It records latency, throughput, loaded and active RSS separately, artifact sizes, environment details, and baseline/native prediction parity. The HTTP benchmark starts a fresh local Uvicorn process per case and measures unique and repeated workloads. The report distinguishes the chosen `native_cache` default from the `native_full` batching candidate and records the ablation results.

## Project map

| Path | Purpose |
|---|---|
| `backend/app.py`, `backend/native_model.py`, `backend/cache.py` | API, lightweight native inference, bounded cache, and optional batch coordination |
| `backend/requirements-runtime.txt` | Lightweight container runtime dependencies; CPU-only XGBoost on amd64 Linux |
| `models/house_prices_native.zip` | Current serving artifact: preprocessing metadata and UBJ booster |
| `models/baseline_xgb_pipeline.pkl` | Frozen Phase 2 reference pipeline used for export and parity checks |
| `models/optimized_xgb_pipeline.pkl` | Historical Phase 3 experimental pipeline; not loaded by the API |
| `models/feature_schema.json` | API/UI feature names and sample record |
| `model_training/export_native.py` | Deterministic native export and full-dataset parity check |
| `model_training/benchmark_models.py`, `model_training/benchmark_api.py` | Model-level and HTTP-level benchmarks |
| `benchmarks/` | Current benchmark outputs and explicitly named historical results |
| `frontend/` | Static prediction form and nginx proxy configuration |
| `docker-compose.yml`, `backend/Dockerfile` | Local container deployment |
| `tests/` | Backend, artifact, and API behavior tests |
| `DESIGN.md` | Design choices, rationale, and milestone history |
| `report/` | Phase 3 report with results, architecture, rubric mapping, and limits |

## Checks

```powershell
ruff check backend tests model_training/*.py
pytest -q
docker compose config
```

These checks do not replace a container runtime smoke test. The report records whether Docker execution was available in the measurement environment.

