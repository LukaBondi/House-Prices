# House Prices — SYS-304

Predict a home's sale price (`SalePrice`) from structural, quality, and location
features — a supervised tabular regression problem on the Kaggle
[House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
dataset (Ames, Iowa).

## Milestone status

- **Milestone 1:** EDA + baseline `XGBRegressor` pipeline (notebook + training script)
- **Milestone 2:** REST API, web UI, Docker Compose, tests, and GitHub Actions CI

Design decisions and rationale live in **[DESIGN.md](DESIGN.md)**.

Use the **`sys-304`** Miniconda environment for local Python commands:

```bash
conda activate sys-304
cd path/to/SYS-304
```

---

## Repo structure

```
.
├── README.md
├── DESIGN.md
├── deploy.sh                     # one-command local deploy
├── docker-compose.yml
├── pyproject.toml                # ruff + pytest config
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data_description.txt
├── model_training/
│   ├── prototype.ipynb           # Phase 1 notebook
│   ├── experimental.ipynb        
│   └── train_baseline.py         
├── models/
│   ├── baseline_xgb_pipeline.pkl
│   └── feature_schema.json
├── backend/
│   ├── app.py                    # FastAPI /predict service
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── index.html / styles.css / app.js
│   ├── nginx.conf
│   └── Dockerfile
├── tests/
│   ├── test_unit_backend.py
│   └── test_integration.py
└── .github/workflows/ci.yml
```

---


### Root / project config

| File | Purpose |
|------|---------|
| [`deploy.sh`](deploy.sh) | One-command local deploy: trains model if missing, then `docker compose up --build -d`. |
| [`docker-compose.yml`](docker-compose.yml) | Defines `backend` (port 8000) and `frontend` (port 3000) services. |
| [`pyproject.toml`](pyproject.toml) | Ruff lint settings + pytest `pythonpath` / test paths. |
| [`.gitignore`](.gitignore) | Ignores caches, venvs, notebook checkpoints, etc. |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | CI on push/PR to `main`: install deps, train model if needed, ruff, pytest. |
| [`DESIGN.md`](DESIGN.md) | Living design-decision log. |

### Model training & artifacts

| File | Purpose |
|------|---------|
| [`model_training/train_baseline.py`](model_training/train_baseline.py) | Trains the baseline pipeline and writes the pickle + schema. |
| [`models/baseline_xgb_pipeline.pkl`](models/baseline_xgb_pipeline.pkl) | Saved preprocessing + `XGBRegressor` pipeline (loaded by the API). |
| [`models/feature_schema.json`](models/feature_schema.json) | Feature name lists + a sample row used by `/schema` and the UI “Load sample” button. |

### Backend API

| File | Purpose |
|------|---------|
| [`backend/app.py`](backend/app.py) | FastAPI app: `/`, `/health`, `/schema`, `/predict`. |
| [`backend/requirements.txt`](backend/requirements.txt) | Python deps for API, model inference, tests, lint. |
| [`backend/Dockerfile`](backend/Dockerfile) | Container image that runs `uvicorn backend.app:app` on port 8000. |
| [`backend/__init__.py`](backend/__init__.py) | Makes `backend` an importable package for pytest. |

### Frontend UI

| File | Purpose |
|------|---------|
| [`frontend/index.html`](frontend/index.html) | Form for key house features + predict button. |
| [`frontend/styles.css`](frontend/styles.css) | Page styling. |
| [`frontend/app.js`](frontend/app.js) | Collects form data, calls API, shows predicted price. |
| [`frontend/config.js`](frontend/config.js) | Local API base (`http://localhost:8000`). |
| [`frontend/config.docker.js`](frontend/config.docker.js) | Docker API base (`/api`); copied to `config.js` in the image. |
| [`frontend/nginx.conf`](frontend/nginx.conf) | Serves static files; proxies `/api/` → backend inside Docker. |
| [`frontend/Dockerfile`](frontend/Dockerfile) | nginx image for the UI on port 80 (mapped to host 3000). |

### Tests

| File | Purpose |
|------|---------|
| [`tests/test_unit_backend.py`](tests/test_unit_backend.py) | Unit tests for feature-frame building / missing-value handling. |
| [`tests/test_integration.py`](tests/test_integration.py) | Integration tests: `/predict` → 200, `/health`, UI assets call `/predict`. |

---

## How to run

### Environment

```bash
conda activate sys-304
# if packages are missing:
pip install -r backend/requirements.txt
```

### Train (or retrain) the model

Only needed if `models/baseline_xgb_pipeline.pkl` is missing or you changed training code:

```bash
python model_training/train_baseline.py
```

### Run without Docker (dev)

**Terminal 1 — API** (from repo root):

```bash
conda activate sys-304
uvicorn backend.app:app --reload --port 8000
```

- Root / docs: http://localhost:8000/ (redirects to `/docs`)
- Health: http://localhost:8000/health

**Terminal 2 — UI:**

```bash
conda activate sys-304
python -m http.server 3000 --directory frontend
```

- UI: http://localhost:3000
- The page calls `http://localhost:8000` directly.

### Run with Docker

**Option A — deploy script** (Git Bash / WSL / macOS / Linux):

```bash
./deploy.sh
```

**Option B — compose directly:**

```bash
docker compose up --build
```

Then open:

- UI: http://localhost:3000
- API: http://localhost:8000
- API docs: http://localhost:8000/docs

Stop:

```bash
docker compose down
```

Inside Docker, the UI uses `/api/...` which nginx proxies to the backend container.

### Call the API manually

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"features\":{\"OverallQual\":7,\"GrLivArea\":1710,\"YearBuilt\":2003,\"Neighborhood\":\"CollgCr\"}}"
```

(On bash/WSL, use `\` line continuations instead of `^`.)

Missing features are allowed; the pipeline imputes them.

### Tests & lint

```bash
conda activate sys-304
ruff check backend tests model_training/train_baseline.py
pytest -q
```

CI runs the same checks on every push/PR to `main`.

---

## API

`POST /predict`

```json
{
  "features": {
    "OverallQual": 7,
    "GrLivArea": 1710,
    "YearBuilt": 2003,
    "Neighborhood": "CollgCr"
  }
}
```

Missing features are filled as missing values and imputed by the saved sklearn pipeline.

---

## Baseline approach

1. **EDA** — `SalePrice` distribution, missingness, correlations, key predictors
2. **Preprocessing** — median impute (numeric); most-frequent + one-hot (categorical)
3. **Model** — untuned `XGBRegressor` (300 trees, max depth 4)
4. **Metrics** — MAE, RMSE, R², RMSLE on a 20% holdout
5. **Serialization** — full pipeline saved with `joblib`

---

## Quick troubleshooting

| Symptom | What to check |
|---------|----------------|
| `Model not found` / 503 | Run `python model_training/train_baseline.py`. |
| `404` on `http://localhost:8000/` | Restart uvicorn after latest `backend/app.py` (root redirects to `/docs`). |
| UI “Prediction failed” / 404 / 501 | Hard-refresh the UI so `config.js` loads. Local UI must call `http://localhost:8000` (not `/api`). Backend must be running on 8000. |
| Docker UI works but predict fails | `docker compose logs backend` — confirm model volume mounted and healthcheck green. |
| Wrong Python packages | Confirm `conda activate sys-304` and `where python` points at `...\.conda\envs\sys-304\python.exe`. |
| Docker: `dockerDesktopLinuxEngine` / pipe not found | Docker Desktop is installed but the **Linux engine is not running**. Start **Docker Desktop** from the Start menu, wait until it says running, then retry `docker compose up --build`. WSL2 distros `docker-desktop` / `Ubuntu` should show as Running (`wsl -l -v`). |

