# House Prices prediction and continual-learning service

Repository: [LukaBondi/House-Prices](https://github.com/LukaBondi/House-Prices)

This repository serves sale-price predictions for the Ames housing dataset and demonstrates a guarded local continual-learning loop. The bootstrap champion is the frozen Phase 2 XGBoost pipeline exported to native UBJ with fitted preprocessing. A gate-approved candidate can replace the active version while the original baseline archive and feature schema remain protected. The service audits requests, monitors input drift, imports delayed labels only for verified adaptation examples, and evaluates scheduled candidates before activation. A real candidate may be rejected; the service does not force an accuracy win.

Design rationale and dated decisions are in [DESIGN.md](DESIGN.md). The architecture report and live-demo runbook are in [report/system_design.md](report/system_design.md) and [report/demo_runbook.md](report/demo_runbook.md). Historical model and optimization measurements remain in [report/phase3_report.md](report/phase3_report.md).

## Start the local Compose stack

Prerequisites: Docker Desktop or another Docker Engine with Linux containers, plus Docker Compose v2. Published ports bind to localhost only.

From PowerShell at the repository root:

~~~powershell
$env:ADMIN_TOKEN = [guid]::NewGuid().ToString('N')
$env:RETRAIN_INTERVAL_SECONDS = '3600'
docker compose config -q
docker compose up --build --detach --wait --wait-timeout 300
~~~

For a short demo, set the scheduler interval to a few seconds before starting the stack, then generate and import at least 32 eligible labels. The default interval is 3,600 seconds. Compose has a conspicuous local-only fallback token if ADMIN_TOKEN is unset; set a private value for any environment accessible by other users. The token is shared only with the API and trusted trainer and is not proxied through nginx.

Open:

- Prediction form: http://localhost:3000/
- Operations dashboard: http://localhost:3000/dashboard.html
- API health: http://localhost:8000/health
- API schema: http://localhost:8000/schema
- Interactive API docs: http://localhost:8000/docs
- Monitoring summary: http://localhost:8000/monitoring/summary?window_minutes=15

The dashboard offers 15-minute, 60-minute, and 24-hour windows and refreshes every five seconds. It labels persisted handler time separately from browser/API round-trip time. An empty window shows its true zero counts; latency quantiles and rates with empty denominators stay unavailable.

To stop containers while preserving the demo database, registry, and active pointer, run:

~~~powershell
docker compose down
~~~

Named volumes survive this command. To deliberately remove all persistent demo state, run:

~~~powershell
docker compose down --volumes --remove-orphans
~~~

That reset deletes request history, imported labels, immutable versions, and the serving pointer. Preserve evidence files first. On Bash/WSL, ./deploy.sh starts Compose and prints the local endpoints; it does not re-export or reset the frozen model.

## Make a prediction

The API accepts a feature object; omitted features use the frozen fitted imputation behavior. The response preserves sale_price and currency and adds a request_id and model_version. The matching UUID is also returned in the X-Request-ID header.

~~~powershell
$body = @{ features = @{ OverallQual = 7; GrLivArea = 1710; YearBuilt = 2003; Neighborhood = 'CollgCr' } } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://localhost:8000/predict' -Method Post -ContentType 'application/json' -Body $body
~~~

Or use curl.exe:

~~~powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features":{"OverallQual":7,"GrLivArea":1710,"YearBuilt":2003,"Neighborhood":"CollgCr"}}'
~~~

A direct prediction does not create a training label. Observation metadata from an HTTP request is untrusted until the private trainer-side importer verifies the exact source row, dataset hash, adaptation membership, persisted successful request, completion time, and label delay.

## Generate monitoring traffic

Host-side workload tools require Python 3.12 and both the API and trainer dependencies. In a fresh development environment, install them once:

~~~powershell
conda activate sys-304
python -m pip install -r backend/requirements.txt
python -m pip install -r model_training/requirements-trainer.txt
python --version
~~~

The main traffic tool sends real HTTP requests to an already-running API. From the repository root:

~~~powershell
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario baseline --requests 160 --seed 42 --run-id baseline-demo --output tmp/baseline-demo.json
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario repeated --requests 160 --seed 42 --run-id repeated-demo --output tmp/repeated-demo.json
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario drift --drift-kind mixed --requests 120 --seed 42 --run-id unlabeled-shift --output tmp/unlabeled-shift.json
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario malformed --requests 12 --seed 42 --run-id malformed-demo --output tmp/malformed-demo.json
~~~

Baseline and repeated scenarios draw from trusted dataset records; repeated requests remain separate audit events but do not become new training observations. Drift and malformed requests are unlabeled. Drift uses shifted inputs and can exercise missing and unknown feature values. The dashboard can show feature distances and missing/unknown counts below 100 valid observations, but marks the sample as insufficient and gates the advisory drift flag until the selected window reaches 100. The configured 0.10 threshold is not an accuracy measurement.

## Exercise delayed labels and scheduled retraining

For a short demo, configure the interval before Compose starts:

~~~powershell
$env:ADMIN_TOKEN = [guid]::NewGuid().ToString('N')
$env:RETRAIN_INTERVAL_SECONDS = '5'
docker compose up --build --detach --wait --wait-timeout 300
~~~

Generate adaptation traffic and keep the private truth file separate from the public run manifest. This example generates 40 distinct requests and delays truth availability for 30 seconds:

~~~powershell
$runId = 'adapt-' + [guid]::NewGuid().ToString('N')
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario adaptation --requests 40 --seed 42 --run-id $runId --label-delay-seconds 30 --output "tmp/$runId.json" --truth-output "tmp/$runId-truth.json"
Start-Sleep -Seconds 30
$trainer = docker compose ps -q trainer
docker cp "tmp/$runId-truth.json" "$($trainer):/tmp/$runId-truth.json"
docker compose exec -T trainer python -m model_training.simulate_truth --run-manifest "/tmp/$runId-truth.json" --split-manifest /app/dataset/split_manifest.json
~~~

Wait for the scheduler and inspect its verdict and active version:

~~~powershell
docker compose logs --tail 100 trainer
Invoke-RestMethod 'http://localhost:8000/monitoring/summary?window_minutes=15' | ConvertTo-Json -Depth 12
Invoke-RestMethod 'http://localhost:8000/health'
~~~

Training requires at least 32 new verified labels from the fixed 146-row adaptation partition. Each timer tick snapshots successful, unchanged, label-available observations at or before a fixed cutoff. The trainer continues a private booster copy for 25 rounds at learning rate 0.01 with frozen preprocessing. The same 73-row promotion gate must show at least 1% relative RMSLE improvement and no RMSE regression. Rejected, skipped, failed, and promoted outcomes are recorded separately. A rejection is a valid outcome and leaves the current model active. The 73-row final audit was evaluated once after model selection and was not used to choose a candidate; its result is descriptive because the rows came from a historically evaluated holdout.

A successful candidate is activated through the token-protected API after immutable artifact checks, expected-parent compare-and-swap, and a live trainer-lease fence. The backend owns the atomic current.json pointer. Restart preserves the request database and serving pointer. In the recorded 40-label live run, the candidate was rejected: RMSLE improved by 0.6658%, below the 1% threshold, while RMSE improved from $24,598.10 to $24,548.88. The frozen bootstrap remained selected. The smoke tool then used an explicitly marked same-model fixture to verify activation, persistence, and rollback; this is an activation-path check and makes no model-performance claim. The selected bootstrap's one-time descriptive 73-row audit gave RMSLE 0.1385911, MAE $16,402.36, and RMSE $24,117.92. The audit partition belonged to a historically evaluated holdout and provides no untouched external accuracy estimate.

For the complete operator flow, expected evidence, rollback example, cleanup, and troubleshooting, see the [live demo runbook](report/demo_runbook.md). The final smoke/audit evidence and matched logging benchmark are indexed in [report/evidence](report/evidence/README.md).


## Benchmark request logging

The matched logging benchmark starts fresh local API processes with isolated database/artifact/state directories. It compares logging enabled and disabled on identical unique and repeated workloads at concurrency 1 and 8, with three trials per case. It reports full client round-trip p50/p95, successful throughput, errors, and raw trial data; it does not record an equivalent per-request handler percentile. Client latency includes audit acknowledgements, HTTP, and network overhead. The dashboard's handler latency is a server-side request measure with a different boundary.

~~~powershell
python tools/benchmark_logging.py --workloads unique repeated --concurrency 1 8 --requests 160 --trials 3 --output benchmarks/logging_overhead.json
~~~

The measured artifact is [benchmarks/logging_overhead.json](benchmarks/logging_overhead.json). Results are a local Windows comparison, not a Docker/Linux latency test or production capacity claim. The system design report gives the environment and measured values.

## Checks

~~~powershell
ruff check backend tests model_training/*.py tools/*.py
pytest -q
node --check frontend/dashboard.js
node --test tests/dashboard_behavior.test.cjs
docker compose config -q
~~~

GitHub Actions also builds and runs a Linux Compose smoke with real nginx/API traffic, delayed-label import, scheduled retraining, pointer/database restart checks, and log artifact upload. A successful local Compose parse is not evidence that the workflow or live smoke passed. See the dated validation status in [report/system_design.md](report/system_design.md).

## Layout

| Path | Purpose |
|---|---|
| backend/ | Native inference API, durable request event store, monitoring and serving-version activation |
| frontend/ | Static prediction form and accessible operations dashboard, served by nginx |
| model_training/ | Split/drift builders, live workload and private truth importer, registry and scheduled trainer |
| models/house_prices_native.zip | Frozen native export used for inference and immutable bootstrap |
| models/baseline_xgb_pipeline.pkl | Protected frozen reference pipeline used by exporter and evaluation |
| models/feature_schema.json | Feature names and UI sample |
| dataset/split_manifest.json | Hash-pinned fit/adaptation/gate/audit membership |
| models/drift_reference.json | Fixed input distribution from the original fit partition |
| docker-compose.yml | Backend, frontend, bootstrap and trainer services with named state volumes |
| tools/smoke_stack.py | Opt-in real HTTP and Compose lifecycle smoke |
| tools/benchmark_logging.py | Matched logging on/off client-latency benchmark |
| tests/ | Backend, training, frontend behavior, and deployment regression coverage |
| DESIGN.md, PLAN.md | Dated architectural rationale and implementation record |
| report/ | System design, demo runbook, and historical Phase 3 report |

## Runtime state and useful settings

Compose stores the request database under /data, immutable model versions under /artifacts, and the backend-owned active pointer under /state. The API mounts artifacts read-only; the trainer may publish new immutable versions but cannot write the active pointer. Local non-Compose defaults live in user-local application data rather than this synced repository.

| Setting | Purpose |
|---|---|
| ADMIN_TOKEN | Shared secret for trainer-to-API activation and rollback |
| RETRAIN_INTERVAL_SECONDS | Compose scheduler interval; 3,600 seconds by default |
| DATA_DIR and DB_PATH | Local event store location |
| MODEL_REGISTRY_PATH | Immutable version registry |
| MODEL_STATE_PATH | Backend-owned serving pointer directory |
| DRIFT_REFERENCE_PATH | Validated frozen drift reference |
| PREDICTION_CACHE_ENABLED | Exact-input TTL/LRU cache; enabled by default |
| DYNAMIC_BATCHING_ENABLED | Optional bounded queue; disabled by default |

Admin endpoints are reachable through the loopback-bound API port, require a bearer token, and are intentionally denied by the public nginx proxy. Running docker compose down preserves state; remove named volumes only when a full reset is intended.
