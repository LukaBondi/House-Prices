# Live demo runbook

This is an operator guide for a local Docker Compose demonstration. It describes commands and the evidence to show; it does not claim a presentation occurred. The acceptance smoke and final descriptive audit were run separately and are recorded in [evidence](evidence/README.md).

## Before you start

Use Docker Desktop with Linux containers and Docker Compose v2. The default ports are localhost-only: the form/dashboard use port 3000 and the API uses port 8000. Use a fresh Compose project name if another demo stack may be running. The model, request database, and current serving pointer live in named volumes.

For workload and truth-import commands, use Python 3.12 with both requirement sets. In the repository root, either activate the configured Conda environment or create a venv and install:

~~~powershell
conda activate sys-304
python -m pip install -r backend/requirements.txt
python -m pip install -r model_training/requirements-trainer.txt
python --version
~~~

Docker-only users can run the truth importer inside the trainer container, as shown below. Set a private token for this local stack and use a short five-second scheduler interval for the timed demo. The normal Compose interval is one hour.

~~~powershell
$env:COMPOSE_PROJECT_NAME = 'house-prices-demo'
$env:ADMIN_TOKEN = [guid]::NewGuid().ToString('N')
$env:RETRAIN_INTERVAL_SECONDS = '5'
docker compose config -q
docker compose up --build --detach --wait --wait-timeout 300
docker compose ps
~~~

Allow a few minutes for the first image build and health checks. Open [the prediction form](http://localhost:3000/), [the operations dashboard](http://localhost:3000/dashboard.html), [API health](http://localhost:8000/health), and [the API docs](http://localhost:8000/docs). The backend admin routes require the bearer token and are deliberately not exposed by nginx.

## Show a prediction and request attribution

Send a normal house feature record directly to the API:

~~~powershell
$body = @{ features = @{ OverallQual = 7; GrLivArea = 1710; YearBuilt = 2003; Neighborhood = 'CollgCr' } } | ConvertTo-Json
$response = Invoke-RestMethod -Uri 'http://localhost:8000/predict' -Method Post -ContentType 'application/json' -Body $body
$response | ConvertTo-Json -Depth 5
~~~

The response includes a server-generated `request_id` and the captured `model_version`. The same UUID is in the `X-Request-ID` response header. A prediction is not a label and cannot be fed back as truth. The dashboard records the request only after the API acknowledges durable admission and completion writes; the prepared status does not prove the client received the response.

On the form, choose **Load sample** and verify `Neighborhood` displays `Sawyer` from the API schema before selecting **Predict**. The categorical select supplies schema values that are not in its original short option list, and the form serializes that selection with the other house features.

## Generate traffic and inspect monitoring

Create a temporary directory, then run the workload generator against the API. Baseline traffic draws from fit-partition records; repeated traffic deliberately replays selected records to exercise exact-input caching while each request remains independently audited. Drift and malformed-input scenarios are unlabeled.

~~~powershell
New-Item -ItemType Directory -Force tmp/demo | Out-Null
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario baseline --requests 160 --seed 42 --run-id demo-baseline --output tmp/demo/baseline.json
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario repeated --requests 160 --seed 42 --run-id demo-repeated --output tmp/demo/repeated.json
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario drift --drift-kind mixed --requests 120 --seed 42 --run-id demo-shift --output tmp/demo/shift.json
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario malformed --requests 12 --seed 42 --run-id demo-malformed --output tmp/demo/malformed.json
~~~

In the dashboard, switch among the 15-minute, 60-minute, and 24-hour windows and show completed, pending, failed, cache, latency, and drift panels. Feature distances and missing/unknown counts can appear below 100 valid observations with an explicit insufficient-sample status; the advisory drift flag is gated until the selected window reaches 100. The drift score is square-root Jensen-Shannon distance against the fixed fit-only reference; the default 0.10 threshold is an advisory input-shift flag, not an accuracy score. Health/dashboard polling does not create prediction traffic. Empty windows keep true zero counts while rates and latency quantiles with empty denominators remain unavailable.

If the backend is stopped, the dashboard should show monitoring unavailable and the timestamp of the last successful update. After the backend recovers, refresh or wait for the next five-second poll; the selected time window should still be explicit.

## Import delayed truth and observe the timer verdict

Only unchanged rows from the 146-row adaptation partition can receive truth. Generate distinct successful requests and keep the private truth file separate from the public manifest. This example waits 30 seconds before importing 40 verified labels:

~~~powershell
$runId = 'adapt-' + [guid]::NewGuid().ToString('N')
python -m model_training.generate_workload --base-url http://127.0.0.1:8000 --scenario adaptation --requests 40 --seed 42 --run-id $runId --label-delay-seconds 30 --output "tmp/demo/$runId.json" --truth-output "tmp/demo/$runId-truth.json"
Start-Sleep -Seconds 30
$trainer = docker compose ps -q trainer
docker cp "tmp/demo/$runId-truth.json" "$($trainer):/tmp/$runId-truth.json"
docker compose exec -T trainer python -m model_training.simulate_truth --run-manifest "/tmp/$runId-truth.json" --split-manifest /app/dataset/split_manifest.json
~~~

The importer verifies the pinned source dataset, adaptation membership, exact features, persisted successful request, run provenance, completion time, and elapsed label delay. Repeated calls to the same house do not add independent training observations. At least 32 distinct verified labels are required. The scheduled trainer snapshots eligible rows at a fixed cutoff, records its lease and run, continues a private copy of the incumbent for 25 rounds at learning rate 0.01, and evaluates the fixed 73-row paired gate. The candidate must improve RMSLE by at least 1%, keep RMSE no worse, pass finite/nonnegative output checks, and match native serving parity.

Follow the terminal trainer output and the dashboard's retraining panel. A real candidate may be rejected; this is a correct outcome. In the October 5 acceptance run, the 40-label candidate improved RMSLE from 0.1527863 to 0.1517690, a 0.6658% relative gain below the 1% requirement. RMSE improved from 24,598.101 to 24,548.884, but the candidate was rejected and the frozen bootstrap stayed active. See the exact run record in [retraining_run.json](evidence/retraining_run.json). These values are a historical run result, not a promise for every run.

## Verify activation, persistence, and rollback

If the real run promotes, inspect health for the active version and inspect the scheduler's run record. Predictions already in flight keep the version they captured. Restart persistence is covered by the acceptance smoke: it pauses the trainer, snapshots the successful request UUID/model version, imported source and observation IDs, and pointer, restarts the backend, then requires those exact values to persist. The request's active version must agree with the pointer before and after restart.

When the real gate rejects, the acceptance smoke uses a distinct, explicitly named same-model fixture to test positive activation. The fixture has no fabricated trainer-run ID or lease and makes no accuracy claim. The fixture activation and its restart check are in [stack_smoke.json](evidence/stack_smoke.json). The verified rollback returned that fixture to the frozen bootstrap; a repeated activation ID returned the original committed result and a stale expected parent returned HTTP 409. Rollback targets are limited to the previously served version or the frozen bootstrap.

Rollback is an explicit local operator action. If a previously served version exists, the following example asks the backend to compare-and-swap from the version currently in the pointer. The route is on port 8000 and requires the token:

~~~powershell
$pointer = docker compose exec -T backend python -c "import json; print(json.dumps(json.load(open('/state/current.json'))))" | ConvertFrom-Json
if (-not $pointer.previous_version) {
    Write-Host 'No previous served version is recorded; no rollback was requested.'
} else {
    $rollbackBody = @{
        target_version = $pointer.previous_version
        expected_parent = $pointer.version
        activation_id = [guid]::NewGuid().ToString()
    } | ConvertTo-Json
    Invoke-RestMethod -Uri 'http://localhost:8000/admin/rollback' -Method Post -Headers @{ Authorization = "Bearer $env:ADMIN_TOKEN" } -ContentType 'application/json' -Body $rollbackBody
}
~~~

The rollback payload field is `target_version`; `expected_parent` must be the current pointer version. A committed activation status describes that transaction, even if another later activation changed the pointer. Read `/health` or the current pointer for the current serving version.

## Reproduce the full acceptance smoke

The smoke harness exercises the actual HTTP routes, delayed import, timer verdict, optional fixture activation, and backend restart. Use a fresh project name. If the regular demo project is running on ports 3000/8000, stop its containers first with `docker compose down` while its project name is selected; this preserves its volumes. Then run the acceptance project. `--exercise-restart` temporarily pauses the acceptance trainer, and the acceptance project's volumes are removed after the run below. Run this when no one is using that project:

~~~powershell
$env:COMPOSE_PROJECT_NAME = 'house-prices-demo'
docker compose down
$env:COMPOSE_PROJECT_NAME = 'house-prices-acceptance-demo'
$env:ADMIN_TOKEN = [guid]::NewGuid().ToString('N')
$env:RETRAIN_INTERVAL_SECONDS = '5'
try {
    docker compose up --build --detach --wait --wait-timeout 300
    python tools/smoke_stack.py --base-url http://127.0.0.1:3000/api --frontend-url http://127.0.0.1:3000 --trainer-service trainer --exercise-request-events --exercise-adaptation --require-timed-retraining --exercise-promotion-fixture --exercise-restart --request-count 40 --minimum-labels 32 --label-delay-seconds 1 --retrain-timeout 240 --trainer-timeout 300
} finally {
    docker compose down --volumes --remove-orphans
}
~~~

Use a disposable project: `down --volumes` deletes that project's database, imported labels, model registry, and serving pointer. The command does not remove volumes from another Compose project. The verified acceptance evidence is preserved in `report/evidence/`.

## Show the CI workflow

Open the repository's [CI workflow](https://github.com/LukaBondi/House-Prices/actions/workflows/ci.yml) and show the latest displayed run, including its timestamp and status. Explain its two jobs: lint/tests/frozen-artifact and split checks/dashboard behavior; then Linux Compose build, real HTTP smoke, log collection, and uploaded smoke artifacts. Remote GitHub Actions was not run during this implementation, so do not present the local 100-test result or Docker acceptance smoke as a passing remote Actions run. If the page has no newer run, state that plainly.

## Shutdown and reset

Ordinary shutdown preserves the database, imported labels, immutable model versions, and serving pointer:

~~~powershell
docker compose down
~~~

To deliberately reset the current project, and only after saving any evidence you need, run:

~~~powershell
docker compose down --volumes --remove-orphans
~~~

Do not use the reset command when you intend to inspect or resume the same demo state.

## Common problems and discussion points

- **Ports 3000 or 8000 are busy:** stop the other local Compose project or choose another demo environment before starting this one; do not inspect another project's volumes.
- **Dashboard says unavailable or logging failed:** check `docker compose ps`, `docker compose logs --tail 100 backend`, and `http://localhost:8000/health`. The API fails a request rather than silently claiming a durable completion when audit acknowledgement fails.
- **Drift panels say insufficient data:** send at least 100 valid predictions within the selected window, then choose a window containing them. Malformed requests do not count as valid drift observations.
- **No eligible training run appears:** confirm the interval was set before container creation, truth import succeeded, at least 32 unique labels came from the adaptation split, and the trainer is running. A past 3,600-second value requires recreating the container after setting the short interval.
- **Truth import rejects a row:** keep generated feature JSON unchanged, wait until its configured delay, use the matching private run manifest, and confirm the source CSV hash/split manifest. Never label drift traffic or use model output as truth.
- **The candidate is rejected:** inspect the paired gate metrics and reason. The real 40-label acceptance candidate did not reach the required 1% RMSLE gain; leave the bootstrap selected. Use the clearly marked fixture only to demonstrate activation mechanics.
- **Admin request returns 401/403:** send the same `ADMIN_TOKEN` used by Compose and call the backend port directly. Nginx intentionally blocks admin routes.
- **A timer sees activation recovery pending:** do not start a competing training run. Let the trainer reconcile the activation by status and serving pointer first.

For the architecture discussion, describe SQLite's serialized writer, the single API worker and process-local cache, shared training CPU/database I/O, local artifact volumes, and growing event history as single-host constraints. A roughly 100x design would need a measured SLO, PostgreSQL plus durable event ingestion, isolated workers, remote immutable artifacts, coordinated rollout, monitoring pre-aggregation/retention, and fresh temporal labels. The matched benchmark in [system_design.md](system_design.md) is a local client-latency comparison, not production capacity.

Useful interview prompts: why does the system accept a rejected real candidate? Which checks prevent bad data or a weak candidate from training or serving? If a faulty version reaches service, how do expected-parent rollback, the previous-version limit, frozen-bootstrap fallback, idempotency, and request-level model snapshots work together? Which measurements would be needed before claiming 100x capacity?
