# Verification evidence

This folder contains the raw outputs summarized by the architecture report and [live demo runbook](../demo_runbook.md). The Compose stack was run under an isolated project; its artifacts are copied here so the result does not depend on a live service.

## Acceptance run

The Linux Compose smoke passed health, schema, real prediction, nginx, malformed-request, event-attribution, delayed-label, timed-retraining, activation-fixture, and restart-persistence checks. It imported 40 delayed labels from the adaptation split and persisted all matching source and observation IDs. The smoke prediction's request UUID and captured model version matched the durable audit row. The state pointer, model version, UUID, and imported IDs matched before and after backend restart. The observed concurrent request group produced one miss and 31 cache hits; the overall request-event phase recorded 32 hits. The actual coalesced count was zero. Fast inference may complete before concurrent requests reach the model, so zero coalescing is reported honestly.

The timed trainer run was `0306a8cc-e89b-47c3-abc9-9c91239cf421`. It used 40 verified labels and 73 paired gate rows. Champion RMSLE was 0.1527862786 and challenger RMSLE was 0.1517689782: a 0.665832% relative gain, below the 1% policy. RMSE improved from 24,598.1010 to 24,548.8839, but both gate conditions must pass. The real candidate was rejected; this is the correct result under the declared policy.

The separately tagged fixture `ci-fixture-50e1187bf7702144` used the same frozen model bytes to exercise positive activation. It had no fabricated trainer run or lease and carries no performance claim. Rollback then restored `bootstrap-4e605ea450b0`; repeating its activation ID returned the historical committed result, and an activation with stale expected parent returned HTTP 409.

## Final audit and model selection

The final audit ran once after selection on the frozen bootstrap model. It used 73 rows and recorded:

| Metric | Frozen bootstrap / selected model |
|---|---:|
| RMSLE | 0.1385910503 |
| MAE | $16,402.36 |
| RMSE | $24,117.92 |

`selection_used` is false in the raw artifact. The rows came from a partition of the previously evaluated 292-row holdout, so this result is descriptive and is not untouched external validation.

## Engineering checks

- Final local suite: 100 passed, six deprecation warnings, 39.16 seconds.
- Global Ruff: passed.
- Dashboard behavior tests: 5 passed.
- Protected native exporter: exact parity on 1,460 original train rows, 1,459 test rows, and three edge cases; maximum absolute difference zero; original frozen archive preserved.
- Linux Compose: images built, rebuilt after final backend pointer changes, and started. Full smoke ran in the isolated acceptance project.
- Dashboard: empty state, insufficient samples, selected 60-minute window, shifted drift, and explicit monitoring unavailability were inspected. Recovery cleared the stale-data banner and displayed the active bootstrap and 40 imported labels. The browser form loaded the API sample with `Sawyer` selected and successfully displayed a $143,950 prediction. The monitoring baseline and shift JSON are included below.
- Matched request logging benchmark: 24 fresh-process cases, 3,840 requests, zero errors. It measured unique/repeated workloads at concurrency 1/8, three trials, logging on/off. It is a local Windows client-latency comparison, not Linux container latency, an SLO, or capacity evidence.
- Remote GitHub Actions was not run; local verification is not described as an Actions result.

## Artifacts

- [Full stack acceptance smoke](stack_smoke.json)
- [Actual timed trainer run and paired gate metrics](retraining_run.json)
- [Rollback, repeat-ID, and stale-parent check](rollback.json)
- [One-time final audit](final_audit.json)
- [Baseline monitoring summary](monitoring_baseline.json)
- [Shifted monitoring summary](monitoring_shift.json)
- [Linux container versions and platform](container_environment.json)
- [Matched logging benchmark and raw trials](../../benchmarks/logging_overhead.json)

The benchmark ran on Windows 11 build 26300 with Python 3.12.14 while the Docker stack was paused. Container validation ran on Linux under WSL2 with Python 3.12.15, FastAPI 0.142.2, NumPy 2.5.3, SciPy 1.18.1, XGBoost 3.4.1, Uvicorn 0.54.0, and SQLite 3.46.1.
