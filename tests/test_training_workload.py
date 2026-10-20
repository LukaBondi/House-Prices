from __future__ import annotations

import pandas as pd
import pytest

from model_training import generate_workload
from model_training.data_partitions import write_manifest
from model_training.generate_workload import build_scenario_calls


def test_adaptation_workload_keeps_labels_out_of_feature_records(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    calls, private_labels, source_hash = build_scenario_calls(
        "adaptation", 4, 19, manifest_path=manifest_path
    )
    assert len(calls) == 4
    assert len(private_labels) == 4
    assert all(call["observation"]["dataset_sha256"] == source_hash for call in calls)
    assert all("SalePrice" not in call["features"] and "Id" not in call["features"] for call in calls)
    assert all("sale_price" in label and "features" not in label for label in private_labels)
    assert all(label["request_ids"] == [] and label["available_at"] is None for label in private_labels)


def test_drift_replay_has_no_observation_or_training_truth(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    calls, private_labels, _ = build_scenario_calls(
        "drift", 6, 9, manifest_path=manifest_path, drift_kind="mixed"
    )
    assert len(calls) == 6
    assert private_labels == []
    assert all(call["observation"] is None for call in calls)
    assert all("SalePrice" not in call["features"] for call in calls)
    assert any(call["features"]["Neighborhood"].startswith("unseen_category_") for call in calls)


def test_unlabeled_scenario_reads_test_features_without_saleprice(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    calls, private_labels, _ = build_scenario_calls(
        "unlabeled", 3, 31, manifest_path=manifest_path
    )
    assert len(calls) == 3
    assert private_labels == []
    assert all(call["observation"] is None for call in calls)
    assert all("SalePrice" not in call["features"] and "Id" not in call["features"] for call in calls)
    assert len(pd.read_csv("dataset/test.csv")) == 1459


def test_malformed_workload_has_no_observation_metadata(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    calls, private_labels, _ = build_scenario_calls("malformed", 5, 4, manifest_path=manifest_path)
    assert all(call["observation"] is None for call in calls)
    assert all("features" not in call for call in calls)
    assert private_labels == []


def test_duration_mode_reconciles_scheduled_calls_and_http_responses(tmp_path, monkeypatch):
    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    sent = []

    def fake_post(base_url, call, run_id, timeout):
        sent.append(call)
        return {
            "request_id": f"request-{len(sent)}",
            "source_id": call.get("source_id"),
            "status_code": 200,
            "model_version": "bootstrap-fixture",
            "response": {"sale_price": 100.0},
            "response_text": None,
            "transport_error": None,
            "client_latency_ms": 1.0,
            "completed_at": 1_800_000_000.0 + len(sent),
        }

    monkeypatch.setattr(generate_workload, "_post", fake_post)
    manifest = generate_workload.run_workload(
        base_url="http://unused",
        scenario="baseline",
        request_count=None,
        duration_seconds=1.2,
        seed=2,
        concurrency=2,
        output_path=tmp_path / "run.json",
        manifest_path=manifest_path,
    )
    assert manifest["requested_calls"] == 3
    assert manifest["responses_received"] == 3
    assert len(sent) == 3
    assert manifest["duration_seconds"] == 1.2
    assert manifest["scheduled_rate_per_second"] == 2.5


def test_baseline_and_unlabeled_scenarios_cycle_to_exact_requested_count(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    baseline, _, _ = build_scenario_calls("baseline", 1173, 12, manifest_path=manifest_path)
    unlabeled, _, _ = build_scenario_calls("unlabeled", 1464, 12, manifest_path=manifest_path)
    assert len(baseline) == 1173
    assert len(unlabeled) == 1464
    assert len({call["source_id"] for call in baseline}) <= 1168
    assert len({call["source_id"] for call in unlabeled}) <= 1459


def test_adaptation_scenario_rejects_repeated_ids_above_reserved_partition(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    write_manifest(manifest_path)
    with pytest.raises(ValueError, match="supports at most 146 unique rows"):
        build_scenario_calls("adaptation", 147, 5, manifest_path=manifest_path)
