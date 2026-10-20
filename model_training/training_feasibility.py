"""Privately check real-data continuation without publishing or activating a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from backend.native_model import NativeModel
from model_training.candidate_export import (
    build_candidate_archive,
    evaluate_gate,
    verify_candidate_parity,
)
from model_training.data_partitions import (
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    FROZEN_BASELINE_SHA256,
    IDENTITY,
    TARGET,
    load_manifest,
    sha256_file,
)
from model_training.retrain import (
    BOOST_ROUNDS,
    LEARNING_RATE,
    MIN_RELATIVE_RMSLE_GAIN,
    _training_parameters,
)
from model_training.training_config import BASELINE_ARCHIVE, SCHEMA_PATH


def run_feasibility(
    *,
    dataset_path: str | Path = DEFAULT_DATASET,
    split_manifest_path: str | Path = DEFAULT_MANIFEST,
    frozen_archive: str | Path = BASELINE_ARCHIVE,
    schema_path: str | Path = SCHEMA_PATH,
    adaptation_label_count: int = 32,
    output_path: str | Path,
) -> dict[str, Any]:
    if adaptation_label_count < 1:
        raise ValueError("adaptation_label_count must be positive")
    split = load_manifest(split_manifest_path)
    source_hash = sha256_file(dataset_path)
    if source_hash != split["source"]["sha256"]:
        raise ValueError("Feasibility input does not match the frozen split manifest")
    if sha256_file("models/baseline_xgb_pipeline.pkl") != FROZEN_BASELINE_SHA256:
        raise ValueError("Frozen baseline pipeline checksum changed")
    adaptation_ids = split["partitions"]["adaptation_ids"][:adaptation_label_count]
    if len(adaptation_ids) != adaptation_label_count:
        raise ValueError("Requested adaptation label count exceeds reserved partition")
    frame = pd.read_csv(dataset_path).set_index(IDENTITY, drop=False)
    feature_names = split["source"]["feature_names"]
    fit_ids = split["partitions"]["original_fit_ids"]
    gate_ids = split["partitions"]["promotion_gate_ids"]
    fit_rows = frame.loc[fit_ids]
    adaptation_rows = frame.loc[adaptation_ids]
    gate_rows = frame.loc[gate_ids]

    pipeline = joblib.load("models/baseline_xgb_pipeline.pkl")
    preprocessor = pipeline.named_steps["preprocessor"]
    original_booster = pipeline.named_steps["regressor"].get_booster()
    champion = NativeModel(frozen_archive)
    training_X = pd.concat(
        [fit_rows.loc[:, feature_names].reset_index(drop=True), adaptation_rows.loc[:, feature_names].reset_index(drop=True)],
        ignore_index=True,
    )
    training_y = np.concatenate([
        fit_rows[TARGET].to_numpy(dtype=np.float64),
        adaptation_rows[TARGET].to_numpy(dtype=np.float64),
    ])
    dtrain = xgb.DMatrix(preprocessor.transform(training_X), label=training_y, nthread=1)
    parameters = _training_parameters(original_booster)
    params = {
        "objective": parameters["objective"],
        "base_score": parameters["base_score"],
        "seed": parameters["seed"],
        "nthread": parameters["nthread"],
        "max_depth": parameters["max_depth"],
        "subsample": parameters["subsample"],
        "colsample_bytree": parameters["colsample_bytree"],
        "min_child_weight": parameters["min_child_weight"],
        "lambda": parameters["lambda"],
        "alpha": parameters["alpha"],
        "gamma": parameters["gamma"],
        "eta": LEARNING_RATE,
        "tree_method": parameters["tree_method"],
        "eval_metric": "rmse",
    }
    booster_copy = xgb.Booster()
    booster_copy.load_model(bytearray(champion.booster.save_raw(raw_format="ubj")))
    challenger = xgb.train(params, dtrain, num_boost_round=BOOST_ROUNDS, xgb_model=booster_copy, verbose_eval=False)

    gate_X = gate_rows.loc[:, feature_names]
    dgate = xgb.DMatrix(preprocessor.transform(gate_X), nthread=1)
    actual = gate_rows[TARGET].to_numpy(dtype=np.float64)
    champion_predictions = np.asarray(champion.booster.predict(dgate), dtype=np.float64)
    challenger_predictions = np.asarray(challenger.predict(dgate), dtype=np.float64)
    gate = evaluate_gate(actual, champion_predictions, challenger_predictions, minimum_relative_rmsle_gain=MIN_RELATIVE_RMSLE_GAIN)

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8-sig"))
    sample = dict(schema.get("sample", {}))
    sample.update({"OverallQual": 0, "GrLivArea": None, "Neighborhood": "not-a-known-neighborhood"})
    edge_records = [sample, {name: None for name in feature_names}, {"OverallQual": None, "GrLivArea": 0, "Neighborhood": "unknown"}]
    gate_records = [
        {name: None if pd.isna(row[name]) else row[name].item() if hasattr(row[name], "item") else row[name] for name in feature_names}
        for _, row in gate_X.iterrows()
    ]
    archive = build_candidate_archive(frozen_archive, challenger)
    parity = verify_candidate_parity(archive, challenger, preprocessor, gate_records + edge_records)
    evidence = {
        "format": "house-prices-training-feasibility-v1",
        "checked_at": time.time(),
        "check_status": "completed",
        "verdict": "gate_passed" if gate["accepted"] else "gate_rejected",
        "promotion_attempted": False,
        "registry_artifact_published": False,
        "live_http_used": False,
        "label_source": "private retrospective use of original train.csv adaptation labels for feasibility only",
        "source_dataset_sha256": source_hash,
        "baseline_pipeline_sha256": sha256_file("models/baseline_xgb_pipeline.pkl"),
        "baseline_native_sha256": sha256_file(frozen_archive),
        "fit_row_count": len(fit_ids),
        "adaptation_label_count": len(adaptation_ids),
        "adaptation_ids": adaptation_ids,
        "promotion_gate_ids": gate_ids,
        "final_audit_used": False,
        "training_parameters": params,
        "tree_count": {"champion": len(champion.booster.get_dump()), "challenger": len(challenger.get_dump())},
        "gate": gate,
        "candidate_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "candidate_native_parity": parity,
        "limitations": [
            "This is a private retrospective feasibility check; it does not exercise delayed-label provenance or activation.",
            "The promotion gate is reused only as the declared gate and provides limited evidence at 73 rows.",
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference", type=Path, default=BASELINE_ARCHIVE)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--adaptation-label-count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = run_feasibility(
        dataset_path=args.dataset,
        split_manifest_path=args.split_manifest,
        frozen_archive=args.reference,
        schema_path=args.schema,
        adaptation_label_count=args.adaptation_label_count,
        output_path=args.output,
    )
    print(json.dumps({"verdict": evidence["verdict"], "candidate_native_parity": evidence["candidate_native_parity"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
