"""Build immutable base-2 Jensen–Shannon reference bins from original fit rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_training.data_partitions import (
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    IDENTITY,
    TARGET,
    _atomic_write,
    canonical_json,
    load_manifest,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "drift_reference.json"
FORMAT = "house-prices-drift-v1"
QUANTILES = tuple(index / 10 for index in range(1, 10))


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "") or bool(pd.isna(value))


def _numeric_bin_index(value: float, *, minimum: float, maximum: float, edges: list[float]) -> int:
    if value < minimum:
        return 1
    if value > maximum:
        return len(edges) + 2
    return 2 + int(np.searchsorted(np.asarray(edges, dtype=np.float64), value, side="left"))


def _finish_feature(spec: dict[str, Any], counts: list[int], sample_count: int) -> dict[str, Any]:
    spec["reference_counts"] = counts
    spec["reference_probabilities"] = [count / sample_count for count in counts]
    spec["counts"] = counts
    spec["probabilities"] = spec["reference_probabilities"]
    spec["sample_count"] = sample_count
    spec["nonmissing_count"] = sample_count - counts[0]
    return spec


def build_reference(
    dataset_path: str | Path = DEFAULT_DATASET,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    manifest = load_manifest(manifest_path, dataset_path=dataset_path)
    source_hash = sha256_file(dataset_path)
    if source_hash != manifest["source"]["sha256"]:
        raise ValueError("Dataset hash changed since split manifest creation")
    frame = pd.read_csv(dataset_path)
    ids = manifest["partitions"]["original_fit_ids"]
    fit = frame.set_index(IDENTITY, drop=False).loc[ids]
    feature_names = manifest["source"]["feature_names"]
    numeric_names = set(manifest["source"]["numeric_features"])
    features: dict[str, Any] = {}
    for name in feature_names:
        series = fit[name]
        missing = series.map(_is_missing).to_numpy(dtype=bool)
        values = series.loc[~missing]
        sample_count = int(len(series))
        if name in numeric_names:
            numeric_values = values.to_numpy(dtype=np.float64)
            if not np.isfinite(numeric_values).all():
                raise ValueError(f"Nonfinite value in original fit feature {name}")
            if len(numeric_values):
                minimum, maximum = float(numeric_values.min()), float(numeric_values.max())
                raw_edges = np.quantile(numeric_values, QUANTILES).astype(np.float64).tolist()
                edges = sorted({float(edge) for edge in raw_edges if minimum <= edge <= maximum} | {maximum})
            else:
                minimum = maximum = 0.0
                edges = []
            bucket_order = [
                "missing",
                "underflow",
                *[f"bin_{index}" for index in range(len(edges))],
                "overflow",
            ]
            counts = [0] * len(bucket_order)
            for missing_value, value in zip(missing, series.to_numpy()):
                if missing_value:
                    counts[0] += 1
                else:
                    counts[_numeric_bin_index(float(value), minimum=minimum, maximum=maximum, edges=edges)] += 1
            numeric_spec = {
                "kind": "numeric",
                "minimum": minimum,
                "maximum": maximum,
                "bin_edges": edges,
                "bins": {"minimum": minimum, "edges": edges},
                "quantile_levels": list(QUANTILES),
                "missing_bucket": "missing",
                "underflow_bucket": "underflow",
                "overflow_bucket": "overflow",
                "bucket_order": bucket_order,
            }
            features[name] = _finish_feature(numeric_spec, counts, sample_count)
            numeric_spec["bins"]["counts"] = counts
            numeric_spec["bins"]["probabilities"] = numeric_spec["reference_probabilities"]
        else:
            categories = sorted({str(value) for value in values.tolist()})
            category_positions = {value: index for index, value in enumerate(categories)}
            bucket_order = ["missing", *[f"category_{index}" for index in range(len(categories))], "unknown"]
            counts = [0] * len(bucket_order)
            for missing_value, value in zip(missing, series.to_numpy()):
                if missing_value:
                    counts[0] += 1
                else:
                    value_text = str(value)
                    category_index = category_positions.get(value_text)
                    if category_index is None:
                        counts[-1] += 1
                    else:
                        counts[1 + category_index] += 1
            features[name] = _finish_feature(
                {
                    "kind": "categorical",
                    "categories": categories,
                    "missing_bucket": "missing",
                    "unknown_bucket": "unknown",
                    "bucket_order": bucket_order,
                },
                counts,
                sample_count,
            )
    identity_payload = {
        "format": FORMAT,
        "source_dataset_sha256": source_hash,
        "dataset_sha256": source_hash,
        "source_split_manifest_sha256": sha256_file(manifest_path),
        "fit_ids": ids,
        "fit_ids_sha256": hashlib.sha256(canonical_json(ids)).hexdigest(),
        "target_excluded": TARGET,
        "identity_excluded": IDENTITY,
        "distance": "jensen-shannon-distance",
        "log_base": 2,
        "features": features,
    }
    identity_payload["reference_id"] = hashlib.sha256(canonical_json(identity_payload)).hexdigest()
    return identity_payload


def write_reference(
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    dataset_path: str | Path = DEFAULT_DATASET,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    output_path = Path(output_path)
    reference = build_reference(dataset_path, manifest_path)
    if output_path.exists():
        current = json.loads(output_path.read_text(encoding="utf-8-sig"))
        if canonical_json(current) != canonical_json(reference):
            raise FileExistsError(f"Refusing to replace immutable drift reference at {output_path}")
        return current
    _atomic_write(
        output_path,
        (json.dumps(reference, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"),
    )
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reference = write_reference(args.output, dataset_path=args.dataset, manifest_path=args.manifest)
    print(json.dumps({"output": str(args.output), "reference_id": reference["reference_id"], "features": len(reference["features"])}))


if __name__ == "__main__":
    main()
