"""Shared paths for local training state, kept outside the synced checkout by default."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LOCAL_DATA = Path(os.getenv("LOCALAPPDATA", tempfile.gettempdir())) / "sys-304-house-prices"
DATA_DIR = Path(os.getenv("DATA_DIR", str(_LOCAL_DATA)))
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "requests.sqlite3")))
MODEL_REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", str(DATA_DIR / "artifacts")))
MODEL_STATE_PATH = Path(os.getenv("MODEL_STATE_PATH", str(DATA_DIR / "state")))
DRIFT_REFERENCE_PATH = Path(os.getenv("DRIFT_REFERENCE_PATH", str(ROOT / "models" / "drift_reference.json")))
BASELINE_ARCHIVE = ROOT / "models" / "house_prices_native.zip"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
DATASET_PATH = ROOT / "dataset" / "train.csv"
SPLIT_MANIFEST_PATH = ROOT / "dataset" / "split_manifest.json"
