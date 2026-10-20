"""Immutable serving snapshots and validated model registry access."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from backend.event_store import canonical_json
from backend.native_model import NativeModel

REGISTRY_FORMAT = "house-prices-registry-v1"
SAFE_VERSION_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024
PREPROCESSING_FIELDS = (
    "format",
    "feature_names",
    "numeric_features",
    "categorical_features",
    "numeric_medians",
    "categorical_modes",
    "categories",
    "categorical_offsets",
    "n_output_features",
    "booster_features",
    "sparse_output",
)


class ArtifactError(RuntimeError):
    """An artifact directory, manifest, or model failed validation."""


@dataclass(frozen=True)
class ServingSnapshot:
    """One immutable reference to the model, schema, and serving version."""

    model: Any
    version: str
    schema_json: str
    feature_names: tuple[str, ...]
    numeric_features: frozenset[str]
    artifact_sha256: str
    preprocessing_sha256: str
    schema_sha256: str
    manifest_json: str | None = None

    @property
    def schema(self) -> dict[str, Any]:
        return json.loads(self.schema_json)

    @property
    def manifest(self) -> dict[str, Any] | None:
        return json.loads(self.manifest_json) if self.manifest_json else None


def safe_version(version: str) -> str:
    if not isinstance(version, str) or not SAFE_VERSION_PATTERN.fullmatch(version):
        raise ArtifactError("version must be a short path-safe identifier")
    return version


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_sha256(schema: Mapping[str, Any] | str | Path) -> str:
    if isinstance(schema, (str, Path)):
        schema_value = json.loads(Path(schema).read_text(encoding="utf-8"))
    else:
        schema_value = dict(schema)
    return sha256_bytes(canonical_json(schema_value).encode("utf-8"))


def _normalize_preprocessing_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        number = float(value)
        if not math.isfinite(number):
            raise ArtifactError("preprocessing metadata contains a non-finite number")
        return 0.0 if number == 0 else number
    if isinstance(value, list):
        return [_normalize_preprocessing_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_preprocessing_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_preprocessing_value(item) for key, item in value.items()}
    return value


def preprocessing_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = {name: metadata[name] for name in PREPROCESSING_FIELDS if name in metadata}
    if len(payload) != len(PREPROCESSING_FIELDS):
        missing = sorted(set(PREPROCESSING_FIELDS) - set(payload))
        raise ArtifactError(f"preprocessing metadata is missing fields: {missing}")
    return _normalize_preprocessing_value(payload)


def preprocessing_sha256(metadata: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(preprocessing_payload(metadata)).encode("utf-8"))


def read_archive_metadata(path: str | Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "metadata.json" not in names or "booster.ubj" not in names:
                raise ArtifactError("model archive must contain metadata.json and booster.ubj")
            if any(info.file_size > MAX_ARCHIVE_MEMBER_BYTES for info in archive.infolist()):
                raise ArtifactError("model archive member exceeds the size limit")
            metadata = json.loads(archive.read("metadata.json"))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not read model archive: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ArtifactError("model metadata must be an object")
    return metadata


def make_snapshot(
    model: Any,
    schema: Mapping[str, Any],
    *,
    version: str,
    artifact_hash: str,
    preprocessor_hash: str,
    manifest: Mapping[str, Any] | None = None,
) -> ServingSnapshot:
    safe_version(version)
    schema_value = dict(schema)
    feature_names = getattr(model, "feature_names", None) or schema_value.get("features", [])
    numeric_features = getattr(model, "numeric_features", [])
    if not isinstance(feature_names, Sequence) or isinstance(feature_names, str):
        raise ArtifactError("model feature names are invalid")
    schema_text = canonical_json(schema_value)
    manifest_text = canonical_json(dict(manifest)) if manifest is not None else None
    return ServingSnapshot(
        model=model,
        version=version,
        schema_json=schema_text,
        feature_names=tuple(str(name) for name in feature_names),
        numeric_features=frozenset(str(name) for name in numeric_features),
        artifact_sha256=artifact_hash,
        preprocessing_sha256=preprocessor_hash,
        schema_sha256=schema_sha256(schema_value),
        manifest_json=manifest_text,
    )


def load_bootstrap_snapshot(model_path: str | Path, schema_path: str | Path) -> ServingSnapshot:
    archive = Path(model_path)
    if not archive.is_file():
        raise ArtifactError(f"bootstrap model does not exist: {archive}")
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not load feature schema: {exc}") from exc
    metadata = read_archive_metadata(archive)
    artifact_hash = sha256_file(archive)
    version = f"bootstrap-{artifact_hash[:12]}"
    model = NativeModel(archive, nthread=1)
    if list(model.feature_names) != list(schema.get("features", [])):
        raise ArtifactError("bootstrap model feature names do not match the schema")
    return make_snapshot(
        model,
        schema,
        version=version,
        artifact_hash=artifact_hash,
        preprocessor_hash=preprocessing_sha256(metadata),
    )


def _manifest_at(version_directory: Path) -> dict[str, Any]:
    manifest_path = version_directory / "manifest.json"
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ArtifactError("registry manifest exceeds the size limit")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not load registry manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError("registry manifest must be a JSON object")
    required = {
        "format",
        "version",
        "artifact_sha256",
        "preprocessing_sha256",
        "schema_sha256",
        "parent_version",
        "gate",
        "training",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ArtifactError(f"registry manifest is missing fields: {missing}")
    if manifest["format"] != REGISTRY_FORMAT:
        raise ArtifactError("unsupported registry manifest format")
    safe_version(manifest["version"])
    parent_version = manifest["parent_version"]
    if parent_version is not None:
        safe_version(parent_version)
    for key in ("artifact_sha256", "preprocessing_sha256", "schema_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ArtifactError(f"manifest {key} must be a SHA-256 hex digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ArtifactError(f"manifest {key} must be a SHA-256 hex digest") from exc
    if not isinstance(manifest["gate"], dict) or not isinstance(manifest["training"], dict):
        raise ArtifactError("manifest gate and training fields must be objects")
    return manifest


def load_registry_snapshot(
    registry_path: str | Path,
    version: str,
    *,
    schema: Mapping[str, Any],
    expected_preprocessing_sha256: str | None = None,
    require_accepted: bool = True,
) -> ServingSnapshot:
    version = safe_version(version)
    root = Path(registry_path).resolve()
    version_directory = (root / version).resolve()
    if version_directory.parent != root or not version_directory.is_dir():
        raise ArtifactError("version directory is absent or outside the registry root")
    manifest = _manifest_at(version_directory)
    if manifest["version"] != version:
        raise ArtifactError("manifest version does not match its directory")
    if require_accepted and manifest["gate"].get("accepted") is not True:
        raise ArtifactError("candidate did not pass the configured promotion gate")
    archive = version_directory / "model.zip"
    if archive.resolve().parent != version_directory or not archive.is_file():
        raise ArtifactError("model archive is absent or outside its version directory")
    artifact_hash = sha256_file(archive)
    if artifact_hash != manifest["artifact_sha256"]:
        raise ArtifactError("model archive checksum does not match the manifest")
    metadata = read_archive_metadata(archive)
    preprocessor_hash = preprocessing_sha256(metadata)
    if preprocessor_hash != manifest["preprocessing_sha256"]:
        raise ArtifactError("preprocessing fingerprint does not match the manifest")
    if (
        expected_preprocessing_sha256 is not None
        and preprocessor_hash != expected_preprocessing_sha256
    ):
        raise ArtifactError("candidate preprocessing differs from the active model")
    expected_schema_hash = schema_sha256(schema)
    if manifest["schema_sha256"] != expected_schema_hash:
        raise ArtifactError("candidate feature schema differs from the active schema")
    model = NativeModel(archive, nthread=1)
    if list(model.feature_names) != list(metadata.get("feature_names", [])):
        raise ArtifactError("native model feature names do not match its metadata")
    if list(model.feature_names) != list(schema.get("features", [])):
        raise ArtifactError("registry model feature names do not match the serving schema")
    _smoke_test(model)
    return make_snapshot(
        model,
        schema,
        version=version,
        artifact_hash=artifact_hash,
        preprocessor_hash=preprocessor_hash,
        manifest=manifest,
    )


def load_version_snapshot(
    version: str,
    *,
    registry_path: str | Path,
    bootstrap_path: str | Path,
    schema_path: str | Path,
    expected_preprocessing_sha256: str | None = None,
    require_accepted: bool = True,
) -> ServingSnapshot:
    safe_version(version)
    bootstrap = load_bootstrap_snapshot(bootstrap_path, schema_path)
    if version == bootstrap.version:
        _smoke_test(bootstrap.model)
        if (
            expected_preprocessing_sha256 is not None
            and bootstrap.preprocessing_sha256 != expected_preprocessing_sha256
        ):
            raise ArtifactError("bootstrap preprocessing differs from the active model")
        return bootstrap
    return load_registry_snapshot(
        registry_path,
        version,
        schema=bootstrap.schema,
        expected_preprocessing_sha256=expected_preprocessing_sha256,
        require_accepted=require_accepted,
    )


def _smoke_test(model: Any) -> None:
    names = list(getattr(model, "feature_names", []))
    if not names:
        raise ArtifactError("candidate model has no feature names")
    numeric = list(getattr(model, "numeric_features", []))
    categorical = list(getattr(model, "categorical_features", []))
    partial = {numeric[0]: 0.0} if numeric else {}
    unknown = {categorical[0]: "__registry_unknown_category__"} if categorical else {}
    records = [partial, unknown, {}, {numeric[0]: None} if numeric else {}]
    try:
        predictions = np.asarray(model.predict_records(records), dtype=np.float64)
    except Exception as exc:
        raise ArtifactError(f"candidate smoke prediction failed: {exc}") from exc
    if predictions.shape != (len(records),) or not np.all(np.isfinite(predictions)):
        raise ArtifactError("candidate smoke predictions must be finite")
    if np.any(predictions < 0):
        raise ArtifactError("candidate smoke predictions must be nonnegative")


def read_pointer(path: str | Path) -> dict[str, Any] | None:
    pointer_path = Path(path)
    if not pointer_path.exists():
        return None
    try:
        value = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"serving pointer is invalid: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("version"), str):
        raise ArtifactError("serving pointer must contain a version")
    safe_version(value["version"])
    activation_id = value.get("activation_id")
    if activation_id is not None and (
        not isinstance(activation_id, str) or len(activation_id) > 128
    ):
        raise ArtifactError("serving pointer activation id is invalid")
    previous = value.get("previous_version")
    if previous is not None:
        safe_version(previous)
    return value


def atomic_write_pointer(
    path: str | Path,
    pointer: Mapping[str, Any],
    *,
    before_replace: Callable[[], None] | None = None,
) -> None:
    pointer_path = Path(path)
    version = safe_version(str(pointer.get("version", "")))
    activation_id = pointer.get("activation_id")
    previous = pointer.get("previous_version")
    value = {
        "version": version,
        "activation_id": activation_id,
        "previous_version": safe_version(previous) if previous is not None else None,
    }
    if activation_id is not None and (not isinstance(activation_id, str) or len(activation_id) > 128):
        raise ArtifactError("activation id is invalid")
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".current-", suffix=".tmp", dir=pointer_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary_path, pointer_path)
        if os.name != "nt":
            directory_fd = os.open(pointer_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
