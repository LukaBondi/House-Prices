"""Immutable local registry format shared by bootstrap and retraining tools."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from backend.artifacts import (
    ArtifactError,
    safe_version,
)
from backend.artifacts import (
    preprocessing_sha256 as serving_preprocessing_sha256,
)
from backend.artifacts import (
    schema_sha256 as serving_schema_sha256,
)

FORMAT = "house-prices-registry-v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    from backend.artifacts import preprocessing_payload as serving_preprocessing_payload

    return serving_preprocessing_payload(metadata)


def preprocessing_sha256(metadata: dict[str, Any]) -> str:
    return serving_preprocessing_sha256(metadata)


def schema_sha256(schema: dict[str, Any] | str | Path) -> str:
    """Hash the canonical schema object used by the serving artifact validator."""
    return serving_schema_sha256(schema)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def archive_metadata(archive_bytes: bytes) -> tuple[dict[str, Any], bytes]:
    import io

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        metadata_bytes = archive.read("metadata.json")
        metadata = json.loads(metadata_bytes)
    return metadata, metadata_bytes


def load_manifest(registry_root: str | Path, version: str, *, verify_artifact: bool = True) -> dict[str, Any]:
    safe_version(version)
    directory = Path(registry_root) / version
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("format") != FORMAT or manifest.get("version") != version:
        raise ValueError("Registry manifest format/version mismatch")
    artifact = directory / "model.zip"
    if verify_artifact and sha256_file(artifact) != manifest.get("artifact_sha256"):
        raise ValueError("Registry artifact checksum mismatch")
    metadata, _ = archive_metadata(artifact.read_bytes())
    if preprocessing_sha256(metadata) != manifest.get("preprocessing_sha256"):
        raise ValueError("Registry preprocessing fingerprint mismatch")
    return manifest


def write_immutable_version(
    registry_root: str | Path,
    archive_bytes: bytes,
    *,
    parent_version: str | None,
    gate: dict[str, Any],
    training: dict[str, Any],
    schema_sha256: str,
    version_prefix: str = "model-",
    version_override: str | None = None,
) -> tuple[str, Path, dict[str, Any]]:
    metadata, _ = archive_metadata(archive_bytes)
    artifact_hash = sha256_bytes(archive_bytes)
    version_material = {
        "artifact_sha256": artifact_hash,
        "parent_version": parent_version,
        "gate": gate,
        "training": training,
    }
    version = version_override or f"{version_prefix}{sha256_bytes(canonical_json(version_material))[:16]}"
    safe_version(version)
    manifest = {
        "format": FORMAT,
        "version": version,
        "artifact_sha256": artifact_hash,
        "preprocessing_sha256": preprocessing_sha256(metadata),
        "schema_sha256": schema_sha256,
        "parent_version": parent_version,
        "gate": gate,
        "training": training,
    }
    root = Path(registry_root)
    root_existed = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    if not root_existed:
        _fsync_directory(root.parent)
    destination = root / version
    if destination.exists():
        existing = load_manifest(root, version)
        if canonical_json(existing) != canonical_json(manifest):
            raise FileExistsError(f"Registry version collision at {destination}")
        if (destination / "model.zip").read_bytes() != archive_bytes:
            raise FileExistsError(f"Registry artifact collision at {destination}")
        return version, destination, manifest
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=root))
    try:
        artifact_path = temporary / "model.zip"
        manifest_path = temporary / "manifest.json"
        with artifact_path.open("wb") as stream:
            stream.write(archive_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        with manifest_path.open("wb") as stream:
            stream.write(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(temporary)
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            existing = load_manifest(root, version)
            if canonical_json(existing) != canonical_json(manifest):
                raise FileExistsError(f"Registry version collision at {destination}")
            shutil.rmtree(temporary, ignore_errors=True)
        _fsync_directory(root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return version, destination, manifest


def current_version(registry_root: str | Path, state_root: str | Path) -> str:
    pointer_path = Path(state_root) / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8-sig")) if pointer_path.exists() else {}
    version = pointer.get("version")
    if not isinstance(version, str):
        raise FileNotFoundError("No valid serving pointer; run model_training.bootstrap_registry first")
    try:
        safe_version(version)
    except ArtifactError as exc:
        raise FileNotFoundError("No valid serving pointer; run model_training.bootstrap_registry first") from exc
    load_manifest(registry_root, version)
    return version
