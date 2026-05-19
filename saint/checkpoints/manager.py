"""Checkpoint helpers for SAINT runtime experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from saint.checkpoints.robust import (
    FORMAT_VERSION as PAYLOAD_FORMAT_VERSION,
    read_matrix_payload_entry,
    read_state_payload,
    sha256_file,
    write_matrix_payload,
    write_state_payload,
)

MANIFEST_FORMAT_VERSION = 2
MIN_READER_VERSION = 1


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_sparse_delta(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    write_json(path, payload)
    value_count = sum(len(values) for values in payload.get("values", {}).values())
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "payload": "delta",
        "format": "saint_sparse_delta_json",
        "values": value_count,
        "matrices": list(payload.get("shapes", {})),
    }


def _write_graft_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    write_json(path, payload)
    trainable = payload.get("trainable_parameters", 0)
    if payload.get("format") == "drm_graft_sequence_payload":
        trainable = sum(
            int(item.get("trainable_parameters", 0))
            for item in payload.get("grafts", [])
        )
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "payload": "delta",
        "format": "drm_graft_payload_json",
        "target_module": payload.get("target_module"),
        "trainable_parameters": trainable,
    }


def _write_consolidation_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    write_json(path, payload)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "payload": "consolidation",
        "format": "drm_graft_consolidation_json",
        "target_module": payload.get("target_module"),
        "state_dict_merge_supported": payload.get("state_dict_merge_supported", False),
    }


def _expand_sparse_delta(
    payload: dict[str, Any],
    matrix_names: set[str] | None = None,
) -> dict[str, list[list[float]]]:
    matrices = {}
    shapes = payload.get("shapes", {})
    values = payload.get("values", {})
    for name, shape in shapes.items():
        if matrix_names is not None and name not in matrix_names:
            continue
        rows, cols = int(shape[0]), int(shape[1])
        matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for row, col, value in values.get(name, []):
            matrix[int(row)][int(col)] = float(value)
        matrices[name] = matrix
    return matrices


def _validate_sparse_delta(payload: dict[str, Any]) -> None:
    shapes = payload.get("shapes", {})
    values = payload.get("values", {})
    for name, shape in shapes.items():
        rows, cols = int(shape[0]), int(shape[1])
        for row, col, _value in values.get(name, []):
            if not (0 <= int(row) < rows and 0 <= int(col) < cols):
                raise ValueError(f"sparse delta index out of bounds: {name}")


def require_sparse_delta_payload(
    checkpoint: dict[str, Any],
    run_dir: str | Path,
    *,
    matrix_names: set[str] | None = None,
) -> dict[str, Any]:
    entry = _file_entry(checkpoint, "delta")
    if entry is None or entry.get("format") != "saint_sparse_delta_json":
        raise ValueError("checkpoint does not contain a sparse delta payload")
    payload = read_json(Path(run_dir) / entry["path"])
    _validate_sparse_delta(payload)
    if matrix_names is None:
        return payload
    names = set(matrix_names)
    return {
        **payload,
        "shapes": {
            name: shape
            for name, shape in payload.get("shapes", {}).items()
            if name in names
        },
        "values": {
            name: values
            for name, values in payload.get("values", {}).items()
            if name in names
        },
    }


def require_graft_payload(checkpoint: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    entry = _file_entry(checkpoint, "delta")
    if entry is None or entry.get("format") != "drm_graft_payload_json":
        raise ValueError("checkpoint does not contain a DRM-G graft payload")
    return read_json(Path(run_dir) / entry["path"])


def read_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("checkpoint payload must be a JSON object")
    return data


def write_jsonl(path: str | Path, events: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(event, sort_keys=True) for event in events]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def checkpoint_payload(result, config, memory_plan) -> dict[str, Any]:
    metadata = {**dict(config.metadata or {}), **dict(result.metadata)}
    delta_payload = metadata.pop("delta_payload", None)
    optimizer_payload = metadata.pop("optimizer_state_payload", None)
    consolidation_payload = metadata.pop("consolidation_payload", None)
    payload = {
        "experiment_name": config.experiment_name,
        "method": result.name,
        "train_loss": result.train_loss,
        "test_loss": result.test_loss,
        "parameter_count": result.parameter_count,
        "optimizer_state_values": result.optimizer_state_values,
        "memory_plan": memory_plan.__dict__,
        "metadata": metadata,
        "has_delta_payload": delta_payload is not None,
        "_delta_payload": delta_payload,
        "_optimizer_state_payload": optimizer_payload,
        "_consolidation_payload": consolidation_payload,
    }
    return payload


def write_checkpoint_bundle(run_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = dict(payload)
    delta_payload = manifest.pop("_delta_payload", None)
    optimizer_payload = manifest.pop("_optimizer_state_payload", None)
    consolidation_payload = manifest.pop("_consolidation_payload", None)
    files = []
    if delta_payload is not None:
        metadata = manifest.get("metadata", {})
        if delta_payload.get("format") in {"drm_graft_payload", "drm_graft_sequence_payload"}:
            files.append(_write_graft_payload(target / "graft.drm-g.json", delta_payload))
        elif delta_payload.get("format") == "saint_sparse_delta":
            files.append(_write_sparse_delta(target / "deltas.saintdelta.json", delta_payload))
        else:
            shard_bytes = metadata.get("checkpoint_shard_bytes")
            files.append(
                write_matrix_payload(
                    target / "deltas.saintbin",
                    delta_payload,
                    dtype=str(metadata.get("checkpoint_dtype", "float32")),
                    shard_bytes=int(shard_bytes) if shard_bytes else None,
                )
            )
    if consolidation_payload is not None:
        files.append(
            _write_consolidation_payload(
                target / "consolidation.drm-g.json",
                consolidation_payload,
            )
        )
    state_payload = optimizer_payload or {
        "optimizer_state_values": manifest["optimizer_state_values"],
        "state": {},
    }
    files.append(write_state_payload(target / "optimizer.saintopt", state_payload))
    manifest["format"] = "saint_checkpoint"
    manifest["format_version"] = MANIFEST_FORMAT_VERSION
    manifest["compatibility"] = {
        "min_reader_version": MIN_READER_VERSION,
        "payload_format_version": PAYLOAD_FORMAT_VERSION,
        "writer_version": MANIFEST_FORMAT_VERSION,
    }
    manifest["files"] = files
    write_json(target / "checkpoint.json", manifest)
    return manifest


def write_metrics(path: str | Path, payload: dict[str, Any]) -> None:
    metrics = {
        key: value
        for key, value in payload.items()
        if not key.startswith("_") and key not in {"delta_payload"}
    }
    write_json(path, metrics)


def _file_entry(checkpoint: dict[str, Any], payload_name: str) -> dict[str, Any] | None:
    for entry in checkpoint.get("files", []):
        if entry.get("payload") == payload_name:
            return entry
    return None


def require_delta_payload(
    checkpoint: dict[str, Any],
    run_dir: str | Path | None = None,
    *,
    matrix_names: set[str] | None = None,
) -> dict[str, Any]:
    payload = checkpoint.get("delta_payload")
    if not isinstance(payload, dict):
        entry = _file_entry(checkpoint, "delta")
        if entry is None or run_dir is None:
            raise ValueError("checkpoint does not contain a delta payload")
        if entry.get("format") == "drm_graft_payload_json":
            return read_json(Path(run_dir) / entry["path"])
        if entry.get("format") == "saint_sparse_delta_json":
            payload = read_json(Path(run_dir) / entry["path"])
            return _expand_sparse_delta(payload, matrix_names)
        return read_matrix_payload_entry(run_dir, entry, matrix_names=matrix_names)
    if matrix_names is None:
        return payload
    return {
        name: matrix
        for name, matrix in payload.items()
        if name in matrix_names
    }


def require_optimizer_state(checkpoint: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    entry = _file_entry(checkpoint, "optimizer_state")
    if entry is None:
        raise ValueError("checkpoint does not contain optimizer state")
    return read_state_payload(
        Path(run_dir) / entry["path"],
        expected_sha256=entry.get("sha256"),
    )


def validate_checkpoint_bundle(
    run_dir: str | Path,
    *,
    validate_payloads: bool = True,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    checkpoint = read_json(run_path / "checkpoint.json")
    checkpoint = migrate_checkpoint_manifest(checkpoint)
    for entry in checkpoint.get("files", []):
        entries = entry.get("shards", [entry])
        for file_entry in entries:
            path = run_path / file_entry["path"]
            if sha256_file(path) != file_entry.get("sha256"):
                raise ValueError(
                    f"checkpoint file checksum mismatch: {file_entry['path']}"
                )
    if validate_payloads and checkpoint.get("has_delta_payload"):
        entry = _file_entry(checkpoint, "delta")
        if entry is not None and entry.get("format") == "saint_sparse_delta_json":
            require_sparse_delta_payload(checkpoint, run_path)
        else:
            require_delta_payload(checkpoint, run_path)
    if validate_payloads:
        require_optimizer_state(checkpoint, run_path)
    return checkpoint


def migrate_checkpoint_manifest(checkpoint: dict[str, Any]) -> dict[str, Any]:
    version = int(checkpoint.get("format_version", -1))
    if version == MANIFEST_FORMAT_VERSION:
        return checkpoint
    if version == 1:
        migrated = dict(checkpoint)
        migrated["format_version"] = MANIFEST_FORMAT_VERSION
        migrated["compatibility"] = {
            "min_reader_version": MIN_READER_VERSION,
            "payload_format_version": PAYLOAD_FORMAT_VERSION,
            "writer_version": 1,
        }
        migrated["migrated_from"] = {
            "format_version": 1,
            "migration": "manifest_v1_to_v2",
        }
        return migrated
    if version > MANIFEST_FORMAT_VERSION:
        raise ValueError(
            "checkpoint format version is newer than this SAINT reader"
        )
    raise ValueError("unsupported checkpoint format version")


__all__ = [
    "checkpoint_payload",
    "MANIFEST_FORMAT_VERSION",
    "migrate_checkpoint_manifest",
    "read_json",
    "require_graft_payload",
    "require_delta_payload",
    "require_optimizer_state",
    "require_sparse_delta_payload",
    "validate_checkpoint_bundle",
    "write_checkpoint_bundle",
    "write_json",
    "write_jsonl",
    "write_metrics",
]
