"""Checkpoint helpers for SAINT runtime experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from saint.checkpoints.robust import (
    FORMAT_VERSION,
    read_matrix_payload_entry,
    read_state_payload,
    sha256_file,
    write_matrix_payload,
    write_state_payload,
)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    }
    return payload


def write_checkpoint_bundle(run_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = dict(payload)
    delta_payload = manifest.pop("_delta_payload", None)
    optimizer_payload = manifest.pop("_optimizer_state_payload", None)
    files = []
    if delta_payload is not None:
        metadata = manifest.get("metadata", {})
        shard_bytes = metadata.get("checkpoint_shard_bytes")
        files.append(
            write_matrix_payload(
                target / "deltas.saintbin",
                delta_payload,
                dtype=str(metadata.get("checkpoint_dtype", "float32")),
                shard_bytes=int(shard_bytes) if shard_bytes else None,
            )
        )
    state_payload = optimizer_payload or {
        "optimizer_state_values": manifest["optimizer_state_values"],
        "state": {},
    }
    files.append(write_state_payload(target / "optimizer.saintopt", state_payload))
    manifest["format"] = "saint_checkpoint"
    manifest["format_version"] = FORMAT_VERSION
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
        require_delta_payload(checkpoint, run_path)
    if validate_payloads:
        require_optimizer_state(checkpoint, run_path)
    return checkpoint


def migrate_checkpoint_manifest(checkpoint: dict[str, Any]) -> dict[str, Any]:
    version = int(checkpoint.get("format_version", -1))
    if version == FORMAT_VERSION:
        return checkpoint
    raise ValueError("unsupported checkpoint format version")


__all__ = [
    "checkpoint_payload",
    "migrate_checkpoint_manifest",
    "read_json",
    "require_delta_payload",
    "require_optimizer_state",
    "validate_checkpoint_bundle",
    "write_checkpoint_bundle",
    "write_json",
    "write_jsonl",
    "write_metrics",
]
