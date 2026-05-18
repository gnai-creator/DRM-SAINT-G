"""Robust compact checkpoint format for SAINT runtime payloads."""

from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path
import struct
from typing import Any
import zlib


FORMAT_VERSION = 1
MATRIX_MAGIC = b"SAINTMAT1\n"
STATE_MAGIC = b"SAINTOPT1\n"
DTYPES = {
    "float32": ("<f", 4),
    "float16": ("<e", 2),
    "int8": ("<b", 1),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(payload: dict[str, list[list[float]]]) -> list[float]:
    return [float(value) for matrix in payload.values() for row in matrix for value in row]


def _quant_scale(values: list[float], dtype: str) -> float:
    if dtype != "int8":
        return 1.0
    max_abs = max((abs(value) for value in values), default=0.0)
    return max_abs / 127.0 if max_abs else 1.0


def _pack_value(value: float, dtype: str, scale: float) -> bytes:
    if dtype == "bfloat16":
        return struct.pack("<f", value)[2:4]
    if dtype == "int8":
        clipped = max(-127, min(127, round(value / scale)))
        return struct.pack("<b", int(clipped))
    fmt = DTYPES[dtype][0]
    return struct.pack(fmt, value)


def _unpack_values(raw: bytes, dtype: str, scale: float) -> list[float]:
    if dtype == "bfloat16":
        return [
            struct.unpack("<f", b"\x00\x00" + raw[index:index + 2])[0]
            for index in range(0, len(raw), 2)
        ]
    if dtype == "int8":
        return [float(item[0]) * scale for item in struct.iter_unpack("<b", raw)]
    fmt, size = DTYPES[dtype]
    return [float(item[0]) for item in struct.iter_unpack(fmt, raw[: len(raw) // size * size])]


def _dtype_size(dtype: str) -> int:
    if dtype == "bfloat16":
        return 2
    return DTYPES[dtype][1]


def _matrix_header(payload: dict[str, list[list[float]]], dtype: str) -> dict[str, Any]:
    matrices = {}
    value_count = 0
    values = _flatten(payload)
    scale = _quant_scale(values, dtype)
    for name, matrix in payload.items():
        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0
        matrices[name] = {
            "rows": rows,
            "cols": cols,
            "offset": value_count,
            "values": rows * cols,
        }
        value_count += rows * cols
    return {
        "format": "saint_matrix_payload",
        "version": FORMAT_VERSION,
        "dtype": dtype,
        "quant_scale": scale,
        "matrices": matrices,
        "value_count": value_count,
    }


def _write_matrix_file(
    path: Path,
    payload: dict[str, list[list[float]]],
    *,
    dtype: str,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = _matrix_header(payload, dtype)
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    with target.open("wb") as handle:
        handle.write(MATRIX_MAGIC)
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for matrix in payload.values():
            for row in matrix:
                for value in row:
                    handle.write(_pack_value(float(value), dtype, header["quant_scale"]))
    return {
        "path": target.name,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "payload": "delta",
        "format": "saint_matrix_payload",
        "dtype": dtype,
    }


def _matrix_value_count(matrix: list[list[float]]) -> int:
    return sum(len(row) for row in matrix)


def _shard_payloads(
    payload: dict[str, list[list[float]]],
    *,
    shard_bytes: int,
    dtype: str,
) -> list[tuple[dict[str, list[list[float]]], list[dict[str, int | str]]]]:
    shards = []
    current = {}
    current_parts = []
    current_bytes = 0
    size = _dtype_size(dtype)

    def flush_current() -> None:
        nonlocal current, current_bytes, current_parts
        if current:
            shards.append((current, current_parts))
            current = {}
            current_parts = []
            current_bytes = 0

    for name, matrix in payload.items():
        matrix_bytes = _matrix_value_count(matrix) * size
        row_values = len(matrix[0]) if matrix else 0
        row_bytes = max(1, row_values * size)
        if matrix_bytes > shard_bytes and len(matrix) > 1:
            flush_current()
            rows_per_shard = max(1, shard_bytes // row_bytes)
            for row_start in range(0, len(matrix), rows_per_shard):
                rows = matrix[row_start:row_start + rows_per_shard]
                part = {
                    "name": name,
                    "row_start": row_start,
                    "rows": len(rows),
                }
                shards.append(({name: rows}, [part]))
            continue
        if current and current_bytes + matrix_bytes > shard_bytes:
            flush_current()
        current[name] = matrix
        current_parts.append(
            {
                "name": name,
                "row_start": 0,
                "rows": len(matrix),
            }
        )
        current_bytes += matrix_bytes
    flush_current()
    return shards


def write_matrix_payload(
    path: str | Path,
    payload: dict[str, list[list[float]]],
    *,
    dtype: str = "float32",
    shard_bytes: int | None = None,
) -> dict[str, Any]:
    if dtype not in {*DTYPES, "bfloat16"}:
        raise ValueError(f"unsupported checkpoint dtype: {dtype}")
    target = Path(path)
    if not shard_bytes:
        return _write_matrix_file(target, payload, dtype=dtype)

    shards = _shard_payloads(payload, shard_bytes=shard_bytes, dtype=dtype)
    entries = []
    for index, (shard, parts) in enumerate(shards):
        shard_path = target.with_name(f"{target.stem}_{index:04d}{target.suffix}")
        entry = _write_matrix_file(shard_path, shard, dtype=dtype)
        entry["index"] = index
        entry["matrix_parts"] = parts
        entries.append(entry)
    return {
        "payload": "delta",
        "format": "saint_matrix_shards",
        "dtype": dtype,
        "shards": entries,
        "bytes": sum(entry["bytes"] for entry in entries),
        "shard_count": len(entries),
    }


def _read_header(handle, magic: bytes) -> dict[str, Any]:
    actual = handle.read(len(magic))
    if actual != magic:
        raise ValueError("checkpoint payload magic mismatch")
    header_size = struct.unpack("<Q", handle.read(8))[0]
    header = json.loads(handle.read(header_size).decode("utf-8"))
    if int(header.get("version", -1)) != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint payload version")
    return header


def read_matrix_payload(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    use_mmap: bool = True,
) -> dict[str, list[list[float]]]:
    source = Path(path)
    if expected_sha256 and sha256_file(source) != expected_sha256:
        raise ValueError("checkpoint payload checksum mismatch")
    with source.open("rb") as handle:
        header = _read_header(handle, MATRIX_MAGIC)
        if use_mmap:
            data_offset = handle.tell()
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                raw = bytes(mapped[data_offset:])
        else:
            raw = handle.read()
    dtype = header.get("dtype", "float32")
    values = _unpack_values(raw, dtype, float(header.get("quant_scale", 1.0)))
    if len(values) != int(header["value_count"]):
        raise ValueError("checkpoint payload value count mismatch")
    matrices = {}
    for name, spec in header["matrices"].items():
        rows = int(spec["rows"])
        cols = int(spec["cols"])
        offset = int(spec["offset"])
        count = int(spec["values"])
        matrix_values = values[offset:offset + count]
        matrices[name] = [
            matrix_values[row * cols:(row + 1) * cols]
            for row in range(rows)
        ]
    return matrices


def read_matrix_payload_entry(run_dir: str | Path, entry: dict[str, Any]) -> dict[str, list[list[float]]]:
    run_path = Path(run_dir)
    if entry.get("format") == "saint_matrix_shards":
        merged: dict[str, list[list[float]]] = {}
        partials: dict[str, list[tuple[int, list[list[float]]]]] = {}
        for shard in entry.get("shards", []):
            shard_payload = read_matrix_payload(
                run_path / shard["path"],
                expected_sha256=shard.get("sha256"),
            )
            for part in shard.get("matrix_parts", []):
                name = str(part["name"])
                row_start = int(part.get("row_start", 0))
                partials.setdefault(name, []).append((row_start, shard_payload[name]))
            if not shard.get("matrix_parts"):
                merged.update(shard_payload)
        for name, pieces in partials.items():
            rows = []
            for _, matrix_rows in sorted(pieces, key=lambda item: item[0]):
                rows.extend(matrix_rows)
            merged[name] = rows
        return merged
    return read_matrix_payload(
        run_path / entry["path"],
        expected_sha256=entry.get("sha256"),
    )


def write_state_payload(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    compressed = zlib.compress(encoded)
    header = {
        "format": "saint_optimizer_state",
        "version": FORMAT_VERSION,
        "codec": "zlib+json",
        "uncompressed_bytes": len(encoded),
    }
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    with target.open("wb") as handle:
        handle.write(STATE_MAGIC)
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(compressed)
    return {
        "path": target.name,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "payload": "optimizer_state",
        "format": "saint_optimizer_state",
    }


def read_state_payload(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    source = Path(path)
    if expected_sha256 and sha256_file(source) != expected_sha256:
        raise ValueError("optimizer state checksum mismatch")
    with source.open("rb") as handle:
        _read_header(handle, STATE_MAGIC)
        compressed = handle.read()
    data = json.loads(zlib.decompress(compressed).decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("optimizer state payload must be an object")
    return data


__all__ = [
    "FORMAT_VERSION",
    "read_matrix_payload",
    "read_matrix_payload_entry",
    "read_state_payload",
    "sha256_file",
    "write_matrix_payload",
    "write_state_payload",
]
