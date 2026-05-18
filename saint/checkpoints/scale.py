"""Scale validation helpers for checkpoint sharding experiments."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import tracemalloc
from typing import Any

from saint.checkpoints.robust import read_matrix_payload_entry, write_matrix_payload


def synthetic_delta_payload(
    *,
    matrix_count: int = 4,
    rows: int = 128,
    cols: int = 128,
) -> dict[str, list[list[float]]]:
    payload = {}
    for matrix_index in range(matrix_count):
        matrix = []
        for row in range(rows):
            matrix.append(
                [
                    ((matrix_index + 1) * 0.001)
                    + ((row % 17) - 8) * 0.0003
                    + ((col % 19) - 9) * 0.0002
                    for col in range(cols)
                ]
            )
        payload[f"matrix_{matrix_index:03d}"] = matrix
    return payload


def _max_abs_error(
    expected: dict[str, list[list[float]]],
    actual: dict[str, list[list[float]]],
) -> float:
    max_error = 0.0
    for name, matrix in expected.items():
        other = actual[name]
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                max_error = max(max_error, abs(value - other[row_index][col_index]))
    return max_error


def benchmark_large_shards(
    run_dir: str | Path,
    *,
    matrix_count: int = 4,
    rows: int = 128,
    cols: int = 128,
    dtype: str = "float16",
    shard_bytes: int = 8192,
) -> dict[str, Any]:
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = synthetic_delta_payload(
        matrix_count=matrix_count,
        rows=rows,
        cols=cols,
    )

    write_start = perf_counter()
    entry = write_matrix_payload(
        target / "large_deltas.saintbin",
        payload,
        dtype=dtype,
        shard_bytes=shard_bytes,
    )
    write_elapsed_s = perf_counter() - write_start

    tracemalloc.start()
    read_start = perf_counter()
    restored = read_matrix_payload_entry(target, entry)
    read_elapsed_s = perf_counter() - read_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    value_count = matrix_count * rows * cols
    return {
        "format": entry["format"],
        "dtype": dtype,
        "matrix_count": matrix_count,
        "rows": rows,
        "cols": cols,
        "value_count": value_count,
        "shard_bytes": shard_bytes,
        "shard_count": int(entry.get("shard_count", 1)),
        "payload_bytes": int(entry["bytes"]),
        "write_elapsed_s": write_elapsed_s,
        "read_elapsed_s": read_elapsed_s,
        "read_peak_bytes": peak_bytes,
        "checksum_validated": True,
        "max_abs_error": _max_abs_error(payload, restored),
    }


__all__ = ["benchmark_large_shards", "synthetic_delta_payload"]
