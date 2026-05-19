"""Checkpoint and smoke-training adapter for drm_transformer matrices.

The core SAINT package stays dependency-free. PyTorch is imported only when a
DRM checkpoint or the autograd smoke path is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from saint.config import RuntimeConfig
from saint.reconstruction.matrix_ops import shape, zeros
from saint.transformer.training import MiniTransformerResult


DEFAULT_KEYWORDS = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.out_proj.weight",
    "ffn.up_proj.weight",
    "ffn.down_proj.weight",
    "dim_gate.gate_net.0.weight",
)


@dataclass(frozen=True)
class DRMCheckpointTask:
    base_weights: dict[str, list[list[float]]]


def _sign(value: float) -> float:
    return -1.0 if value < 0.0 else 1.0


def _metadata(config: RuntimeConfig) -> dict[str, Any]:
    return dict(config.metadata or {})


def _keywords(metadata: dict[str, Any]) -> tuple[str, ...]:
    values = metadata.get("keywords", DEFAULT_KEYWORDS)
    if isinstance(values, list):
        return tuple(str(item) for item in values)
    return DEFAULT_KEYWORDS


def _tensor_name_matches(name: str, keywords: tuple[str, ...]) -> bool:
    return not keywords or any(keyword in name for keyword in keywords)


def _load_state_dict(checkpoint: str | Path) -> dict[str, Any]:
    path = Path(checkpoint)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("model", "model_state_dict", "state_dict"):
                if isinstance(data.get(key), dict):
                    return data[key]
            return data
        raise ValueError("JSON checkpoint must contain an object")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to load drm_transformer checkpoints."
        ) from exc

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(state.get(key), dict):
                return state[key]
        return state
    raise ValueError("drm_transformer checkpoint did not contain a state dict")


def _is_2d_payload(value: Any) -> bool:
    if hasattr(value, "ndim"):
        return int(value.ndim) == 2
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and row for row in value)
    )


def _payload_shape(value: Any) -> tuple[int, int]:
    if hasattr(value, "shape"):
        return int(value.shape[0]), int(value.shape[1])
    return len(value), len(value[0])


def _slice_2d(value: Any, rows: int, cols: int) -> list[list[float]]:
    if hasattr(value, "float"):
        return value[:rows, :cols].float().tolist()
    return [
        [float(value[row][col]) for col in range(cols)]
        for row in range(rows)
    ]


def _generated_matrix_payload(config: RuntimeConfig) -> dict[str, list[list[float]]]:
    from saint.adapters.drm_autograd import generated_matrix_payload

    return generated_matrix_payload(config)


def _matrix_payload(config: RuntimeConfig) -> dict[str, list[list[float]]]:
    metadata = _metadata(config)
    checkpoint = metadata.get("checkpoint")
    if not checkpoint:
        if config.method == "drm_saint_autograd_smoke":
            return _generated_matrix_payload(config)
        raise ValueError("drm_transformer adapter requires metadata.checkpoint")

    max_dim = int(metadata.get("max_dim", 64))
    max_matrices = int(metadata.get("max_matrices", 16))
    keywords = _keywords(metadata)
    state_dict = _load_state_dict(checkpoint)
    matrices: dict[str, list[list[float]]] = {}

    for name, tensor in state_dict.items():
        if not _tensor_name_matches(name, keywords):
            continue
        if not _is_2d_payload(tensor):
            continue
        tensor_rows, tensor_cols = _payload_shape(tensor)
        rows = min(tensor_rows, max_dim)
        cols = min(tensor_cols, max_dim)
        matrices[name] = _slice_2d(tensor, rows, cols)
        if len(matrices) >= max_matrices:
            break

    if not matrices:
        raise ValueError("no matching 2D drm_transformer matrices found")
    return matrices


def make_task(config: RuntimeConfig) -> DRMCheckpointTask:
    return DRMCheckpointTask(base_weights=_matrix_payload(config))


def _block_regions(weights: dict[str, list[list[float]]], block_size: int) -> list[dict]:
    regions = []
    for matrix_name, matrix in weights.items():
        rows, cols = shape(matrix)
        for row in range(0, rows, block_size):
            for col in range(0, cols, block_size):
                row_end = min(row + block_size, rows)
                col_end = min(col + block_size, cols)
                magnitude = sum(
                    abs(matrix[r][c])
                    for r in range(row, row_end)
                    for c in range(col, col_end)
                )
                regions.append(
                    {
                        "matrix": matrix_name,
                        "row": row,
                        "col": col,
                        "rows": row_end - row,
                        "cols": col_end - col,
                        "magnitude": magnitude,
                    }
                )
    return sorted(regions, key=lambda item: item["magnitude"], reverse=True)


def _delta_for_regions(
    weights: dict[str, list[list[float]]],
    regions: list[dict],
    *,
    parameter_budget: int,
    delta_scale: float,
) -> tuple[dict[str, list[list[float]]], list[dict]]:
    deltas = {name: zeros(*shape(matrix)) for name, matrix in weights.items()}
    selected = []
    used = 0
    for region in regions:
        cost = region["rows"] * region["cols"]
        if used + cost > parameter_budget:
            continue
        matrix = weights[region["matrix"]]
        for row in range(region["row"], region["row"] + region["rows"]):
            for col in range(region["col"], region["col"] + region["cols"]):
                deltas[region["matrix"]][row][col] = (
                    delta_scale * 1e-3 * _sign(matrix[row][col])
                )
        selected.append({key: region[key] for key in ("matrix", "row", "col", "rows", "cols")})
        used += cost
        if used >= parameter_budget:
            break
    return deltas, selected


def _validate_delta_shapes(
    weights: dict[str, list[list[float]]],
    deltas: dict[str, list[list[float]]],
) -> bool:
    return all(
        name in deltas and shape(deltas[name]) == shape(matrix)
        for name, matrix in weights.items()
    )


def _run_drm_delta_smoke(config: RuntimeConfig) -> MiniTransformerResult:
    start = perf_counter()
    metadata = _metadata(config)
    block_size = int(metadata.get("block_size", 2))
    task = make_task(config)
    regions = _block_regions(task.base_weights, block_size)
    deltas, selected = _delta_for_regions(
        task.base_weights,
        regions,
        parameter_budget=max(1, config.parameter_budget),
        delta_scale=config.delta_scale,
    )
    parameter_count = sum(region["rows"] * region["cols"] for region in selected)
    return MiniTransformerResult(
        name="drm_saint_delta_smoke",
        train_loss=0.0,
        test_loss=0.0,
        parameter_count=parameter_count,
        optimizer_state_values=0,
        elapsed_s=perf_counter() - start,
        metadata={
            "delta_payload": deltas,
            "block_size": block_size,
            "selected_regions": selected,
            "available_regions": len(regions),
            "shape_validation": _validate_delta_shapes(task.base_weights, deltas),
            "checkpoint": metadata.get("checkpoint"),
            "autograd": False,
            "marco": "fase_9_marco_1",
        },
    )


def run_method(config: RuntimeConfig) -> MiniTransformerResult:
    if config.method == "drm_g_saint_phi_eval":
        from saint.adapters.drm_grafting_eval import run_drm_graft_eval

        return run_drm_graft_eval(config)
    if config.method == "drm_g_saint_phi_graft":
        from saint.adapters.drm_grafting import run_drm_graft

        return run_drm_graft(config)
    if config.method == "drm_g_saint_phi_progressive":
        from saint.adapters.drm_grafting_progressive import run_drm_graft_progressive

        return run_drm_graft_progressive(config)
    if config.method == "drm_saint_autograd_smoke":
        from saint.adapters.drm_autograd import run_drm_autograd

        return run_drm_autograd(config)
    if config.method == "drm_saint_delta_smoke":
        return _run_drm_delta_smoke(config)
    raise NotImplementedError(f"unknown drm_transformer method: {config.method}")


def inspect_model(config: RuntimeConfig) -> dict:
    if config.method in {"drm_g_saint_phi_graft", "drm_g_saint_phi_progressive"}:
        from saint.adapters.drm_grafting import inspect_graft_model

        return inspect_graft_model(config)
    matrices = {
        name: {
            "rows": len(matrix),
            "cols": len(matrix[0]) if matrix else 0,
            "parameters": sum(len(row) for row in matrix),
        }
        for name, matrix in _matrix_payload(config).items()
    }
    return {
        "task": config.task,
        "adapter": "drm_transformer_checkpoint",
        "matrices": matrices,
        "total_parameters": sum(item["parameters"] for item in matrices.values()),
    }


__all__ = ["DRMCheckpointTask", "inspect_model", "make_task", "run_method"]
