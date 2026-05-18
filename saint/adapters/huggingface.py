"""Dependency-optional Hugging Face adapter for small local checkpoints."""

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
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "out_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "wte.weight",
)


@dataclass(frozen=True)
class HuggingFaceTask:
    base_weights: dict[str, list[list[float]]]
    model_source: str


def _metadata(config: RuntimeConfig) -> dict[str, Any]:
    return dict(config.metadata or {})


def _keywords(metadata: dict[str, Any]) -> tuple[str, ...]:
    values = metadata.get("keywords", DEFAULT_KEYWORDS)
    if isinstance(values, list):
        return tuple(str(item) for item in values)
    return DEFAULT_KEYWORDS


def _matches(name: str, keywords: tuple[str, ...]) -> bool:
    return not keywords or any(keyword in name for keyword in keywords)


def _load_json_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Hugging Face JSON checkpoint must contain an object")
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(data.get(key), dict):
            return data[key]
    return data


def _load_torch_state(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load binary checkpoints") from exc
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(state.get(key), dict):
                return state[key]
        return state
    raise ValueError("binary checkpoint did not contain a state dict")


def _load_transformers_state(path: Path) -> dict[str, Any]:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for model_name_or_path loading"
        ) from exc
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True)
    return dict(model.state_dict())


def _load_state_dict(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    if path.is_file() and path.suffix.lower() == ".json":
        return _load_json_state(path)
    if path.is_file() and path.suffix.lower() in {".bin", ".pt", ".pth"}:
        return _load_torch_state(path)
    if path.is_dir():
        json_path = path / "state_dict.json"
        if json_path.exists():
            return _load_json_state(json_path)
        for name in ("pytorch_model.bin", "model.bin", "model.pt"):
            candidate = path / name
            if candidate.exists():
                return _load_torch_state(candidate)
        return _load_transformers_state(path)
    raise ValueError(f"unsupported Hugging Face model source: {source}")


def _is_2d(value: Any) -> bool:
    if hasattr(value, "ndim"):
        return int(value.ndim) == 2
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and row for row in value)
    )


def _shape(value: Any) -> tuple[int, int]:
    if hasattr(value, "shape"):
        return int(value.shape[0]), int(value.shape[1])
    return len(value), len(value[0])


def _slice(value: Any, rows: int, cols: int) -> list[list[float]]:
    if hasattr(value, "float"):
        return value[:rows, :cols].float().tolist()
    return [[float(value[row][col]) for col in range(cols)] for row in range(rows)]


def _matrix_payload(config: RuntimeConfig) -> tuple[str, dict[str, list[list[float]]]]:
    metadata = _metadata(config)
    source = metadata.get("model_name_or_path") or metadata.get("checkpoint")
    if not source:
        raise ValueError("Hugging Face adapter requires metadata.model_name_or_path")
    max_dim = int(metadata.get("max_dim", 64))
    max_matrices = int(metadata.get("max_matrices", 16))
    keywords = _keywords(metadata)
    matrices = {}
    for name, tensor in _load_state_dict(source).items():
        if not _matches(name, keywords) or not _is_2d(tensor):
            continue
        tensor_rows, tensor_cols = _shape(tensor)
        rows = min(tensor_rows, max_dim)
        cols = min(tensor_cols, max_dim)
        matrices[name] = _slice(tensor, rows, cols)
        if len(matrices) >= max_matrices:
            break
    if not matrices:
        raise ValueError("no matching 2D Hugging Face matrices found")
    return str(source), matrices


def make_task(config: RuntimeConfig) -> HuggingFaceTask:
    source, matrices = _matrix_payload(config)
    return HuggingFaceTask(base_weights=matrices, model_source=source)


def _sign(value: float) -> float:
    return -1.0 if value < 0.0 else 1.0


def _regions(weights: dict[str, list[list[float]]], block_size: int) -> list[dict[str, Any]]:
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


def _delta_payload(
    weights: dict[str, list[list[float]]],
    regions: list[dict[str, Any]],
    *,
    parameter_budget: int,
    delta_scale: float,
) -> tuple[dict[str, list[list[float]]], list[dict[str, Any]]]:
    deltas = {name: zeros(*shape(matrix)) for name, matrix in weights.items()}
    selected = []
    used = 0
    for region in regions:
        cost = int(region["rows"]) * int(region["cols"])
        if used + cost > parameter_budget:
            continue
        matrix = weights[str(region["matrix"])]
        for row in range(int(region["row"]), int(region["row"]) + int(region["rows"])):
            for col in range(int(region["col"]), int(region["col"]) + int(region["cols"])):
                deltas[str(region["matrix"])][row][col] = (
                    delta_scale * 1e-3 * _sign(matrix[row][col])
                )
        selected.append({key: region[key] for key in ("matrix", "row", "col", "rows", "cols")})
        used += cost
    return deltas, selected


def run_method(config: RuntimeConfig) -> MiniTransformerResult:
    if config.method == "hf_saint_autograd_smoke":
        from saint.adapters.huggingface_autograd import run_hf_autograd

        return run_hf_autograd(config)
    if config.method != "hf_saint_delta_smoke":
        raise ValueError(f"unknown Hugging Face method: {config.method}")
    start = perf_counter()
    metadata = _metadata(config)
    task = make_task(config)
    block_size = int(metadata.get("block_size", 2))
    regions = _regions(task.base_weights, block_size)
    deltas, selected = _delta_payload(
        task.base_weights,
        regions,
        parameter_budget=max(1, config.parameter_budget),
        delta_scale=config.delta_scale,
    )
    parameter_count = sum(region["rows"] * region["cols"] for region in selected)
    return MiniTransformerResult(
        name="hf_saint_delta_smoke",
        train_loss=0.0,
        test_loss=0.0,
        parameter_count=parameter_count,
        optimizer_state_values=0,
        elapsed_s=perf_counter() - start,
        metadata={
            "delta_payload": deltas,
            "model_source": task.model_source,
            "block_size": block_size,
            "selected_regions": selected,
            "available_regions": len(regions),
            "adapter": "huggingface_causal_lm",
            "marco": "fase_13_marco_1",
        },
    )


def inspect_model(config: RuntimeConfig) -> dict[str, Any]:
    task = make_task(config)
    matrices = {
        name: {
            "rows": len(matrix),
            "cols": len(matrix[0]) if matrix else 0,
            "parameters": sum(len(row) for row in matrix),
        }
        for name, matrix in task.base_weights.items()
    }
    return {
        "task": config.task,
        "adapter": "huggingface_causal_lm",
        "model_source": task.model_source,
        "matrices": matrices,
        "total_parameters": sum(item["parameters"] for item in matrices.values()),
    }


__all__ = ["HuggingFaceTask", "inspect_model", "make_task", "run_method"]
