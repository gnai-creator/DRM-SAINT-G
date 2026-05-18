"""PyTorch autograd path for local Hugging Face checkpoint experiments."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from saint.config import RuntimeConfig
from saint.reconstruction.matrix_ops import shape, zeros
from saint.transformer.training import MiniTransformerResult


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for hf_saint_autograd_smoke."
        ) from exc
    return torch


def _metadata(config: RuntimeConfig) -> dict[str, Any]:
    return dict(config.metadata or {})


def _target_delta(matrix: list[list[float]], scale: float) -> list[list[float]]:
    return [
        [
            scale * 1e-3 * (-1.0 if value < 0.0 else 1.0)
            for value in row
        ]
        for row in matrix
    ]


def _coords_by_magnitude(
    weights: dict[str, list[list[float]]],
) -> list[tuple[str, int, int]]:
    coords = [
        (name, row, col)
        for name, matrix in weights.items()
        for row in range(shape(matrix)[0])
        for col in range(shape(matrix)[1])
    ]
    return sorted(
        coords,
        key=lambda coord: abs(weights[coord[0]][coord[1]][coord[2]]),
        reverse=True,
    )


def _masked_tensors(torch, weights, coords, *, scale: float):
    trainable = set(coords)
    params = {}
    targets = {}
    masks = {}
    for name, matrix in weights.items():
        rows, cols = shape(matrix)
        target = _target_delta(matrix, scale)
        mask = zeros(rows, cols)
        for row in range(rows):
            for col in range(cols):
                if (name, row, col) in trainable:
                    mask[row][col] = 1.0
        params[name] = torch.zeros((rows, cols), dtype=torch.float32, requires_grad=True)
        targets[name] = torch.tensor(target, dtype=torch.float32)
        masks[name] = torch.tensor(mask, dtype=torch.float32)
    return params, targets, masks


def _loss(torch, params, targets, masks):
    total = torch.tensor(0.0, dtype=torch.float32)
    count = 0.0
    for name, param in params.items():
        error = (param - targets[name]) * masks[name]
        total = total + torch.sum(error * error)
        count += float(torch.sum(masks[name]).item())
    return total / max(count, 1.0)


def _payload(params, masks) -> dict[str, list[list[float]]]:
    deltas = {}
    for name, param in params.items():
        materialized = (param.detach() * masks[name]).cpu().tolist()
        deltas[name] = [[float(value) for value in row] for row in materialized]
    return deltas


def run_hf_autograd(config: RuntimeConfig) -> MiniTransformerResult:
    torch = _require_torch()
    from saint.adapters.huggingface import make_task

    start = perf_counter()
    metadata = _metadata(config)
    task = make_task(config)
    coords = _coords_by_magnitude(task.base_weights)
    trainable = coords[: max(1, min(config.parameter_budget, len(coords)))]
    params, targets, masks = _masked_tensors(
        torch,
        task.base_weights,
        trainable,
        scale=config.delta_scale,
    )
    optimizer = torch.optim.AdamW(list(params.values()), lr=float(metadata.get("learning_rate", 0.1)))
    initial_loss = float(_loss(torch, params, targets, masks).detach().cpu().item())
    for _ in range(max(1, int(config.steps))):
        optimizer.zero_grad()
        loss = _loss(torch, params, targets, masks)
        loss.backward()
        optimizer.step()
    final_loss = float(_loss(torch, params, targets, masks).detach().cpu().item())
    return MiniTransformerResult(
        name="hf_saint_autograd_smoke",
        train_loss=final_loss,
        test_loss=final_loss,
        parameter_count=len(trainable),
        optimizer_state_values=len(trainable) * 2,
        elapsed_s=perf_counter() - start,
        metadata={
            "delta_payload": _payload(params, masks),
            "adapter": "huggingface_causal_lm",
            "autograd": True,
            "initial_loss": initial_loss,
            "model_source": task.model_source,
            "selected_parameters": len(trainable),
            "marco": "fase_13_marco_2",
        },
    )


__all__ = ["run_hf_autograd"]
