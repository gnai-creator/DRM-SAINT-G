"""SAINT-style mini-transformer delta adapter."""

from __future__ import annotations

from saint.reconstruction.matrix_ops import shape, zeros
from saint.transformer.model import MatrixDict, MiniTransformerTask
from saint.transformer.training import (
    MiniTransformerResult,
    _block_coords,
    _blocks,
    _finite_gradient,
    _initial_gradients,
    _loss,
    _result,
    _coords,
)


def _block_values(gradients: dict, coords: list[tuple[str, int, int]]) -> list[float]:
    scale = sum(abs(gradients[coord]) for coord in coords) or 1.0
    return [round(gradients[coord] / scale, 2) for coord in coords]


def _bucket(values: list[float], count: int) -> int:
    raw = sum((index + 1) * int(value * 100) for index, value in enumerate(values))
    return abs(raw) % max(count, 1)


def _zero_like(weights: MatrixDict) -> MatrixDict:
    return {name: zeros(*shape(matrix)) for name, matrix in weights.items()}


def train_mini_saint_delta(
    task: MiniTransformerTask,
    *,
    parameter_budget: int = 48,
    block_size: int = 2,
    max_prototypes: int = 6,
    residual_fraction: float = 0.25,
    steps: int = 8,
    learning_rate: float = 0.25,
    epsilon: float = 1e-4,
    name: str = "mini_saint_dynamic_delta",
    share_scope: str = "global",
) -> MiniTransformerResult:
    """Train a tiny SAINT codebook adapter against global transformer loss."""

    from time import perf_counter

    start = perf_counter()
    all_coords = _coords(task.base_weights)
    initial = _initial_gradients(task, all_coords, epsilon=epsilon)
    ranked_blocks = []
    for matrix_name, row, col in _blocks(task.base_weights, block_size):
        rows, cols = shape(task.base_weights[matrix_name])
        coords = _block_coords(matrix_name, row, col, rows, cols, block_size)
        ranked_blocks.append((sum(abs(initial[coord]) for coord in coords), matrix_name, row, col, coords))
    prototype_cost = max_prototypes * block_size * block_size
    block_budget = max(1, (parameter_budget - prototype_cost) // 3)
    selected = sorted(ranked_blocks, reverse=True)[:block_budget]
    assignments = {}
    for _score, matrix_name, row, col, coords in selected:
        bucket = _bucket(_block_values(initial, coords), max_prototypes)
        prototype_key = "global" if share_scope == "global" else matrix_name
        assignments[(matrix_name, row, col)] = (prototype_key, bucket)
    prototype_groups = sorted({key for key, _bucket_id in assignments.values()}) or ["global"]
    prototypes = {
        key: [zeros(block_size, block_size) for _ in range(max_prototypes)]
        for key in prototype_groups
    }
    scales = {position: 1.0 for position in assignments}
    biases = {position: 0.0 for position in assignments}
    residual_count = max(1, int(len(selected) * residual_fraction))
    residual_blocks = {
        (matrix_name, row, col)
        for _score, matrix_name, row, col, _coords_block in selected[:residual_count]
    }
    residuals = {position: zeros(block_size, block_size) for position in residual_blocks}

    def materialize() -> MatrixDict:
        deltas = _zero_like(task.base_weights)
        for matrix_name, row, col in assignments:
            prototype_key, prototype_id = assignments[(matrix_name, row, col)]
            prototype = prototypes[prototype_key][prototype_id]
            scale = scales[(matrix_name, row, col)]
            bias = biases[(matrix_name, row, col)]
            rows, cols = shape(deltas[matrix_name])
            for r_offset in range(min(block_size, rows - row)):
                for c_offset in range(min(block_size, cols - col)):
                    deltas[matrix_name][row + r_offset][col + c_offset] = (
                        prototype[r_offset][c_offset] * scale + bias
                    )
        for matrix_name, row, col in residuals:
            rows, cols = shape(deltas[matrix_name])
            residual = residuals[(matrix_name, row, col)]
            for r_offset in range(min(block_size, rows - row)):
                for c_offset in range(min(block_size, cols - col)):
                    deltas[matrix_name][row + r_offset][col + c_offset] += residual[r_offset][c_offset]
        return deltas

    def param_refs():
        refs = []
        for prototype_key, group in prototypes.items():
            for prototype_id in range(len(group)):
                for row in range(block_size):
                    for col in range(block_size):
                        refs.append(("prototype", prototype_key, prototype_id, row, col))
        for position in assignments:
            refs.append(("scale", position))
            refs.append(("bias", position))
        for position in residuals:
            for row in range(block_size):
                for col in range(block_size):
                    refs.append(("residual", position, row, col))
        return refs

    def get_value(ref):
        if ref[0] == "prototype":
            return prototypes[ref[1]][ref[2]][ref[3]][ref[4]]
        if ref[0] == "scale":
            return scales[ref[1]]
        if ref[0] == "bias":
            return biases[ref[1]]
        return residuals[ref[1]][ref[2]][ref[3]]

    def set_value(ref, value: float) -> None:
        if ref[0] == "prototype":
            prototypes[ref[1]][ref[2]][ref[3]][ref[4]] = value
        elif ref[0] == "scale":
            scales[ref[1]] = value
        elif ref[0] == "bias":
            biases[ref[1]] = value
        else:
            residuals[ref[1]][ref[2]][ref[3]] = value

    def ref_gradient(ref) -> float:
        original = get_value(ref)
        set_value(ref, original + epsilon)
        plus = _loss(task, materialize(), train=True)
        set_value(ref, original - epsilon)
        minus = _loss(task, materialize(), train=True)
        set_value(ref, original)
        return (plus - minus) / (2.0 * epsilon)

    refs = param_refs()
    for _ in range(steps):
        gradients = {ref: ref_gradient(ref) for ref in refs}
        for ref, gradient in gradients.items():
            set_value(ref, get_value(ref) - learning_rate * gradient)

    return _result(
        name,
        task,
        materialize(),
        len(refs),
        start,
        {
            "parameter_budget": parameter_budget,
            "block_size": block_size,
            "prototype_count": sum(len(group) for group in prototypes.values()),
            "prototype_groups": len(prototypes),
            "share_scope": share_scope,
            "assigned_blocks": len(assignments),
            "residual_blocks": len(residuals),
            "global_loss": True,
        },
    )


def train_mini_saint_per_matrix_delta(
    task: MiniTransformerTask,
    **kwargs,
) -> MiniTransformerResult:
    kwargs.setdefault("name", "mini_saint_per_matrix_delta")
    kwargs["share_scope"] = "matrix"
    return train_mini_saint_delta(task, **kwargs)


__all__ = ["train_mini_saint_delta", "train_mini_saint_per_matrix_delta"]
