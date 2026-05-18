"""LoRA baseline for the phase-5 mini-transformer."""

from __future__ import annotations

from random import Random
from time import perf_counter

from saint.reconstruction.matrix_ops import shape, zeros
from saint.transformer.model import MatrixDict, MiniTransformerTask
from saint.transformer.training import MiniTransformerResult, _loss, _result


def _lora_delta(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    rows, rank = shape(left)
    _rank, cols = shape(right)
    delta = zeros(rows, cols)
    for row in range(rows):
        for hidden in range(rank):
            left_value = left[row][hidden]
            for col in range(cols):
                delta[row][col] += left_value * right[hidden][col]
    return delta


def _materialize(base: MatrixDict, factors: dict) -> MatrixDict:
    deltas = {name: zeros(*shape(matrix)) for name, matrix in base.items()}
    for name, (left, right) in factors.items():
        deltas[name] = _lora_delta(left, right)
    return deltas


def train_mini_lora_delta(
    task: MiniTransformerTask,
    *,
    target_matrices: tuple[str, ...] = ("w_q", "w_v", "w_o", "w_head"),
    rank: int = 1,
    steps: int = 8,
    learning_rate: float = 0.25,
    epsilon: float = 1e-4,
    seed: int = 23,
    name: str = "mini_lora_rank_1",
) -> MiniTransformerResult:
    """Train LoRA factors on selected transformer matrices with global loss."""

    start = perf_counter()
    rng = Random(seed)
    factors = {}
    refs = []
    for matrix_name in target_matrices:
        rows, cols = shape(task.base_weights[matrix_name])
        left = zeros(rows, rank)
        right = [[rng.uniform(-0.02, 0.02) for _col in range(cols)] for _hidden in range(rank)]
        factors[matrix_name] = (left, right)
        for row in range(rows):
            for hidden in range(rank):
                refs.append(("left", matrix_name, row, hidden))
        for hidden in range(rank):
            for col in range(cols):
                refs.append(("right", matrix_name, hidden, col))

    def get_value(ref):
        side, matrix_name, row, col = ref
        left, right = factors[matrix_name]
        return left[row][col] if side == "left" else right[row][col]

    def set_value(ref, value: float) -> None:
        side, matrix_name, row, col = ref
        left, right = factors[matrix_name]
        if side == "left":
            left[row][col] = value
        else:
            right[row][col] = value

    def ref_gradient(ref) -> float:
        original = get_value(ref)
        set_value(ref, original + epsilon)
        plus = _loss(task, _materialize(task.base_weights, factors), train=True)
        set_value(ref, original - epsilon)
        minus = _loss(task, _materialize(task.base_weights, factors), train=True)
        set_value(ref, original)
        return (plus - minus) / (2.0 * epsilon)

    for _ in range(steps):
        gradients = {ref: ref_gradient(ref) for ref in refs}
        for ref, gradient in gradients.items():
            set_value(ref, get_value(ref) - learning_rate * gradient)

    return _result(
        name,
        task,
        _materialize(task.base_weights, factors),
        len(refs),
        start,
        {
            "rank": rank,
            "target_matrices": target_matrices,
            "global_loss": True,
        },
    )


__all__ = ["train_mini_lora_delta"]
