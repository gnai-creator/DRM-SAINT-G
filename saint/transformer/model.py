"""Tiny dependency-free transformer model for phase 5 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt, tanh
from random import Random

from saint.reconstruction.matrix_ops import add, copy_matrix, matvec, shape, zeros


MatrixDict = dict[str, list[list[float]]]


@dataclass(frozen=True)
class MiniTransformerTask:
    base_weights: MatrixDict
    target_weights: MatrixDict
    train_sequences: list[list[int]]
    test_sequences: list[list[int]]
    d_model: int
    vocab_size: int


def _random_matrix(rows: int, cols: int, rng: Random, scale: float) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _repeated_delta(rows: int, cols: int, rng: Random, scale: float) -> list[list[float]]:
    prototypes = (
        ((0.04, -0.02), (0.01, 0.03)),
        ((-0.03, 0.02), (0.02, -0.01)),
        ((0.0, 0.0), (0.0, 0.0)),
    )
    delta = zeros(rows, cols)
    for row in range(0, rows, 2):
        for col in range(0, cols, 2):
            prototype = prototypes[rng.randrange(len(prototypes))]
            factor = rng.choice((0.5, 1.0, 1.5))
            for r_offset in range(min(2, rows - row)):
                for c_offset in range(min(2, cols - col)):
                    delta[row + r_offset][col + c_offset] = (
                        prototype[r_offset][c_offset] * factor * scale
                    )
    return delta


def _dense_delta(rows: int, cols: int, rng: Random, scale: float) -> list[list[float]]:
    return _random_matrix(rows, cols, rng, 0.035 * scale)


def _make_weights(vocab_size: int, d_model: int, rng: Random) -> MatrixDict:
    return {
        "embed": _random_matrix(vocab_size, d_model, rng, 0.15),
        "w_q": _random_matrix(d_model, d_model, rng, 0.12),
        "w_k": _random_matrix(d_model, d_model, rng, 0.12),
        "w_v": _random_matrix(d_model, d_model, rng, 0.12),
        "w_o": _random_matrix(d_model, d_model, rng, 0.12),
        "w_mlp1": _random_matrix(d_model, d_model, rng, 0.12),
        "w_mlp2": _random_matrix(d_model, d_model, rng, 0.12),
        "w_head": _random_matrix(vocab_size, d_model, rng, 0.12),
    }


def _add_delta(weights: MatrixDict, mode: str, rng: Random, delta_scale: float) -> MatrixDict:
    target = {}
    for name, matrix in weights.items():
        rows, cols = shape(matrix)
        delta = (
            _repeated_delta(rows, cols, rng, delta_scale)
            if mode == "repeated"
            else _dense_delta(rows, cols, rng, delta_scale)
        )
        target[name] = add(matrix, delta)
    return target


def _sequences(count: int, seq_len: int, vocab_size: int, rng: Random) -> list[list[int]]:
    return [[rng.randrange(vocab_size) for _ in range(seq_len)] for _ in range(count)]


def make_mini_transformer_task(
    *,
    vocab_size: int = 8,
    d_model: int = 4,
    seq_len: int = 4,
    train_samples: int = 16,
    test_samples: int = 8,
    seed: int = 31,
    delta_mode: str = "repeated",
    delta_scale: float = 3.0,
) -> MiniTransformerTask:
    rng = Random(seed)
    base = _make_weights(vocab_size, d_model, rng)
    target = _add_delta(base, delta_mode, rng, delta_scale)
    return MiniTransformerTask(
        base_weights=base,
        target_weights=target,
        train_sequences=_sequences(train_samples, seq_len, vocab_size, rng),
        test_sequences=_sequences(test_samples, seq_len, vocab_size, rng),
        d_model=d_model,
        vocab_size=vocab_size,
    )


def combine_weights(base: MatrixDict, deltas: MatrixDict | None = None) -> MatrixDict:
    if deltas is None:
        return {name: copy_matrix(matrix) for name, matrix in base.items()}
    return {
        name: add(matrix, deltas.get(name, zeros(*shape(matrix))))
        for name, matrix in base.items()
    }


def zero_deltas(weights: MatrixDict) -> MatrixDict:
    return {name: zeros(*shape(matrix)) for name, matrix in weights.items()}


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    shifted = [exp(value - peak) for value in values]
    total = sum(shifted) or 1.0
    return [value / total for value in shifted]


def forward_logits(weights: MatrixDict, sequence: list[int]) -> list[float]:
    embeddings = [weights["embed"][token] for token in sequence]
    q_last = matvec(weights["w_q"], embeddings[-1])
    keys = [matvec(weights["w_k"], embedding) for embedding in embeddings]
    values = [matvec(weights["w_v"], embedding) for embedding in embeddings]
    scale = sqrt(len(q_last)) or 1.0
    scores = [
        sum(query * key for query, key in zip(q_last, key_vector)) / scale
        for key_vector in keys
    ]
    attention = softmax(scores)
    context = [
        sum(weight * value[index] for weight, value in zip(attention, values))
        for index in range(len(q_last))
    ]
    projected = matvec(weights["w_o"], context)
    hidden = [tanh(value + residual) for value, residual in zip(projected, embeddings[-1])]
    mlp = [tanh(value) for value in matvec(weights["w_mlp1"], hidden)]
    hidden2 = [
        tanh(value + residual)
        for value, residual in zip(matvec(weights["w_mlp2"], mlp), hidden)
    ]
    return matvec(weights["w_head"], hidden2)


def distillation_loss(weights: MatrixDict, teacher: MatrixDict, sequences: list[list[int]]) -> float:
    total = 0.0
    for sequence in sequences:
        logits = forward_logits(weights, sequence)
        target = forward_logits(teacher, sequence)
        for value, expected in zip(logits, target):
            error = value - expected
            total += error * error
    return total / (len(sequences) * len(next(iter(teacher.values()))))
