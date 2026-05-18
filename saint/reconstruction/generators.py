"""Deterministic synthetic matrices for reconstruction benchmarks."""

from __future__ import annotations

import random

from .matrix_ops import matmul


def gaussian_matrix(rows: int, cols: int, *, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def low_rank_matrix(
    rows: int,
    cols: int,
    *,
    rank: int = 2,
    seed: int = 0,
) -> list[list[float]]:
    rng = random.Random(seed)
    left = [[rng.gauss(0.0, 1.0) for _ in range(rank)] for _ in range(rows)]
    right = [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rank)]
    return matmul(left, right)


def sparse_matrix(
    rows: int,
    cols: int,
    *,
    density: float = 0.1,
    seed: int = 0,
) -> list[list[float]]:
    rng = random.Random(seed)
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(rng.gauss(0.0, 1.0) if rng.random() < density else 0.0)
        matrix.append(row)
    return matrix


def repeated_block_matrix(
    rows: int,
    cols: int,
    *,
    block_size: int = 2,
    prototypes: int = 4,
    seed: int = 0,
) -> list[list[float]]:
    rng = random.Random(seed)
    proto_blocks = [
        [
            [rng.randint(-3, 3) for _ in range(block_size)]
            for _ in range(block_size)
        ]
        for _ in range(prototypes)
    ]
    matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for row_start in range(0, rows, block_size):
        for col_start in range(0, cols, block_size):
            block = proto_blocks[rng.randrange(prototypes)]
            for r_offset in range(block_size):
                for c_offset in range(block_size):
                    row = row_start + r_offset
                    col = col_start + c_offset
                    if row < rows and col < cols:
                        matrix[row][col] = float(block[r_offset][c_offset])
    return matrix
