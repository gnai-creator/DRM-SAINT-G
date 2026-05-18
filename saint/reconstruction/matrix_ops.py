"""Small pure-Python matrix helpers for dependency-free benchmarks."""

from __future__ import annotations

from math import sqrt

from saint.blocks.partition import Matrix


def shape(matrix: Matrix) -> tuple[int, int]:
    if not matrix:
        raise ValueError("matrix must contain at least one row")
    return len(matrix), len(matrix[0])


def zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def transpose(matrix: Matrix) -> list[list[float]]:
    rows, cols = shape(matrix)
    return [[float(matrix[row][col]) for row in range(rows)] for col in range(cols)]


def matmul(left: Matrix, right: Matrix) -> list[list[float]]:
    left_rows, left_cols = shape(left)
    right_rows, right_cols = shape(right)
    if left_cols != right_rows:
        raise ValueError("matrix shapes are not compatible for multiplication")

    result = zeros(left_rows, right_cols)
    for i in range(left_rows):
        for k in range(left_cols):
            left_value = float(left[i][k])
            if left_value == 0.0:
                continue
            for j in range(right_cols):
                result[i][j] += left_value * float(right[k][j])
    return result


def matvec(matrix: Matrix, vector: list[float]) -> list[float]:
    rows, cols = shape(matrix)
    if cols != len(vector):
        raise ValueError("matrix and vector shapes are not compatible")
    return [
        sum(float(matrix[row][col]) * vector[col] for col in range(cols))
        for row in range(rows)
    ]


def vector_norm(vector: list[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def normalize(vector: list[float]) -> list[float]:
    norm = vector_norm(vector)
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def outer(left: list[float], right: list[float]) -> list[list[float]]:
    return [[left_value * right_value for right_value in right] for left_value in left]


def subtract(left: Matrix, right: Matrix) -> list[list[float]]:
    rows, cols = shape(left)
    if shape(right) != (rows, cols):
        raise ValueError("matrix shapes differ")
    return [
        [float(left[row][col]) - float(right[row][col]) for col in range(cols)]
        for row in range(rows)
    ]


def add(left: Matrix, right: Matrix) -> list[list[float]]:
    rows, cols = shape(left)
    if shape(right) != (rows, cols):
        raise ValueError("matrix shapes differ")
    return [
        [float(left[row][col]) + float(right[row][col]) for col in range(cols)]
        for row in range(rows)
    ]


def scale(matrix: Matrix, value: float) -> list[list[float]]:
    return [[float(cell) * value for cell in row] for row in matrix]


def copy_matrix(matrix: Matrix) -> list[list[float]]:
    return [[float(cell) for cell in row] for row in matrix]
