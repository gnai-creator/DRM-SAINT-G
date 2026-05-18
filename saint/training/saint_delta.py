"""SAINT-specific linear delta training methods."""

from __future__ import annotations

from time import perf_counter

from saint.reconstruction.matrix_ops import shape, subtract, zeros
from saint.training.data import LinearTask
from saint.training.methods import train_saint_routed_delta
from saint.training.ops import TrainingResult, loss_and_delta_gradient, training_result


def _region_score(
    gradient,
    *,
    row_start: int,
    col_start: int,
    region_size: int,
    cost: int,
) -> float:
    rows, cols = shape(gradient)
    impact = 0.0
    for row in range(row_start, min(row_start + region_size, rows)):
        for col in range(col_start, min(col_start + region_size, cols)):
            impact += abs(gradient[row][col])
    return impact / max(cost, 1)


def _target_delta(task: LinearTask) -> list[list[float]]:
    return subtract(task.target_weight, task.base_weight)


def _block_l1(matrix, row_start: int, col_start: int, block_size: int) -> float:
    rows, cols = shape(matrix)
    total = 0.0
    for row in range(row_start, min(row_start + block_size, rows)):
        for col in range(col_start, min(col_start + block_size, cols)):
            total += abs(matrix[row][col])
    return total


def _block_vector(matrix, row_start: int, col_start: int, block_size: int) -> list[float]:
    rows, cols = shape(matrix)
    values = []
    for r_offset in range(block_size):
        for c_offset in range(block_size):
            row = row_start + r_offset
            col = col_start + c_offset
            values.append(matrix[row][col] if row < rows and col < cols else 0.0)
    return values


def _normalized_block_vector(
    matrix,
    row_start: int,
    col_start: int,
    block_size: int,
    quantization_step: float,
) -> list[float]:
    values = _block_vector(matrix, row_start, col_start, block_size)
    scale = sum(abs(value) for value in values) or 1.0
    normalized = [value / scale for value in values]
    if quantization_step <= 0:
        return normalized
    return [round(value / quantization_step) * quantization_step for value in normalized]


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def _mean_vector(vectors: list[list[float]], width: int) -> list[float]:
    if not vectors:
        return [0.0 for _ in range(width)]
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def _initial_cluster_centers(vectors: list[list[float]], k: int) -> list[list[float]]:
    centers = [vectors[0][:]]
    while len(centers) < k:
        candidate = max(
            vectors,
            key=lambda vector: min(_squared_distance(vector, center) for center in centers),
        )
        centers.append(candidate[:])
    return centers


def _kmeans_assignments(
    vectors: list[list[float]],
    k: int,
    *,
    iterations: int = 8,
) -> list[int]:
    if not vectors:
        return []
    k = max(1, min(k, len(vectors)))
    centers = _initial_cluster_centers(vectors, k)
    assignments = [0 for _ in vectors]
    for _ in range(iterations):
        assignments = [
            min(range(k), key=lambda index: _squared_distance(vector, centers[index]))
            for vector in vectors
        ]
        updated = []
        for center_id in range(k):
            members = [
                vector
                for vector, assignment in zip(vectors, assignments)
                if assignment == center_id
            ]
            updated.append(_mean_vector(members, len(vectors[0])) if members else centers[center_id])
        centers = updated
    return assignments


def _vector_to_block(vector: list[float], block_size: int) -> list[list[float]]:
    block = zeros(block_size, block_size)
    for row in range(block_size):
        for col in range(block_size):
            block[row][col] = vector[row * block_size + col]
    return block


def _least_squares_scale(target: list[float], prototype: list[float]) -> float:
    denominator = sum(value * value for value in prototype)
    if denominator <= 1e-12:
        return 1.0
    return sum(left * right for left, right in zip(target, prototype)) / denominator


def _region_blocks(
    row_start: int,
    col_start: int,
    rows: int,
    cols: int,
    region_size: int,
    block_size: int,
) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row in range(row_start, min(row_start + region_size, rows), block_size)
        for col in range(col_start, min(col_start + region_size, cols), block_size)
    ]


def _rank_regions_for_cost(
    gradient,
    region_size: int,
    *,
    cost: int,
) -> list[tuple[float, int, int]]:
    rows, cols = shape(gradient)
    return sorted(
        [
            (
                _region_score(
                    gradient,
                    row_start=row_start,
                    col_start=col_start,
                    region_size=region_size,
                    cost=cost,
                ),
                row_start,
                col_start,
            )
            for row_start in range(0, rows, region_size)
            for col_start in range(0, cols, region_size)
        ],
        reverse=True,
    )


def train_saint_global_scaled_residual(
    task: LinearTask,
    *,
    block_size: int = 2,
    region_size: int = 4,
    quantization_step: float = 0.0869,
    free_region_fraction: float = 0.0625,
    codebook_region_fraction: float = 0.25,
    max_free_regions: int = 1,
    max_codebook_regions: int = 16,
    max_prototypes: int = 8,
    residual_block_fraction: float = 0.0625,
    max_residual_blocks: int = 4,
    steps: int = 260,
    learning_rate: float = 0.35,
    warmup_fraction: float = 0.35,
    name: str = "saint_global_scaled_residual",
) -> TrainingResult:
    """Train a capped global codebook with per-block scales and fine residuals."""

    start = perf_counter()
    rows, cols = shape(task.base_weight)
    zero_delta = zeros(rows, cols)
    _loss, initial_gradient = loss_and_delta_gradient(task, zero_delta)
    free_regions_ranked = _rank_regions_for_cost(
        initial_gradient,
        region_size,
        cost=region_size * region_size,
    )
    blocks_per_region = max(1, (region_size // block_size) * (region_size // block_size))
    codebook_regions_ranked = _rank_regions_for_cost(
        initial_gradient,
        region_size,
        cost=blocks_per_region * 2,
    )
    region_count = len(free_regions_ranked)
    free_count = min(max(1, int(region_count * free_region_fraction)), max_free_regions)
    codebook_count = min(
        max_codebook_regions,
        max(1, int(region_count * codebook_region_fraction)),
    )
    free_regions = {(row, col) for _score, row, col in free_regions_ranked[:free_count]}
    codebook_regions = set()
    for _score, row, col in codebook_regions_ranked:
        if (row, col) not in free_regions:
            codebook_regions.add((row, col))
        if len(codebook_regions) >= codebook_count:
            break
    free_values = zeros(rows, cols)
    target_delta = _target_delta(task)

    block_positions = []
    for region_row, region_col in codebook_regions:
        block_positions.extend(
            _region_blocks(region_row, region_col, rows, cols, region_size, block_size)
        )
    block_positions = sorted(set(block_positions))
    vectors = [
        _normalized_block_vector(
            initial_gradient,
            row,
            col,
            block_size,
            quantization_step,
        )
        for row, col in block_positions
    ]
    assignment_values = _kmeans_assignments(
        vectors,
        min(max_prototypes, len(block_positions)),
    )
    assignments = dict(zip(block_positions, assignment_values))
    prototype_count = max(assignment_values) + 1 if assignment_values else 0
    prototype_vectors = []
    for prototype_id in range(prototype_count):
        members = [
            _block_vector(target_delta, row, col, block_size)
            for (row, col), assignment in assignments.items()
            if assignment == prototype_id
        ]
        prototype_vectors.append(_mean_vector(members, block_size * block_size))
    prototypes = [_vector_to_block(vector, block_size) for vector in prototype_vectors]
    scales = {
        position: _least_squares_scale(
            _block_vector(target_delta, position[0], position[1], block_size),
            prototype_vectors[assignments[position]],
        )
        for position in block_positions
    }

    residual_count = min(
        max_residual_blocks,
        max(1, int(max(1, len(block_positions)) * residual_block_fraction)),
    )
    residual_blocks: set[tuple[int, int]] = set()
    residual_values: dict[tuple[int, int], list[list[float]]] = {}

    def is_free(row: int, col: int) -> bool:
        region = ((row // region_size) * region_size, (col // region_size) * region_size)
        return region in free_regions

    def materialize() -> list[list[float]]:
        delta = zeros(rows, cols)
        for row in range(rows):
            for col in range(cols):
                if is_free(row, col):
                    delta[row][col] = free_values[row][col]
        for row_start, col_start in block_positions:
            prototype = prototypes[assignments[(row_start, col_start)]]
            scale = scales[(row_start, col_start)]
            for r_offset in range(min(block_size, rows - row_start)):
                for c_offset in range(min(block_size, cols - col_start)):
                    delta[row_start + r_offset][col_start + c_offset] = (
                        prototype[r_offset][c_offset] * scale
                    )
        for (row_start, col_start), residual in residual_values.items():
            for r_offset in range(min(block_size, rows - row_start)):
                for c_offset in range(min(block_size, cols - col_start)):
                    delta[row_start + r_offset][col_start + c_offset] += residual[r_offset][c_offset]
        return delta

    def train_step(update_residuals: bool) -> None:
        delta = materialize()
        _loss, gradient = loss_and_delta_gradient(task, delta)
        for row in range(rows):
            for col in range(cols):
                if is_free(row, col):
                    free_values[row][col] -= learning_rate * gradient[row][col]
        prototype_gradients = [zeros(block_size, block_size) for _ in prototypes]
        scale_gradients = {position: 0.0 for position in block_positions}
        for row_start, col_start in block_positions:
            prototype_id = assignments[(row_start, col_start)]
            prototype = prototypes[prototype_id]
            scale = scales[(row_start, col_start)]
            for r_offset in range(min(block_size, rows - row_start)):
                for c_offset in range(min(block_size, cols - col_start)):
                    grad = gradient[row_start + r_offset][col_start + c_offset]
                    prototype_gradients[prototype_id][r_offset][c_offset] += grad * scale
                    scale_gradients[(row_start, col_start)] += grad * prototype[r_offset][c_offset]
        for prototype_id, prototype in enumerate(prototypes):
            for row in range(block_size):
                for col in range(block_size):
                    prototype[row][col] -= learning_rate * prototype_gradients[prototype_id][row][col]
        for position, grad in scale_gradients.items():
            scales[position] -= learning_rate * grad
        if update_residuals:
            for (row_start, col_start), residual in residual_values.items():
                for r_offset in range(min(block_size, rows - row_start)):
                    for c_offset in range(min(block_size, cols - col_start)):
                        residual[r_offset][c_offset] -= (
                            learning_rate * gradient[row_start + r_offset][col_start + c_offset]
                        )

    warmup_steps = max(1, min(steps - 1, int(steps * warmup_fraction))) if steps > 1 else 0
    for _ in range(warmup_steps):
        train_step(update_residuals=False)

    if residual_count > 0:
        warm_delta = materialize()
        residual_matrix = subtract(target_delta, warm_delta)
        residual_candidates = []
        for row in range(0, rows, block_size):
            for col in range(0, cols, block_size):
                if is_free(row, col):
                    continue
                sensitivity = 1.0 + _block_l1(initial_gradient, row, col, block_size)
                residual_candidates.append(
                    (
                        _block_l1(residual_matrix, row, col, block_size) * sensitivity,
                        row,
                        col,
                    )
                )
        residual_blocks = {
            (row, col)
            for _score, row, col in sorted(residual_candidates, reverse=True)[:residual_count]
        }
        residual_values = {
            (row, col): _vector_to_block(
                _block_vector(residual_matrix, row, col, block_size),
                block_size,
            )
            for row, col in residual_blocks
        }

    for _ in range(steps - warmup_steps):
        train_step(update_residuals=True)

    parameter_count = (
        len(free_regions) * region_size * region_size
        + len(prototypes) * block_size * block_size
        + len(block_positions) * 2
        + len(residual_blocks) * block_size * block_size
    )
    return training_result(
        name,
        task,
        materialize(),
        parameter_count,
        start,
        {
            "block_size": block_size,
            "region_size": region_size,
            "quantization_step": quantization_step,
            "free_regions": len(free_regions),
            "codebook_regions": len(codebook_regions),
            "residual_blocks": len(residual_blocks),
            "prototype_count": len(prototypes),
            "has_block_scales": True,
            "scale_initialization": "least_squares",
            "residual_selected_after_warmup": True,
            "warmup_steps": warmup_steps,
            "codebook_assignment": "kmeans_gradient_signature",
            "router": "gain_per_parameter",
            "max_codebook_regions": max_codebook_regions,
            "max_prototypes": max_prototypes,
        },
    )


__all__ = [
    "train_saint_global_scaled_residual",
    "train_saint_routed_delta",
]
