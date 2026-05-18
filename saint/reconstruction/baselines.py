"""Reconstruction baselines for phase-2 benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from saint.blocks import (
    MatrixBlock,
    analyze_block_reuse,
    group_blocks_by_signature,
    partition_matrix,
    reconstruct_matrix,
    reconstruction_error,
)
from saint.blocks.partition import Matrix

from .matrix_ops import (
    add,
    copy_matrix,
    matvec,
    normalize,
    outer,
    scale,
    shape,
    subtract,
    transpose,
    zeros,
)


@dataclass(frozen=True)
class ReconstructionResult:
    name: str
    reconstructed: list[list[float]]
    parameter_count: int
    elapsed_s: float
    metadata: dict = field(default_factory=dict)


def original_reconstruction(matrix: Matrix) -> ReconstructionResult:
    start = perf_counter()
    rows, cols = shape(matrix)
    return ReconstructionResult(
        name="original",
        reconstructed=copy_matrix(matrix),
        parameter_count=rows * cols,
        elapsed_s=perf_counter() - start,
    )


def uniform_quantization_reconstruction(
    matrix: Matrix,
    *,
    step: float = 0.1,
) -> ReconstructionResult:
    start = perf_counter()
    if step <= 0:
        raise ValueError("quantization step must be positive")

    reconstructed = [
        [round(float(value) / step) * step for value in row]
        for row in matrix
    ]
    unique_values = {value for row in reconstructed for value in row}
    rows, cols = shape(matrix)
    return ReconstructionResult(
        name=f"quantization_step_{step:g}",
        reconstructed=reconstructed,
        parameter_count=len(unique_values) + rows * cols,
        elapsed_s=perf_counter() - start,
        metadata={"unique_values": len(unique_values), "step": step},
    )


def block_codebook_reconstruction(
    matrix: Matrix,
    *,
    block_size: int | tuple[int, int],
    signature_mode: str = "quantized",
    quantization_step: float = 0.1,
) -> ReconstructionResult:
    start = perf_counter()
    rows, cols = shape(matrix)
    blocks = partition_matrix(matrix, block_size=block_size)
    groups = group_blocks_by_signature(
        blocks,
        mode=signature_mode,
        quantization_step=quantization_step,
    )
    prototype_by_position: dict[tuple[int, int], MatrixBlock] = {}
    prototype_values = 0
    for group in groups.values():
        prototype = group[0]
        prototype_values += prototype.height * prototype.width
        for block in group:
            prototype_by_position[(block.row, block.col)] = prototype

    reconstructed_blocks = [
        MatrixBlock(
            row=block.row,
            col=block.col,
            row_start=block.row_start,
            col_start=block.col_start,
            height=block.height,
            width=block.width,
            values=prototype_by_position[(block.row, block.col)].values,
        )
        for block in blocks
    ]
    reconstructed = reconstruct_matrix(reconstructed_blocks, original_shape=(rows, cols))
    return ReconstructionResult(
        name=f"block_codebook_{block_size}",
        reconstructed=[[float(value) for value in row] for row in reconstructed],
        parameter_count=prototype_values + len(blocks),
        elapsed_s=perf_counter() - start,
        metadata={
            "block_size": block_size,
            "prototype_count": len(groups),
            "block_count": len(blocks),
        },
    )


def multi_scale_codebook_reconstruction(
    matrix: Matrix,
    *,
    block_sizes: tuple[int, ...] = (16, 8, 4, 2),
    signature_mode: str = "quantized",
    quantization_step: float = 0.1,
) -> ReconstructionResult:
    """Select the best block-codebook result by error, then parameters.

    This is the first phase-2 multi-scale baseline. It is intentionally simple:
    run several fixed-size codebooks and select the strongest reconstruction.
    Later phases can replace this with a true hierarchical router.
    """

    start = perf_counter()
    candidates = [
        block_codebook_reconstruction(
            matrix,
            block_size=size,
            signature_mode=signature_mode,
            quantization_step=quantization_step,
        )
        for size in block_sizes
    ]
    best = min(
        candidates,
        key=lambda result: (
            reconstruction_error(matrix, result.reconstructed).relative_l1_error,
            result.parameter_count,
        ),
    )
    return ReconstructionResult(
        name="multi_scale_codebook",
        reconstructed=best.reconstructed,
        parameter_count=best.parameter_count,
        elapsed_s=perf_counter() - start,
        metadata={
            "selected": best.name,
            "candidates": [
                {
                    "name": candidate.name,
                    "parameter_count": candidate.parameter_count,
                    "relative_l1_error": reconstruction_error(
                        matrix,
                        candidate.reconstructed,
                    ).relative_l1_error,
                }
                for candidate in candidates
            ],
        },
    )


def hierarchical_codebook_reconstruction(
    matrix: Matrix,
    *,
    block_sizes: tuple[int, ...] = (16, 8, 4, 2),
    signature_mode: str = "quantized",
    quantization_step: float = 0.1,
) -> ReconstructionResult:
    """Hierarchical multi-scale block-codebook reconstruction.

    The algorithm starts with large blocks. A block is represented at the
    current scale when its signature is reused elsewhere, or when it reached
    the smallest scale. Otherwise, the region is split into the next smaller
    block size. This is the first real multi-scale baseline for SAINT.
    """

    start = perf_counter()
    rows, cols = shape(matrix)
    sizes = tuple(sorted(set(block_sizes), reverse=True))
    if not sizes:
        raise ValueError("block_sizes must not be empty")

    blocks_by_size: dict[int, list[MatrixBlock]] = {
        size: partition_matrix(matrix, block_size=size)
        for size in sizes
    }
    groups_by_size = {
        size: group_blocks_by_signature(
            blocks,
            mode=signature_mode,
            quantization_step=quantization_step,
        )
        for size, blocks in blocks_by_size.items()
    }
    signature_by_position: dict[tuple[int, int, int], tuple] = {}
    block_by_position: dict[tuple[int, int, int], MatrixBlock] = {}
    group_size_by_signature: dict[tuple[int, tuple], int] = {}
    prototype_by_signature: dict[tuple[int, tuple], MatrixBlock] = {}

    for size, blocks in blocks_by_size.items():
        for signature, group in groups_by_size[size].items():
            group_size_by_signature[(size, signature)] = len(group)
            prototype_by_signature[(size, signature)] = group[0]
        for block in blocks:
            signature = next(
                signature
                for signature, group in groups_by_size[size].items()
                if block in group
            )
            signature_by_position[(size, block.row_start, block.col_start)] = signature
            block_by_position[(size, block.row_start, block.col_start)] = block

    reconstructed = zeros(rows, cols)
    used_prototypes: set[tuple[int, tuple]] = set()
    leaf_count = 0

    def write_block(block: MatrixBlock, prototype: MatrixBlock) -> None:
        for r_offset, row_values in enumerate(prototype.values):
            target_row = block.row_start + r_offset
            if target_row >= rows:
                continue
            for c_offset, value in enumerate(row_values):
                target_col = block.col_start + c_offset
                if target_col >= cols:
                    continue
                reconstructed[target_row][target_col] = float(value)

    def visit(block: MatrixBlock, size_index: int) -> None:
        nonlocal leaf_count
        size = sizes[size_index]
        signature = signature_by_position[(size, block.row_start, block.col_start)]
        group_size = group_size_by_signature[(size, signature)]
        is_smallest = size_index == len(sizes) - 1

        if group_size > 1 or is_smallest:
            used_prototypes.add((size, signature))
            write_block(block, prototype_by_signature[(size, signature)])
            leaf_count += 1
            return

        next_size = sizes[size_index + 1]
        row_end = min(block.row_start + block.height, rows)
        col_end = min(block.col_start + block.width, cols)
        for row_start in range(block.row_start, row_end, next_size):
            for col_start in range(block.col_start, col_end, next_size):
                child = block_by_position[(next_size, row_start, col_start)]
                visit(child, size_index + 1)

    largest = sizes[0]
    for root in blocks_by_size[largest]:
        visit(root, 0)

    prototype_values = 0
    for size, signature in used_prototypes:
        prototype = prototype_by_signature[(size, signature)]
        prototype_values += prototype.height * prototype.width

    return ReconstructionResult(
        name="hierarchical_codebook",
        reconstructed=reconstructed,
        parameter_count=prototype_values + leaf_count,
        elapsed_s=perf_counter() - start,
        metadata={
            "block_sizes": sizes,
            "leaf_count": leaf_count,
            "prototype_count": len(used_prototypes),
        },
    )


def low_rank_reconstruction(
    matrix: Matrix,
    *,
    rank: int = 1,
    iterations: int = 20,
) -> ReconstructionResult:
    """Approximate a matrix with a small number of rank-1 components.

    This dependency-free baseline is not a full SVD implementation. It is a
    power-iteration/deflation approximation used as an early low-rank baseline.
    """

    start = perf_counter()
    rows, cols = shape(matrix)
    residual = copy_matrix(matrix)
    approximation = zeros(rows, cols)

    for component in range(rank):
        vector = normalize([1.0 + component for _ in range(cols)])
        residual_t = transpose(residual)
        for _ in range(iterations):
            left = normalize(matvec(residual, vector))
            vector = normalize(matvec(residual_t, left))

        left_raw = matvec(residual, vector)
        sigma = sum(left_raw[i] * left[i] for i in range(len(left)))
        component_matrix = scale(outer(left, vector), sigma)
        approximation = add(approximation, component_matrix)
        residual = subtract(residual, component_matrix)

    return ReconstructionResult(
        name=f"low_rank_{rank}",
        reconstructed=approximation,
        parameter_count=rank * (rows + cols + 1),
        elapsed_s=perf_counter() - start,
        metadata={"rank": rank, "iterations": iterations},
    )
