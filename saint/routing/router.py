"""Heuristic block router based on reconstruction error by region."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from saint.blocks import MatrixBlock, group_blocks_by_signature, partition_matrix
from saint.blocks.metrics import reconstruction_error
from saint.blocks.partition import Matrix, reconstruct_matrix
from saint.reconstruction.baselines import ReconstructionResult, block_codebook_reconstruction
from saint.reconstruction.matrix_ops import shape, zeros


@dataclass(frozen=True)
class RoutedRegion:
    row_start: int
    col_start: int
    height: int
    width: int
    method: str
    relative_l1_error: float
    parameter_count: int


@dataclass(frozen=True)
class RoutingPlan:
    regions: tuple[RoutedRegion, ...]
    total_parameter_count: int
    metadata: dict


def _slice_region(
    matrix: Matrix,
    row_start: int,
    col_start: int,
    height: int,
    width: int,
) -> list[list[float]]:
    rows, cols = shape(matrix)
    row_end = min(row_start + height, rows)
    col_end = min(col_start + width, cols)
    return [
        [float(matrix[row][col]) for col in range(col_start, col_end)]
        for row in range(row_start, row_end)
    ]


def _write_region(
    target: list[list[float]],
    row_start: int,
    col_start: int,
    values: Matrix,
) -> None:
    for r_offset, row in enumerate(values):
        for c_offset, value in enumerate(row):
            target[row_start + r_offset][col_start + c_offset] = float(value)


def _best_region_candidate(
    region: Matrix,
    *,
    candidate_block_sizes: tuple[int, ...],
    error_threshold: float,
    quantization_step: float,
) -> tuple[str, list[list[float]], float, int]:
    candidates = []
    for block_size in candidate_block_sizes:
        result = block_codebook_reconstruction(
            region,
            block_size=block_size,
            signature_mode="quantized",
            quantization_step=quantization_step,
        )
        error = reconstruction_error(region, result.reconstructed).relative_l1_error
        candidates.append((f"codebook_{block_size}", result.reconstructed, error, result.parameter_count))

    acceptable = [
        candidate for candidate in candidates if candidate[2] <= error_threshold
    ]
    if acceptable:
        return min(acceptable, key=lambda candidate: (candidate[3], candidate[2]))

    rows, cols = shape(region)
    return ("free_delta", [[float(value) for value in row] for row in region], 0.0, rows * cols)


def route_matrix_regions(
    matrix: Matrix,
    *,
    region_size: int = 8,
    candidate_block_sizes: tuple[int, ...] = (4, 2),
    error_threshold: float = 0.1,
    quantization_step: float = 0.05,
) -> tuple[RoutingPlan, list[list[float]]]:
    """Route each region to the cheapest candidate below an error threshold."""

    rows, cols = shape(matrix)
    reconstructed = zeros(rows, cols)
    regions: list[RoutedRegion] = []

    for row_start in range(0, rows, region_size):
        for col_start in range(0, cols, region_size):
            region = _slice_region(matrix, row_start, col_start, region_size, region_size)
            height, width = shape(region)
            method, region_recon, error, params = _best_region_candidate(
                region,
                candidate_block_sizes=candidate_block_sizes,
                error_threshold=error_threshold,
                quantization_step=quantization_step,
            )
            _write_region(reconstructed, row_start, col_start, region_recon)
            regions.append(
                RoutedRegion(
                    row_start=row_start,
                    col_start=col_start,
                    height=height,
                    width=width,
                    method=method,
                    relative_l1_error=error,
                    parameter_count=params,
                )
            )

    plan = RoutingPlan(
        regions=tuple(regions),
        total_parameter_count=sum(region.parameter_count for region in regions),
        metadata={
            "region_size": region_size,
            "candidate_block_sizes": candidate_block_sizes,
            "error_threshold": error_threshold,
            "quantization_step": quantization_step,
        },
    )
    return plan, reconstructed


def routed_codebook_reconstruction(
    matrix: Matrix,
    *,
    region_size: int = 8,
    candidate_block_sizes: tuple[int, ...] = (4, 2),
    error_threshold: float = 0.1,
    quantization_step: float = 0.05,
) -> ReconstructionResult:
    """Reconstruction baseline using the heuristic region router."""

    start = perf_counter()
    plan, reconstructed = route_matrix_regions(
        matrix,
        region_size=region_size,
        candidate_block_sizes=candidate_block_sizes,
        error_threshold=error_threshold,
        quantization_step=quantization_step,
    )
    method_counts: dict[str, int] = {}
    for region in plan.regions:
        method_counts[region.method] = method_counts.get(region.method, 0) + 1
    return ReconstructionResult(
        name="routed_codebook",
        reconstructed=reconstructed,
        parameter_count=plan.total_parameter_count,
        elapsed_s=perf_counter() - start,
        metadata={
            **plan.metadata,
            "region_count": len(plan.regions),
            "method_counts": method_counts,
        },
    )
