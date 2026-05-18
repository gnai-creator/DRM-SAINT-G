"""Benchmark runner for matrix reconstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from saint.blocks import analyze_block_reuse, reconstruction_error
from saint.blocks.partition import Matrix

from .baselines import ReconstructionResult
from .matrix_ops import shape


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    matrix: Matrix


@dataclass(frozen=True)
class BenchmarkResult:
    case_name: str
    method_name: str
    l1_error: float
    l2_error: float
    relative_l1_error: float
    max_abs_error: float
    parameter_count: int
    compression_ratio: float
    elapsed_s: float
    metadata: dict


Baseline = Callable[[Matrix], ReconstructionResult]


def run_reconstruction_benchmark(
    cases: list[BenchmarkCase],
    baselines: list[Baseline],
    *,
    reuse_block_size: int | tuple[int, int] = 2,
) -> list[BenchmarkResult]:
    """Run reconstruction baselines over benchmark cases."""

    results: list[BenchmarkResult] = []
    for case in cases:
        rows, cols = shape(case.matrix)
        original_parameters = rows * cols
        reuse = analyze_block_reuse(case.matrix, block_size=reuse_block_size)
        for baseline in baselines:
            reconstruction = baseline(case.matrix)
            errors = reconstruction_error(case.matrix, reconstruction.reconstructed)
            results.append(
                BenchmarkResult(
                    case_name=case.name,
                    method_name=reconstruction.name,
                    l1_error=errors.l1_error,
                    l2_error=errors.l2_error,
                    relative_l1_error=errors.relative_l1_error,
                    max_abs_error=errors.max_abs_error,
                    parameter_count=reconstruction.parameter_count,
                    compression_ratio=(
                        original_parameters / reconstruction.parameter_count
                        if reconstruction.parameter_count > 0
                        else 0.0
                    ),
                    elapsed_s=reconstruction.elapsed_s,
                    metadata={
                        **reconstruction.metadata,
                        "case_shape": (rows, cols),
                        "reuse_ratio": reuse.reuse_metrics.reuse_ratio,
                    },
                )
            )
    return results
