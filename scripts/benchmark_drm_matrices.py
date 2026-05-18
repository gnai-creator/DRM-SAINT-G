"""Benchmark SAINT reconstruction methods on real drm_transformer matrices.

This script intentionally keeps PyTorch optional for the core package. It only
uses torch here to load an external checkpoint and convert selected tensors into
plain Python matrices for SAINT's dependency-free benchmark runner.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

from saint.reconstruction import (
    BenchmarkCase,
    block_codebook_reconstruction,
    hierarchical_codebook_reconstruction,
    low_rank_reconstruction,
    multi_scale_codebook_reconstruction,
    original_reconstruction,
    run_reconstruction_benchmark,
    uniform_quantization_reconstruction,
)


DEFAULT_KEYWORDS = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.out_proj.weight",
    "ffn.up_proj.weight",
    "ffn.down_proj.weight",
    "dim_gate.gate_net.0.weight",
)


def _safe_case_name(name: str) -> str:
    return (
        name.replace(".weight", "")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _load_state_dict(checkpoint: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required only for this checkpoint-loading script. "
            "Run it in an environment with torch installed."
        ) from exc

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint did not contain a state dict")
    return state


def load_real_matrix_cases(
    checkpoint: Path,
    *,
    max_cases: int = 8,
    max_dim: int = 64,
    keywords: tuple[str, ...] = DEFAULT_KEYWORDS,
) -> list[BenchmarkCase]:
    state_dict = _load_state_dict(checkpoint)
    cases: list[BenchmarkCase] = []

    for name, tensor in state_dict.items():
        if not any(keyword in name for keyword in keywords):
            continue
        if not hasattr(tensor, "ndim") or tensor.ndim != 2:
            continue

        rows = min(int(tensor.shape[0]), max_dim)
        cols = min(int(tensor.shape[1]), max_dim)
        sample = tensor[:rows, :cols].float().tolist()
        cases.append(BenchmarkCase(_safe_case_name(name), sample))
        if len(cases) >= max_cases:
            break

    if not cases:
        raise ValueError("no matching 2D tensors found in checkpoint")
    return cases


def _result_to_dict(result) -> dict[str, Any]:
    return {
        "case": result.case_name,
        "method": result.method_name,
        "l1_error": result.l1_error,
        "l2_error": result.l2_error,
        "relative_l1_error": result.relative_l1_error,
        "max_abs_error": result.max_abs_error,
        "parameter_count": result.parameter_count,
        "compression_ratio": result.compression_ratio,
        "elapsed_s": result.elapsed_s,
        "metadata": result.metadata,
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DRM Matrix Reconstruction Benchmark",
        "",
        "| Case | Method | Rel L1 | Params | Compression | Reuse | Time s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {method} | {rel:.4f} | {params} | {comp:.2f} | {reuse:.2f} | {time:.4f} |".format(
                case=row["case"],
                method=row["method"],
                rel=row["relative_l1_error"],
                params=row["parameter_count"],
                comp=row["compression_ratio"],
                reuse=row["metadata"].get("reuse_ratio", 0.0),
                time=row["elapsed_s"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="runs/reconstruction")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--max-dim", type=int, default=64)
    parser.add_argument("--quantization-step", type=float, default=0.05)
    parser.add_argument("--low-rank", type=int, default=4)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_real_matrix_cases(
        checkpoint,
        max_cases=args.max_cases,
        max_dim=args.max_dim,
    )
    baselines = [
        original_reconstruction,
        partial(
            uniform_quantization_reconstruction,
            step=args.quantization_step,
        ),
        partial(
            low_rank_reconstruction,
            rank=args.low_rank,
            iterations=16,
        ),
        partial(
            block_codebook_reconstruction,
            block_size=2,
            signature_mode="quantized",
            quantization_step=args.quantization_step,
        ),
        partial(
            block_codebook_reconstruction,
            block_size=4,
            signature_mode="quantized",
            quantization_step=args.quantization_step,
        ),
        partial(
            multi_scale_codebook_reconstruction,
            block_sizes=(8, 4, 2),
            signature_mode="quantized",
            quantization_step=args.quantization_step,
        ),
        partial(
            hierarchical_codebook_reconstruction,
            block_sizes=(8, 4, 2),
            signature_mode="quantized",
            quantization_step=args.quantization_step,
        ),
    ]
    results = run_reconstruction_benchmark(cases, baselines)
    rows = [_result_to_dict(result) for result in results]

    json_path = out_dir / "drm_matrix_benchmark.json"
    md_path = out_dir / "drm_matrix_benchmark.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_markdown(md_path, rows)

    print(f"cases={len(cases)} results={len(rows)}")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
