"""Run the phase-4 linear-layer learning benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saint.training import (
    evaluate_phase4_closure,
    evaluate_phase4_success,
    evaluate_phase4_regime_success,
    make_linear_delta_task,
    run_linear_phase4_benchmark,
    run_linear_phase4_regime_sweep,
    run_linear_phase4_sweep,
    summarize_phase4_rows,
)


def _result_to_dict(result) -> dict[str, Any]:
    return {
        "method": result.name,
        "train_loss": result.train_loss,
        "test_loss": result.test_loss,
        "weight_relative_l1_error": result.weight_relative_l1_error,
        "parameter_count": result.parameter_count,
        "optimizer_state_values": result.optimizer_state_values,
        "elapsed_s": result.elapsed_s,
        "metadata": result.metadata,
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase 4 Linear Training Benchmark",
        "",
        "| Method | Test Loss | Weight Rel L1 | Params | Optimizer Values | Time s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {test_loss:.6f} | {weight_error:.4f} | {params} | {optim} | {time:.4f} |".format(
                method=row["method"],
                test_loss=row["test_loss"],
                weight_error=row["weight_relative_l1_error"],
                params=row["parameter_count"],
                optim=row["optimizer_state_values"],
                time=row["elapsed_s"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sweep_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> None:
    lines = [
        "# Phase 4 Linear Training Sweep",
        "",
        "| Method | Runs | Avg Test Loss | Avg Weight Rel L1 | Avg Params | Avg Optimizer Values | Avg Gain/Param |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {method} | {runs} | {loss:.6f} | {weight_error:.4f} | {params:.1f} | {optim:.1f} | {gain:.8f} |".format(
                method=row["method"],
                runs=row["runs"],
                loss=row["avg_test_loss"],
                weight_error=row["avg_weight_relative_l1_error"],
                params=row["avg_parameter_count"],
                optim=row["avg_optimizer_state_values"],
                gain=row["avg_gain_per_parameter"],
            )
        )
    lines.extend(["", "## Decisions", ""])
    for decision in decisions:
        if "compared_method" not in decision:
            lines.append(
                "- {saint}: {status} ({reason})".format(
                    saint=decision["saint_method"],
                    status="passed" if decision["passed"] else "failed",
                    reason=decision["reason"],
                )
            )
            continue
        lines.append(
            "- {saint} vs {compared}: {status} ({reason})".format(
                saint=decision["saint_method"],
                compared=decision["compared_method"],
                status="passed" if decision["passed"] else "failed",
                reason=decision["reason"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/phase4_linear")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--test-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--regime-sweep", action="store_true")
    parser.add_argument("--seeds", default="11,12,13,14,15")
    parser.add_argument("--sizes", default="8,16,32")
    parser.add_argument("--delta-modes", default="repeated,dense")
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--lora-steps", type=int, default=140)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.regime_sweep:
        seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
        sizes = tuple(int(size.strip()) for size in args.sizes.split(",") if size.strip())
        delta_modes = tuple(
            mode.strip() for mode in args.delta_modes.split(",") if mode.strip()
        )
        rows = run_linear_phase4_regime_sweep(
            seeds=seeds,
            sizes=sizes,
            delta_modes=delta_modes,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            steps=args.steps,
            lora_steps=args.lora_steps,
        )
        summaries = summarize_phase4_rows(rows)
        decisions = [
            evaluate_phase4_success(
                summaries,
                saint_method="saint_dynamic_delta",
                compared_method="lora_tuned_rank_2",
            ).__dict__,
            *evaluate_phase4_regime_success(
                rows,
                saint_method="saint_dynamic_delta",
                compared_method="lora_tuned_rank_2",
            ),
            *evaluate_phase4_regime_success(
                rows,
                saint_method="saint_dynamic_delta",
                compared_method="lora_tuned_rank_4",
            ),
            *evaluate_phase4_regime_success(
                rows,
                saint_method="saint_dynamic_delta",
                compared_method="block_budgeted_delta_for_saint_dynamic_delta",
            ),
            *evaluate_phase4_regime_success(
                rows,
                saint_method="saint_dynamic_delta",
                compared_method="budgeted_full_delta_for_saint_dynamic_delta",
            ),
            evaluate_phase4_closure(rows, saint_method="saint_dynamic_delta"),
        ]

        rows_path = out_dir / "linear_training_regime_rows.json"
        summary_path = out_dir / "linear_training_regime_summary.json"
        decisions_path = out_dir / "linear_training_regime_decisions.json"
        md_path = out_dir / "linear_training_regime.md"
        rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        decisions_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
        _write_sweep_markdown(md_path, summaries, decisions)

        print(f"rows={len(rows)}")
        print(f"summary={summary_path}")
        print(f"decisions={decisions_path}")
        print(f"markdown={md_path}")
    elif args.sweep:
        seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
        rows = run_linear_phase4_sweep(
            seeds=seeds,
            rows=args.rows,
            cols=args.cols,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            steps=args.steps,
            lora_steps=args.lora_steps,
        )
        summaries = summarize_phase4_rows(rows)
        decisions = [
            evaluate_phase4_success(
                summaries,
                saint_method="saint_dynamic_delta",
                compared_method="lora_tuned_rank_2",
            ).__dict__,
            evaluate_phase4_success(
                summaries,
                saint_method="saint_global_capped",
                compared_method="lora_rank_2",
            ).__dict__,
            evaluate_phase4_success(
                summaries,
                saint_method="saint_routed_f50_c25",
                compared_method="lora_rank_2",
            ).__dict__,
            evaluate_phase4_success(
                summaries,
                saint_method="saint_routed_f25_c50",
                compared_method="lora_rank_2",
            ).__dict__,
            evaluate_phase4_success(
                summaries,
                saint_method="saint_routed_f25_c25",
                compared_method="lora_rank_1",
                max_parameter_ratio=2.0,
            ).__dict__,
            evaluate_phase4_success(
                summaries,
                saint_method="saint_global_capped",
                compared_method="budgeted_full_delta_for_saint_global_capped",
            ).__dict__,
            evaluate_phase4_success(
                summaries,
                saint_method="saint_dynamic_delta",
                compared_method="block_budgeted_delta_for_saint_dynamic_delta",
            ).__dict__,
            evaluate_phase4_success(
                summaries,
                saint_method="saint_dynamic_delta",
                compared_method="budgeted_full_delta_for_saint_dynamic_delta",
            ).__dict__,
        ]

        rows_path = out_dir / "linear_training_sweep_rows.json"
        summary_path = out_dir / "linear_training_sweep_summary.json"
        decisions_path = out_dir / "linear_training_sweep_decisions.json"
        md_path = out_dir / "linear_training_sweep.md"
        rows_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        decisions_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")
        _write_sweep_markdown(md_path, summaries, decisions)

        print(f"rows={len(rows)}")
        print(f"summary={summary_path}")
        print(f"decisions={decisions_path}")
        print(f"markdown={md_path}")
    else:
        task = make_linear_delta_task(
            rows=args.rows,
            cols=args.cols,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            seed=args.seed,
        )
        results = run_linear_phase4_benchmark(task)
        rows = [_result_to_dict(result) for result in results]

        json_path = out_dir / "linear_training_benchmark.json"
        md_path = out_dir / "linear_training_benchmark.md"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        _write_markdown(md_path, rows)

        print(f"results={len(rows)}")
        print(f"json={json_path}")
        print(f"markdown={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
