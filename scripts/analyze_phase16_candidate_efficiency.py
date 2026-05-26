#!/usr/bin/env python
"""Run Phase 16 Marco 4P-A offline candidate efficiency reranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saint.adapters.drm_grafting_candidate_efficiency import (
    EfficiencyWeights,
    analyze_runs,
    default_checkpoint_bytes_delta,
    dense_params_per_graft,
    render_markdown_report,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Completed dense Marco 4N-B run directory. Repeat for multiple seeds.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where 4P-A offline efficiency artifacts will be written.",
    )
    parser.add_argument("--d-model", type=int, default=96, help="Dense graft d_model used to infer params_per_graft.")
    parser.add_argument("--hidden-size", type=int, default=25889, help="Dense graft hidden size used to infer params_per_graft.")
    parser.add_argument("--params-per-graft", type=int, default=None, help="Override inferred dense graft parameter count.")
    parser.add_argument("--checkpoint-bytes-delta", type=int, default=None, help="Override estimated bytes added per graft.")
    parser.add_argument("--lambda-params", type=float, default=0.0)
    parser.add_argument("--lambda-bytes", type=float, default=0.0)
    parser.add_argument("--lambda-time", type=float, default=0.0)
    parser.add_argument("--lambda-ntk-risk", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    params_per_graft = args.params_per_graft or dense_params_per_graft(d_model=args.d_model, hidden_size=args.hidden_size)
    checkpoint_bytes_delta = args.checkpoint_bytes_delta or default_checkpoint_bytes_delta(params_per_graft)
    weights = EfficiencyWeights(
        lambda_params=args.lambda_params,
        lambda_bytes=args.lambda_bytes,
        lambda_time=args.lambda_time,
        lambda_ntk_risk=args.lambda_ntk_risk,
    )
    rows, summary = analyze_runs(
        args.run_dir,
        weights,
        params_per_graft=params_per_graft,
        checkpoint_bytes_delta=checkpoint_bytes_delta,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "candidate_efficiency_rows.json", rows)
    write_json(output_dir / "candidate_efficiency_summary.json", summary)
    write_csv(output_dir / "candidate_efficiency_table.csv", rows)
    (output_dir / "candidate_efficiency.md").write_text(
        render_markdown_report(summary, rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} efficiency rows to {output_dir}")
    print(
        "mean_gain={gain:.9f} mean_grafts={grafts:.3f} match_rate={match:.3f}".format(
            gain=float(summary.get("mean_accumulated_gain", 0.0) or 0.0),
            grafts=float(summary.get("mean_accepted_grafts", 0.0) or 0.0),
            match=float(summary.get("efficiency_actual_target_match_rate", 0.0) or 0.0),
        )
    )
    print("recommendations=" + ",".join(summary.get("recommendations", [])))


if __name__ == "__main__":
    main()
