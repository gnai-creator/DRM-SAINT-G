"""Offline cost-aware candidate efficiency analysis for Phase 16 Marco 4P-A.

This module re-ranks completed dense Marco 4N-B candidate metrics without running
CUDA training. It starts the 4P line by asking whether the already-probed dense
candidates remain attractive after subtracting redundancy, NTK risk, parameter,
checkpoint, and probe-time costs.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

JsonDict = dict[str, Any]

REQUIRED_ARTIFACTS = (
    "summary.json",
    "stage_metrics.json",
    "candidate_metrics.json",
)


@dataclass(frozen=True)
class EfficiencyWeights:
    lambda_params: float = 0.0
    lambda_bytes: float = 0.0
    lambda_time: float = 0.0
    lambda_ntk_risk: float = 1.0


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required 4P-A input artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_dir(run_dir: Path) -> str:
    return run_dir.name.rsplit("seed", 1)[-1] if "seed" in run_dir.name else "unknown"


def _float(row: JsonDict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return default if value is None else float(value)


def dense_params_per_graft(*, d_model: int = 96, hidden_size: int = 25889) -> int:
    """Return DRMGraftBlock parameter count: up + down + scalar scale."""
    return (2 * int(d_model) * int(hidden_size)) + 1


def default_checkpoint_bytes_delta(params_per_graft: int) -> int:
    """Approximate fp32 checkpoint bytes added by one dense graft block."""
    return int(params_per_graft) * 4


def efficiency_score(row: JsonDict, weights: EfficiencyWeights) -> float:
    """Return cost-aware score for one already-evaluated candidate row.

    Formula:
        candidate_composed_gain
        - redundancy_penalty
        - lambda_ntk_risk * ntk_hybrid_penalty
        - lambda_params * log1p(params_per_graft)
        - lambda_bytes * log1p(checkpoint_bytes_delta)
        - lambda_time * probe_seconds
    """
    gain = _float(row, "candidate_composed_gain", 0.0)
    redundancy = _float(row, "redundancy_penalty", 0.0)
    ntk_risk = _float(row, "ntk_hybrid_penalty", 0.0)
    params = max(0.0, _float(row, "params_per_graft", 0.0))
    bytes_delta = max(0.0, _float(row, "checkpoint_bytes_delta", 0.0))
    probe_seconds = max(0.0, _float(row, "elapsed_s", _float(row, "probe_seconds", 0.0)))
    return (
        gain
        - redundancy
        - (float(weights.lambda_ntk_risk) * ntk_risk)
        - (float(weights.lambda_params) * math.log1p(params))
        - (float(weights.lambda_bytes) * math.log1p(bytes_delta))
        - (float(weights.lambda_time) * probe_seconds)
    )


def _enrich_candidate(row: JsonDict, weights: EfficiencyWeights, *, params_per_graft: int, checkpoint_bytes_delta: int) -> JsonDict:
    enriched = dict(row)
    enriched.setdefault("params_per_graft", int(params_per_graft))
    enriched.setdefault("checkpoint_bytes_delta", int(checkpoint_bytes_delta))
    enriched["probe_seconds"] = _float(enriched, "elapsed_s", _float(enriched, "probe_seconds", 0.0))
    enriched["ntk_risk_penalty"] = _float(enriched, "ntk_hybrid_penalty", 0.0)
    enriched["gain_per_million_params"] = (
        _float(enriched, "candidate_composed_gain", 0.0) / (float(enriched["params_per_graft"]) / 1_000_000.0)
        if float(enriched["params_per_graft"]) > 0.0
        else 0.0
    )
    enriched["gain_per_mb_checkpoint"] = (
        _float(enriched, "candidate_composed_gain", 0.0) / (float(enriched["checkpoint_bytes_delta"]) / 1_000_000.0)
        if float(enriched["checkpoint_bytes_delta"]) > 0.0
        else 0.0
    )
    enriched["gain_per_probe_second"] = (
        _float(enriched, "candidate_composed_gain", 0.0) / enriched["probe_seconds"]
        if float(enriched["probe_seconds"]) > 0.0
        else 0.0
    )
    enriched["efficiency_score"] = efficiency_score(enriched, weights)
    return enriched


def rank_stage_candidates(rows: list[JsonDict], weights: EfficiencyWeights) -> list[JsonDict]:
    """Return candidates sorted by cost-aware score, descending."""
    return sorted(
        [{**row, "efficiency_score": efficiency_score(row, weights)} for row in rows],
        key=lambda row: (
            float(row.get("efficiency_score", 0.0) or 0.0),
            float(row.get("candidate_composed_gain", 0.0) or 0.0),
        ),
        reverse=True,
    )


def _stage_actual(stage_metrics: list[JsonDict]) -> dict[int, JsonDict]:
    return {int(row.get("stage", 0)): row for row in stage_metrics}


def _group_deep_candidate_rows(candidate_metrics: list[JsonDict]) -> dict[int, list[JsonDict]]:
    grouped: dict[int, list[JsonDict]] = defaultdict(list)
    for row in candidate_metrics:
        if row.get("pass") != "deep":
            continue
        grouped[int(row.get("stage", 0))].append(row)
    return dict(grouped)


def analyze_run(
    run_dir: str | Path,
    weights: EfficiencyWeights,
    *,
    params_per_graft: int,
    checkpoint_bytes_delta: int,
) -> tuple[list[JsonDict], JsonDict]:
    root = Path(run_dir)
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"run directory {root} is missing: {', '.join(missing)}")

    summary = _load_json(root / "summary.json")
    stage_metrics = _load_json(root / "stage_metrics.json")
    candidate_metrics = _load_json(root / "candidate_metrics.json")
    if not stage_metrics or not candidate_metrics:
        raise ValueError(f"run directory {root} has empty stage/candidate metrics")

    seed = _seed_from_dir(root)
    actual_by_stage = _stage_actual(stage_metrics)
    deep_by_stage = _group_deep_candidate_rows(candidate_metrics)
    rows: list[JsonDict] = []
    stage_winners: list[JsonDict] = []

    for stage in sorted(deep_by_stage):
        enriched = [
            _enrich_candidate(row, weights, params_per_graft=params_per_graft, checkpoint_bytes_delta=checkpoint_bytes_delta)
            for row in deep_by_stage[stage]
        ]
        ranked = rank_stage_candidates(enriched, weights)
        actual = actual_by_stage.get(stage, {})
        for rank, row in enumerate(ranked, start=1):
            output_row = dict(row)
            output_row.update(
                {
                    "seed": seed,
                    "run_dir": str(root),
                    "efficiency_rank": rank,
                    "actual_target": actual.get("selected_target"),
                    "actual_decision": actual.get("decision"),
                    "actual_gain": actual.get("stage_gain"),
                    "matches_actual_target": row.get("candidate_target") == actual.get("selected_target"),
                }
            )
            rows.append(output_row)
        if ranked:
            winner = dict(ranked[0])
            winner.update(
                {
                    "seed": seed,
                    "run_dir": str(root),
                    "stage": stage,
                    "actual_target": actual.get("selected_target"),
                    "actual_decision": actual.get("decision"),
                    "actual_gain": actual.get("stage_gain"),
                    "matches_actual_target": winner.get("candidate_target") == actual.get("selected_target"),
                }
            )
            stage_winners.append(winner)

    approved_stages = [row for row in stage_metrics if row.get("decision") == "approved"]
    matches = [row for row in stage_winners if row.get("matches_actual_target")]
    positive_efficiency_winners = [row for row in stage_winners if float(row.get("efficiency_score", 0.0) or 0.0) > 0.0]
    best_winner = max(stage_winners, key=lambda row: float(row.get("efficiency_score", 0.0) or 0.0)) if stage_winners else {}
    run_summary = {
        "seed": seed,
        "run_dir": str(root),
        "base_loss": summary.get("base_loss"),
        "composed_loss": summary.get("composed_loss"),
        "accumulated_gain": summary.get("accumulated_gain"),
        "accepted_grafts": int(summary.get("accepted_grafts", 0) or 0),
        "target_by_graft": summary.get("target_by_graft", {}),
        "recompose_abs_diff": summary.get("recompose_abs_diff"),
        "stage_count": len(stage_metrics),
        "approved_stage_count": len(approved_stages),
        "deep_candidate_count": sum(len(rows) for rows in deep_by_stage.values()),
        "efficiency_target_match_rate": len(matches) / len(stage_winners) if stage_winners else None,
        "positive_efficiency_winner_count": len(positive_efficiency_winners),
        "best_efficiency_target": best_winner.get("candidate_target"),
        "best_efficiency_stage": best_winner.get("stage"),
        "best_efficiency_score": best_winner.get("efficiency_score"),
        "best_efficiency_gain": best_winner.get("candidate_composed_gain"),
        "best_gain_per_million_params": best_winner.get("gain_per_million_params"),
        "stage_winners": stage_winners,
    }
    return rows, run_summary


def _has_exact_recompose(summary: JsonDict) -> bool:
    value = summary.get("recompose_abs_diff")
    return value is not None and float(value) == 0.0


def summarize_efficiency(rows: list[JsonDict], run_summaries: list[JsonDict], weights: EfficiencyWeights, *, params_per_graft: int, checkpoint_bytes_delta: int) -> JsonDict:
    stage_winners = [winner for run in run_summaries for winner in run.get("stage_winners", [])]
    target_counter = Counter(str(row.get("candidate_target")) for row in stage_winners)
    positive_winners = [row for row in stage_winners if float(row.get("efficiency_score", 0.0) or 0.0) > 0.0]
    matches = [row for row in stage_winners if row.get("matches_actual_target")]
    return {
        "phase": "16",
        "marco": "4p_a_offline_candidate_efficiency",
        "status": "completed_offline_no_cuda_training",
        "run_count": len(run_summaries),
        "seeds": [summary.get("seed") for summary in run_summaries],
        "params_per_graft": int(params_per_graft),
        "checkpoint_bytes_delta": int(checkpoint_bytes_delta),
        "weights": {
            "lambda_params": float(weights.lambda_params),
            "lambda_bytes": float(weights.lambda_bytes),
            "lambda_time": float(weights.lambda_time),
            "lambda_ntk_risk": float(weights.lambda_ntk_risk),
        },
        "exact_recompose_runs": sum(1 for summary in run_summaries if _has_exact_recompose(summary)),
        "mean_accumulated_gain": mean([float(summary.get("accumulated_gain", 0.0) or 0.0) for summary in run_summaries]) if run_summaries else 0.0,
        "mean_accepted_grafts": mean([float(summary.get("accepted_grafts", 0.0) or 0.0) for summary in run_summaries]) if run_summaries else 0.0,
        "stage_decisions": len(stage_winners),
        "positive_efficiency_winner_count": len(positive_winners),
        "efficiency_actual_target_match_rate": len(matches) / len(stage_winners) if stage_winners else None,
        "top_efficiency_targets": target_counter.most_common(),
        "mean_winner_efficiency_score": mean([float(row.get("efficiency_score", 0.0) or 0.0) for row in stage_winners]) if stage_winners else 0.0,
        "mean_winner_gain_per_million_params": mean([float(row.get("gain_per_million_params", 0.0) or 0.0) for row in stage_winners]) if stage_winners else 0.0,
        "run_summaries": run_summaries,
        "recommendations": [
            "use_4p_a_as_offline_gate_before_cuda_score_changes",
            "tune_lambda_params_bytes_time_against_4n_b_stage_winner_preservation",
            "prefer_cost_aware_score_only_if_it_preserves_positive_4n_b_winners",
            "next_cuda_step_4p_b_short_dense_cost_aware_routing_if_offline_match_rate_is_acceptable",
        ],
    }


def analyze_runs(
    run_dirs: list[str | Path],
    weights: EfficiencyWeights,
    *,
    params_per_graft: int,
    checkpoint_bytes_delta: int | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    bytes_delta = default_checkpoint_bytes_delta(params_per_graft) if checkpoint_bytes_delta is None else int(checkpoint_bytes_delta)
    all_rows: list[JsonDict] = []
    run_summaries: list[JsonDict] = []
    for run_dir in run_dirs:
        rows, run_summary = analyze_run(
            run_dir,
            weights,
            params_per_graft=int(params_per_graft),
            checkpoint_bytes_delta=bytes_delta,
        )
        all_rows.extend(rows)
        run_summaries.append(run_summary)
    summary = summarize_efficiency(all_rows, run_summaries, weights, params_per_graft=int(params_per_graft), checkpoint_bytes_delta=bytes_delta)
    return all_rows, summary


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[JsonDict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row if not isinstance(row.get(key), (dict, list))})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _fmt_float(value: Any, digits: int = 9) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_markdown_report(summary: JsonDict, rows: list[JsonDict]) -> str:
    lines = [
        "# Phase 16 Marco 4P-A - Offline Candidate Efficiency Reranking",
        "",
        "Status: completed offline from Marco 4N-B artifacts; no CUDA training performed.",
        "",
        "## Score",
        "",
        "```text",
        "efficiency_score = candidate_composed_gain",
        "  - redundancy_penalty",
        "  - lambda_ntk_risk * ntk_hybrid_penalty",
        "  - lambda_params * log1p(params_per_graft)",
        "  - lambda_bytes  * log1p(checkpoint_bytes_delta)",
        "  - lambda_time   * probe_seconds",
        "```",
        "",
        "## Configuration",
        "",
        f"- params_per_graft: {summary.get('params_per_graft')}",
        f"- checkpoint_bytes_delta: {summary.get('checkpoint_bytes_delta')}",
        f"- weights: `{summary.get('weights')}`",
        "",
        "## Run Summary",
        "",
        "| seed | gain | accepted_grafts | stages | efficiency_match_rate | best_efficiency_target | best_efficiency_score | recompose_abs_diff |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for run in summary.get("run_summaries", []):
        rate = run.get("efficiency_target_match_rate")
        lines.append(
            "| {seed} | {gain} | {grafts} | {stages} | {rate} | {target} | {score} | {recompose} |".format(
                seed=run.get("seed"),
                gain=_fmt_float(run.get("accumulated_gain")),
                grafts=run.get("accepted_grafts"),
                stages=run.get("stage_count"),
                rate="n/a" if rate is None else f"{float(rate):.3f}",
                target=run.get("best_efficiency_target"),
                score=_fmt_float(run.get("best_efficiency_score")),
                recompose=run.get("recompose_abs_diff"),
            )
        )
    lines.extend([
        "",
        "## Aggregate Verdict",
        "",
        f"- exact_recompose_runs: {summary.get('exact_recompose_runs')}/{summary.get('run_count')}",
        f"- mean_accumulated_gain: {_fmt_float(summary.get('mean_accumulated_gain'))}",
        f"- mean_accepted_grafts: {float(summary.get('mean_accepted_grafts', 0.0) or 0.0):.3f}",
        f"- stage_decisions: {summary.get('stage_decisions')}",
        f"- positive_efficiency_winner_count: {summary.get('positive_efficiency_winner_count')}",
        f"- efficiency_actual_target_match_rate: {_fmt_float(summary.get('efficiency_actual_target_match_rate'), 3)}",
        f"- mean_winner_efficiency_score: {_fmt_float(summary.get('mean_winner_efficiency_score'))}",
        f"- mean_winner_gain_per_million_params: {_fmt_float(summary.get('mean_winner_gain_per_million_params'))}",
        "",
        "## Stage Winners by Efficiency Score",
        "",
        "| seed | stage | winner | gain | redundancy | ntk_risk | probe_s | gain/M params | efficiency_score | actual |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    winners = [winner for run in summary.get("run_summaries", []) for winner in run.get("stage_winners", [])]
    for row in sorted(winners, key=lambda item: (str(item.get("seed")), int(item.get("stage", 0)))):
        lines.append(
            "| {seed} | {stage} | {target} | {gain} | {redundancy} | {ntk} | {time} | {gpm} | {score} | {actual} |".format(
                seed=row.get("seed"),
                stage=row.get("stage"),
                target=row.get("candidate_target"),
                gain=_fmt_float(row.get("candidate_composed_gain")),
                redundancy=_fmt_float(row.get("redundancy_penalty")),
                ntk=_fmt_float(row.get("ntk_risk_penalty")),
                time=_fmt_float(row.get("probe_seconds"), 3),
                gpm=_fmt_float(row.get("gain_per_million_params")),
                score=_fmt_float(row.get("efficiency_score")),
                actual=f"{row.get('actual_target')}/{row.get('actual_decision')}",
            )
        )
    lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    for rec in summary.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "EfficiencyWeights",
    "analyze_run",
    "analyze_runs",
    "default_checkpoint_bytes_delta",
    "dense_params_per_graft",
    "efficiency_score",
    "rank_stage_candidates",
    "render_markdown_report",
    "summarize_efficiency",
    "write_csv",
    "write_json",
]
