"""Offline score ablation for Phase 16 Marco 4N-C.

This module re-scores completed Marco 4N-B candidate metrics without running any
CUDA training. It asks whether NTK-hybrid bonuses changed stage ordering, whether
zero-gain candidates received optimistic positive scores, and whether a stricter
acceptance epsilon would preserve the useful grafts.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

JsonDict = dict[str, Any]

POLICIES = (
    "composed_gain",
    "composed_gain_orthogonal",
    "ntk_hybrid_current",
    "ntk_hybrid_no_bonus",
    "ntk_hybrid_half_bonus",
    "ntk_hybrid_double_anti_saturation",
    "ntk_hybrid_gain_gated_bonus",
)

THRESHOLDS = (0.0, 0.00002, 0.00005)

REQUIRED_ARTIFACTS = (
    "summary.json",
    "stage_metrics.json",
    "candidate_metrics.json",
)


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required 4N-C input artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_dir(run_dir: Path) -> str:
    return run_dir.name.rsplit("seed", 1)[-1] if "seed" in run_dir.name else "unknown"


def _float(row: JsonDict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return default if value is None else float(value)


def _accepted_before(row: JsonDict) -> int:
    return int(row.get("accepted_grafts_on_target_before_stage", 0) or 0)


def _orthogonal_penalty(row: JsonDict, orthogonal_weight: float = 0.00001) -> float:
    return orthogonal_weight * float(_accepted_before(row))


def _anti_saturation_penalty(row: JsonDict) -> float:
    return _float(row, "ntk_hybrid_penalty", 0.0)


def rescore_candidate(row: JsonDict, policy: str, *, orthogonal_weight: float = 0.00001) -> float:
    """Return the offline score for one already-evaluated candidate row."""
    gain = _float(row, "candidate_composed_gain", 0.0)
    orthogonal = _orthogonal_penalty(row, orthogonal_weight)
    anti = _anti_saturation_penalty(row)
    bonus = _float(row, "ntk_hybrid_bonus", 0.0)
    current_penalty = _float(row, "redundancy_penalty", orthogonal + anti)

    if policy == "composed_gain":
        return gain
    if policy == "composed_gain_orthogonal":
        return gain - orthogonal
    if policy == "ntk_hybrid_current":
        return gain - current_penalty + bonus
    if policy == "ntk_hybrid_no_bonus":
        return gain - current_penalty
    if policy == "ntk_hybrid_half_bonus":
        return gain - current_penalty + (0.5 * bonus)
    if policy == "ntk_hybrid_double_anti_saturation":
        return gain - orthogonal - (2.0 * anti) + bonus
    if policy == "ntk_hybrid_gain_gated_bonus":
        gated_bonus = bonus if gain > 0.0 else 0.0
        return gain - current_penalty + gated_bonus
    raise ValueError(f"unknown 4N-C policy: {policy}")


def _candidate_rank(row: JsonDict, policy: str, threshold: float) -> tuple[int, float]:
    gain = _float(row, "candidate_composed_gain", 0.0)
    return (1 if gain > threshold else 0, rescore_candidate(row, policy))


def _select_winner(rows: list[JsonDict], policy: str, threshold: float) -> JsonDict:
    if not rows:
        raise ValueError("cannot select a 4N-C winner from an empty candidate set")
    return max(rows, key=lambda row: _candidate_rank(row, policy, threshold))


def _stage_actual(stage_metrics: list[JsonDict]) -> dict[int, JsonDict]:
    return {int(row.get("stage", 0)): row for row in stage_metrics}


def _group_candidate_rows(candidate_metrics: list[JsonDict], pass_name: str) -> dict[int, list[JsonDict]]:
    grouped: dict[int, list[JsonDict]] = defaultdict(list)
    for row in candidate_metrics:
        if row.get("pass") != pass_name:
            continue
        grouped[int(row.get("stage", 0))].append(row)
    return dict(grouped)


def _top_targets(rows: list[JsonDict], policy: str, threshold: float, *, limit: int = 3) -> list[str]:
    ranked = sorted(rows, key=lambda row: _candidate_rank(row, policy, threshold), reverse=True)
    return [
        "{target}({gain:.6g},{score:.6g})".format(
            target=row.get("candidate_target"),
            gain=_float(row, "candidate_composed_gain", 0.0),
            score=rescore_candidate(row, policy),
        )
        for row in ranked[:limit]
    ]


def analyze_run(run_dir: str | Path) -> tuple[list[JsonDict], JsonDict]:
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
    deep_by_stage = _group_candidate_rows(candidate_metrics, "deep")
    probe_by_stage = _group_candidate_rows(candidate_metrics, "probe")
    rows: list[JsonDict] = []
    zero_gain_positive_score_count = 0
    zero_gain_positive_score_stages: list[int] = []

    for stage in sorted(deep_by_stage):
        deep_rows = deep_by_stage[stage]
        actual = actual_by_stage.get(stage, {})
        current_zero_positive = [
            row for row in deep_rows
            if _float(row, "candidate_composed_gain", 0.0) <= 0.0
            and rescore_candidate(row, "ntk_hybrid_current") > 0.0
        ]
        zero_gain_positive_score_count += len(current_zero_positive)
        if current_zero_positive:
            zero_gain_positive_score_stages.append(stage)
        for policy in POLICIES:
            for threshold in THRESHOLDS:
                winner = _select_winner(deep_rows, policy, threshold)
                gain = _float(winner, "candidate_composed_gain", 0.0)
                row = {
                    "seed": seed,
                    "run_dir": str(root),
                    "stage": stage,
                    "policy": policy,
                    "accept_threshold": threshold,
                    "winner_target": winner.get("candidate_target"),
                    "winner_tag": winner.get("candidate_tag"),
                    "winner_gain": gain,
                    "winner_score": rescore_candidate(winner, policy),
                    "winner_ntk_rank": winner.get("ntk_rank"),
                    "winner_accepted_before": winner.get("accepted_grafts_on_target_before_stage"),
                    "would_approve": gain > threshold,
                    "actual_target": actual.get("selected_target"),
                    "actual_decision": actual.get("decision"),
                    "actual_gain": actual.get("stage_gain"),
                    "matches_actual_target": winner.get("candidate_target") == actual.get("selected_target"),
                    "top3": _top_targets(deep_rows, policy, threshold),
                    "deep_candidate_count": len(deep_rows),
                    "probe_candidate_count": len(probe_by_stage.get(stage, [])),
                    "zero_gain_positive_current_score_count": len(current_zero_positive),
                }
                rows.append(row)

    accepted_grafts = int(summary.get("accepted_grafts", 0) or 0)
    run_summary = {
        "seed": seed,
        "run_dir": str(root),
        "base_loss": summary.get("base_loss"),
        "composed_loss": summary.get("composed_loss"),
        "accumulated_gain": summary.get("accumulated_gain"),
        "accepted_groups": summary.get("accepted_groups"),
        "accepted_grafts": accepted_grafts,
        "target_by_graft": summary.get("target_by_graft", {}),
        "recompose_abs_diff": summary.get("recompose_abs_diff"),
        "stage_count": len(stage_metrics),
        "approved_stage_count": sum(1 for row in stage_metrics if row.get("decision") == "approved"),
        "zero_gain_positive_current_score_count": zero_gain_positive_score_count,
        "zero_gain_positive_current_score_stages": sorted(set(zero_gain_positive_score_stages)),
    }
    return rows, run_summary


def _has_exact_recompose(summary: JsonDict) -> bool:
    value = summary.get("recompose_abs_diff")
    return value is not None and float(value) == 0.0


def summarize_ablation(rows: list[JsonDict], run_summaries: list[JsonDict]) -> JsonDict:
    by_policy: dict[tuple[str, float], list[JsonDict]] = defaultdict(list)
    for row in rows:
        by_policy[(str(row["policy"]), float(row["accept_threshold"]))].append(row)

    policy_summaries = []
    for (policy, threshold), policy_rows in sorted(by_policy.items()):
        approved = [row for row in policy_rows if row.get("would_approve")]
        matches = [row for row in policy_rows if row.get("matches_actual_target")]
        zero_gain_winners = [row for row in policy_rows if float(row.get("winner_gain", 0.0) or 0.0) <= 0.0]
        policy_summaries.append({
            "policy": policy,
            "accept_threshold": threshold,
            "stage_decisions": len(policy_rows),
            "would_approve_count": len(approved),
            "winner_zero_gain_count": len(zero_gain_winners),
            "actual_target_match_rate": len(matches) / len(policy_rows) if policy_rows else None,
            "mean_winner_gain": mean([float(row.get("winner_gain", 0.0) or 0.0) for row in policy_rows]) if policy_rows else 0.0,
        })

    return {
        "phase": "16",
        "marco": "4n_c_offline_score_ablation",
        "run_count": len(run_summaries),
        "seeds": [summary.get("seed") for summary in run_summaries],
        "positive_runs": sum(1 for summary in run_summaries if float(summary.get("accumulated_gain", 0.0) or 0.0) > 0.0),
        "exact_recompose_runs": sum(1 for summary in run_summaries if _has_exact_recompose(summary)),
        "mean_accumulated_gain": mean([float(summary.get("accumulated_gain", 0.0) or 0.0) for summary in run_summaries]) if run_summaries else 0.0,
        "mean_accepted_grafts": mean([float(summary.get("accepted_grafts", 0.0) or 0.0) for summary in run_summaries]) if run_summaries else 0.0,
        "seeds_with_five_or_more_grafts": [summary.get("seed") for summary in run_summaries if int(summary.get("accepted_grafts", 0) or 0) >= 5],
        "zero_gain_positive_current_score_total": sum(int(summary.get("zero_gain_positive_current_score_count", 0) or 0) for summary in run_summaries),
        "run_summaries": run_summaries,
        "policy_summaries": policy_summaries,
        "recommendations": [
            "keep_4k_as_best_loss_checkpoint_for_seed42",
            "treat_4n_b_as_structural_routing_improvement_not_robust_quality_win",
            "gate_ntk_bonus_when_candidate_composed_gain_is_zero_or_below_epsilon",
            "test_accept_min_gain_epsilon_2e-5_before_more_cuda",
            "prefer_offline_ablation_or_4o_svd_before_new_large_routing_sweep",
        ],
    }


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
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_report(summary: JsonDict, rows: list[JsonDict]) -> str:
    lines = [
        "# Phase 16 Marco 4N-C - Offline NTK-Hybrid Score Ablation",
        "",
        "Status: completed offline from Marco 4N-B artifacts; no CUDA training performed.",
        "",
        "## Run Summary",
        "",
        "| seed | gain | accepted_grafts | stages | zero-gain positive current scores | route | recompose_abs_diff |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for run in summary.get("run_summaries", []):
        route = ", ".join(f"{key}->{value}" for key, value in sorted((run.get("target_by_graft") or {}).items(), key=lambda item: int(item[0])))
        lines.append(
            "| {seed} | {gain:.9f} | {grafts} | {stages} | {zero} | {route} | {recompose} |".format(
                seed=run.get("seed"),
                gain=float(run.get("accumulated_gain", 0.0) or 0.0),
                grafts=run.get("accepted_grafts"),
                stages=run.get("stage_count"),
                zero=run.get("zero_gain_positive_current_score_count"),
                route=route or "n/a",
                recompose=run.get("recompose_abs_diff"),
            )
        )
    lines.extend([
        "",
        "## Aggregate Verdict",
        "",
        f"- positive_runs: {summary.get('positive_runs')}/{summary.get('run_count')}",
        f"- exact_recompose_runs: {summary.get('exact_recompose_runs')}/{summary.get('run_count')}",
        f"- mean_accumulated_gain: {float(summary.get('mean_accumulated_gain', 0.0)):.9f}",
        f"- mean_accepted_grafts: {float(summary.get('mean_accepted_grafts', 0.0)):.3f}",
        f"- seeds_with_five_or_more_grafts: {', '.join(summary.get('seeds_with_five_or_more_grafts', [])) or 'none'}",
        f"- zero_gain_positive_current_score_total: {summary.get('zero_gain_positive_current_score_total')}",
        "",
        "Interpretation: 4N-B is a structural routing improvement, especially for seed 42, but it did not solve multi-seed robustness. Seeds 7 and 123 still stop at four accepted grafts.",
        "",
        "## Policy Summary",
        "",
        "| policy | threshold | approve_count | zero_gain_winners | target_match_rate | mean_winner_gain |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for policy in summary.get("policy_summaries", []):
        rate = policy.get("actual_target_match_rate")
        lines.append(
            "| {policy} | {threshold:.5f} | {approve} | {zero} | {rate} | {mean_gain:.9f} |".format(
                policy=policy.get("policy"),
                threshold=float(policy.get("accept_threshold", 0.0) or 0.0),
                approve=policy.get("would_approve_count"),
                zero=policy.get("winner_zero_gain_count"),
                rate="n/a" if rate is None else f"{float(rate):.3f}",
                mean_gain=float(policy.get("mean_winner_gain", 0.0) or 0.0),
            )
        )
    lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    for rec in summary.get("recommendations", []):
        lines.append(f"- {rec}")
    lines.extend([
        "",
        "## Stage Winners by Current and Gated Policy",
        "",
        "| seed | stage | policy | threshold | winner | gain | score | would_approve | actual |",
        "|---|---:|---|---:|---|---:|---:|---|---|",
    ])
    interesting = [
        row for row in rows
        if row.get("policy") in {"ntk_hybrid_current", "ntk_hybrid_gain_gated_bonus", "composed_gain"}
        and float(row.get("accept_threshold", 0.0) or 0.0) in {0.0, 0.00002}
    ]
    for row in sorted(interesting, key=lambda item: (str(item.get("seed")), int(item.get("stage", 0)), str(item.get("policy")), float(item.get("accept_threshold", 0.0)))):
        lines.append(
            "| {seed} | {stage} | {policy} | {threshold:.5f} | {winner} | {gain:.9f} | {score:.9f} | {approve} | {actual} |".format(
                seed=row.get("seed"),
                stage=row.get("stage"),
                policy=row.get("policy"),
                threshold=float(row.get("accept_threshold", 0.0) or 0.0),
                winner=row.get("winner_target"),
                gain=float(row.get("winner_gain", 0.0) or 0.0),
                score=float(row.get("winner_score", 0.0) or 0.0),
                approve=str(bool(row.get("would_approve"))).lower(),
                actual=f"{row.get('actual_target')}/{row.get('actual_decision')}",
            )
        )
    lines.append("")
    return "\n".join(lines)


def analyze_runs(run_dirs: list[str | Path]) -> tuple[list[JsonDict], JsonDict]:
    all_rows: list[JsonDict] = []
    run_summaries: list[JsonDict] = []
    for run_dir in run_dirs:
        rows, run_summary = analyze_run(run_dir)
        all_rows.extend(rows)
        run_summaries.append(run_summary)
    summary = summarize_ablation(all_rows, run_summaries)
    return all_rows, summary


__all__ = [
    "POLICIES",
    "THRESHOLDS",
    "analyze_run",
    "analyze_runs",
    "render_markdown_report",
    "rescore_candidate",
    "summarize_ablation",
    "write_csv",
    "write_json",
]
