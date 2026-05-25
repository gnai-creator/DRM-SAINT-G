"""Small utilities for validation-routed DRM-G graftblock runs."""

from __future__ import annotations

from typing import Any


def marco_name(args) -> str:
    if getattr(args, "candidate_score_mode", "composed_gain") == "composed_gain_ntk_hybrid_conservative":
        return "4n_b_ntk_hybrid_conservative_routing"
    if int(getattr(args, "ntk_activation_probe_batches", 0) or 0) > 0:
        return "4m_ntkmirror_activation_gate_probe"
    if int(getattr(args, "candidate_top_k", 0) or 0) > 0:
        return "4j_two_pass_candidate_pruning"
    if getattr(args, "candidate_score_mode", "composed_gain") == "composed_gain_orthogonal":
        return "4i_residual_orthogonal_routing"
    if int(getattr(args, "post_first_stage_size", 0) or 0) > 0:
        return "4h_fine_grained_second_stage"
    grid_args = (args.candidate_learning_rates, args.candidate_init_scales, args.candidate_activations)
    return "4g_candidate_grid_routed_grafts" if any(grid_args) else "4f_validation_routed_staged_grafts"


def candidate_rank(row: dict[str, Any], min_gain: float) -> tuple[int, float]:
    positive = row["candidate_composed_gain"] > float(min_gain)
    return (1 if positive else 0, float(row["candidate_score"]))


def ntk_feature_map(ntk_rows: list[dict[str, Any]], previous_ntk_by_target: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
    previous_ntk_by_target = previous_ntk_by_target or {}
    result = {}
    for row in ntk_rows:
        target = str(row.get("target"))
        raw = float(row.get("ntk_activation_score", 0.0) or 0.0)
        previous = previous_ntk_by_target.get(target)
        delta = None if previous is None else raw - float(previous)
        result[target] = {
            "ntk_activation_score": raw,
            "ntk_rank": int(row.get("ntk_rank", 999)),
            "previous_ntk_activation_score": previous,
            "ntk_delta_from_previous_stage": delta,
            "ntk_delta_abs": None if delta is None else abs(delta),
        }
    return result


def candidate_score(
    args,
    target: str,
    gain: float,
    accepted_target_map: dict[int, str],
    *,
    ntk_features: dict[str, Any] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    mode = getattr(args, "candidate_score_mode", "composed_gain")
    overlap = sum(1 for accepted_target in accepted_target_map.values() if accepted_target == target)
    orthogonal_penalty = float(getattr(args, "orthogonal_penalty", 0.0)) * float(overlap)
    if mode == "composed_gain":
        return float(gain), 0.0, {}
    if mode == "composed_gain_orthogonal":
        return float(gain) - orthogonal_penalty, orthogonal_penalty, {}
    if mode != "composed_gain_ntk_hybrid_conservative":
        return float(gain), 0.0, {}
    features = dict(ntk_features or {})
    raw_ntk = float(features.get("ntk_activation_score", 0.0) or 0.0)
    delta_abs = float(features.get("ntk_delta_abs", 0.0) or 0.0)
    saturation_adjusted = raw_ntk / (1.0 + float(overlap))
    anti_saturation_penalty = float(getattr(args, "ntk_hybrid_anti_saturation_penalty", 0.0)) * float(overlap)
    ntk_bonus = (
        float(getattr(args, "ntk_hybrid_saturation_weight", 0.0)) * saturation_adjusted
        + float(getattr(args, "ntk_hybrid_residual_delta_weight", 0.0)) * delta_abs
    )
    penalty = orthogonal_penalty + anti_saturation_penalty
    details = {
        **features,
        "accepted_grafts_on_target_before_stage": overlap,
        "saturation_adjusted_ntk": saturation_adjusted,
        "ntk_hybrid_bonus": ntk_bonus,
        "ntk_hybrid_penalty": anti_saturation_penalty,
    }
    return float(gain) - penalty + ntk_bonus, penalty, details


def select_stage_candidates(
    probe_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    args,
    *,
    min_gain: float,
    ntk_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ranked = sorted(probe_rows, key=lambda item: candidate_rank(item[0], min_gain), reverse=True)
    selected = [candidate for _row, candidate in ranked[: int(getattr(args, "candidate_top_k", 0) or 0)]]
    selected_tags = {str(candidate.get("tag")) for candidate in selected}
    if getattr(args, "candidate_score_mode", "composed_gain") != "composed_gain_ntk_hybrid_conservative":
        return selected
    keep_ranks = int(getattr(args, "ntk_hybrid_keep_ranks", 0) or 0)
    keep_targets = {str(row.get("target")) for row in ntk_rows or [] if int(row.get("ntk_rank", 999)) <= keep_ranks}
    for target in keep_targets:
        matches = [item for item in ranked if str(item[1].get("target")) == target]
        if not matches:
            continue
        candidate = matches[0][1]
        tag = str(candidate.get("tag"))
        if tag not in selected_tags:
            selected.append(candidate)
            selected_tags.add(tag)
    return selected


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Phase 16 {summary['marco']}",
        "",
        f"- base_loss: {summary['base_loss']:.6f}",
        f"- composed_loss: {summary['composed_loss']:.6f}",
        f"- accumulated_gain: {summary['accumulated_gain']:.6f}",
        f"- accepted_groups: {summary['accepted_groups']}",
        f"- accepted_grafts: {summary['accepted_grafts']}",
        "",
        "| stage | target | lr | init_scale | activation | decision | gain | best |",
        "|---:|---|---:|---:|---|---|---:|---:|",
    ]
    for row in summary["stage_metrics"]:
        lines.append(
            "| {stage} | {selected_target} | {learning_rate:.2e} | "
            "{init_scale:.2e} | {activation} | {decision} | "
            "{stage_gain:.6f} | {stage_best_loss:.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


__all__ = ["candidate_score", "candidate_rank", "marco_name", "markdown", "ntk_feature_map", "select_stage_candidates"]
