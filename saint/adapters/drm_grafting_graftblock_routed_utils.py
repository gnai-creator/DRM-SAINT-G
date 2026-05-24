"""Small utilities for validation-routed DRM-G graftblock runs."""

from __future__ import annotations

from typing import Any


def marco_name(args) -> str:
    if int(getattr(args, "ntk_activation_probe_batches", 0) or 0) > 0:
        return "4m_ntkmirror_activation_gate_probe"
    if int(getattr(args, "candidate_top_k", 0) or 0) > 0:
        return "4j_two_pass_candidate_pruning"
    if getattr(args, "candidate_score_mode", "composed_gain") == "composed_gain_orthogonal":
        return "4i_residual_orthogonal_routing"
    if int(getattr(args, "post_first_stage_size", 0) or 0) > 0:
        return "4h_fine_grained_second_stage"
    grid_args = (
        args.candidate_learning_rates,
        args.candidate_init_scales,
        args.candidate_activations,
    )
    return "4g_candidate_grid_routed_grafts" if any(grid_args) else "4f_validation_routed_staged_grafts"


def candidate_score(args, target: str, gain: float, accepted_target_map: dict[int, str]) -> tuple[float, float]:
    if getattr(args, "candidate_score_mode", "composed_gain") != "composed_gain_orthogonal":
        return float(gain), 0.0
    overlap = sum(1 for accepted_target in accepted_target_map.values() if accepted_target == target)
    penalty = float(getattr(args, "orthogonal_penalty", 0.0)) * float(overlap)
    return float(gain) - penalty, penalty


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


__all__ = ["candidate_score", "marco_name", "markdown"]
