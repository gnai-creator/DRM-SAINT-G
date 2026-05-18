"""Success criteria for phase-4 linear experiments."""

from __future__ import annotations

from dataclasses import dataclass

from saint.training.sweeps import summarize_phase4_rows


@dataclass(frozen=True)
class Phase4Decision:
    passed: bool
    reason: str
    saint_method: str
    compared_method: str
    saint_avg_test_loss: float
    compared_avg_test_loss: float
    saint_avg_parameter_count: float
    compared_avg_parameter_count: float
    saint_avg_gain_per_parameter: float
    compared_avg_gain_per_parameter: float


def evaluate_phase4_success(
    summaries: list[dict],
    *,
    saint_method: str = "saint_routed_f25_c50",
    compared_method: str = "lora_rank_2",
    max_loss_ratio: float = 1.0,
    max_parameter_ratio: float = 2.0,
    min_gain_per_parameter_ratio: float = 1.0,
) -> Phase4Decision:
    by_method = {summary["method"]: summary for summary in summaries}
    saint = by_method[saint_method]
    compared = by_method[compared_method]
    loss_ratio = saint["avg_test_loss"] / compared["avg_test_loss"]
    parameter_ratio = saint["avg_parameter_count"] / compared["avg_parameter_count"]
    gain_ratio = (
        saint["avg_gain_per_parameter"] / compared["avg_gain_per_parameter"]
        if compared["avg_gain_per_parameter"] > 0
        else float("inf")
    )
    passed = (
        loss_ratio <= max_loss_ratio
        and parameter_ratio <= max_parameter_ratio
        and gain_ratio >= min_gain_per_parameter_ratio
    )
    reason = (
        "passed"
        if passed
        else (
            "failed thresholds: "
            f"loss_ratio={loss_ratio:.4f} max={max_loss_ratio:.4f}, "
            f"parameter_ratio={parameter_ratio:.4f} max={max_parameter_ratio:.4f}, "
            f"gain_per_parameter_ratio={gain_ratio:.4f} min={min_gain_per_parameter_ratio:.4f}"
        )
    )
    return Phase4Decision(
        passed=passed,
        reason=reason,
        saint_method=saint_method,
        compared_method=compared_method,
        saint_avg_test_loss=saint["avg_test_loss"],
        compared_avg_test_loss=compared["avg_test_loss"],
        saint_avg_parameter_count=saint["avg_parameter_count"],
        compared_avg_parameter_count=compared["avg_parameter_count"],
        saint_avg_gain_per_parameter=saint["avg_gain_per_parameter"],
        compared_avg_gain_per_parameter=compared["avg_gain_per_parameter"],
    )


def evaluate_phase4_regime_success(
    rows: list[dict],
    *,
    saint_method: str = "saint_routed_f50_c25",
    compared_method: str = "lora_rank_2",
) -> list[dict]:
    regimes = sorted(
        {
            (row["rows"], row["cols"], row["delta_mode"])
            for row in rows
        }
    )
    decisions = []
    for regime_rows, regime_cols, delta_mode in regimes:
        group = [
            row
            for row in rows
            if row["rows"] == regime_rows
            and row["cols"] == regime_cols
            and row["delta_mode"] == delta_mode
        ]
        decision = evaluate_phase4_success(
            summarize_phase4_rows(group),
            saint_method=saint_method,
            compared_method=compared_method,
        )
        decisions.append(
            {
                **decision.__dict__,
                "rows": regime_rows,
                "cols": regime_cols,
                "delta_mode": delta_mode,
            }
        )
    return decisions


def evaluate_phase4_closure(
    rows: list[dict],
    *,
    saint_method: str = "saint_dynamic_delta",
) -> dict:
    """Evaluate the stricter phase-4 closure rule across regimes."""

    lora_rank_2 = evaluate_phase4_regime_success(
        rows,
        saint_method=saint_method,
        compared_method="lora_tuned_rank_2",
    )
    lora_rank_4 = evaluate_phase4_regime_success(
        rows,
        saint_method=saint_method,
        compared_method="lora_tuned_rank_4",
    )
    block_budgeted = evaluate_phase4_regime_success(
        rows,
        saint_method=saint_method,
        compared_method=f"block_budgeted_delta_for_{saint_method}",
    )
    full_budgeted = evaluate_phase4_regime_success(
        rows,
        saint_method=saint_method,
        compared_method=f"budgeted_full_delta_for_{saint_method}",
    )
    block_passes = sum(1 for decision in block_budgeted if decision["passed"])
    full_passes = sum(1 for decision in full_budgeted if decision["passed"])
    passed = (
        all(decision["passed"] for decision in lora_rank_2)
        and all(decision["passed"] for decision in lora_rank_4)
        and block_passes >= 1
        and full_passes >= 2
    )
    return {
        "passed": passed,
        "saint_method": saint_method,
        "lora_rank_2_all_passed": all(decision["passed"] for decision in lora_rank_2),
        "lora_rank_4_all_passed": all(decision["passed"] for decision in lora_rank_4),
        "block_budgeted_passes": block_passes,
        "budgeted_full_delta_passes": full_passes,
        "required_budgeted_full_delta_passes": 2,
        "reason": "passed" if passed else "phase 4 closure thresholds not met",
    }
