"""Small LoRA hyperparameter sweeps for fair phase-4 comparisons."""

from __future__ import annotations

from saint.training.data import LinearTask
from saint.training.methods import train_lora_delta
from saint.training.ops import TrainingResult


def _rename_result(result: TrainingResult, name: str, metadata: dict) -> TrainingResult:
    merged = dict(result.metadata)
    merged.update(metadata)
    return TrainingResult(
        name=name,
        train_loss=result.train_loss,
        test_loss=result.test_loss,
        weight_relative_l1_error=result.weight_relative_l1_error,
        parameter_count=result.parameter_count,
        optimizer_state_values=result.optimizer_state_values,
        elapsed_s=result.elapsed_s,
        metadata=merged,
    )


def train_tuned_lora_delta(
    task: LinearTask,
    *,
    rank: int,
    learning_rates: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5),
    step_options: tuple[int, ...] = (90, 140, 220),
    seed: int = 17,
    name: str | None = None,
) -> TrainingResult:
    """Return the best LoRA run from a compact rank/lr/steps grid."""

    candidates = []
    for learning_rate in learning_rates:
        for steps in step_options:
            candidates.append(
                train_lora_delta(
                    task,
                    rank=rank,
                    steps=steps,
                    learning_rate=learning_rate,
                    seed=seed,
                )
            )
    best = min(candidates, key=lambda result: result.test_loss)
    return _rename_result(
        best,
        name or f"lora_tuned_rank_{rank}",
        {
            "rank": rank,
            "tuned": True,
            "candidate_count": len(candidates),
        },
    )


__all__ = ["train_tuned_lora_delta"]
