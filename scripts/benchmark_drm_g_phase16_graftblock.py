"""Phase 16 Marco 4B: scale DRM 5M with repeatable 5M graft blocks."""

from __future__ import annotations

import argparse, sys
import json
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_drm_g_marco5c_phi_variants as phi_bench
from saint.adapters.drm_grafting import (
    _freeze,
    _import_drm,
    _load_optional_state,
    _loss,
    _tokens,
    load_drm_baseline_config,
)
from saint.adapters.drm_grafting_graftblock import (
    attach_graft_blocks,
    graft_checkpoint_payload,
    make_graft_blocks,
    plan_graft_blocks,
    set_progressive_state,
)

DEFAULT_CONFIG = "configs/scaling/multilingual/5m.yaml"
DEFAULT_CHECKPOINT = "checkpoints/multilingual_5m/smoke_819k/final.pt"
DEFAULT_DATA = "data/multilingual_125m"

def _metadata(args, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "baseline_config": args.baseline_config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "use_real_tokens": True,
        "real_data_dir": args.data_dir,
        "validation_split": "val",
        "validation_batches": args.validation_batches,
        "data_seed": seed,
        "validation_seed": args.validation_seed_offset + seed,
    }


def _batch(metadata: dict[str, Any], index: int, *, split: str = "train") -> dict[str, Any]:
    local = dict(metadata)
    key = "train_token_offset" if split == "train" else f"{split}_token_offset"
    local[key] = int(local.get(key, 0)) + int(index) * 4096
    return local


def _base_params(model) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _cuda_peak(torch, device: str) -> int | None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _reset_cuda(torch, device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _plan_rows(base_parameters: int, target_total: int, d_model: int) -> list[dict[str, int]]:
    rows = []
    for graft_count in (1, 2, 4, 8, 16, 24):
        plan = plan_graft_blocks(
            base_parameters=base_parameters,
            target_total_parameters=target_total,
            d_model=d_model,
            graft_count=graft_count,
        )
        rows.append(plan.__dict__)
    return rows


def _optimizer(torch, grafts, args):
    groups = []
    for index, graft in enumerate(grafts):
        lr = args.learning_rate / (1.0 + (index * args.lr_decay))
        groups.append({"params": list(graft.parameters()), "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _active_count(args, step: int, graft_count: int) -> int:
    if args.training_mode != "progressive":
        return graft_count
    if args.steps <= 1:
        return graft_count
    return max(1, min(graft_count, 1 + (step * graft_count) // args.steps))


def _save_artifact(
    torch,
    out_dir: Path,
    grafts,
    args,
    row: dict[str, Any],
    *,
    prefix: str = "graftblock",
) -> Path:
    artifact = out_dir / f"{prefix}_g{row['graft_count']}_seed{row['seed']}.pt"
    payload = graft_checkpoint_payload(
        grafts=grafts,
        target_modules=args.targets,
        metadata={
            "baseline_config": args.baseline_config,
            "checkpoint": args.checkpoint,
            "data_dir": args.data_dir,
            "row": row,
        },
    )
    torch.save(payload, artifact)
    return artifact


def _append_metric(out_dir: Path, row: dict[str, Any]) -> None:
    path = out_dir / "training_metrics.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _early_stop(best_loss: float, eval_loss: float, bad_evals: int, args) -> tuple[float, int, bool]:
    improved = eval_loss < (best_loss - args.early_stopping_min_delta)
    if improved:
        return eval_loss, 0, False
    bad_evals += 1
    should_stop = args.early_stopping_patience > 0 and bad_evals >= args.early_stopping_patience
    return best_loss, bad_evals, should_stop


def _recomposed_loss(torch, config_cls, model_cls, artifact: Path, metadata: dict[str, Any]) -> float:
    payload = torch.load(str(artifact), map_location="cpu", weights_only=False)
    row = payload["metadata"]["row"]
    device = str(metadata.get("device", "cpu"))
    model = model_cls(load_drm_baseline_config(metadata, config_cls)).to(device)
    _load_optional_state(model, metadata, torch)
    _freeze(model)
    grafts = make_graft_blocks(
        torch,
        d_model=int(row["d_model"]),
        hidden_size=int(row["hidden_size"]),
        graft_count=int(row["graft_count"]),
        seed=int(row["seed"]),
        init_scale=1.0,
        activation=str(row["activation"]),
        device=device,
    )
    for graft, state in zip(grafts, payload["grafts"]):
        graft.load_state_dict(state, device)
    handles = attach_graft_blocks(model, payload["target_modules"], grafts)
    try:
        return phi_bench._mean_eval(torch, model, model.config, metadata)
    finally:
        for handle in handles:
            handle.remove()


def _train_one(args, seed: int, graft_count: int, out_dir: Path) -> dict[str, Any]:
    metadata = _metadata(args, seed)
    torch, config_cls, model_cls, _root = _import_drm(metadata)
    device = args.device
    torch.manual_seed(seed)
    _reset_cuda(torch, device)
    model = model_cls(load_drm_baseline_config(metadata, config_cls)).to(device)
    _load_optional_state(model, metadata, torch)
    drm_config = model.config
    _freeze(model)
    base_parameters = _base_params(model)
    plan = plan_graft_blocks(
        base_parameters=base_parameters,
        target_total_parameters=args.target_total_parameters,
        d_model=int(drm_config.d_model),
        graft_count=graft_count,
    )
    hidden_size = args.hidden_size or plan.hidden_size
    grafts = make_graft_blocks(
        torch,
        d_model=int(drm_config.d_model),
        hidden_size=hidden_size,
        graft_count=graft_count,
        seed=seed,
        init_scale=args.init_scale,
        activation=args.activation,
        device=device,
    )
    handles = attach_graft_blocks(model, args.targets, grafts)
    optimizer = _optimizer(torch, grafts, args)
    base_loss = phi_bench._mean_eval(torch, model, drm_config, metadata)
    start = perf_counter()
    trained_steps = 0
    last_eval_loss = base_loss
    best_eval_loss = base_loss
    best_eval_step = 0
    best_eval_elapsed = 0.0
    best_artifact: Path | None = None
    bad_evals = 0
    stopped_early = False
    try:
        max_steps = max(1, args.steps)
        for step in range(max_steps):
            active = _active_count(args, step, graft_count)
            set_progressive_state(grafts, active, args.scale_warmup_grafts)
            local = _batch(metadata, step % max(1, args.train_batches))
            inputs, targets = _tokens(torch, local, drm_config.vocab_size, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, inputs, targets)
            loss.backward()
            optimizer.step()
            trained_steps = step + 1
            elapsed = perf_counter() - start
            if args.eval_every_steps and trained_steps % args.eval_every_steps == 0:
                set_progressive_state(grafts, graft_count, 1)
                last_eval_loss = phi_bench._mean_eval(torch, model, drm_config, metadata)
                best_eval_loss, bad_evals, stopped_early = _early_stop(
                    best_eval_loss,
                    last_eval_loss,
                    bad_evals,
                    args,
                )
                if bad_evals == 0:
                    best_eval_step = trained_steps
                    best_eval_elapsed = elapsed
                    if args.save_best_checkpoint:
                        preview = {
                            "seed": seed,
                            "graft_count": graft_count,
                            "d_model": int(drm_config.d_model),
                            "hidden_size": hidden_size,
                            "activation": args.activation,
                            "eval_loss": last_eval_loss,
                            "step": trained_steps,
                        }
                        best_artifact = _save_artifact(
                            torch,
                            out_dir,
                            grafts,
                            args,
                            preview,
                            prefix="best_graftblock",
                        )
                _append_metric(out_dir, {
                    "seed": seed,
                    "graft_count": graft_count,
                    "step": trained_steps,
                    "elapsed_s": elapsed,
                    "eval_loss": last_eval_loss,
                    "validation_gain": base_loss - last_eval_loss,
                    "best_eval_loss": best_eval_loss,
                    "bad_evals": bad_evals,
                    "stopped_early": stopped_early,
                })
                if stopped_early:
                    break
            if args.max_train_seconds > 0 and elapsed >= args.max_train_seconds:
                break
        set_progressive_state(grafts, graft_count, 1)
        train_s = perf_counter() - start
        final_loss = phi_bench._mean_eval(torch, model, drm_config, metadata)
    finally:
        for handle in handles:
            handle.remove()
    trainable = sum(graft.parameter_count() for graft in grafts)
    row = {
        "method": "drm_graftblock",
        "seed": seed,
        "base_loss": base_loss,
        "final_loss": final_loss,
        "validation_gain": base_loss - final_loss,
        "gain_per_parameter": (base_loss - final_loss) / max(1, trainable),
        "base_parameters": base_parameters,
        "trainable_parameters": trainable,
        "effective_total_parameters": base_parameters + trainable,
        "target_total_parameters": args.target_total_parameters,
        "graft_count": graft_count,
        "hidden_size": hidden_size,
        "parameters_per_graft": grafts[0].parameter_count(),
        "targets": args.targets,
        "d_model": int(drm_config.d_model),
        "activation": args.activation,
        "training_mode": args.training_mode,
        "lr_decay": args.lr_decay,
        "steps": args.steps,
        "trained_steps": trained_steps,
        "max_train_seconds": args.max_train_seconds,
        "train_batches": args.train_batches,
        "validation_batches": args.validation_batches,
        "train_s": train_s,
        "cuda_peak_bytes": _cuda_peak(torch, device),
        "full_125m_smoke_loss": args.full_125m_smoke_loss,
        "distance_to_full_125m_smoke": final_loss - args.full_125m_smoke_loss,
        "last_eval_loss": last_eval_loss,
        "best_eval_loss": best_eval_loss,
        "best_eval_gain": base_loss - best_eval_loss,
        "best_eval_step": best_eval_step,
        "best_eval_elapsed_s": best_eval_elapsed,
        "best_distance_to_full_125m_smoke": best_eval_loss - args.full_125m_smoke_loss,
        "stopped_early": stopped_early,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
    }
    if best_artifact:
        row["best_graft_checkpoint"] = str(best_artifact)
        row["best_graft_checkpoint_bytes"] = best_artifact.stat().st_size
        row["best_recomposed_loss"] = _recomposed_loss(torch, config_cls, model_cls, best_artifact, metadata)
        row["best_recompose_abs_diff"] = abs(row["best_recomposed_loss"] - row["best_eval_loss"])
    if args.save_graft_checkpoint:
        artifact = _save_artifact(torch, out_dir, grafts, args, row)
        row["graft_checkpoint"] = str(artifact)
        row["graft_checkpoint_bytes"] = artifact.stat().st_size
        row["recomposed_loss"] = _recomposed_loss(torch, config_cls, model_cls, artifact, metadata)
        row["recompose_abs_diff"] = abs(row["recomposed_loss"] - row["final_loss"])
    return row


def _summary(rows: list[dict[str, Any]], plan_rows: list[dict[str, int]], args) -> dict[str, Any]:
    best = max(rows, key=lambda row: row.get("best_eval_gain", row["validation_gain"]))
    marco = "4d_early_stopping_best_checkpoint" if args.save_best_checkpoint else "4c_5m_graftblock_training"
    return {
        "phase": "16",
        "marco": marco,
        "baseline_config": args.baseline_config,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "target_total_parameters": args.target_total_parameters,
        "plan": plan_rows,
        "rows": len(rows),
        "mean_base_loss": sum(row["base_loss"] for row in rows) / len(rows),
        "mean_final_loss": sum(row["final_loss"] for row in rows) / len(rows),
        "mean_gain": sum(row["validation_gain"] for row in rows) / len(rows),
        "mean_best_gain": sum(row.get("best_eval_gain", row["validation_gain"]) for row in rows) / len(rows),
        "mean_gain_per_parameter": sum(row["gain_per_parameter"] for row in rows) / len(rows),
        "positive_runs": sum(1 for row in rows if row["validation_gain"] > 0.0),
        "full_125m_smoke_loss": args.full_125m_smoke_loss,
        "best_distance_to_full_125m_smoke": best.get(
            "best_distance_to_full_125m_smoke",
            best["distance_to_full_125m_smoke"],
        ),
        "best": best,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 16 Marco 4D - 5M GraftBlock Best Checkpoint",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- data_dir: `{summary['data_dir']}`",
        f"- target_total_parameters: {summary['target_total_parameters']}",
        f"- mean_gain: {summary['mean_gain']:.6f}",
        f"- mean_best_gain: {summary['mean_best_gain']:.6f}",
        f"- mean_gain_per_parameter: {summary['mean_gain_per_parameter']:.6e}",
        f"- positive_runs: {summary['positive_runs']}/{summary['rows']}",
        f"- full_125m_smoke_loss: {summary['full_125m_smoke_loss']:.6f}",
        f"- best_distance_to_full_125m_smoke: {summary['best_distance_to_full_125m_smoke']:.6f}",
        "",
        "## Parameter Plan",
        "",
        "| grafts | hidden | params/graft | graft params | effective total | remaining gap |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["plan"]:
        lines.append(
            "| {graft_count} | {hidden_size} | {parameters_per_graft} | "
            "{graft_parameters} | {effective_total_parameters} | {remaining_gap} |".format(**row)
        )
    best = summary["best"]
    lines.extend([
        "",
        "## Best Smoke Run",
        "",
        f"- graft_count: {best['graft_count']}",
        f"- hidden_size: {best['hidden_size']}",
        f"- trainable_parameters: {best['trainable_parameters']}",
        f"- base_loss: {best['base_loss']:.6f}",
        f"- final_loss: {best['final_loss']:.6f}",
        f"- best_eval_loss: {best.get('best_eval_loss', best['final_loss']):.6f}",
        f"- validation_gain: {best['validation_gain']:.6f}",
        f"- best_eval_gain: {best.get('best_eval_gain', best['validation_gain']):.6f}",
        f"- distance_to_full_125m_smoke: {best['distance_to_full_125m_smoke']:.6f}",
        f"- train_s: {best['train_s']:.2f}",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/phase16_marco4b_graftblock")
    parser.add_argument("--baseline-config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--validation-seed-offset", type=int, default=7000)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--lr-decay", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-total-parameters", type=int, default=125_000_000)
    parser.add_argument("--graft-count", type=int, default=1)
    parser.add_argument("--graft-counts", nargs="*", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--activation", choices=["silu", "gelu", "relu"], default="silu")
    parser.add_argument(
        "--training-mode",
        choices=["simultaneous", "progressive", "staged", "validation_routed_staged"],
        default="simultaneous",
    )
    parser.add_argument("--scale-warmup-grafts", type=int, default=2)
    parser.add_argument("--stage-size", type=int, default=4)
    parser.add_argument("--post-first-stage-size", type=int, default=0)
    parser.add_argument("--max-stages", type=int, default=6)
    parser.add_argument("--freeze-accepted-stages", action="store_true")
    parser.add_argument("--stage-accept-min-gain", type=float, default=0.0)
    parser.add_argument("--candidate-targets", nargs="*", default=None)
    parser.add_argument("--candidate-learning-rates", nargs="*", type=float, default=None); parser.add_argument("--candidate-init-scales", nargs="*", type=float, default=None)
    parser.add_argument("--candidate-activations", nargs="*", default=None); parser.add_argument("--candidate-score-mode", choices=["composed_gain", "composed_gain_orthogonal"], default="composed_gain")
    parser.add_argument("--orthogonal-penalty", type=float, default=0.0); parser.add_argument("--candidate-probe-steps", type=int, default=0); parser.add_argument("--candidate-probe-max-train-seconds", type=float, default=0.0); parser.add_argument("--candidate-top-k", type=int, default=0)
    parser.add_argument("--full-125m-smoke-loss", type=float, default=9.049912414550782)
    parser.add_argument("--save-graft-checkpoint", action="store_true")
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=[
            "blocks.1",
            "blocks.2",
            "blocks.3",
            "blocks.4",
            "blocks.5",
        ],
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(args, args.seeds[0])
    torch, config_cls, model_cls, _root = _import_drm(metadata)
    drm_config = load_drm_baseline_config(metadata, config_cls)
    temp_model = model_cls(drm_config)
    plan_rows = _plan_rows(_base_params(temp_model), args.target_total_parameters, int(drm_config.d_model))
    if args.training_mode == "staged":
        from saint.adapters.drm_grafting_graftblock_staged import run_staged_graft_growth

        if not args.hidden_size:
            args.hidden_size = plan_graft_blocks(
                base_parameters=_base_params(temp_model),
                target_total_parameters=args.target_total_parameters,
                d_model=int(drm_config.d_model),
                graft_count=args.graft_count,
            ).hidden_size
        run_staged_graft_growth(torch, config_cls, model_cls, drm_config, metadata, args, out_dir)
        return 0
    if args.training_mode == "validation_routed_staged":
        from saint.adapters.drm_grafting_graftblock_routed import run_validation_routed_staged

        if not args.hidden_size:
            args.hidden_size = plan_graft_blocks(
                base_parameters=_base_params(temp_model),
                target_total_parameters=args.target_total_parameters,
                d_model=int(drm_config.d_model),
                graft_count=args.graft_count,
            ).hidden_size
        run_validation_routed_staged(torch, config_cls, model_cls, drm_config, metadata, args, out_dir)
        return 0
    graft_counts = args.graft_counts or [args.graft_count]
    rows = [
        _train_one(args, seed, graft_count, out_dir)
        for graft_count in graft_counts
        for seed in args.seeds
    ]
    summary = _summary(rows, plan_rows, args)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
