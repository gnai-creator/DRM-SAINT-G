"""Phase 16 Marco 4: graft DRM 5M on the multilingual 125M dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import benchmark_drm_g_marco5c_phi_variants as phi_bench
from saint.adapters.drm_grafting import (
    _freeze,
    _import_drm,
    _load_optional_state,
    _loss,
    _target_module,
    _tokens,
    load_drm_baseline_config,
)
from saint.adapters.drm_grafting_phi_variants import (
    capture_activation_gradient,
    make_phi_variant,
)


DEFAULT_CONFIG = "configs/scaling/multilingual/5m.yaml"
DEFAULT_CHECKPOINT = "checkpoints/multilingual_5m/smoke_819k/final.pt"
DEFAULT_DATA = "data/multilingual_125m"


def _metadata(args, seed: int, learning_rate: float) -> dict[str, Any]:
    return {
        "seed": seed,
        "baseline_config": args.baseline_config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": learning_rate,
        "use_real_tokens": True,
        "real_data_dir": args.data_dir,
        "validation_split": "val",
        "validation_batches": args.validation_batches,
        "data_seed": seed,
        "validation_seed": args.validation_seed_offset + seed,
    }


def _variants(d_model: int, steps: int) -> list[dict[str, Any]]:
    return [
        {"name": "phi_zero_full_rank", "rank": d_model, "init": "zero", "steps": steps},
        {
            "name": "phi_ls_full_rank",
            "rank": d_model,
            "init": "least_squares_gradient",
            "steps": steps,
        },
        {
            "name": "phi_ls_residual_full_rank",
            "rank": max(1, d_model - 1),
            "init": "least_squares_gradient",
            "residual_k": max(1, (d_model * 2) - 1),
            "steps": steps,
        },
        {
            "name": "phi_ls_train_ab_half_rank",
            "rank": max(1, d_model // 2),
            "init": "least_squares_gradient",
            "train_ab": True,
            "steps": steps,
            "lr": 0.002,
        },
    ]


def _batch(metadata: dict[str, Any], index: int) -> dict[str, Any]:
    local = dict(metadata)
    local["train_token_offset"] = int(metadata.get("train_token_offset", 0)) + index * 4096
    return local


def _train_variant(torch, model_cls, drm_config, metadata, variant, target):
    device = str(metadata.get("device", "cpu"))
    torch.manual_seed(int(metadata["seed"]))
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    model.eval()
    init_inputs, init_targets = _tokens(
        torch,
        _batch(metadata, 0),
        drm_config.vocab_size,
        device,
    )
    module = _target_module(model, target)
    activation, gradient = capture_activation_gradient(
        torch,
        model,
        module,
        _loss,
        init_inputs,
        init_targets,
    )
    _freeze(model)
    graft = make_phi_variant(
        torch,
        activation,
        gradient,
        int(variant["rank"]),
        init=str(variant["init"]),
        train_ab=bool(variant.get("train_ab", False)),
        residual_k=int(variant.get("residual_k", 0)),
        step_scale=float(variant.get("step_scale", metadata.get("learning_rate", 0.005))),
    ).to(device)
    handle = module.register_forward_hook(graft.hook)
    optimizer = torch.optim.AdamW(
        graft.parameters(),
        lr=float(variant.get("lr", metadata.get("learning_rate", 0.005))),
        weight_decay=float(variant.get("weight_decay", 0.0)),
    )
    from time import perf_counter

    start = perf_counter()
    try:
        for step in range(max(1, int(variant.get("steps", 8)))):
            local = _batch(metadata, step % max(1, int(metadata.get("train_batches", 1))))
            inputs, targets = _tokens(torch, local, drm_config.vocab_size, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, inputs, targets)
            loss.backward()
            optimizer.step()
        final = phi_bench._mean_eval(torch, model, drm_config, metadata)
    finally:
        handle.remove()
    return {
        "method": str(variant["name"]),
        "seed": int(metadata["seed"]),
        "target_module": target,
        "final_loss": final,
        "trainable_parameters": graft.parameter_count(),
        "train_s": perf_counter() - start,
        "rank": int(variant["rank"]),
        "init": str(variant["init"]),
        "train_ab": bool(variant.get("train_ab", False)),
        "residual_k": int(variant.get("residual_k", 0)),
        "steps": int(variant.get("steps", 8)),
    }


def _train_full_module(torch, model_cls, drm_config, metadata, target, steps):
    device = str(metadata.get("device", "cpu"))
    torch.manual_seed(int(metadata["seed"]))
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    model.eval()
    _freeze(model)
    module = _target_module(model, target)
    module.weight.requires_grad_(True)
    optimizer = torch.optim.AdamW([module.weight], lr=float(metadata.get("learning_rate", 0.005)))
    from time import perf_counter

    start = perf_counter()
    for step in range(max(1, steps)):
        local = _batch(metadata, step % max(1, int(metadata.get("train_batches", 1))))
        inputs, targets = _tokens(torch, local, drm_config.vocab_size, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
    return {
        "method": "full_module_linear",
        "seed": int(metadata["seed"]),
        "target_module": target,
        "final_loss": phi_bench._mean_eval(torch, model, drm_config, metadata),
        "trainable_parameters": int(module.weight.numel()),
        "train_s": perf_counter() - start,
        "rank": None,
        "init": "full",
        "train_ab": True,
        "residual_k": 0,
        "steps": steps,
    }


def _run_one(args, seed: int, target: str, learning_rate: float) -> list[dict[str, Any]]:
    metadata = _metadata(args, seed, learning_rate)
    torch, config_cls, model_cls, _drm_root = _import_drm(metadata)
    drm_config = load_drm_baseline_config(metadata, config_cls)
    base = phi_bench._base_loss(torch, model_cls, drm_config, metadata)
    rows = []
    for variant in _variants(int(drm_config.d_model), args.steps):
        row = _train_variant(torch, model_cls, drm_config, metadata, variant, target)
        row["base_loss"] = base
        row["validation_gain"] = base - row["final_loss"]
        row["gain_per_parameter"] = row["validation_gain"] / max(1, row["trainable_parameters"])
        row["d_model"] = int(drm_config.d_model)
        row["checkpoint"] = args.checkpoint
        row["learning_rate"] = learning_rate
        rows.append(row)
    full = _train_full_module(torch, model_cls, drm_config, metadata, target, args.steps)
    full["base_loss"] = base
    full["validation_gain"] = base - full["final_loss"]
    full["gain_per_parameter"] = full["validation_gain"] / max(1, full["trainable_parameters"])
    full["d_model"] = int(drm_config.d_model)
    full["checkpoint"] = args.checkpoint
    full["learning_rate"] = learning_rate
    rows.append(full)
    return rows


def _run_seed(args, seed: int) -> list[dict[str, Any]]:
    rows = []
    for target in args.targets:
        for learning_rate in args.learning_rates:
            rows.extend(_run_one(args, seed, target, learning_rate))
    return rows


def _method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            "mean_base_loss": sum(row["base_loss"] for row in subset) / len(subset),
            "mean_final_loss": sum(row["final_loss"] for row in subset) / len(subset),
            "mean_gain": sum(row["validation_gain"] for row in subset) / len(subset),
            "mean_gain_per_parameter": sum(row["gain_per_parameter"] for row in subset) / len(subset),
            "positive_runs": sum(1 for row in subset if row["validation_gain"] > 0.0),
            "run_count": len(subset),
            "params": int(subset[0]["trainable_parameters"]),
        })
    summary.sort(key=lambda row: row["mean_gain"], reverse=True)
    return summary


def _summary(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    methods = _method_summary(rows)
    best = max(rows, key=lambda row: row["validation_gain"])
    phi_rows = [row for row in rows if row["method"] != "full_module_linear"]
    best_phi = max(phi_rows, key=lambda row: row["validation_gain"])
    best_full = max(
        [row for row in rows if row["method"] == "full_module_linear"],
        key=lambda row: row["validation_gain"],
    )
    return {
        "phase": "16",
        "marco": "4_grafted_5m_budget_target",
        "baseline_config": args.baseline_config,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "targets": args.targets,
        "learning_rates": args.learning_rates,
        "train_batches": args.train_batches,
        "row_count": len(rows),
        "method_summary": methods,
        "best": best,
        "best_phi": best_phi,
        "best_full_module": best_full,
        "phi_best_beats_full_best": best_phi["validation_gain"] > best_full["validation_gain"],
        "any_phi_positive": any(row["validation_gain"] > 0.0 for row in phi_rows),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 16 Marco 4 - DRM 5M Grafted",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- data_dir: `{summary['data_dir']}`",
        f"- any_phi_positive: {summary['any_phi_positive']}",
        f"- phi_best_beats_full_best: {summary['phi_best_beats_full_best']}",
        "",
        "| method | mean base | mean final | mean gain | gain/param | positive | params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_summary"]:
        lines.append(
            "| {method} | {mean_base_loss:.6f} | {mean_final_loss:.6f} | "
            "{mean_gain:.6f} | {mean_gain_per_parameter:.6e} | "
            "{positive_runs}/{run_count} | {params} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/phase16_marco4_5m_graft")
    parser.add_argument("--baseline-config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--validation-seed-offset", type=int, default=5000)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rates", nargs="*", type=float, default=[0.005])
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["blocks.1.attn.out_proj"],
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rows.extend(_run_seed(args, seed))
    summary = _summary(rows, args)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
