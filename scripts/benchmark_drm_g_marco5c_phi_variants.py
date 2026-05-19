"""Benchmark DRM-G A Phi B parametrization variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

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


TARGET = "blocks.1.attn.out_proj"


def _metadata(seed: int, validation_batches: int, batch_size: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "baseline_config": "configs/baselines/small_3.5M.yaml",
        "device": "cpu",
        "batch_size": batch_size,
        "seq_len": 8,
        "learning_rate": 0.005,
        "use_real_tokens": True,
        "real_data_dir": "data/baseline",
        "validation_split": "val",
        "validation_batches": validation_batches,
        "data_seed": seed,
        "validation_seed": 3000 + seed,
    }


def _mean_eval(torch, model, drm_config, metadata: dict[str, Any]) -> float:
    total = 0.0
    device = str(metadata.get("device", "cpu"))
    batches = max(1, int(metadata.get("validation_batches", 1)))
    model.eval()
    for index in range(batches):
        local = dict(metadata)
        split = str(local.get("validation_split", "val"))
        local[f"{split}_token_offset"] = int(local.get(f"{split}_token_offset", 0)) + index * 4096
        inputs, targets = _tokens(torch, local, drm_config.vocab_size, device, seed_key="validation_seed")
        total += float(_loss(model, inputs, targets).detach().cpu().item())
    return total / batches


def _base_loss(torch, model_cls, drm_config, metadata: dict[str, Any]) -> float:
    device = str(metadata.get("device", "cpu"))
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    _freeze(model)
    return _mean_eval(torch, model, drm_config, metadata)


def _train_variant(
    torch,
    model_cls,
    drm_config,
    metadata: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    device = str(metadata.get("device", "cpu"))
    seed = int(metadata["seed"])
    torch.manual_seed(seed)
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    model.eval()
    inputs, targets = _tokens(torch, metadata, drm_config.vocab_size, device)
    module = _target_module(model, TARGET)
    activation, gradient = capture_activation_gradient(torch, model, module, _loss, inputs, targets)
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
    start = perf_counter()
    try:
        for _ in range(max(1, int(variant.get("steps", 8)))):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, inputs, targets)
            loss.backward()
            optimizer.step()
        final = _mean_eval(torch, model, drm_config, metadata)
    finally:
        handle.remove()
    return {
        "method": str(variant["name"]),
        "seed": seed,
        "target_module": TARGET,
        "final_loss": final,
        "trainable_parameters": graft.parameter_count(),
        "train_s": perf_counter() - start,
        "rank": int(variant["rank"]),
        "init": str(variant["init"]),
        "train_ab": bool(variant.get("train_ab", False)),
        "residual_k": int(variant.get("residual_k", 0)),
        "steps": int(variant.get("steps", 8)),
    }


def _train_full_module(torch, model_cls, drm_config, metadata: dict[str, Any], steps: int) -> dict[str, Any]:
    device = str(metadata.get("device", "cpu"))
    torch.manual_seed(int(metadata["seed"]))
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    model.eval()
    _freeze(model)
    module = _target_module(model, TARGET)
    module.weight.requires_grad_(True)
    params = [module.weight]
    inputs, targets = _tokens(torch, metadata, drm_config.vocab_size, device)
    optimizer = torch.optim.AdamW(params, lr=float(metadata.get("learning_rate", 0.005)))
    start = perf_counter()
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
    return {
        "method": "full_module_linear",
        "seed": int(metadata["seed"]),
        "target_module": TARGET,
        "final_loss": _mean_eval(torch, model, drm_config, metadata),
        "trainable_parameters": int(module.weight.numel()),
        "train_s": perf_counter() - start,
        "rank": None,
        "init": "full",
        "train_ab": True,
        "residual_k": 0,
        "steps": steps,
    }


def _variants(steps: int) -> list[dict[str, Any]]:
    return [
        {"name": "phi_zero_4096", "rank": 64, "init": "zero", "steps": steps},
        {"name": "phi_ls_4096", "rank": 64, "init": "least_squares_gradient", "steps": steps},
        {
            "name": "phi_ls_residual_4096",
            "rank": 63,
            "init": "least_squares_gradient",
            "residual_k": 127,
            "steps": steps,
        },
        {
            "name": "phi_ls_train_ab",
            "rank": 32,
            "init": "least_squares_gradient",
            "train_ab": True,
            "steps": steps,
            "lr": 0.002,
        },
    ]


def _run_seed(seed: int, args) -> list[dict[str, Any]]:
    metadata = _metadata(seed, args.validation_batches, args.batch_size)
    torch, config_cls, model_cls, _drm_root = _import_drm(metadata)
    drm_config = load_drm_baseline_config(metadata, config_cls)
    base = _base_loss(torch, model_cls, drm_config, metadata)
    rows = []
    for variant in _variants(args.steps):
        row = _train_variant(torch, model_cls, drm_config, metadata, variant)
        row["base_loss"] = base
        row["validation_gain"] = base - row["final_loss"]
        row["gain_per_parameter"] = row["validation_gain"] / max(1, row["trainable_parameters"])
        rows.append(row)
    full = _train_full_module(torch, model_cls, drm_config, metadata, args.steps)
    full["base_loss"] = base
    full["validation_gain"] = base - full["final_loss"]
    full["gain_per_parameter"] = full["validation_gain"] / max(1, full["trainable_parameters"])
    rows.append(full)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_gain = max(rows, key=lambda row: row["validation_gain"])
    best_phi = max(
        [row for row in rows if row["method"] != "full_module_linear"],
        key=lambda row: row["validation_gain"],
    )
    full_best = max(
        [row for row in rows if row["method"] == "full_module_linear"],
        key=lambda row: row["validation_gain"],
    )
    return {
        "row_count": len(rows),
        "best_by_gain": best_by_gain,
        "best_phi": best_phi,
        "best_full_module": full_best,
        "phi_beats_full_module": best_phi["validation_gain"] > full_best["validation_gain"],
    }


def _markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# DRM-G Marco 5C Phi Variants",
        "",
        f"- phi_beats_full_module: {summary['phi_beats_full_module']}",
        "",
        "| method | seed | gain | gain/param | params | steps | train_s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["validation_gain"], reverse=True):
        lines.append(
            "| {method} | {seed} | {validation_gain:.6f} | {gain_per_parameter:.6e} | "
            "{trainable_parameters} | {steps} | {train_s:.3f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/drm_g_marco5c_phi_variants")
    parser.add_argument("--seeds", nargs="*", type=int, default=[31, 32, 33, 34])
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        rows.extend(_run_seed(seed, args))
    summary = _summary(rows)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(rows, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
