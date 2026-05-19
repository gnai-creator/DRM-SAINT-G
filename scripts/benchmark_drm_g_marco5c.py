"""DRM-G Marco 5C stronger full baselines."""

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
from saint.adapters.drm_grafting_full_budget import FullBudgetLinearGraft
from saint.adapters.drm_grafting_progressive import _mean_eval
from saint.config import RuntimeConfig
from saint.runtime.runner import train_runtime


LINEAR_SEQUENCE = [
    {"target_module": "blocks.1.attn.out_proj", "projection_init": "gradient"},
    {"target_module": "blocks.2.attn.out_proj", "projection_init": "gradient"},
    {"target_module": "blocks.3.attn.out_proj", "projection_init": "gradient"},
]


def _metadata(seed: int, validation_batches: int, batch_size: int) -> dict[str, Any]:
    return {
        "baseline_config": "configs/baselines/small_3.5M.yaml",
        "device": "cpu",
        "batch_size": batch_size,
        "seq_len": 8,
        "phi_rank": 8,
        "graft_scale": 1.0,
        "learning_rate": 0.005,
        "use_real_tokens": True,
        "real_data_dir": "data/baseline",
        "validation_split": "val",
        "validation_batches": validation_batches,
        "data_seed": seed,
        "validation_seed": 3000 + seed,
        "old_validation_seed": 1000 + seed,
        "require_beats_dense": True,
        "min_validation_gain": 0.0,
        "min_gain_per_parameter": 0.0,
        "defer_gain_floor": -0.00005,
        "grafts": LINEAR_SEQUENCE,
    }


def _run_saint(
    seed: int,
    out_dir: Path,
    validation_batches: int,
    batch_size: int,
    phi_rank: int,
    name: str,
) -> dict[str, Any]:
    run_dir = out_dir / f"{name}_seed{seed}"
    metadata = _metadata(seed, validation_batches, batch_size)
    metadata["phi_rank"] = phi_rank
    metadata["grafts"] = [
        {"target_module": "blocks.1.attn.out_proj", "projection_init": "gradient"}
    ]
    config = RuntimeConfig(
        experiment_name=f"drm_g_marco5c_{name}_seed{seed}",
        task="drm_transformer",
        method="drm_g_saint_phi_progressive",
        output_dir=str(run_dir),
        seed=seed,
        steps=2,
        parameter_budget=phi_rank * phi_rank,
        metadata=metadata,
    )
    manifest = train_runtime(config)
    meta = manifest["metadata"]
    return {
        "method": name,
        "seed": seed,
        "target_module": "blocks.1.attn.out_proj",
        "base_loss": meta["base_loss"],
        "final_loss": meta["final_loss"],
        "validation_gain": meta["sequence_gain"],
        "gain_per_parameter": meta["sequence_gain_per_parameter"],
        "trainable_parameters": manifest["parameter_count"],
        "attempted_parameter_budget": phi_rank * phi_rank,
        "checkpoint_bytes": sum(int(item.get("bytes", 0)) for item in manifest.get("files", [])),
        "old_regression": meta["old_regression"],
        "train_s": meta["train_s"],
        "eval_s": meta["eval_s"],
        "routing_s": meta["routing_s"],
        "approved_grafts": meta["approved_grafts"],
        "run_dir": str(run_dir),
    }


def _train_full_budget(
    torch,
    model_cls,
    drm_config,
    metadata: dict[str, Any],
    target: str,
    budget: int,
    steps: int,
) -> tuple[float, int, float]:
    device = str(metadata.get("device", "cpu"))
    torch.manual_seed(int(metadata["seed"]))
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    model.eval()
    _freeze(model)
    inputs, targets = _tokens(torch, metadata, drm_config.vocab_size, device)
    module = _target_module(model, target)
    graft = FullBudgetLinearGraft(torch, module, budget).to(device)
    handle = module.register_forward_hook(graft.hook)
    optimizer = torch.optim.AdamW(graft.parameters(), lr=float(metadata.get("learning_rate", 0.005)))
    start = perf_counter()
    try:
        for _ in range(max(1, steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, inputs, targets)
            loss.backward()
            optimizer.step()
    finally:
        handle.remove()
    train_s = perf_counter() - start
    return _eval_full_budget(torch, model_cls, drm_config, metadata, target, graft), budget, train_s


def _eval_full_budget(torch, model_cls, drm_config, metadata, target: str, graft) -> float:
    device = str(metadata.get("device", "cpu"))
    total = 0.0
    batches = max(1, int(metadata.get("validation_batches", 1)))
    for index in range(batches):
        local = dict(metadata)
        split = str(local.get("validation_split", "val"))
        local[f"{split}_token_offset"] = int(local.get(f"{split}_token_offset", 0)) + index * 4096
        inputs, targets = _tokens(torch, local, drm_config.vocab_size, device, seed_key="validation_seed")
        model = model_cls(drm_config).to(device)
        _load_optional_state(model, local, torch)
        model.eval()
        _freeze(model)
        module = _target_module(model, target)
        graft.to(device)
        handle = module.register_forward_hook(graft.hook)
        try:
            total += float(_loss(model, inputs, targets).detach().cpu().item())
        finally:
            handle.remove()
    return total / batches


def _train_full_module(
    torch,
    model_cls,
    drm_config,
    metadata: dict[str, Any],
    target: str,
    steps: int,
) -> tuple[float, int, float]:
    device = str(metadata.get("device", "cpu"))
    torch.manual_seed(int(metadata["seed"]))
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    model.eval()
    _freeze(model)
    module = _target_module(model, target)
    module.weight.requires_grad_(True)
    params = [module.weight]
    if getattr(module, "bias", None) is not None:
        module.bias.requires_grad_(True)
        params.append(module.bias)
    inputs, targets = _tokens(torch, metadata, drm_config.vocab_size, device)
    optimizer = torch.optim.AdamW(params, lr=float(metadata.get("learning_rate", 0.005)))
    start = perf_counter()
    for _ in range(max(1, steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
    train_s = perf_counter() - start
    return _eval_model(torch, model, drm_config, metadata), sum(param.numel() for param in params), train_s


def _eval_model(torch, model, drm_config, metadata: dict[str, Any]) -> float:
    device = str(metadata.get("device", "cpu"))
    total = 0.0
    batches = max(1, int(metadata.get("validation_batches", 1)))
    model.eval()
    for index in range(batches):
        local = dict(metadata)
        split = str(local.get("validation_split", "val"))
        local[f"{split}_token_offset"] = int(local.get(f"{split}_token_offset", 0)) + index * 4096
        inputs, targets = _tokens(torch, local, drm_config.vocab_size, device, seed_key="validation_seed")
        total += float(_loss(model, inputs, targets).detach().cpu().item())
    return total / batches


def _baseline_rows(
    seed: int,
    validation_batches: int,
    batch_size: int,
    steps: int,
    budgets: list[int],
) -> list[dict[str, Any]]:
    metadata = _metadata(seed, validation_batches, batch_size)
    metadata["seed"] = seed
    torch, config_cls, model_cls, _drm_root = _import_drm(metadata)
    drm_config = load_drm_baseline_config(metadata, config_cls)
    base_loss = _mean_eval(torch, model_cls, drm_config, metadata, "cpu", [])
    rows = []
    for target in ["blocks.1.attn.out_proj"]:
        for budget in budgets:
            start = perf_counter()
            final, params, train_s = _train_full_budget(
                torch, model_cls, drm_config, metadata, target, budget, steps
            )
            gain = base_loss - final
            rows.append({
                "method": f"full_budget_linear_{budget}",
                "seed": seed,
                "target_module": target,
                "base_loss": base_loss,
                "final_loss": final,
                "validation_gain": gain,
                "gain_per_parameter": gain / max(1, params),
                "trainable_parameters": int(params),
                "checkpoint_bytes_estimate": int(params) * 4,
                "train_s": train_s,
                "elapsed_s": perf_counter() - start,
            })
        start = perf_counter()
        final, params, train_s = _train_full_module(
            torch, model_cls, drm_config, metadata, target, steps
        )
        gain = base_loss - final
        rows.append({
            "method": "full_module_linear",
            "seed": seed,
            "target_module": target,
            "base_loss": base_loss,
            "final_loss": final,
            "validation_gain": gain,
            "gain_per_parameter": gain / max(1, params),
            "trainable_parameters": int(params),
            "checkpoint_bytes_estimate": int(params) * 4,
            "train_s": train_s,
            "elapsed_s": perf_counter() - start,
        })
    return rows


def _summary(saint_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_saint = max(saint_rows, key=lambda row: row["gain_per_parameter"])
    best_baseline = max(baseline_rows, key=lambda row: row["gain_per_parameter"])
    equal_budget = [
        row for row in saint_rows + baseline_rows
        if int(row.get("attempted_parameter_budget", row["trainable_parameters"])) >= 4096
    ]
    best_equal = max(equal_budget, key=lambda row: row["validation_gain"])
    stronger_than_full = best_saint["gain_per_parameter"] > best_baseline["gain_per_parameter"]
    return {
        "saint_runs": len(saint_rows),
        "baseline_runs": len(baseline_rows),
        "best_saint": best_saint,
        "best_baseline": best_baseline,
        "best_equal_budget_by_gain": best_equal,
        "saint_beats_best_full_by_gain_per_parameter": stronger_than_full,
        "saint_4096_beats_full_module": best_equal["method"] == "drm_saint_g_4096",
        "phase_5c_passed": stronger_than_full or best_equal["method"] == "drm_saint_g_4096",
    }


def _markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# DRM-G Marco 5C Strong Full Baselines",
        "",
        f"- phase_5c_passed: {summary['phase_5c_passed']}",
        f"- saint_beats_best_full_by_gain_per_parameter: "
        f"{summary['saint_beats_best_full_by_gain_per_parameter']}",
        "",
        "| method | seed | target | gain | gain/param | params | train_s |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["gain_per_parameter"], reverse=True):
        lines.append(
            "| {method} | {seed} | {target_module} | {validation_gain:.6f} | "
            "{gain_per_parameter:.6e} | {params} | {train_s:.3f} |".format(
                **{**row, "target_module": row.get("target_module", "sequence")},
                params=row.get("attempted_parameter_budget", row["trainable_parameters"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/drm_g_marco5c_full_baselines")
    parser.add_argument("--seeds", nargs="*", type=int, default=[31, 32, 33, 34])
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--baseline-steps", type=int, default=8)
    parser.add_argument("--saint-ranks", nargs="*", type=int, default=[8, 64])
    parser.add_argument("--full-budgets", nargs="*", type=int, default=[128, 4096])
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saint_rows = []
    for seed in args.seeds:
        for rank in args.saint_ranks:
            saint_rows.append(
                _run_saint(
                    seed,
                    out_dir,
                    args.validation_batches,
                    args.batch_size,
                    rank,
                    f"drm_saint_g_{rank * rank}",
                )
            )
    baseline_rows = []
    for seed in args.seeds:
        baseline_rows.extend(
            _baseline_rows(
                seed,
                args.validation_batches,
                args.batch_size,
                args.baseline_steps,
                args.full_budgets,
            )
        )
    rows = saint_rows + baseline_rows
    summary = _summary(saint_rows, baseline_rows)
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(summary, rows), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
