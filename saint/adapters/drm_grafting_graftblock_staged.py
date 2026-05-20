"""Staged graft growth for Phase 16 Marco 4E."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from scripts import benchmark_drm_g_marco5c_phi_variants as phi_bench
from saint.adapters.drm_grafting import _freeze, _load_optional_state, _loss, _tokens
from saint.adapters.drm_grafting_graftblock import (
    attach_graft_blocks,
    graft_checkpoint_payload,
    make_graft_blocks,
)


def _batch(metadata: dict[str, Any], index: int) -> dict[str, Any]:
    local = dict(metadata)
    local["train_token_offset"] = int(local.get("train_token_offset", 0)) + index * 4096
    return local


def _cuda_peak(torch, device: str) -> int | None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _set_stage_state(grafts, accepted: set[int], current: set[int]) -> None:
    for index, graft in enumerate(grafts):
        active = index in accepted or index in current
        trainable = index in current
        graft.enabled = active
        graft.runtime_scale = 1.0 if active else 0.0
        for param in graft.parameters():
            param.requires_grad_(trainable)


def _optimizer(torch, grafts, indices: list[int], args):
    groups = []
    for offset, index in enumerate(indices):
        lr = args.learning_rate / (1.0 + (offset * args.lr_decay))
        groups.append({"params": list(grafts[index].parameters()), "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _save_checkpoint(torch, out_dir: Path, grafts, args, row: dict[str, Any], name: str) -> Path:
    path = out_dir / name
    payload = graft_checkpoint_payload(
        grafts=grafts,
        target_modules=args.targets,
        metadata={
            "baseline_config": args.baseline_config,
            "checkpoint": args.checkpoint,
            "data_dir": args.data_dir,
            "row": row,
            "accepted_graft_ids": list(row.get("accepted_graft_ids", [])),
        },
    )
    torch.save(payload, path)
    return path


def _load_graft_states(torch, grafts, artifact: Path, device: str) -> None:
    payload = torch.load(str(artifact), map_location="cpu", weights_only=False)
    for graft, state in zip(grafts, payload["grafts"]):
        graft.load_state_dict(state, device)


def _recompose_loss(torch, model_cls, drm_config, metadata, artifact: Path, args) -> float:
    payload = torch.load(str(artifact), map_location="cpu", weights_only=False)
    accepted = set(payload["metadata"].get("accepted_graft_ids", []))
    device = str(metadata["device"])
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    _freeze(model)
    grafts = make_graft_blocks(
        torch,
        d_model=int(drm_config.d_model),
        hidden_size=int(args.hidden_size),
        graft_count=int(args.graft_count),
        seed=int(metadata["seed"]),
        init_scale=float(args.init_scale),
        activation=str(args.activation),
        device=device,
    )
    for graft, state in zip(grafts, payload["grafts"]):
        graft.load_state_dict(state, device)
    _set_stage_state(grafts, accepted, set())
    handles = attach_graft_blocks(model, payload["target_modules"], grafts)
    try:
        return _eval(torch, model, drm_config, metadata)
    finally:
        for handle in handles:
            handle.remove()


def _eval(torch, model, drm_config, metadata: dict[str, Any]) -> float:
    return phi_bench._mean_eval(torch, model, drm_config, metadata)


def _append_stage_metric(out_dir: Path, row: dict[str, Any]) -> None:
    with (out_dir / "stage_training_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _train_stage(torch, model, drm_config, metadata, grafts, stage, indices, accepted, args, out_dir):
    current = set(indices)
    _set_stage_state(grafts, accepted, current)
    optimizer = _optimizer(torch, grafts, indices, args)
    previous_loss = _eval(torch, model, drm_config, metadata)
    best_loss = previous_loss
    best_step = 0
    best_elapsed = 0.0
    bad_evals = 0
    best_artifact = None
    start = perf_counter()
    trained_steps = 0
    stopped_early = False
    for step in range(max(1, args.steps)):
        local = _batch(metadata, step % max(1, args.train_batches))
        inputs, targets = _tokens(torch, local, drm_config.vocab_size, str(metadata["device"]))
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
        trained_steps = step + 1
        elapsed = perf_counter() - start
        if args.eval_every_steps and trained_steps % args.eval_every_steps == 0:
            eval_loss = _eval(torch, model, drm_config, metadata)
            improved = eval_loss < (best_loss - args.early_stopping_min_delta)
            if improved:
                best_loss = eval_loss
                best_step = trained_steps
                best_elapsed = elapsed
                bad_evals = 0
                preview = {
                    "stage": stage,
                    "graft_count": len(grafts),
                    "seed": int(metadata["seed"]),
                    "d_model": int(drm_config.d_model),
                    "hidden_size": int(grafts[0].hidden_size),
                    "activation": args.activation,
                    "accepted_graft_ids": sorted(accepted | current),
                }
                best_artifact = _save_checkpoint(
                    torch,
                    out_dir,
                    grafts,
                    args,
                    preview,
                    f"stage{stage}_best_graftblock.pt",
                )
            else:
                bad_evals += 1
            _append_stage_metric(out_dir, {
                "stage": stage,
                "step": trained_steps,
                "elapsed_s": elapsed,
                "eval_loss": eval_loss,
                "best_loss": best_loss,
                "bad_evals": bad_evals,
            })
            if args.early_stopping_patience > 0 and bad_evals >= args.early_stopping_patience:
                stopped_early = True
                break
        if args.max_train_seconds > 0 and elapsed >= args.max_train_seconds:
            break
    stage_gain = previous_loss - best_loss
    decision = "approved" if stage_gain > args.stage_accept_min_gain else "rejected"
    if decision == "approved" and best_artifact:
        _load_graft_states(torch, grafts, best_artifact, str(metadata["device"]))
        accepted.update(current)
    else:
        current.clear()
    _set_stage_state(grafts, accepted, set())
    final_loss = _eval(torch, model, drm_config, metadata)
    return {
        "stage": stage,
        "graft_start": min(indices),
        "graft_end": max(indices) + 1,
        "previous_best_loss": previous_loss,
        "stage_best_loss": best_loss,
        "stage_gain": stage_gain,
        "stage_final_loss": final_loss,
        "decision": decision,
        "best_step": best_step,
        "best_elapsed_s": best_elapsed,
        "trained_steps": trained_steps,
        "stopped_early": stopped_early,
        "accepted_graft_ids": sorted(accepted),
        "best_checkpoint": str(best_artifact) if best_artifact else None,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 16 Marco 4E - Staged Graft Growth",
        "",
        f"- base_loss: {summary['base_loss']:.6f}",
        f"- composed_loss: {summary['composed_loss']:.6f}",
        f"- accumulated_gain: {summary['accumulated_gain']:.6f}",
        f"- accepted_stages: {summary['accepted_stages']}",
        f"- accepted_grafts: {summary['accepted_grafts']}",
        f"- checkpoint: `{summary['composed_checkpoint']}`",
        "",
        "| stage | grafts | decision | previous | best | gain | final |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["stage_metrics"]:
        lines.append(
            "| {stage} | {graft_start}-{graft_end} | {decision} | "
            "{previous_best_loss:.6f} | {stage_best_loss:.6f} | "
            "{stage_gain:.6f} | {stage_final_loss:.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def run_staged_graft_growth(torch, config_cls, model_cls, drm_config, metadata, args, out_dir: Path):
    device = str(metadata["device"])
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    _freeze(model)
    base_loss = _eval(torch, model, drm_config, metadata)
    grafts = make_graft_blocks(
        torch,
        d_model=int(drm_config.d_model),
        hidden_size=int(args.hidden_size),
        graft_count=int(args.graft_count),
        seed=int(metadata["seed"]),
        init_scale=float(args.init_scale),
        activation=str(args.activation),
        device=device,
    )
    handles = attach_graft_blocks(model, args.targets, grafts)
    accepted: set[int] = set()
    stage_metrics = []
    try:
        for stage in range(1, int(args.max_stages) + 1):
            start = (stage - 1) * int(args.stage_size)
            end = min(start + int(args.stage_size), len(grafts))
            if start >= len(grafts):
                break
            row = _train_stage(
                torch,
                model,
                drm_config,
                metadata,
                grafts,
                stage,
                list(range(start, end)),
                accepted,
                args,
                out_dir,
            )
            stage_metrics.append(row)
            if row["decision"] != "approved":
                break
    finally:
        for handle in handles:
            handle.remove()
    _set_stage_state(grafts, accepted, set())
    handles = attach_graft_blocks(model, args.targets, grafts)
    try:
        composed_loss = _eval(torch, model, drm_config, metadata)
    finally:
        for handle in handles:
            handle.remove()
    summary = {
        "phase": "16",
        "marco": "4e_staged_graft_growth",
        "base_loss": base_loss,
        "composed_loss": composed_loss,
        "accumulated_gain": base_loss - composed_loss,
        "accepted_stages": sum(1 for row in stage_metrics if row["decision"] == "approved"),
        "accepted_grafts": len(accepted),
        "accepted_graft_ids": sorted(accepted),
        "cuda_peak_bytes": _cuda_peak(torch, device),
        "stage_metrics": stage_metrics,
    }
    checkpoint = _save_checkpoint(
        torch,
        out_dir,
        grafts,
        args,
        {
            "seed": int(metadata["seed"]),
            "graft_count": len(grafts),
            "d_model": int(drm_config.d_model),
            "hidden_size": int(args.hidden_size),
            "activation": args.activation,
            "accepted_graft_ids": sorted(accepted),
        },
        "composed_graft_checkpoint.pt",
    )
    summary["composed_checkpoint"] = str(checkpoint)
    summary["composed_checkpoint_bytes"] = checkpoint.stat().st_size
    summary["recomposed_loss"] = _recompose_loss(torch, model_cls, drm_config, metadata, checkpoint, args)
    summary["recompose_abs_diff"] = abs(summary["recomposed_loss"] - composed_loss)
    (out_dir / "stage_metrics.json").write_text(json.dumps(stage_metrics, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
