"""Validation-routed staged graft growth for Phase 16 Marco 4F."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from scripts import benchmark_drm_g_marco5c_phi_variants as phi_bench
from saint.adapters.drm_grafting import _freeze, _load_optional_state, _loss, _tokens
from saint.adapters.drm_grafting_graftblock import (
    graft_checkpoint_payload,
    make_graft_blocks,
)


def _batch(metadata: dict[str, Any], index: int) -> dict[str, Any]:
    local = dict(metadata)
    local["train_token_offset"] = int(local.get("train_token_offset", 0)) + index * 4096
    return local


def _eval(torch, model, drm_config, metadata: dict[str, Any]) -> float:
    return phi_bench._mean_eval(torch, model, drm_config, metadata)


def _set_state(grafts, accepted: set[int], current: set[int]) -> None:
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


def _new_model(torch, model_cls, drm_config, metadata):
    model = model_cls(drm_config).to(str(metadata["device"]))
    _load_optional_state(model, metadata, torch)
    _freeze(model)
    return model


def _new_grafts(torch, drm_config, metadata, args):
    return make_graft_blocks(
        torch,
        d_model=int(drm_config.d_model),
        hidden_size=int(args.hidden_size),
        graft_count=int(args.graft_count),
        seed=int(metadata["seed"]),
        init_scale=float(args.init_scale),
        activation=str(args.activation),
        device=str(metadata["device"]),
    )


def _copy_states(torch, dst, src, device: str) -> None:
    for out, inc in zip(dst, src):
        out.load_state_dict(inc.state_dict(), device)


def _copy_indices(dst, src, indices: set[int], device: str) -> None:
    for index in indices:
        dst[index].load_state_dict(src[index].state_dict(), device)


def _state_payload(grafts) -> list[dict[str, Any]]:
    return [graft.state_dict() for graft in grafts]


def _load_payload(grafts, payload: list[dict[str, Any]], device: str) -> None:
    for graft, state in zip(grafts, payload):
        graft.load_state_dict(state, device)


def _attach_target_map(model, grafts, target_by_graft: dict[int, str]):
    modules = dict(model.named_modules())
    handles = []
    for index, target in sorted(target_by_graft.items()):
        if target not in modules:
            raise ValueError(f"unknown graft target_module: {target}")
        handles.append(modules[target].register_forward_hook(grafts[int(index)].hook))
    return handles


def _compose_target_map(
    accepted_target_map: dict[int, str],
    candidate: str | None = None,
    indices: list[int] | None = None,
) -> dict[int, str]:
    target_map = dict(accepted_target_map)
    if candidate and indices:
        for index in indices:
            target_map[int(index)] = str(candidate)
    return target_map


def _candidate_grid(args) -> list[dict[str, Any]]:
    targets = list(args.candidate_targets or args.targets)
    lrs = list(args.candidate_learning_rates or [args.learning_rate])
    scales = list(args.candidate_init_scales or [args.init_scale])
    activations = list(args.candidate_activations or [args.activation])
    candidates = []
    for target in targets:
        for lr in lrs:
            for scale in scales:
                for activation in activations:
                    candidates.append({
                        "target": str(target),
                        "learning_rate": float(lr),
                        "init_scale": float(scale),
                        "activation": str(activation),
                        "tag": (
                            f"{target}|lr={float(lr):.2e}|"
                            f"scale={float(scale):.2e}|act={activation}"
                        ),
                    })
    return candidates


def _candidate_args(args, candidate: dict[str, Any]):
    values = vars(args).copy()
    values["learning_rate"] = float(candidate["learning_rate"])
    values["init_scale"] = float(candidate["init_scale"])
    values["activation"] = str(candidate["activation"])
    return SimpleNamespace(**values)


def _marco_name(args) -> str:
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


def _candidate_score(args, target: str, gain: float, accepted_target_map: dict[int, str]) -> tuple[float, float]:
    if getattr(args, "candidate_score_mode", "composed_gain") != "composed_gain_orthogonal":
        return float(gain), 0.0
    overlap = sum(1 for accepted_target in accepted_target_map.values() if accepted_target == target)
    penalty = float(getattr(args, "orthogonal_penalty", 0.0)) * float(overlap)
    return float(gain) - penalty, penalty


def _candidate_rank(row: dict[str, Any], min_gain: float) -> tuple[int, float]:
    positive = row["candidate_composed_gain"] > float(min_gain)
    return (1 if positive else 0, float(row["candidate_score"]))


def _stage_indices(args, start: int, stage: int) -> tuple[list[int], int]:
    size = int(args.stage_size)
    if stage > 1 and int(getattr(args, "post_first_stage_size", 0) or 0) > 0:
        size = int(args.post_first_stage_size)
    end = min(start + max(1, size), int(args.graft_count))
    return list(range(start, end)), end


def _train_current(torch, model, drm_config, metadata, grafts, indices, accepted, args, out_dir, stage, tag):
    current = set(indices)
    _set_state(grafts, accepted, current)
    optimizer = _optimizer(torch, grafts, indices, args)
    previous = _eval(torch, model, drm_config, metadata)
    best = previous
    best_payload = _state_payload(grafts)
    best_step = 0
    bad = 0
    trained = 0
    start = perf_counter()
    for step in range(max(1, int(args.steps))):
        local = _batch(metadata, step % max(1, int(args.train_batches)))
        inputs, targets = _tokens(torch, local, drm_config.vocab_size, str(metadata["device"]))
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
        trained = step + 1
        if args.eval_every_steps and trained % int(args.eval_every_steps) == 0:
            value = _eval(torch, model, drm_config, metadata)
            improved = value < (best - float(args.early_stopping_min_delta))
            if improved:
                best = value
                best_payload = _state_payload(grafts)
                best_step = trained
                bad = 0
            else:
                bad += 1
            with (out_dir / "candidate_training_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "stage": stage,
                    "tag": tag,
                    "step": trained,
                    "eval_loss": value,
                    "best_loss": best,
                    "bad_evals": bad,
                }) + "\n")
            if int(args.early_stopping_patience) > 0 and bad >= int(args.early_stopping_patience):
                break
        if args.max_train_seconds > 0 and perf_counter() - start >= float(args.max_train_seconds):
            break
    final = _eval(torch, model, drm_config, metadata)
    return {
        "previous_loss": previous,
        "best_loss": best,
        "final_loss": final,
        "gain": previous - best,
        "state_payload": best_payload,
        "final_state_payload": _state_payload(grafts),
        "best_step": best_step,
        "trained_steps": trained,
        "elapsed_s": perf_counter() - start,
    }


def _composed_loss_for(torch, model_cls, drm_config, metadata, args, states, active, target_map):
    model = _new_model(torch, model_cls, drm_config, metadata)
    grafts = _new_grafts(torch, drm_config, metadata, args)
    _load_payload(grafts, states, str(metadata["device"]))
    _set_state(grafts, active, set())
    handles = _attach_target_map(model, grafts, target_map)
    try:
        return _eval(torch, model, drm_config, metadata)
    finally:
        for handle in handles:
            handle.remove()


def _save_composed(torch, out_dir: Path, grafts, target_map: dict[int, str], args, row: dict[str, Any]) -> Path:
    path = out_dir / "composed_graft_checkpoint.pt"
    payload = graft_checkpoint_payload(
        grafts=grafts,
        target_modules=sorted(set(target_map.values())) or args.targets,
        metadata={
            "baseline_config": args.baseline_config,
            "checkpoint": args.checkpoint,
            "data_dir": args.data_dir,
            "row": row,
            "accepted_graft_ids": row["accepted_graft_ids"],
            "target_by_graft": {str(key): value for key, value in sorted(target_map.items())},
        },
    )
    torch.save(payload, path)
    return path


def _recompose_loss(torch, model_cls, drm_config, metadata, artifact: Path, args) -> float:
    payload = torch.load(str(artifact), map_location="cpu", weights_only=False)
    accepted = set(payload["metadata"].get("accepted_graft_ids", []))
    target_by_graft = {
        int(key): value
        for key, value in payload["metadata"].get("target_by_graft", {}).items()
    }
    model = _new_model(torch, model_cls, drm_config, metadata)
    grafts = _new_grafts(torch, drm_config, metadata, args)
    for graft, state in zip(grafts, payload["grafts"]):
        graft.load_state_dict(state, str(metadata["device"]))
    _set_state(grafts, accepted, set())
    handles = _attach_target_map(model, grafts, target_by_graft)
    try:
        return _eval(torch, model, drm_config, metadata)
    finally:
        for handle in handles:
            handle.remove()


def _markdown(summary: dict[str, Any]) -> str:
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


def run_validation_routed_staged(torch, config_cls, model_cls, drm_config, metadata, args, out_dir: Path):
    base_model = _new_model(torch, model_cls, drm_config, metadata)
    base_loss = _eval(torch, base_model, drm_config, metadata)
    accepted: set[int] = set()
    accepted_target_map: dict[int, str] = {}
    accepted_states = _new_grafts(torch, drm_config, metadata, args)
    current_composed_loss = base_loss
    stage_metrics = []
    candidate_metrics = []
    candidates = _candidate_grid(args)
    start = 0
    for stage in range(1, int(args.max_stages) + 1):
        if start >= int(args.graft_count):
            break
        indices, end = _stage_indices(args, start, stage)
        previous_composed_loss = current_composed_loss
        probes = []
        best_payload = None
        best_gain = None
        for candidate in candidates:
            candidate_args = _candidate_args(args, candidate)
            target = candidate["target"]
            model = _new_model(torch, model_cls, drm_config, metadata)
            grafts = _new_grafts(torch, drm_config, metadata, candidate_args)
            _copy_indices(grafts, accepted_states, accepted, str(metadata["device"]))
            candidate_target_map = _compose_target_map(accepted_target_map, target, indices)
            handles = _attach_target_map(model, grafts, candidate_target_map)
            try:
                result = _train_current(
                    torch, model, drm_config, metadata, grafts, indices, accepted,
                    candidate_args, out_dir, stage, candidate["tag"],
                )
            finally:
                for handle in handles:
                    handle.remove()
            state_payload = result.pop("state_payload")
            result.pop("final_state_payload", None)
            candidate_active = accepted | set(indices)
            candidate_composed_loss = _composed_loss_for(
                torch,
                model_cls,
                drm_config,
                metadata,
                candidate_args,
                state_payload,
                candidate_active,
                candidate_target_map,
            )
            result["candidate_composed_loss"] = candidate_composed_loss
            result["candidate_composed_gain"] = current_composed_loss - candidate_composed_loss
            score, penalty = _candidate_score(
                args,
                target,
                result["candidate_composed_gain"],
                accepted_target_map,
            )
            result["candidate_score"] = score
            result["redundancy_penalty"] = penalty
            result["previous_composed_loss"] = current_composed_loss
            result["candidate_target_by_graft"] = {
                str(key): value for key, value in sorted(candidate_target_map.items())
            }
            result.update({
                "stage": stage,
                "candidate_target": target,
                "learning_rate": candidate["learning_rate"],
                "init_scale": candidate["init_scale"],
                "activation": candidate["activation"],
                "candidate_tag": candidate["tag"],
                "indices": indices,
            })
            probes.append(result)
            candidate_metrics.append(result)
            rank = _candidate_rank(result, float(args.stage_accept_min_gain))
            if best_gain is None or rank > best_gain:
                best_payload = state_payload
                best_gain = rank
        best = max(probes, key=lambda row: _candidate_rank(row, float(args.stage_accept_min_gain)))
        decision = "approved" if best["candidate_composed_gain"] > float(args.stage_accept_min_gain) else "rejected"
        if decision == "approved":
            _load_payload(accepted_states, best_payload, str(metadata["device"]))
            for index in indices:
                accepted_target_map[int(index)] = best["candidate_target"]
            accepted.update(indices)
            current_composed_loss = best["candidate_composed_loss"]
        stage_metrics.append({
            "stage": stage,
            "selected_target": best["candidate_target"],
            "decision": decision,
            "graft_start": start,
            "graft_end": end,
            "stage_best_loss": best["best_loss"],
            "stage_gain": best["candidate_composed_gain"],
            "stage_score": best["candidate_score"],
            "redundancy_penalty": best["redundancy_penalty"],
            "learning_rate": best["learning_rate"],
            "init_scale": best["init_scale"],
            "activation": best["activation"],
            "candidate_tag": best["candidate_tag"],
            "candidate_composed_loss": best["candidate_composed_loss"],
            "previous_composed_loss": previous_composed_loss,
            "target_by_graft": {str(key): value for key, value in sorted(accepted_target_map.items())},
            "accepted_graft_ids": sorted(accepted),
        })
        if decision != "approved":
            break
        start = end
    final_model = _new_model(torch, model_cls, drm_config, metadata)
    _set_state(accepted_states, accepted, set())
    handles = _attach_target_map(final_model, accepted_states, accepted_target_map)
    try:
        composed_loss = _eval(torch, final_model, drm_config, metadata)
    finally:
        for handle in handles:
            handle.remove()
    summary = {
        "phase": "16",
        "marco": _marco_name(args),
        "base_loss": base_loss,
        "composed_loss": composed_loss,
        "accumulated_gain": base_loss - composed_loss,
        "accepted_groups": sum(1 for row in stage_metrics if row["decision"] == "approved"),
        "accepted_grafts": len(accepted),
        "accepted_graft_ids": sorted(accepted),
        "target_by_graft": {str(key): value for key, value in sorted(accepted_target_map.items())},
        "candidate_score_mode": getattr(args, "candidate_score_mode", "composed_gain"),
        "orthogonal_penalty": float(getattr(args, "orthogonal_penalty", 0.0)),
        "stage_metrics": stage_metrics,
    }
    checkpoint = _save_composed(torch, out_dir, accepted_states, accepted_target_map, args, summary)
    summary["composed_checkpoint"] = str(checkpoint)
    summary["composed_checkpoint_bytes"] = checkpoint.stat().st_size
    summary["recomposed_loss"] = _recompose_loss(torch, model_cls, drm_config, metadata, checkpoint, args)
    summary["recompose_abs_diff"] = abs(summary["recomposed_loss"] - composed_loss)
    (out_dir / "candidate_metrics.json").write_text(json.dumps(candidate_metrics, indent=2), encoding="utf-8")
    (out_dir / "stage_metrics.json").write_text(json.dumps(stage_metrics, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
