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
from saint.adapters.drm_grafting_graftblock_routed_utils import (
    candidate_score as _candidate_score,
    marco_name as _marco_name,
    markdown as _markdown,
)
from saint.adapters.drm_grafting_ntk_probe import (
    activation_gate_scores_for_loss as _activation_gate_scores_for_loss,
    run_activation_probe_stage as _run_ntk_activation_probe_stage,
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


def _candidate_args(args, candidate: dict[str, Any], overrides: dict[str, Any] | None = None):
    values = vars(args).copy()
    values["learning_rate"] = float(candidate["learning_rate"])
    values["init_scale"] = float(candidate["init_scale"])
    values["activation"] = str(candidate["activation"])
    if overrides:
        values.update(overrides)
    return SimpleNamespace(**values)


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


def _ntk_activation_probe_stage(
    torch,
    model_cls,
    drm_config,
    metadata,
    args,
    accepted_states,
    accepted,
    accepted_target_map,
    targets: list[str],
    stage: int,
) -> list[dict[str, Any]]:
    if int(getattr(args, "ntk_activation_probe_batches", 0) or 0) <= 0:
        return []
    model = _new_model(torch, model_cls, drm_config, metadata)
    grafts = _new_grafts(torch, drm_config, metadata, args)
    _copy_indices(grafts, accepted_states, accepted, str(metadata["device"]))
    return _run_ntk_activation_probe_stage(
        torch,
        model,
        grafts,
        metadata,
        drm_config,
        args,
        accepted,
        accepted_target_map,
        targets,
        stage,
        copy_indices=_copy_indices,
        set_state=_set_state,
        attach_target_map=_attach_target_map,
        batch_fn=_batch,
        tokens_fn=_tokens,
        loss_fn=_loss,
    )


def _evaluate_candidate(
    torch,
    model_cls,
    drm_config,
    metadata,
    args,
    out_dir,
    accepted_states,
    accepted,
    accepted_target_map,
    candidate,
    indices,
    stage,
    pass_name,
    use_final_state=False,
):
    candidate_args = _candidate_args(args, candidate)
    target = candidate["target"]
    model = _new_model(torch, model_cls, drm_config, metadata)
    grafts = _new_grafts(torch, drm_config, metadata, candidate_args)
    _copy_indices(grafts, accepted_states, accepted, str(metadata["device"]))
    target_map = _compose_target_map(accepted_target_map, target, indices)
    handles = _attach_target_map(model, grafts, target_map)
    try:
        result = _train_current(
            torch, model, drm_config, metadata, grafts, indices, accepted,
            candidate_args, out_dir, stage, candidate["tag"],
        )
    finally:
        for handle in handles:
            handle.remove()
    payload_key = "final_state_payload" if use_final_state else "state_payload"
    state_payload = result.pop(payload_key)
    result.pop("state_payload", None)
    result.pop("final_state_payload", None)
    active = accepted | set(indices)
    loss = _composed_loss_for(torch, model_cls, drm_config, metadata, candidate_args, state_payload, active, target_map)
    result["candidate_composed_loss"] = loss
    result["candidate_composed_gain"] = args._current_composed_loss - loss
    score, penalty = _candidate_score(args, target, result["candidate_composed_gain"], accepted_target_map)
    result["candidate_score"] = score
    result["redundancy_penalty"] = penalty
    result["previous_composed_loss"] = args._current_composed_loss
    result["candidate_target_by_graft"] = {str(key): value for key, value in sorted(target_map.items())}
    result.update({
        "stage": stage,
        "pass": pass_name,
        "candidate_target": target,
        "learning_rate": candidate["learning_rate"],
        "init_scale": candidate["init_scale"],
        "activation": candidate["activation"],
        "candidate_tag": candidate["tag"],
        "indices": indices,
    })
    return result, state_payload


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


def run_validation_routed_staged(torch, config_cls, model_cls, drm_config, metadata, args, out_dir: Path):
    base_model = _new_model(torch, model_cls, drm_config, metadata)
    base_loss = _eval(torch, base_model, drm_config, metadata)
    accepted: set[int] = set()
    accepted_target_map: dict[int, str] = {}
    accepted_states = _new_grafts(torch, drm_config, metadata, args)
    current_composed_loss = base_loss
    stage_metrics = []
    candidate_metrics = []
    ntk_activation_probe_metrics = []
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
        stage_candidates = candidates
        args._current_composed_loss = current_composed_loss
        ntk_rows = _ntk_activation_probe_stage(
            torch,
            model_cls,
            drm_config,
            metadata,
            args,
            accepted_states,
            accepted,
            accepted_target_map,
            list(args.candidate_targets or args.targets),
            stage,
        )
        ntk_activation_probe_metrics.extend(ntk_rows)
        if int(getattr(args, "candidate_top_k", 0) or 0) > 0:
            probe_args = SimpleNamespace(**vars(args))
            probe_args.steps = int(getattr(args, "candidate_probe_steps", 0) or args.eval_every_steps or 1)
            probe_args.max_train_seconds = float(getattr(args, "candidate_probe_max_train_seconds", 0.0) or 0.0)
            probe_args.early_stopping_patience = 0
            probe_args._current_composed_loss = current_composed_loss
            probe_rows = []
            for candidate in candidates:
                result, _payload = _evaluate_candidate(
                    torch, model_cls, drm_config, metadata, probe_args, out_dir, accepted_states,
                    accepted, accepted_target_map, candidate, indices, stage, "probe", use_final_state=True,
                )
                candidate_metrics.append(result)
                probe_rows.append((result, candidate))
            probe_rows.sort(key=lambda item: _candidate_rank(item[0], float(args.stage_accept_min_gain)), reverse=True)
            stage_candidates = [candidate for _row, candidate in probe_rows[: int(args.candidate_top_k)]]
        for candidate in stage_candidates:
            result, state_payload = _evaluate_candidate(
                torch, model_cls, drm_config, metadata, args, out_dir, accepted_states,
                accepted, accepted_target_map, candidate, indices, stage, "deep",
            )
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
            "ntk_activation_probe": ntk_rows,
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
        "candidate_probe_steps": int(getattr(args, "candidate_probe_steps", 0) or 0),
        "candidate_probe_max_train_seconds": float(getattr(args, "candidate_probe_max_train_seconds", 0.0) or 0.0),
        "candidate_top_k": int(getattr(args, "candidate_top_k", 0) or 0),
        "ntk_activation_probe_batches": int(getattr(args, "ntk_activation_probe_batches", 0) or 0),
        "ntk_activation_probe_split": str(getattr(args, "ntk_activation_probe_split", "train") or "train"),
        "stage_metrics": stage_metrics,
    }
    checkpoint = _save_composed(torch, out_dir, accepted_states, accepted_target_map, args, summary)
    summary["composed_checkpoint"] = str(checkpoint)
    summary["composed_checkpoint_bytes"] = checkpoint.stat().st_size
    summary["recomposed_loss"] = _recompose_loss(torch, model_cls, drm_config, metadata, checkpoint, args)
    summary["recompose_abs_diff"] = abs(summary["recomposed_loss"] - composed_loss)
    (out_dir / "candidate_metrics.json").write_text(json.dumps(candidate_metrics, indent=2), encoding="utf-8")
    (out_dir / "ntk_activation_probe_metrics.json").write_text(
        json.dumps(ntk_activation_probe_metrics, indent=2),
        encoding="utf-8",
    )
    (out_dir / "stage_metrics.json").write_text(json.dumps(stage_metrics, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "results.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary
