"""Progressive DRM-G graft cycles."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from saint.adapters.drm_grafting import (
    _baseline_path,
    _freeze,
    _import_drm,
    _load_optional_state,
    _loss,
    _projection_init,
    _target_module,
    _tokens,
    inspect_graft_model,
    load_drm_baseline_config,
)
from saint.adapters.drm_grafting_decision import evaluate_graft_decision
from saint.adapters.drm_grafting_modules import DenseBudgetGraft, PhiHiddenGraft
from saint.adapters.drm_grafting_optimizer import optimizer_to_payload
from saint.config import RuntimeConfig
from saint.transformer.training import MiniTransformerResult


def _default_candidates(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = metadata.get("grafts")
    if isinstance(candidates, list) and candidates:
        return [dict(item) for item in candidates if isinstance(item, dict)]
    return [
        {"target_module": "blocks.1", "projection_init": "gradient"},
        {"target_module": "blocks.2", "projection_init": "gradient"},
        {"target_module": "final_norm", "projection_init": "activation"},
    ]


def _attach_payloads(torch, model, payloads: list[dict[str, Any]], device: str):
    handles = []
    grafts = []
    for payload in payloads:
        graft = PhiHiddenGraft.from_payload(torch, payload).to(device)
        handle = _target_module(model, str(payload["target_module"])).register_forward_hook(graft.hook)
        grafts.append(graft)
        handles.append(handle)
    return handles, grafts


def _eval_with_payloads(
    torch,
    model_cls,
    drm_config,
    metadata: dict[str, Any],
    device: str,
    eval_inputs,
    eval_targets,
    payloads: list[dict[str, Any]],
) -> float:
    seed = int(metadata.get("seed", 0))
    torch.manual_seed(seed)
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    model.eval()
    _freeze(model)
    handles, _grafts = _attach_payloads(torch, model, payloads, device)
    try:
        return float(_loss(model, eval_inputs, eval_targets).detach().cpu().item())
    finally:
        for handle in handles:
            handle.remove()


def _cuda_peak(torch, device: str) -> int:
    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return 0


def _mean_eval(
    torch,
    model_cls,
    drm_config,
    metadata: dict[str, Any],
    device: str,
    payloads: list[dict[str, Any]],
    *,
    seed_key: str = "validation_seed",
) -> float:
    total = 0.0
    batches = max(1, int(metadata.get("validation_batches", 1)))
    for index in range(batches):
        local = dict(metadata)
        split = str(local.get("validation_split", "val"))
        local[f"{split}_token_offset"] = int(local.get(f"{split}_token_offset", 0)) + index * 4096
        inputs, targets = _tokens(torch, local, drm_config.vocab_size, device, seed_key=seed_key)
        total += _eval_with_payloads(
            torch, model_cls, drm_config, local, device, inputs, targets, payloads
        )
    return total / batches


def _queue_decision(decision: dict[str, Any], gain: float, metadata: dict[str, Any]) -> str:
    if decision["approved"]:
        return "approve"
    defer_floor = float(metadata.get("defer_gain_floor", -1e-4))
    return "defer" if gain > defer_floor else "reject"


def _train_candidate(
    torch,
    model_cls,
    drm_config,
    metadata: dict[str, Any],
    device: str,
    inputs,
    targets,
    eval_inputs,
    eval_targets,
    approved: list[dict[str, Any]],
    graft,
    target_module: str,
) -> tuple[float, dict[str, Any]]:
    seed = int(metadata.get("seed", 0))
    torch.manual_seed(seed)
    model = model_cls(drm_config).to(device)
    _load_optional_state(model, metadata, torch)
    model.eval()
    _freeze(model)
    base_handles, _base_grafts = _attach_payloads(torch, model, approved, device)
    graft.to(device)
    handle = _target_module(model, target_module).register_forward_hook(graft.hook)
    optimizer = torch.optim.AdamW(graft.parameters(), lr=float(metadata.get("learning_rate", 0.005)))
    train_batches = max(1, int(metadata.get("train_batches", 1)))
    try:
        for _ in range(max(1, int(metadata.get("graft_steps", metadata.get("steps", 2))))):
            optimizer.zero_grad(set_to_none=True)
            for _batch in range(train_batches):
                loss = _loss(model, inputs, targets) / train_batches
                loss.backward()
            optimizer.step()
        final = float(_loss(model, eval_inputs, eval_targets).detach().cpu().item())
        return final, optimizer_to_payload(optimizer)
    finally:
        handle.remove()
        for base_handle in base_handles:
            base_handle.remove()


def _make_phi(
    torch,
    model_cls,
    drm_config,
    metadata: dict[str, Any],
    device: str,
    inputs,
    targets,
    candidate: dict[str, Any],
    index: int,
):
    rank = int(candidate.get("phi_rank", metadata.get("phi_rank", 8)))
    seed = int(candidate.get("seed", int(metadata.get("seed", 0)) + index))
    scale = float(candidate.get("graft_scale", metadata.get("graft_scale", 1.0)))
    local_metadata = {**metadata, **candidate}
    projections = _projection_init(
        torch,
        model_cls,
        drm_config,
        local_metadata,
        device,
        inputs,
        targets,
        rank,
        seed,
    )
    return PhiHiddenGraft(torch, drm_config.d_model, rank, scale, seed, projections)


def run_drm_graft_progressive(config: RuntimeConfig) -> MiniTransformerResult:
    start = perf_counter()
    routing_s = 0.0
    train_s = 0.0
    eval_s = 0.0
    metadata = dict(config.metadata or {})
    metadata["steps"] = config.steps
    torch, config_cls, model_cls, drm_root = _import_drm(metadata)
    device = str(metadata.get("device", "cpu"))
    seed = int(metadata.get("seed", config.seed))
    metadata["seed"] = seed
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    drm_config = load_drm_baseline_config(metadata, config_cls)
    inputs, targets = _tokens(torch, metadata, drm_config.vocab_size, device)
    eval_inputs, eval_targets = _tokens(
        torch, metadata, drm_config.vocab_size, device, seed_key="validation_seed"
    )
    approved: list[dict[str, Any]] = []
    rows = []
    optimizer_states = []
    eval_start = perf_counter()
    base_loss = _mean_eval(torch, model_cls, drm_config, metadata, device, approved)
    old_base_loss = _mean_eval(
        torch, model_cls, drm_config, {**metadata, "validation_seed": metadata.get("old_validation_seed", seed)}, device, approved
    )
    eval_s += perf_counter() - eval_start
    current_loss = base_loss
    candidates = _default_candidates(metadata)
    for index, candidate in enumerate(candidates, start=1):
        target = str(candidate.get("target_module", "final_norm"))
        init = str(candidate.get("projection_init", metadata.get("projection_init", "gradient")))
        loss_before = current_loss
        route_start = perf_counter()
        phi = _make_phi(torch, model_cls, drm_config, metadata, device, inputs, targets, candidate, index)
        routing_s += perf_counter() - route_start
        train_start = perf_counter()
        _candidate_batch_loss, optimizer_payload = _train_candidate(
            torch, model_cls, drm_config, metadata, device, inputs, targets,
            eval_inputs, eval_targets, approved, phi, target
        )
        train_s += perf_counter() - train_start
        payload = phi.payload(target, init)
        eval_start = perf_counter()
        graft_loss = _mean_eval(
            torch, model_cls, drm_config, metadata, device, approved + [payload]
        )
        eval_s += perf_counter() - eval_start
        dense = DenseBudgetGraft(torch, drm_config.d_model, int(phi.phi.shape[0]), float(phi.scale))
        train_start = perf_counter()
        dense_loss, _dense_optimizer = _train_candidate(
            torch, model_cls, drm_config, metadata, device, inputs, targets,
            eval_inputs, eval_targets, approved, dense, target
        )
        train_s += perf_counter() - train_start
        gain = loss_before - graft_loss
        dense_gain = loss_before - dense_loss
        params = int(phi.phi.numel())
        decision = evaluate_graft_decision({
            **metadata,
            "validation_gain": gain,
            "validation_gain_per_parameter": gain / max(1, params),
            "dense_budget_gain": dense_gain,
        })
        queue_status = _queue_decision(decision, gain, metadata)
        if queue_status == "approve":
            approved.append(payload)
            optimizer_states.append(optimizer_payload)
            current_loss = graft_loss
        rows.append({
            "index": index,
            "target_module": target,
            "projection_init": init,
            "loss_before": loss_before,
            "candidate_loss": graft_loss,
            "dense_budget_loss": dense_loss,
            "validation_gain": gain,
            "dense_budget_gain": dense_gain,
            "decision": queue_status,
            "approved": queue_status == "approve",
        })
    eval_start = perf_counter()
    final_loss = _mean_eval(torch, model_cls, drm_config, metadata, device, approved)
    old_final_loss = _mean_eval(
        torch, model_cls, drm_config, {**metadata, "validation_seed": metadata.get("old_validation_seed", seed)}, device, approved
    )
    eval_s += perf_counter() - eval_start
    payload = {
        "format": "drm_graft_sequence_payload",
        "grafts": approved,
        "rejected": [row for row in rows if not row["approved"]],
    }
    params = sum(int(graft.get("trainable_parameters", 0)) for graft in approved)
    approved_count = sum(1 for row in rows if row["decision"] == "approve")
    rejected_count = sum(1 for row in rows if row["decision"] == "reject")
    deferred_count = sum(1 for row in rows if row["decision"] == "defer")
    conflict_count = sum(
        1 for row in rows if row["index"] > 1 and row["validation_gain"] <= 0.0
    )
    return MiniTransformerResult(
        name="drm_g_saint_phi_progressive",
        train_loss=final_loss,
        test_loss=final_loss,
        parameter_count=params,
        optimizer_state_values=params * 2,
        elapsed_s=perf_counter() - start,
        metadata={
            "baseline_config": str(_baseline_path(metadata, drm_root)),
            "base_parameters": inspect_graft_model(config)["base_parameters"],
            "base_loss": base_loss,
            "final_loss": final_loss,
            "sequence_gain": base_loss - final_loss,
            "sequence_gain_per_parameter": (base_loss - final_loss) / max(1, params),
            "old_base_loss": old_base_loss,
            "old_final_loss": old_final_loss,
            "old_regression": old_final_loss - old_base_loss,
            "approved_grafts": len(approved),
            "rejected_grafts": rejected_count,
            "deferred_grafts": deferred_count,
            "approval_rate": approved_count / max(1, len(rows)),
            "conflict_count": conflict_count,
            "candidate_count": len(candidates),
            "progressive_rows": rows,
            "delta_payload": payload,
            "optimizer_state_payload": {
                "format": "drm_graft_sequence_adamw_state",
                "optimizer": "AdamW",
                "states": optimizer_states,
            },
            "drm_g": True,
            "cuda_peak_bytes": _cuda_peak(torch, device),
            "routing_s": routing_s,
            "train_s": train_s,
            "eval_s": eval_s,
            "marco": "drm_g_marco_4_progressive",
        },
    )


__all__ = ["run_drm_graft_progressive"]
