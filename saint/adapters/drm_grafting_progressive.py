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
    try:
        for _ in range(max(1, int(metadata.get("graft_steps", metadata.get("steps", 2))))):
            optimizer.zero_grad()
            loss = _loss(model, inputs, targets)
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
    metadata = dict(config.metadata or {})
    metadata["steps"] = config.steps
    torch, config_cls, model_cls, drm_root = _import_drm(metadata)
    device = str(metadata.get("device", "cpu"))
    seed = int(metadata.get("seed", config.seed))
    metadata["seed"] = seed
    torch.manual_seed(seed)
    drm_config = load_drm_baseline_config(metadata, config_cls)
    inputs, targets = _tokens(torch, metadata, drm_config.vocab_size, device)
    eval_inputs, eval_targets = _tokens(
        torch, metadata, drm_config.vocab_size, device, seed_key="validation_seed"
    )
    approved: list[dict[str, Any]] = []
    rows = []
    optimizer_states = []
    base_loss = _eval_with_payloads(
        torch, model_cls, drm_config, metadata, device, eval_inputs, eval_targets, approved
    )
    current_loss = base_loss
    candidates = _default_candidates(metadata)
    for index, candidate in enumerate(candidates, start=1):
        target = str(candidate.get("target_module", "final_norm"))
        init = str(candidate.get("projection_init", metadata.get("projection_init", "gradient")))
        loss_before = current_loss
        phi = _make_phi(torch, model_cls, drm_config, metadata, device, inputs, targets, candidate, index)
        graft_loss, optimizer_payload = _train_candidate(
            torch, model_cls, drm_config, metadata, device, inputs, targets,
            eval_inputs, eval_targets, approved, phi, target
        )
        dense = DenseBudgetGraft(torch, drm_config.d_model, int(phi.phi.shape[0]), float(phi.scale))
        dense_loss, _dense_optimizer = _train_candidate(
            torch, model_cls, drm_config, metadata, device, inputs, targets,
            eval_inputs, eval_targets, approved, dense, target
        )
        gain = loss_before - graft_loss
        dense_gain = loss_before - dense_loss
        params = int(phi.phi.numel())
        decision = evaluate_graft_decision({
            **metadata,
            "validation_gain": gain,
            "validation_gain_per_parameter": gain / max(1, params),
            "dense_budget_gain": dense_gain,
        })
        payload = phi.payload(target, init)
        if decision["approved"]:
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
            "decision": decision["decision"],
            "approved": decision["approved"],
        })
    final_loss = _eval_with_payloads(
        torch, model_cls, drm_config, metadata, device, eval_inputs, eval_targets, approved
    )
    payload = {
        "format": "drm_graft_sequence_payload",
        "grafts": approved,
        "rejected": [row for row in rows if not row["approved"]],
    }
    params = sum(int(graft.get("trainable_parameters", 0)) for graft in approved)
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
            "approved_grafts": len(approved),
            "candidate_count": len(candidates),
            "progressive_rows": rows,
            "delta_payload": payload,
            "optimizer_state_payload": {
                "format": "drm_graft_sequence_adamw_state",
                "optimizer": "AdamW",
                "states": optimizer_states,
            },
            "drm_g": True,
            "marco": "drm_g_marco_4_progressive",
        },
    )


__all__ = ["run_drm_graft_progressive"]
