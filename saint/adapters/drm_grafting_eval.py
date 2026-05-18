"""Evaluation path for recomposed DRM-G graft payloads."""

from __future__ import annotations

from time import perf_counter

from saint.adapters.drm_grafting import (
    _baseline_path,
    _freeze,
    _import_drm,
    _loss,
    _target_module,
    _consolidation_summary,
    load_drm_baseline_config,
)
from saint.adapters.drm_grafting_data import token_batch
from saint.adapters.drm_grafting_decision import consolidation_payload
from saint.adapters.drm_grafting_modules import PhiHiddenGraft
from saint.config import RuntimeConfig
from saint.transformer.training import MiniTransformerResult


def _eval_payload(
    torch,
    model_cls,
    drm_config,
    payload: dict,
    device: str,
    eval_inputs,
    eval_targets,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    model = model_cls(drm_config).to(device)
    model.eval()
    _freeze(model)
    graft = PhiHiddenGraft.from_payload(torch, payload).to(device)
    handle = _target_module(model, str(payload["target_module"])).register_forward_hook(graft.hook)
    try:
        return float(_loss(model, eval_inputs, eval_targets).detach().cpu().item())
    finally:
        handle.remove()


def run_drm_graft_eval(config: RuntimeConfig) -> MiniTransformerResult:
    start = perf_counter()
    metadata = dict(config.metadata or {})
    run_dir = metadata.get("graft_run")
    if not run_dir:
        raise ValueError("drm_g_saint_phi_eval requires metadata.graft_run")
    from saint.checkpoints import require_graft_payload, validate_checkpoint_bundle

    checkpoint = validate_checkpoint_bundle(run_dir)
    payload = require_graft_payload(checkpoint, run_dir)
    torch, config_cls, model_cls, drm_root = _import_drm(metadata)
    device = str(metadata.get("device", "cpu"))
    seed = int(metadata.get("seed", config.seed))
    drm_config = load_drm_baseline_config(metadata, config_cls)
    eval_inputs, eval_targets = token_batch(
        torch,
        metadata,
        drm_config.vocab_size,
        device,
        seed_key="validation_seed",
    )
    torch.manual_seed(seed)
    base = model_cls(drm_config).to(device)
    base.eval()
    base_loss = float(_loss(base, eval_inputs, eval_targets).detach().cpu().item())
    del base
    graft_loss = _eval_payload(
        torch, model_cls, drm_config, payload, device, eval_inputs, eval_targets, seed
    )
    consolidation_model = model_cls(drm_config)
    consolidation = consolidation_payload(torch, consolidation_model, payload)
    gain = base_loss - graft_loss
    params = int(payload.get("trainable_parameters", 0))
    return MiniTransformerResult(
        name="drm_g_saint_phi_eval",
        train_loss=graft_loss,
        test_loss=graft_loss,
        parameter_count=params,
        optimizer_state_values=0,
        elapsed_s=perf_counter() - start,
        metadata={
            "baseline_config": str(_baseline_path(metadata, drm_root)),
            "graft_run": str(run_dir),
            "base_loss": base_loss,
            "graft_loss": graft_loss,
            "validation_gain": gain,
            "validation_gain_per_parameter": gain / max(1, params),
            "target_module": payload.get("target_module"),
            "projection_init": payload.get("projection_init"),
            "payload_recomposed": True,
            "consolidation": _consolidation_summary(consolidation),
            "consolidation_payload": consolidation,
            "marco": "drm_g_marco_3",
        },
    )


__all__ = ["run_drm_graft_eval"]
