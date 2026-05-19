"""Consolidated artifact export for DRM-G grafts."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from saint.adapters.drm_grafting import (
    _baseline_path,
    _import_drm,
    _loss,
    load_drm_baseline_config,
)
from saint.adapters.drm_grafting_data import token_batch
from saint.adapters.drm_grafting_eval import _mean_eval_payload
from saint.adapters.drm_grafting_merge import merge_linear_grafts_into_state
from saint.checkpoints import require_graft_payload, validate_checkpoint_bundle
from saint.checkpoints.robust import sha256_file
from saint.config import RuntimeConfig
from saint.transformer.training import MiniTransformerResult


def _state_dict_from_file(torch, path: Path) -> dict:
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(state.get(key), dict):
                return state[key]
        if all(hasattr(value, "shape") for value in state.values()):
            return state
    raise ValueError("consolidated artifact does not contain a state_dict")


def _eval_saved_model(torch, model_cls, drm_config, artifact: Path, metadata: dict, device: str, seed: int) -> float:
    total = 0.0
    batches = max(1, int(metadata.get("validation_batches", 1)))
    state_dict = _state_dict_from_file(torch, artifact)
    for index in range(batches):
        local = dict(metadata)
        split = str(local.get("validation_split", "val"))
        local[f"{split}_token_offset"] = int(local.get(f"{split}_token_offset", 0)) + index * 4096
        inputs, targets = token_batch(
            torch, local, drm_config.vocab_size, device, seed_key="validation_seed"
        )
        torch.manual_seed(seed)
        model = model_cls(drm_config).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        total += float(_loss(model, inputs, targets).detach().cpu().item())
    return total / batches


def run_drm_graft_consolidate(config: RuntimeConfig) -> MiniTransformerResult:
    start = perf_counter()
    metadata = dict(config.metadata or {})
    run_dir = metadata.get("graft_run")
    if not run_dir:
        raise ValueError("drm_g_saint_phi_consolidate requires metadata.graft_run")
    checkpoint = validate_checkpoint_bundle(run_dir)
    payload = require_graft_payload(checkpoint, run_dir)
    torch, config_cls, model_cls, drm_root = _import_drm(metadata)
    device = str(metadata.get("device", "cpu"))
    seed = int(metadata.get("seed", config.seed))
    drm_config = load_drm_baseline_config(metadata, config_cls)
    base_loss, hook_loss = _mean_eval_payload(
        torch, model_cls, drm_config, payload, metadata, device, seed
    )
    torch.manual_seed(seed)
    model = model_cls(drm_config)
    merge_summary = merge_linear_grafts_into_state(torch, model, payload)
    artifact = Path(config.output_dir) / str(metadata.get("artifact_name", "consolidated_model.pt"))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "merge_summary": merge_summary}, str(artifact))
    saved_loss = _eval_saved_model(torch, model_cls, drm_config, artifact, metadata, device, seed)
    params = sum(int(item.get("trainable_parameters", 0)) for item in payload.get("grafts", []))
    return MiniTransformerResult(
        name="drm_g_saint_phi_consolidate",
        train_loss=saved_loss,
        test_loss=saved_loss,
        parameter_count=params,
        optimizer_state_values=0,
        elapsed_s=perf_counter() - start,
        metadata={
            "baseline_config": str(_baseline_path(metadata, drm_root)),
            "graft_run": str(run_dir),
            "artifact_path": str(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "artifact_sha256": sha256_file(artifact),
            "base_loss": base_loss,
            "hook_loss": hook_loss,
            "saved_loss": saved_loss,
            "validation_gain": base_loss - saved_loss,
            "saved_loss_abs_diff": abs(saved_loss - hook_loss),
            "state_merge": merge_summary,
            "marco": "drm_g_marco_5a_consolidated_artifact",
        },
    )


__all__ = ["run_drm_graft_consolidate"]
