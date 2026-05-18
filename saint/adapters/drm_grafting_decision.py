"""Decision and consolidation helpers for DRM-G grafting."""

from __future__ import annotations

from typing import Any


def evaluate_graft_decision(metadata: dict[str, Any]) -> dict[str, Any]:
    gain = float(metadata.get("validation_gain", 0.0))
    gain_per_param = float(metadata.get("validation_gain_per_parameter", 0.0))
    dense_gain = float(metadata.get("dense_budget_gain", 0.0))
    min_gain = float(metadata.get("min_validation_gain", 0.0))
    min_gain_per_param = float(metadata.get("min_gain_per_parameter", 0.0))
    require_beats_dense = bool(metadata.get("require_beats_dense", True))
    reasons = []
    if gain <= min_gain:
        reasons.append(f"validation_gain <= {min_gain}")
    if gain_per_param <= min_gain_per_param:
        reasons.append(f"gain_per_parameter <= {min_gain_per_param}")
    if require_beats_dense and gain < dense_gain:
        reasons.append("does not beat dense budget baseline")
    return {
        "decision": "reject" if reasons else "approve",
        "approved": not reasons,
        "validation_gain": gain,
        "validation_gain_per_parameter": gain_per_param,
        "dense_budget_gain": dense_gain,
        "require_beats_dense": require_beats_dense,
        "reasons": reasons or ["passes configured thresholds"],
    }


def graft_equivalent_matrix(torch, payload: dict[str, Any]):
    left = torch.tensor(payload["left"], dtype=torch.float32)
    phi = torch.tensor(payload["phi"], dtype=torch.float32)
    right = torch.tensor(payload["right"], dtype=torch.float32)
    return float(payload.get("scale", 1.0)) * left.matmul(phi).matmul(right)


def consolidation_payload(torch, model, payload: dict[str, Any]) -> dict[str, Any]:
    target_module = str(payload.get("target_module", ""))
    modules = dict(model.named_modules())
    target = modules.get(target_module)
    matrix = graft_equivalent_matrix(torch, payload)
    supported = target is not None and hasattr(target, "weight")
    result: dict[str, Any] = {
        "format": "drm_graft_consolidation",
        "target_module": target_module,
        "equivalent_matrix_shape": list(matrix.shape),
        "state_dict_merge_supported": bool(supported),
        "merge_kind": "right_output_transform" if supported else "hook_required",
        "note": (
            "Exact state merge is supported only for affine modules whose output "
            "is directly transformed by the graft."
        ),
    }
    if supported:
        weight = target.weight.detach().cpu().float()
        if weight.ndim == 2 and weight.shape[0] == matrix.shape[0]:
            delta = matrix.transpose(0, 1).matmul(weight)
            result["delta_weight_shape"] = list(delta.shape)
            result["delta_weight"] = delta.tolist()
        else:
            result["state_dict_merge_supported"] = False
            result["merge_kind"] = "hook_required"
            result["reason"] = "target weight shape is incompatible with graft matrix"
    return result


__all__ = ["consolidation_payload", "evaluate_graft_decision", "graft_equivalent_matrix"]
