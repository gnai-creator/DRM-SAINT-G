"""Optimizer serialization for DRM-G grafting."""

from __future__ import annotations

from typing import Any


def _tensor_to_json(value: Any) -> Any:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data": tensor.tolist(),
        }
    if isinstance(value, dict):
        return {str(key): _tensor_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_to_json(item) for item in value]
    return value


def optimizer_to_payload(optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    return {
        "format": "drm_graft_adamw_state",
        "optimizer": "AdamW",
        "state_dict": _tensor_to_json(state),
    }


__all__ = ["optimizer_to_payload"]
