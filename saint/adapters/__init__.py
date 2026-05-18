"""Model adapters for SAINT runtime."""

from saint.adapters import drm_transformer, huggingface, mini_transformer


def _select(config):
    if config.task == "mini_transformer":
        return mini_transformer
    if config.task == "drm_transformer":
        return drm_transformer
    if config.task == "huggingface_causal_lm":
        return huggingface
    raise ValueError(f"unknown runtime task: {config.task}")


def inspect_model(config):
    return _select(config).inspect_model(config)


def make_task(config):
    return _select(config).make_task(config)


def run_method(config):
    return _select(config).run_method(config)


__all__ = ["inspect_model", "make_task", "run_method"]
