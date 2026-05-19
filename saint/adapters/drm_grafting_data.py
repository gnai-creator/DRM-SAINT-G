"""Token loading helpers for DRM-G grafting."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _default_drm_root() -> Path:
    return Path(__file__).resolve().parents[3] / "drm_transformer"


def real_token_batch(
    torch,
    metadata: dict[str, Any],
    vocab_size: int,
    device: str,
    *,
    split: str,
    seed_key: str,
) -> tuple[Any, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to load real DRM token shards.") from exc

    drm_root = Path(metadata.get("drm_root") or _default_drm_root()).resolve()
    data_dir = Path(metadata.get("real_data_dir", "data/baseline"))
    if not data_dir.is_absolute():
        data_dir = drm_root / data_dir
    split_dir = data_dir / split
    shards = sorted(split_dir.glob("*.npy"))
    if not shards:
        shards = sorted(data_dir.glob("shard_*.npy"))
    if not shards:
        raise FileNotFoundError(f"no .npy token shards found in {split_dir} or {data_dir}")

    batch_size = int(metadata.get("batch_size", 1))
    seq_len = int(metadata.get("seq_len", 16))
    seed = int(metadata.get(seed_key, metadata.get("data_seed", 991)))
    span = batch_size * (seq_len + 1)
    shard = np.load(shards[seed % len(shards)], mmap_mode="r")
    flat = shard.reshape(-1)
    if flat.shape[0] <= span:
        raise ValueError(f"token shard is too small for batch: {shards[0]}")
    offset = int(metadata.get(f"{split}_token_offset", seed * 997)) % (flat.shape[0] - span)
    values = flat[offset : offset + span].astype("int64", copy=False)
    data = torch.tensor(values, dtype=torch.long).reshape(batch_size, seq_len + 1)
    data = data.remainder(int(vocab_size))
    return data[:, :-1].contiguous().to(device), data[:, 1:].contiguous().to(device)


def token_batch(
    torch,
    metadata: dict[str, Any],
    vocab_size: int,
    device: str,
    *,
    seed_key: str = "data_seed",
):
    split = str(metadata.get("validation_split", "val" if seed_key == "validation_seed" else "train"))
    if bool(metadata.get("use_real_tokens", False)):
        return real_token_batch(
            torch,
            metadata,
            vocab_size,
            device,
            split=split,
            seed_key=seed_key,
        )
    batch_size = int(metadata.get("batch_size", 1))
    seq_len = int(metadata.get("seq_len", 16))
    seed = int(metadata.get(seed_key, metadata.get("data_seed", 991)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    data = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=generator)
    return data[:, :-1].contiguous().to(device), data[:, 1:].contiguous().to(device)


__all__ = ["real_token_batch", "token_batch"]
