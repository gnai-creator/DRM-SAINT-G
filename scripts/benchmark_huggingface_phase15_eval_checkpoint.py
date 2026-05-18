"""Evaluate a Phase 15 SAINT sparse checkpoint in a separate process."""

from __future__ import annotations

import argparse
from json import dumps
from pathlib import Path
from typing import Any

from saint.adapters.huggingface_validation import (
    _evaluate_sparse_delta,
    load_text_corpus,
    split_texts,
)
from saint.checkpoints import require_sparse_delta_payload, validate_checkpoint_bundle


def _hf_metadata(args) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "hf_device_map": args.hf_device_map,
            "hf_max_memory": args.hf_max_memory,
            "hf_offload_folder": args.hf_offload_folder,
        }.items()
        if value
    }


def run(args) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    checkpoint = validate_checkpoint_bundle(run_dir)
    payload = require_sparse_delta_payload(checkpoint, run_dir)
    texts = load_text_corpus(args.corpus)
    _, validation_texts = split_texts(texts)
    result = _evaluate_sparse_delta(
        args.model,
        payload,
        validation_texts=validation_texts[: max(1, args.validation_texts)],
        device_name=args.device,
        max_length=args.max_length,
        model_dtype=args.model_dtype,
        hf_load_metadata=_hf_metadata(args),
    )
    result.update(
        {
            "status": "ok",
            "run_dir": str(run_dir),
            "parameter_count": checkpoint.get("parameter_count"),
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--corpus", default="data/tinyshakespeare_phase13.txt")
    parser.add_argument("--out", default="runs/phase15_eval_checkpoint.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--validation-texts", type=int, default=1)
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument("--hf-max-memory", default=None)
    parser.add_argument("--hf-offload-folder", default=None)
    args = parser.parse_args()
    print(dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
