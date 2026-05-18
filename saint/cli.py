"""Command line interface for the SAINT runtime."""

from __future__ import annotations

import argparse
import json

from saint.config import RuntimeConfig, load_config
from saint.runtime import (
    estimate_runtime,
    inspect_runtime,
    load_and_train,
    merge_runtime,
    resume_runtime,
)


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _config_from_optional(path: str | None) -> RuntimeConfig:
    return load_config(path) if path else RuntimeConfig()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="saint")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("--config")

    estimate_parser = sub.add_parser("estimate")
    estimate_parser.add_argument("--config")
    estimate_parser.add_argument("--vram-gb", type=float)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)

    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--run", required=True)

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--run", required=True)
    merge_parser.add_argument("--matrix", action="append")

    args = parser.parse_args(argv)
    if args.command == "inspect":
        _print(inspect_runtime(_config_from_optional(args.config)))
    elif args.command == "estimate":
        config = _config_from_optional(args.config)
        if args.vram_gb is not None:
            config = RuntimeConfig(**{**config.__dict__, "vram_gb": args.vram_gb})
        _print(estimate_runtime(config))
    elif args.command == "train":
        _print(load_and_train(args.config))
    elif args.command == "resume":
        _print(resume_runtime(args.run))
    elif args.command == "merge":
        matrices = set(args.matrix) if args.matrix else None
        _print(merge_runtime(args.run, matrix_names=matrices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
