# Security Policy

SAINT-G is experimental research software. It is not production-hardened and
should not be used to train or serve sensitive models without additional review.

## Supported Versions

Only the current `main` branch is considered active.

## Reporting a Vulnerability

Please report security issues privately to the repository maintainer before
opening a public issue. Include:

- affected file or command;
- reproduction steps;
- expected impact;
- relevant logs or stack traces;
- whether external model files, checkpoints, or datasets are required.

## Areas of Concern

Pay particular attention to:

- unsafe checkpoint loading;
- arbitrary code execution through model adapters;
- path traversal in runtime, checkpoint, or merge commands;
- unsafe deserialization;
- accidental inclusion of private datasets, model weights, or API keys;
- untrusted third-party model files.

## Checkpoint Safety

Checkpoint formats such as PyTorch `.pt` files may execute or trigger unsafe
deserialization behavior depending on how they are loaded. Only load checkpoints
from trusted sources.

## Secret Handling

Do not commit:

- API keys;
- private datasets;
- proprietary checkpoints;
- credentials;
- local machine paths that reveal sensitive information.

## Dependency Policy

The core package should remain dependency-light. New dependencies must be
justified by the phase that introduces them and should be documented.
