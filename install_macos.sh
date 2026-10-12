#!/usr/bin/env bash
# Install the macOS port's pinned runtime and tests with uv.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "This setup targets Apple silicon macOS." >&2
    exit 1
fi
if ! command -v uv >/dev/null; then
    echo "Install uv first (for example: brew install uv)." >&2
    exit 1
fi
if [[ ! -d .venv ]]; then
    uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r requirements-macos.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c 'import torch; assert torch.backends.mps.is_available(), "PyTorch MPS is unavailable"'
echo "Environment ready: source .venv/bin/activate"
echo "Run tests with: python -m pytest tests -q"
echo "Single-image, multiview and dynamic execution verified; Metal fine-tuning uses FP32."
echo "See docs/macos-port.md for model-cache setup and demo commands."
