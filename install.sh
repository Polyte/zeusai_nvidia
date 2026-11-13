#!/bin/bash
set -e
# install via uv if available, else pip
if command -v uv >/dev/null 2>&1; then
  uv venv .venv 2>/dev/null || true
  uv pip install -e .
  ln -sf "$(pwd)/.venv/bin/zeusai" ~/.local/bin/zeusai 2>/dev/null || true
  mkdir -p ~/.local/bin
  ln -sf "$(pwd)/.venv/bin/zeusai" ~/.local/bin/zeusai
  echo "✓ zeusai installed: ~/.local/bin/zeusai"
else
  pip install -e .
  echo "✓ zeusai installed"
fi
echo "Next: zeusai configure --profile laguna --api-key YOUR_KEY && zeusai doctor"
echo "Ensemble default: zeusai use ensemble"
