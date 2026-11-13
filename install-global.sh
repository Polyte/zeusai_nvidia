#!/bin/bash
# Global CLI install for zeusai (opencode-style)
set -e
cd "$(dirname "$0")"
echo "Installing zeusai globally..."
if command -v uv >/dev/null 2>&1; then
  uv venv .venv 2>/dev/null || true
  uv pip install -e .
  mkdir -p ~/.local/bin
  ln -sf "$(pwd)/.venv/bin/zeusai" ~/.local/bin/zeusai
  echo "✓ Linked ~/.local/bin/zeusai -> $(pwd)/.venv/bin/zeusai"
else
  pip install -e .
  echo "✓ pip install done"
fi
export PATH="$HOME/.local/bin:$PATH"
echo ""
echo "Check:"
zeusai --help 2>&1 | head -20
echo ""
echo "Configure (once):"
echo "  zeusai configure --profile laguna --api-key YOUR_KEY"
echo "  zeusai doctor"
echo ""
echo "Use like opencode:"
echo "  zeusai                    # full-screen TUI"
echo "  zeusai ~/code/my-project  # TUI in that dir"
echo "  zeusai mini               # light REPL"
echo "  zeusai run \"explain repo\" # automation"
