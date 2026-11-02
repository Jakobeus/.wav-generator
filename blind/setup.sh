#!/usr/bin/env bash
# Create venv + install deps for the blind tone pipeline.
set -e
cd "$(dirname "$0")"
uv venv .venv
uv pip install -p .venv/bin/python \
  demucs \
  audio-separator \
  neural-amp-modeler \
  soundfile scipy numpy
echo "Done. Activate with: source .venv/bin/activate"
