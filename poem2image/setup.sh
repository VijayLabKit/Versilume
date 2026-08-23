#!/usr/bin/env bash
# One-time setup: install dependencies and download required model weights.
# Run this on a machine with internet access to huggingface.co (this
# sandbox that built the code does NOT have that access -- see README).
set -euo pipefail

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading spaCy English model (for visual-element / entity extraction)..."
python -m spacy download en_core_web_sm

echo "Done. Copy .env.example to .env and configure MODEL_PROVIDER / MODEL_NAME "
echo "and IMAGE_MODEL_PROVIDER / IMAGE_MODEL_NAME before running the API."
