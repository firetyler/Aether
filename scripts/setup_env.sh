#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
if [ -f "ai_Model/requirements.txt" ]; then
  pip install -r ai_Model/requirements.txt
else
  echo "ai_Model/requirements.txt not found"
fi
pip install -e .
echo "Setup complete. Activate with: source .venv/bin/activate"
