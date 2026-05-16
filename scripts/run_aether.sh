#!/usr/bin/env bash
if [ ! -f ".venv/bin/activate" ]; then
  echo "Virtualenv not found. Run scripts/setup_env.sh first."
  exit 1
fi
source .venv/bin/activate
python -m ai_Model.aether2 "$@"
