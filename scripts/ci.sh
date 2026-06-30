#!/usr/bin/env bash
set -euo pipefail

if [[ -f .venv/bin/activate ]]; then
    # Local agent/developer workflow: use the project virtualenv when present.
    source .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pytest tests/ -v --cov=dragon3_pipelines --cov-report=xml --cov-report=term
