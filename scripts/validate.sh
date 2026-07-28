#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:src"
python3 tools/audit_dataset.py --data data/tuning_set.v2.jsonl \
  --report artifacts/results/dataset_audit.json
python3 tools/build_case_provenance.py
python3 -m tools.analyze_fixed_split
python3 -m pytest
python3 -m compileall -q src benchmark tools tests
