#!/usr/bin/env bash
# Move one suspect LOCO fold aside so the resumable driver regenerates it cleanly.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 SIZE_TAG FOLD_DIR QUARANTINE_DIR" >&2
  exit 2
fi
SIZE_TAG="$1"
FOLD_DIR="$2"
QUARANTINE_DIR="$3"
ROOT="artifacts/loco/fixed_${SIZE_TAG}"
mkdir -p "$QUARANTINE_DIR"
mv "$ROOT/adapters/$FOLD_DIR" "$QUARANTINE_DIR/adapter"
mv "$ROOT/predictions/${FOLD_DIR}_adapter.jsonl" "$QUARANTINE_DIR/adapter.jsonl"
mv "$ROOT/predictions/${FOLD_DIR}_schema.jsonl" "$QUARANTINE_DIR/schema.jsonl"
mv "$ROOT/predictions/${FOLD_DIR}_fewshot.jsonl" "$QUARANTINE_DIR/fewshot.jsonl"
