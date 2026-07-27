#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CODE_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$CODE_ROOT/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/submission"
STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/decomp-adapt-package.XXXXXX")"
STAGE_CODE="$STAGE_ROOT/decomp-adapt"
STAGE_TRANSFER="$STAGE_CODE/reverse-harness-transfer"

cleanup() {
  rm -rf -- "$STAGE_ROOT"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR" "$STAGE_CODE" "$STAGE_TRANSFER"

# zip updates an existing archive and retains entries removed from the source tree.
# Recreate every deliverable so deleted or renamed audit templates cannot survive.
rm -f -- \
  "$OUTPUT_DIR/decomp-adapt-anonymous-artifact.tar.gz" \
  "$OUTPUT_DIR/decomp-adapt-sbseg-anonymous-source.zip" \
  "$OUTPUT_DIR/decomp-adapt-expert-audit.zip" \
  "$OUTPUT_DIR/SHA256SUMS"

rsync -a \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.log' \
  --exclude 'collect_atadev.sh' \
  --exclude 'package_anonymous_submission.sh' \
  --exclude 'hf' \
  "$CODE_ROOT/README.md" \
  "$CODE_ROOT/RELEASE_RISK.md" \
  "$CODE_ROOT/Dockerfile" \
  "$CODE_ROOT/LICENSE" \
  "$CODE_ROOT/LICENSE-DATA" \
  "$CODE_ROOT/pyproject.toml" \
  "$CODE_ROOT/requirements-lock.txt" \
  "$CODE_ROOT/benchmark" \
  "$CODE_ROOT/configs" \
  "$CODE_ROOT/data" \
  "$CODE_ROOT/models" \
  "$CODE_ROOT/artifacts" \
  "$CODE_ROOT/scripts" \
  "$CODE_ROOT/src" \
  "$CODE_ROOT/tests" \
  "$CODE_ROOT/tools" \
  "$STAGE_CODE/"

perl -pi -e 's/Joao Paolo Cavalcante Martins Oliveira/Anonymous Authors/g' \
  "$STAGE_CODE/LICENSE" "$STAGE_CODE/pyproject.toml"
perl -pi -e 's/Designed to run inside the paolo-dev container on atadev:/Designed to run inside the reference CUDA container:/g' \
  "$STAGE_CODE/src/satsec/training/train_lora.py"

# Curated transfer evidence used by the manuscript. Nested paper drafts, Git
# history, and transport-only launch failures are intentionally excluded.
(cd "$PROJECT_ROOT/reverse-harness" && rsync -aR \
  package.json \
  package-lock.json \
  src/decomposition-metrics.mjs \
  src/schema-registry.mjs \
  scripts/run-openai-decomposition-arm.mjs \
  scripts/aggregate-format-neutral-decomposition-arm.mjs \
  scripts/aggregate-decomposition-arm.mjs \
  scripts/score-decomposition.mjs \
  scripts/validate-instance.mjs \
  schemas/differential-lab-validation.schema.json \
  labs/cve-2025-32433 \
  experiments/base-planner-inside-harness \
  experiments/real-program-erlang-ssh \
  experiments/table-i-crackme-bridge/README.md \
  experiments/table-i-crackme-bridge/prompts.jsonl \
  experiments/table-i-crackme-bridge/seeds.json \
  experiments/table-i-crackme-bridge/references \
  experiments/table-i-crackme-bridge/internal-glm-5.2-seeded-format-neutral.json \
  experiments/table-i-crackme-bridge/runs/internal-glm-5.2-seeded-final \
  "$STAGE_TRANSFER/")

find "$STAGE_TRANSFER" -type f -exec perl -pi -e \
  's#/Users/paolo/Workspace/Papers/msc/decomp-adapt/reverse-harness/#reverse-harness-transfer/#g;
   s#/home/paolo/reverse-harness-cve-repro-20260727#<workspace>#g;
   s/\batadev\b/remote Docker host/g;
   s/dev-coyote1/reproduction-host/g;
   s/Teske.s Lab/Anonymous Lab/g;
   s/teskeslab\.dev/anonymous.invalid/g' {} +

(cd "$STAGE_TRANSFER" && shasum -a 256 \
  experiments/real-program-erlang-ssh/runs/reproduction-20260727/validation.json \
  experiments/real-program-erlang-ssh/runs/reproduction-20260727/verify-lab-2.log \
  experiments/real-program-erlang-ssh/runs/reproduction-20260727/environment.txt \
  labs/cve-2025-32433/compose.yaml \
  labs/cve-2025-32433/scripts/verify-lab.sh \
  labs/cve-2025-32433/oracle/bounded_probe.py \
  labs/cve-2025-32433/server/lab_ssh_server.escript \
  > experiments/real-program-erlang-ssh/runs/reproduction-20260727/SHA256SUMS)

if rg -n -i 'Joao Paolo|/Users/paolo|/home/paolo|paolo-dev|atadev|dev-coyote|Teske|teskeslab' "$STAGE_CODE" \
    --glob '!vocab.json' --glob '!tokenizer.json' --glob '!*.safetensors'; then
  echo "identity check failed; archive not created" >&2
  exit 1
fi

tar -czf "$OUTPUT_DIR/decomp-adapt-anonymous-artifact.tar.gz" \
  -C "$STAGE_ROOT" decomp-adapt

zip -q -j "$OUTPUT_DIR/decomp-adapt-sbseg-anonymous-source.zip" \
  "$PROJECT_ROOT/paper/paper.tex" \
  "$PROJECT_ROOT/paper/references.bib" \
  "$PROJECT_ROOT/paper/sbc-template.sty"

(cd "$CODE_ROOT/artifacts" && zip -q -r \
  "$OUTPUT_DIR/decomp-adapt-expert-audit.zip" expert_audit)

(cd "$OUTPUT_DIR" && shasum -a 256 \
  decomp-adapt-anonymous-artifact.tar.gz \
  decomp-adapt-sbseg-anonymous-source.zip \
  decomp-adapt-expert-audit.zip \
  > SHA256SUMS)

echo "wrote anonymous packages to $OUTPUT_DIR"
