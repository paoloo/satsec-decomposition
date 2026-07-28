# SatSec grounded decomposition

This repository is the reproducibility package for *A Controlled Candidate-Set Benchmark
for Offline Satellite-Security Plan Decomposition*. The released dataset excludes plan text from model
inputs: each prompt contains only a one-line high-level objective and eight shuffled
SPARTA grounding candidates. The held-out split is disjoint by case.

The package includes the source corpus, per-case provenance, deterministic dataset builder, leakage and
gold audits, completion-masked LoRA training, prompting and non-learned baselines,
autoregressive next-step evaluation, offline scoring, and locations for raw outputs.
It is limited to authorized development-time or emulated security testing.
See `RELEASE_RISK.md` for the dual-use assessment and `data/EVALUATION_PROVENANCE.md`
for the fixed cases' source-to-reference ledger. `python tools/build_case_provenance.py`
rebuilds the complete machine-readable 24-case provenance manifest.

## Dataset

- 24 authored full decompositions and 83 mechanically derived next-step rows across 24 cases.
- 82 train and 25 test examples; six cases are held out entirely.
- Internal prompt-policy tag: `objective-only-v2`. No reference narrative or future step title is
  included in the objective.
- Each grounding block has eight distinct, shuffled, format-matched SPARTA candidates.
- GNSS spoofing maps signal generation, overpowering, and carry-off to `EX-0014.04`.
  `EX-0002` is geofenced execution and is not part of that gold plan.

The data is CC BY 4.0 (`LICENSE-DATA`); code is MIT (`LICENSE`).
The dataset DOI is [10.57967/hf/9586](https://doi.org/10.57967/hf/9586).

## Authors

- João Paolo Cavalcante Martins Oliveira — Universidade Federal do Rio Grande do Norte
  (UFRN) and SETI Institute — [ORCID 0000-0003-4117-953X](https://orcid.org/0000-0003-4117-953X)
- Lucas Teske — TeskesLab — [ORCID 0009-0002-8526-7662](https://orcid.org/0009-0002-8526-7662)
- Paulo Matias — Universidade Federal de São Carlos —
  [ORCID 0000-0002-6504-5141](https://orcid.org/0000-0002-6504-5141)

## Citation

```bibtex
@dataset{satsecdecomp2026,
  author    = {Oliveira, João Paolo Cavalcante Martins and Teske, Lucas and Matias, Paulo},
  title     = {SatSec Grounded Objective-Decomposition Dataset},
  year      = {2026},
  url       = {https://huggingface.co/datasets/paolocmo/satsec-decomposition},
  doi       = {10.57967/hf/9586},
  publisher = {Hugging Face}
}
```

## Quick validation

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
./scripts/build_dataset.sh
./scripts/validate.sh
```

The audit must report `status: pass`. The generated file's SHA-256 is recorded by
every training run in `run_manifest.json`.

## Training

Immutable model revisions are in `configs/models.json`; all hyperparameters are in
`configs/experiment.json`. Example:

```bash
REV=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
python -m satsec.training.train_lora \
  --data data/tuning_set.v2.jsonl \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --model-revision "$REV" \
  --out models/qwen2.5-1.5b-v2
```

This run uses no quantization, bfloat16 on CUDA, rank 16, alpha 32, dropout 0.05,
all Q/K/V/O and MLP projection modules, 10 epochs, effective batch 8, AdamW,
learning rate 2e-4, linear scheduling, and seed 42.

Completion masking means prompt token positions have label `-100`. The assistant loss
still backpropagates through context representations; therefore the method neither
keeps facts outside weights nor guarantees absence of memorization.

## Evaluation and artifacts

`scripts/run_fixed_eval.sh MODEL ADAPTER REV OUTDIR` runs five stochastic seeds for
adapter, candidate-only, schema, two-shot, positional-copy, and autoregressive
next-step conditions. Two-shot exemplars are held fixed by `--exemplar-seed 0`, so
generation seeds vary decoding only. Omit the adapter argument as an empty string only for a base
smoke test. Raw JSONL records include exact prompts, selected few-shot case IDs,
generation settings, model revision, and outputs. Score with:

```bash
python -m satsec.training.decomp_score \
  --data data/tuning_set.v2.jsonl \
  --pred artifacts/raw_predictions/all.jsonl
```

The paper's inferential unit is the held-out case, not a decoding seed. Reproduce the
paired case differences, exact case-resampling intervals, sign-flip tests, ordering
support, and formatting diagnostics with:

```bash
python -m tools.analyze_fixed_split
python -m tools.analyze_format_neutral
```

This writes machine-readable and reviewable summaries to
`artifacts/results/fixed_case_analysis.{json,md}`. Intervals describe sensitivity over
the six fixed, development-facing cases; they do not estimate deployment performance.
The format-neutral report extracts SPARTA identifiers anywhere in an output and therefore
separates identifier selection from compliance with the requested Plan/Technique layout.
The fixed-case report also recomputes every paired mean after omitting each authored
reference case in turn, exposing whether a direction depends on one case definition.
The manuscript's primary score files are `fixed_{size}_controlled_scores.json`, built from
`fixed_{size}_controlled_all.jsonl`. Files without `controlled` preserve the original
seed-varying-exemplar diagnostic and must not be substituted into the main table.

Additional split-sensitivity and stability controls are:

```bash
scripts/run_fixed_fewshot.sh MODEL MODEL_REVISION OUTDIR
python -m tools.analyze_fixed_fewshot
scripts/run_training_seed_eval.sh MODEL MODEL_REVISION SIZE_TAG PRIMARY_ADAPTER OUTDIR
python -m tools.analyze_training_seeds
scripts/run_loco_greedy.sh MODEL MODEL_REVISION SIZE_TAG OUTDIR
```

The training-seed run evaluates three independently trained adapters greedily, avoiding
decoding noise. The case-level LOCO driver trains a fresh adapter for each of 24 held-out cases and
evaluates adapter, schema, and fixed-exemplar two-shot conditions greedily.

Final LOCO reports are in `artifacts/loco/fixed_{0.5b,1.5b}/loco_analysis.{json,md}`.
The analyzer validates all 24 folds, greedy decoding, exemplar exclusion, training seed,
and each training manifest's fold-data hash before reporting means and paired
win/tie/loss counts. This is a census-style case-disjoint diagnostic over the authored
corpus, not family-disjoint validation or an estimate for a target population. It also
reports format-neutral identifier extraction from the same
outputs so layout compliance cannot silently drive the LOCO comparison.

The training manifests used by the paper are tracked under `artifacts/manifests/`:
three primary fixed-split manifests, six additional training-seed manifests, and all
48 LOCO fold manifests. Adapter weights are hosted separately and indexed by immutable
revision and SHA-256 in `artifacts/manifests/index.json`; base-model weights are not
redistributed.

## Transfer and execution diagnostic

The self-contained public packet under `artifacts/transfer_diagnostic/` contains the
three crackme prompts and references, 15 sanitized GLM-5.2-NVFP4 API request/response
pairs, the bounded CVE-2025-32433 differential lab, retained logs, environment metadata,
validation, and checksums. Recompute the manuscript values with:

```bash
python -m tools.analyze_transfer_diagnostic
```

The released responses support exact metric recomputation. The original service did not
expose an immutable GLM weight revision or fingerprint, so token-identical generation is
not claimed. `python -m tools.run_transfer_diagnostic --help` documents how to repeat the
protocol against a user-supplied OpenAI-compatible endpoint.

`tools/loco_split.py` constructs each leave-one-case-out fold. A fresh adapter must be
trained per fold; never reuse the fixed-split adapter. `tools/factsep_relabel.py` is a
controlled grounding-relabeling probe. It measures whether outputs track a supplied
identifier mapping, not whether facts are absent from weights.

Commit raw predictions, score tables, run manifests, and training logs for the paper's
final runs. Publish adapters in the linked Hugging Face model repositories; do not commit
model caches, adapter weights, or base weights to GitHub.

## Reproducibility notes

- Reference environment: Python 3.10, PyTorch 2.4.1+cu121, Transformers 4.57.6,
  PEFT 0.19.1, Datasets 3.6.0, Accelerate 1.14.0.
- `Dockerfile` starts from the corresponding official PyTorch CUDA image.
- Results from an earlier internal pilot are invalid because its inputs exposed the reference
  sequence. Only the leakage-controlled results in this package may be reported.
- Technique-page existence and title matching can be checked online with
  `python tools/audit_sparta.py --online`. Semantic mappings still require expert
  review; the audit does not pretend to automate that judgment.
- References are author-defined benchmark operationalizations collectively reviewed against
  the recorded sources and SPARTA, not independently certified ground truth. The complete
  provenance report gives every case a source and audits step fields, candidate inclusion,
  and all 34 unique reference identifiers. The blank expert-audit packet is an optional
  future author/non-author comparison, not a paper acceptance gate; its analyzer fails closed
  only if that separate study is undertaken and incompletely reported.

## AI-use disclosure

Generative AI tools assisted with code scaffolding, dataset and manuscript review, and
language revision. The authors verified the dataset construction, semantic mappings,
experimental results, references, and final manuscript and take full responsibility for
the work. This disclosure should remain consistent with the paper and submission form.
