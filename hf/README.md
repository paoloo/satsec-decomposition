---
license: cc-by-4.0
task_categories:
  - text-generation
language:
  - en
tags:
  - satellite-security
  - sparta
  - defensive-security
  - grounded-generation
pretty_name: SatSec Grounded Objective-Decomposition v2
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: test
        path: data/test-*
---

# SatSec Grounded Objective-Decomposition v2

Version 2.0 is a leakage-controlled replacement for the original dataset used in
*A Controlled Candidate-Set Benchmark for Offline Satellite-Security Plan Decomposition*.
It contains 24 authored full decompositions and 83 mechanically derived next-step rows
across 24 cases.
There are 82 train rows and 25 test rows; the six test cases never occur in training.

## Important v2 correction

The original version exposed prose describing the expected sequence inside the user
prompt. Consequently, its decomposition and next-step results measured extraction and
formatting rather than planning from a high-level objective. Do not use v1 scores as
evidence for decomposition performance.

In v2 each input starts with only `Objective: <one-line high-level objective>`, followed
by eight shuffled SPARTA candidates. Automated audits reject legacy sequence markers,
future-step titles in next-step prompts, non-disjoint case splits, malformed grounding,
and non-self-consistent gold plans.

The GNSS gold was also corrected: `EX-0002` is PNT geofencing. Signal generation,
overpowering, and controlled carry-off are represented by `EX-0014.04` PNT Spoofing.

## Schema

| Column | Meaning |
|---|---|
| `messages` | system/user/assistant chat turns |
| `type` | `decompose` or `next_step` |
| `case` | source case identifier |
| `root` | corpus attack-tree root |
| `step` | 1-based next-step index, otherwise null |
| `split` | case-disjoint `train` or `test` |
| `dataset_version` | `2.0.0` |
| `prompt_policy` | `objective-only-v2` |

The raw source-of-truth JSONL is under `raw/tuning_set.v2.jsonl`.

## Intended use and limitations

Use only for authorized, development-time testing on emulated or consented systems.
Do not use it against operational or on-orbit assets. The dataset contains concrete
identifiers and actions in assistant targets. Completion-only label masking does not
guarantee that facts remain outside learned weights or that memorization cannot occur.

Exact-identifier scoring has one curated reference per case and may penalize valid
alternative decompositions. SPARTA page checks establish identifier/name consistency,
not the correctness of every semantic mapping; expert review remains necessary.

## Loading

```python
from datasets import load_dataset
ds = load_dataset("paolocmo/satsec-decomposition")
```

## Licensing and disclosure

Dataset: CC BY 4.0. Grounding text is derived from public incident sources and SPARTA;
no ECSS standard text is redistributed. Generative AI assisted code scaffolding,
dataset/paper review, and language revision; the authors remain responsible for source
verification, mappings, experiments, and final text.

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
