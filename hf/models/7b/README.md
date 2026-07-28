---
base_model: Qwen/Qwen2.5-7B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
  - peft
  - lora
  - satellite-security
  - reproducibility
---

# SatSec decomposition Qwen2.5-7B adapters

This repository contains the three Qwen2.5-7B-Instruct LoRA adapters used in
*A Controlled Candidate-Set Benchmark for Offline Satellite-Security Plan
Decomposition*: the fixed-split seed-42 adapter and two additional training-seed
adapters.

The immutable base-model revision is
`a09a35458c702b33eeacc393d103063234e8bc28`. Adapter subfolders contain only
`adapter_model.safetensors`, `adapter_config.json`, and `run_manifest.json`.
The manifests are mirrored and indexed in the
[reproducibility repository](https://github.com/paoloo/satsec-decomposition/tree/main/artifacts/manifests).

Use is limited to authorized development-time or emulated security testing.
The adapter selects from supplied candidates and does not retrieve or execute actions.
