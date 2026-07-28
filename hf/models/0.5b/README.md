---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
  - peft
  - lora
  - satellite-security
  - reproducibility
---

# SatSec decomposition Qwen2.5-0.5B adapters

This repository contains the 27 Qwen2.5-0.5B-Instruct LoRA adapters used in
*A Controlled Candidate-Set Benchmark for Offline Satellite-Security Plan
Decomposition*: one fixed-split seed-42 adapter, two additional training-seed
adapters, and 24 case-level LOCO adapters.

The immutable base-model revision is
`7ae557604adf67be50417f59c2c2f167def9a775`. Adapter subfolders contain only
`adapter_model.safetensors`, `adapter_config.json`, and `run_manifest.json`.
The manifests are mirrored and indexed in the
[reproducibility repository](https://github.com/paoloo/satsec-decomposition/tree/main/artifacts/manifests).

Use is limited to authorized development-time or emulated security testing.
The adapter selects from supplied candidates and does not retrieve or execute actions.
