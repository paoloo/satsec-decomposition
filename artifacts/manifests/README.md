# Training manifests

This tree publishes the exact `run_manifest.json` files for every adapter-backed
result in the paper:

- `fixed/`: three seed-42 fixed-split adapters;
- `training_seed/`: the six additional seed-43 and seed-44 adapters (the three
  seed-42 members are the fixed adapters above);
- `loco/`: 48 seed-42 leave-one-case-out adapters across 0.5B and 1.5B.

There are 57 distinct trained adapters and 57 manifests. Each manifest records the
base-model revision, dataset path and SHA-256, software versions, LoRA configuration,
precision, target modules, and training seed. The exact 57 PEFT configuration files are
retained beside their corresponding manifests.

`adapter_files.json` records the SHA-256 and byte count of every adapter weight.
`index.json` binds those records and every manifest SHA-256 to an immutable Hugging
Face commit and subfolder. Adapter weights and base-model weights are intentionally
excluded from GitHub. The three model repositories and the dataset also carry the
release tag `v2.0.1`.
