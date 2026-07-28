# Transfer and execution diagnostic capsule

This directory is the complete public evidence capsule for the manuscript's
peripheral transfer and execution diagnostic. It is deliberately independent of
the larger reverse-engineering harness from which the experiment originated.

## What can be reproduced

The crackme packet contains the three prompts and references, all 15 API requests,
sanitized API responses, seeds, sampling settings, and a deterministic analyzer.
Recompute the reported values from the released responses with:

```bash
python -m tools.analyze_transfer_diagnostic
```

The command must report 15/15 format-neutral JSON runs, 2/15 strict-contract
runs, completeness 0.933, and precision, ordering, grounding validity, and
alias-normalized check presence of 1.000. The exact-contract mean is 0.133 with
population standard deviation 0.163 across the five seed-level macro averages.

To repeat the generation protocol against a user-supplied OpenAI-compatible
endpoint, use:

```bash
python -m tools.run_transfer_diagnostic \
  --base-url https://your-authorized-endpoint.example/v1 \
  --model YOUR_MODEL \
  --out /tmp/transfer-diagnostic-rerun
```

The original endpoint resolved the requested model to `GLM-5.2-NVFP4`, but it
did not expose an immutable weight revision or model fingerprint. The released
responses therefore support exact metric recomputation; token-identical model
regeneration is not claimed.

The CVE packet contains the complete bounded differential lab, its schema, the
successful validation, environment record, logs, and checksums. See
`cve-2025-32433/lab/README.md`. The oracle accepts only the Compose service names
`vulnerable` and `patched`, writes one fixed marker, and has no caller-controlled
host or command input.

Verify the integrity of the complete capsule with:

```bash
(cd artifacts/transfer_diagnostic && shasum -a 256 -c SHA256SUMS)
```

## Deliberate exclusions

No private endpoint, credential, internal hostname, unrelated experiment,
general-purpose exploit interface, or larger harness source tree is included.
The response packets retain only the assistant content and ordinary completion
metadata; provider reasoning fields and routing metadata were removed.
