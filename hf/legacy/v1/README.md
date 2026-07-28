# Legacy v1 snapshot

`tuning_set.jsonl` is the superseded 110-row snapshot retained only for historical
audit. It exposed prose describing the expected sequence inside user prompts and must
not be used as evidence for decomposition performance.

The leakage-controlled source used by the paper is
`../../raw/tuning_set.v2.jsonl` (107 rows, dataset version 2.0.0, prompt policy
`objective-only-v2`). The default Parquet configuration is generated from that v2
source.
