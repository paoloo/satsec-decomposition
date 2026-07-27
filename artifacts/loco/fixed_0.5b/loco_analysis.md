# Greedy leave-one-case-out analysis

The 24 authored cases are the complete benchmark corpus, not a population sample.

| Configuration | Recall | Precision | Ordering | Candidate validity | Check presence |
|---|---:|---:|---:|---:|---:|
| adapter | 0.306 | 0.353 | 0.708 | 0.938 | 1.000 |
| schema | 0.278 | 0.154 | 0.208 | 0.419 | 0.875 |
| two-shot | 0.321 | 0.335 | 0.667 | 0.717 | 0.991 |

## Format-neutral identifier extraction

| Configuration | Recall | Precision | Ordering | Candidate validity |
|---|---:|---:|---:|---:|
| adapter | 0.306 | 0.353 | 0.667 | 0.944 |
| schema | 0.319 | 0.175 | 0.257 | 0.418 |
| two-shot | 0.321 | 0.335 | 0.667 | 0.697 |

| Baseline | Metric | Mean delta | Win/tie/loss cases |
|---|---|---:|---:|
| schema | completeness | 0.028 | 12/3/9 |
| schema | precision | 0.199 | 16/4/4 |
| schema | ordering | 0.500 | 16/5/3 |
| schema | grounding | 0.518 | 17/6/1 |
| schema | check | 0.125 | 17/7/0 |
| two-shot | completeness | -0.015 | 5/12/7 |
| two-shot | precision | 0.018 | 9/6/9 |
| two-shot | ordering | 0.042 | 5/15/4 |
| two-shot | grounding | 0.221 | 15/7/2 |
| two-shot | check | 0.009 | 2/22/0 |
