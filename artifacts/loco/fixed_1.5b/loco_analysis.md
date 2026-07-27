# Greedy leave-one-case-out analysis

The 24 authored cases are the complete benchmark corpus, not a population sample.

| Configuration | Recall | Precision | Ordering | Candidate validity | Check presence |
|---|---:|---:|---:|---:|---:|
| adapter | 0.586 | 0.583 | 0.715 | 1.000 | 1.000 |
| schema | 0.665 | 0.475 | 0.736 | 0.943 | 0.911 |
| two-shot | 0.660 | 0.480 | 0.618 | 0.958 | 0.984 |

## Format-neutral identifier extraction

| Configuration | Recall | Precision | Ordering | Candidate validity |
|---|---:|---:|---:|---:|
| adapter | 0.586 | 0.583 | 0.688 | 1.000 |
| schema | 0.654 | 0.475 | 0.653 | 0.946 |
| two-shot | 0.660 | 0.473 | 0.660 | 0.947 |

| Baseline | Metric | Mean delta | Win/tie/loss cases |
|---|---|---:|---:|
| schema | completeness | -0.078 | 6/9/9 |
| schema | precision | 0.107 | 14/3/7 |
| schema | ordering | -0.021 | 7/11/6 |
| schema | grounding | 0.057 | 3/21/0 |
| schema | check | 0.089 | 15/9/0 |
| two-shot | completeness | -0.074 | 5/7/12 |
| two-shot | precision | 0.103 | 13/3/8 |
| two-shot | ordering | 0.097 | 10/8/6 |
| two-shot | grounding | 0.042 | 5/19/0 |
| two-shot | check | 0.016 | 3/21/0 |
