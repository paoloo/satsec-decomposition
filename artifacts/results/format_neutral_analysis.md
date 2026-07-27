# Format-neutral identifier analysis

The first occurrence of each SPARTA identifier is extracted anywhere in an output.
No Plan/Technique/Action/Check layout is required. Actions and checks are not judged.

## 0.5b

| Configuration | Recall | Precision | Ordering | Candidate validity |
|---|---:|---:|---:|---:|
| two-shot | 0.281 | 0.200 | 0.500 | 0.597 |
| candidate-only | 0.300 | 0.150 | 0.250 | 0.378 |
| positional-copy | 1.000 | 0.396 | 0.444 | 1.000 |
| adapter | 0.264 | 0.273 | 0.733 | 0.968 |
| schema | 0.261 | 0.189 | 0.367 | 0.386 |

## 1.5b

| Configuration | Recall | Precision | Ordering | Candidate validity |
|---|---:|---:|---:|---:|
| positional-copy | 1.000 | 0.396 | 0.444 | 1.000 |
| adapter | 0.450 | 0.442 | 0.739 | 0.936 |
| schema | 0.614 | 0.362 | 0.711 | 0.810 |
| candidate-only | 0.431 | 0.217 | 0.500 | 0.562 |
| two-shot | 0.431 | 0.297 | 0.644 | 0.830 |

## 7b

| Configuration | Recall | Precision | Ordering | Candidate validity |
|---|---:|---:|---:|---:|
| candidate-only | 0.742 | 0.470 | 0.794 | 0.970 |
| schema | 0.703 | 0.589 | 0.744 | 1.000 |
| positional-copy | 1.000 | 0.396 | 0.444 | 1.000 |
| adapter | 0.767 | 0.672 | 0.672 | 1.000 |
| two-shot | 0.661 | 0.410 | 0.628 | 0.898 |

