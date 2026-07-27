# Training-seed stability

Each cell is mean +/- sample standard deviation across three independently trained
adapters. Every adapter is evaluated greedily on the same six fixed cases.

| Size | Recall | Precision | Ordering | Candidate validity | Check presence |
|---|---:|---:|---:|---:|---:|
| 0.5b | 0.287 +/- 0.049 | 0.347 +/- 0.069 | 0.667 +/- 0.167 | 0.981 +/- 0.032 | 1.000 +/- 0.000 |
| 1.5b | 0.463 +/- 0.113 | 0.481 +/- 0.099 | 0.648 +/- 0.140 | 0.972 +/- 0.024 | 1.000 +/- 0.000 |
| 7b | 0.806 +/- 0.042 | 0.727 +/- 0.098 | 0.806 +/- 0.073 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
