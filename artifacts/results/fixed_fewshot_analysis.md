# Fixed-exemplar two-shot control

The controlled row fixes the two training exemplars while varying only decoding seed.

| Size | Condition | Recall | Precision | Ordering | Candidate validity | Check presence |
|---|---|---:|---:|---:|---:|---:|
| 0.5b | varying exemplars | 0.317 +/- 0.117 | 0.273 +/- 0.092 | 0.600 +/- 0.190 | 0.719 +/- 0.144 | 0.961 +/- 0.053 |
| 0.5b | fixed exemplars | 0.281 +/- 0.081 | 0.200 +/- 0.051 | 0.500 +/- 0.118 | 0.591 +/- 0.059 | 0.932 +/- 0.049 |
| 1.5b | varying exemplars | 0.508 +/- 0.201 | 0.318 +/- 0.123 | 0.600 +/- 0.181 | 0.745 +/- 0.168 | 0.895 +/- 0.094 |
| 1.5b | fixed exemplars | 0.319 +/- 0.237 | 0.217 +/- 0.157 | 0.378 +/- 0.224 | 0.664 +/- 0.315 | 0.686 +/- 0.344 |
| 7b | varying exemplars | 0.711 +/- 0.041 | 0.464 +/- 0.058 | 0.761 +/- 0.179 | 0.921 +/- 0.025 | 0.941 +/- 0.044 |
| 7b | fixed exemplars | 0.661 +/- 0.043 | 0.412 +/- 0.037 | 0.744 +/- 0.205 | 0.917 +/- 0.023 | 0.920 +/- 0.019 |
