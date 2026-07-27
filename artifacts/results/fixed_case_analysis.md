# Corrected v2 fixed-split exploratory analysis

The six held-out cases are the paired units. Generation seeds are averaged within
each case. Intervals resample these six cases; they do not establish population-level
generalization. Exact sign-flip tests have very low resolution at n=6 and are reported
as diagnostics, not as a pass/fail significance filter.

## 0.5b

### Adapter minus prompted baseline

| Baseline | Metric | Mean delta | 95% case-resampling interval | Exact p |
|---|---:|---:|---:|---:|
| schema | completeness | 0.075 | [-0.078, 0.219] | 0.406 |
| schema | precision | 0.124 | [-0.047, 0.310] | 0.312 |
| schema | ordering | 0.378 | [0.178, 0.544] | 0.062 |
| schema | grounding | 0.641 | [0.478, 0.764] | 0.031 |
| schema | check | 0.081 | [0.030, 0.145] | 0.062 |
| two-shot | completeness | -0.017 | [-0.186, 0.111] | 0.938 |
| two-shot | precision | 0.073 | [-0.072, 0.223] | 0.438 |
| two-shot | ordering | 0.233 | [0.000, 0.433] | 0.188 |
| two-shot | grounding | 0.384 | [0.259, 0.506] | 0.031 |
| two-shot | check | 0.068 | [0.028, 0.114] | 0.062 |

### Ordering support

| Configuration | Predictions | Mean matched steps | >=2 matches | Comparable pairs | Pair-weighted ordering |
|---|---:|---:|---:|---:|---:|
| positional-copy | 30 | 3.167 | 30 | 110 | 0.364 |
| adapter | 30 | 0.867 | 4 | 4 | 1.000 |
| candidate-only | 30 | 0.033 | 0 | 0 | N/A |
| schema | 30 | 0.533 | 3 | 5 | 0.600 |
| two-shot | 60 | 0.800 | 12 | 12 | 0.500 |

### Authored-reference case sensitivity

Each range is the adapter-minus-baseline mean after omitting each of the six
authored fixed-case references in turn.

| Baseline | Metric | Leave-one-case-out range | Stable sign |
|---|---:|---:|---:|
| schema | completeness | [0.030, 0.130] | yes |
| schema | precision | [0.042, 0.185] | yes |
| schema | ordering | [0.333, 0.453] | yes |
| schema | grounding | [0.596, 0.712] | yes |
| schema | check | [0.052, 0.097] | yes |
| two-shot | completeness | [-0.050, 0.060] | no |
| two-shot | precision | [0.019, 0.117] | yes |
| two-shot | ordering | [0.160, 0.320] | yes |
| two-shot | grounding | [0.340, 0.433] | yes |
| two-shot | check | [0.050, 0.082] | yes |

### Formatting diagnostic

| Configuration | Parsed predictions | Fully structured predictions | Technique fields | Complete steps |
|---|---:|---:|---:|---:|
| positional-copy | 1.000 | 1.000 | 1.000 | 1.000 |
| adapter | 1.000 | 1.000 | 1.000 | 1.000 |
| candidate-only | 0.567 | 0.000 | 0.068 | 0.000 |
| schema | 1.000 | 0.300 | 0.553 | 0.531 |
| two-shot | 1.000 | 0.533 | 0.917 | 0.845 |

## 1.5b

### Adapter minus prompted baseline

| Baseline | Metric | Mean delta | 95% case-resampling interval | Exact p |
|---|---:|---:|---:|---:|
| schema | completeness | -0.139 | [-0.333, 0.044] | 0.375 |
| schema | precision | 0.089 | [-0.034, 0.211] | 0.281 |
| schema | ordering | -0.006 | [-0.267, 0.233] | 1.000 |
| schema | grounding | 0.093 | [-0.116, 0.308] | 0.469 |
| schema | check | 0.056 | [0.006, 0.105] | 0.188 |
| two-shot | completeness | 0.114 | [-0.014, 0.250] | 0.219 |
| two-shot | precision | 0.226 | [0.099, 0.342] | 0.062 |
| two-shot | ordering | 0.328 | [-0.039, 0.700] | 0.250 |
| two-shot | grounding | 0.250 | [0.113, 0.396] | 0.031 |
| two-shot | check | 0.297 | [0.181, 0.398] | 0.031 |

### Ordering support

| Configuration | Predictions | Mean matched steps | >=2 matches | Comparable pairs | Pair-weighted ordering |
|---|---:|---:|---:|---:|---:|
| positional-copy | 30 | 3.167 | 30 | 110 | 0.364 |
| two-shot | 60 | 1.067 | 24 | 42 | 0.571 |
| adapter | 30 | 1.467 | 15 | 22 | 0.636 |
| schema | 30 | 1.867 | 18 | 45 | 0.667 |
| candidate-only | 30 | 0.067 | 0 | 0 | N/A |

### Authored-reference case sensitivity

Each range is the adapter-minus-baseline mean after omitting each of the six
authored fixed-case references in turn.

| Baseline | Metric | Leave-one-case-out range | Stable sign |
|---|---:|---:|---:|
| schema | completeness | [-0.207, -0.060] | yes |
| schema | precision | [0.045, 0.130] | yes |
| schema | ordering | [-0.060, 0.100] | no |
| schema | grounding | [0.008, 0.168] | yes |
| schema | check | [0.038, 0.073] | yes |
| two-shot | completeness | [0.057, 0.157] | yes |
| two-shot | precision | [0.186, 0.278] | yes |
| two-shot | ordering | [0.193, 0.433] | yes |
| two-shot | grounding | [0.195, 0.285] | yes |
| two-shot | check | [0.264, 0.346] | yes |

### Formatting diagnostic

| Configuration | Parsed predictions | Fully structured predictions | Technique fields | Complete steps |
|---|---:|---:|---:|---:|
| positional-copy | 1.000 | 1.000 | 1.000 | 1.000 |
| two-shot | 0.767 | 0.367 | 0.965 | 0.865 |
| adapter | 1.000 | 0.967 | 1.000 | 0.990 |
| schema | 1.000 | 0.300 | 0.879 | 0.816 |
| candidate-only | 0.133 | 0.000 | 0.667 | 0.000 |

## 7b

### Adapter minus prompted baseline

| Baseline | Metric | Mean delta | 95% case-resampling interval | Exact p |
|---|---:|---:|---:|---:|
| schema | completeness | 0.064 | [-0.092, 0.200] | 0.562 |
| schema | precision | 0.082 | [-0.143, 0.248] | 0.406 |
| schema | ordering | -0.161 | [-0.333, -0.000] | 0.125 |
| schema | grounding | 0.000 | [0.000, 0.000] | 1.000 |
| schema | check | 0.000 | [0.000, 0.000] | 1.000 |
| two-shot | completeness | 0.106 | [-0.061, 0.256] | 0.312 |
| two-shot | precision | 0.260 | [0.222, 0.305] | 0.031 |
| two-shot | ordering | -0.061 | [-0.194, 0.056] | 0.500 |
| two-shot | grounding | 0.083 | [0.019, 0.156] | 0.125 |
| two-shot | check | 0.080 | [0.043, 0.114] | 0.062 |

### Ordering support

| Configuration | Predictions | Mean matched steps | >=2 matches | Comparable pairs | Pair-weighted ordering |
|---|---:|---:|---:|---:|---:|
| schema | 30 | 2.267 | 24 | 55 | 0.818 |
| candidate-only | 30 | 0.333 | 2 | 6 | 0.500 |
| positional-copy | 30 | 3.167 | 30 | 110 | 0.364 |
| adapter | 30 | 2.433 | 27 | 62 | 0.742 |
| two-shot | 60 | 2.200 | 40 | 120 | 0.783 |

### Authored-reference case sensitivity

Each range is the adapter-minus-baseline mean after omitting each of the six
authored fixed-case references in turn.

| Baseline | Metric | Leave-one-case-out range | Stable sign |
|---|---:|---:|---:|
| schema | completeness | [0.037, 0.130] | yes |
| schema | precision | [0.046, 0.185] | yes |
| schema | ordering | [-0.220, -0.087] | yes |
| schema | grounding | [0.000, 0.000] | no |
| schema | check | [0.000, 0.000] | no |
| two-shot | completeness | [0.060, 0.177] | yes |
| two-shot | precision | [0.240, 0.272] | yes |
| two-shot | ordering | [-0.100, -0.007] | yes |
| two-shot | grounding | [0.056, 0.100] | yes |
| two-shot | check | [0.067, 0.096] | yes |

### Formatting diagnostic

| Configuration | Parsed predictions | Fully structured predictions | Technique fields | Complete steps |
|---|---:|---:|---:|---:|
| schema | 1.000 | 0.967 | 0.994 | 0.994 |
| candidate-only | 0.333 | 0.000 | 0.603 | 0.000 |
| positional-copy | 1.000 | 1.000 | 1.000 | 1.000 |
| adapter | 1.000 | 1.000 | 1.000 | 1.000 |
| two-shot | 1.000 | 0.333 | 0.966 | 0.903 |

