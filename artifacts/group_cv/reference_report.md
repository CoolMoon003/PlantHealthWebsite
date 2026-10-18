# Reference (non-CNN) baselines under group-aware CV

## Protocol (computed)
- Images: 203; independent generation sessions: 80.
- Outer folds: 3 group-disjoint folds, 3 repeat(s) with different partitions; verified leakage-free for every repeat.
- **LIMITATION** Healthy: only 1 independent generation session(s) are tested per fold (threshold 3). Per-fold and pooled metrics for this class are driven by very few independent samples and are NOT statistically reliable.

## Chance baselines (exact, on this label distribution)
- Majority class (Potassium_Deficiency): accuracy 33.5%, macro-F1 12.5%
- Uniform random guess accuracy: 25.0%; frequency-matched random accuracy: 30.0%

## Results (percent; mean ± std over repeats of the pooled-over-folds metric)

| Config | Test acc | Test macro-F1 | Test bal. acc | Macro-F1 95% session-bootstrap CI | Inner-val macro-F1 | Healthy recall | Mean best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `majority_class` | 33.0 ± 0.4 | 19.3 ± 0.2 | 25.0 ± 0.2 | 15.8%–27.4% | 15.7 ± 1.5 | 0.0 ± 0.0 | n/a |
| `color_hist_logreg_unweighted` | 70.0 ± 3.2 | 66.8 ± 5.0 | 64.4 ± 4.9 | 57.0%–75.2% | 79.2 ± 11.8 | 41.7 ± 13.6 | n/a |
| `color_hist_logreg_balanced` | 69.0 ± 3.2 | 68.8 ± 2.8 | 68.2 ± 1.8 | 58.9%–76.6% | 78.0 ± 10.8 | 63.9 ± 3.9 | n/a |

## Per-class recall / precision (percent, mean ± std over repeats)

| Config | Healthy R | Nitrogen_Deficiency R | Phosphorus_Deficiency R | Potassium_Deficiency R | Healthy P | Nitrogen_Deficiency P | Phosphorus_Deficiency P | Potassium_Deficiency P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `majority_class` | 0.0 ± 0.0 | 0.0 ± 0.0 | 44.4 ± 15.7 | 55.4 ± 16.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 32.5 ± 0.2 | 33.3 ± 0.4 |
| `color_hist_logreg_unweighted` | 41.7 ± 13.6 | 75.4 ± 1.4 | 76.8 ± 0.7 | 63.7 ± 7.8 | 87.5 ± 10.2 | 60.7 ± 3.2 | 83.2 ± 7.4 | 66.6 ± 1.2 |
| `color_hist_logreg_balanced` | 63.9 ± 3.9 | 73.7 ± 6.2 | 76.3 ± 1.4 | 58.8 ± 11.6 | 75.6 ± 8.7 | 60.5 ± 1.3 | 81.2 ± 10.6 | 66.2 ± 2.6 |

## Confusion matrices (summed over repeats; rows = true, columns = predicted)
Order: Healthy, Nitrogen_Deficiency, Phosphorus_Deficiency, Potassium_Deficiency

`majority_class`
```
   0     0    16    20
   0     0    76    95
   0     0    88   110
   0     0    91   113
```
`color_hist_logreg_unweighted`
```
  15    14     4     3
   2   129     6    34
   0    18   152    28
   0    52    22   130
```
`color_hist_logreg_balanced`
```
  23     9     3     1
   8   126     5    32
   0    19   151    28
   0    54    30   120
```

## Notes and limitations
- Reference models only: 30 hand-made colour statistics + logistic regression (C picked on inner validation), and a majority-class predictor.
- Images are AI-generated. Any score here measures separability of generated images, not plant diagnosis.
