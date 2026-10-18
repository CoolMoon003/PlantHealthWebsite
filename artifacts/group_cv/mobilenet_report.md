# MobileNetV3-Small under group-aware cross-validation

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
| `A_legacy_frozen` | 67.0 ± 2.8 | 65.2 ± 3.0 | 68.9 ± 4.1 | 53.9%–73.1% | 74.4 ± 10.3 | 72.2 ± 20.8 | 9.3 |
| `B_frozen_bn_eval` | 82.9 ± 4.8 | 75.0 ± 6.2 | 74.0 ± 6.0 | 64.6%–86.6% | 89.7 ± 7.7 | 38.9 ± 10.4 | 7.1 |
| `C_frozen_weighted` | 82.6 ± 3.4 | 76.4 ± 4.7 | 77.0 ± 4.7 | 65.3%–86.8% | 89.8 ± 5.8 | 55.6 ± 10.4 | 6.7 |
| `D_frozen_weighted_lr3e-4` | 80.3 ± 2.6 | 75.8 ± 4.0 | 77.6 ± 4.5 | 66.5%–83.8% | 85.2 ± 11.1 | 66.7 ± 11.8 | 8.7 |
| `E_frozen_weighted_jitter_wd` | 82.1 ± 2.5 | 76.3 ± 3.8 | 76.6 ± 4.7 | 68.3%–85.4% | 89.0 ± 7.1 | 55.6 ± 14.2 | 7.8 |
| `F_finetune2_weighted` | 76.0 ± 1.3 | 72.4 ± 2.3 | 76.9 ± 3.7 | 62.2%–80.8% | 80.3 ± 9.8 | 77.8 ± 14.2 | 9.0 |
| `G_finetune2_weighted_jitter` | 73.2 ± 3.0 | 70.6 ± 2.8 | 74.2 ± 3.5 | 60.5%–78.4% | 79.5 ± 13.5 | 75.0 ± 6.8 | 9.1 |
| `ref:majority_class` | 33.0 ± 0.4 | 19.3 ± 0.2 | 25.0 ± 0.2 | 15.8%–27.4% | 15.7 ± 1.5 | 0.0 ± 0.0 | n/a |
| `ref:color_hist_logreg_unweighted` | 70.0 ± 3.2 | 66.8 ± 5.0 | 64.4 ± 4.9 | 57.0%–75.2% | 79.2 ± 11.8 | 41.7 ± 13.6 | n/a |
| `ref:color_hist_logreg_balanced` | 69.0 ± 3.2 | 68.8 ± 2.8 | 68.2 ± 1.8 | 58.9%–76.6% | 78.0 ± 10.8 | 63.9 ± 3.9 | n/a |

## Per-class recall / precision (percent, mean ± std over repeats)

| Config | Healthy R | Nitrogen_Deficiency R | Phosphorus_Deficiency R | Potassium_Deficiency R | Healthy P | Nitrogen_Deficiency P | Phosphorus_Deficiency P | Potassium_Deficiency P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_legacy_frozen` | 72.2 ± 20.8 | 89.5 ± 3.8 | 55.1 ± 6.2 | 58.8 ± 3.2 | 57.2 ± 20.2 | 59.0 ± 3.0 | 87.8 ± 6.8 | 72.1 ± 9.3 |
| `B_frozen_bn_eval` | 38.9 ± 10.4 | 85.4 ± 6.8 | 85.9 ± 7.2 | 85.8 ± 1.4 | 56.0 ± 13.8 | 76.1 ± 4.9 | 92.0 ± 3.5 | 85.1 ± 7.1 |
| `C_frozen_weighted` | 55.6 ± 10.4 | 80.7 ± 4.3 | 85.4 ± 6.3 | 86.3 ± 0.7 | 49.7 ± 12.8 | 78.9 ± 2.7 | 90.3 ± 1.6 | 85.7 ± 5.2 |
| `D_frozen_weighted_lr3e-4` | 66.7 ± 11.8 | 82.5 ± 3.8 | 81.3 ± 4.7 | 79.9 ± 3.9 | 51.0 ± 8.2 | 75.4 ± 2.8 | 88.1 ± 2.2 | 85.2 ± 4.3 |
| `E_frozen_weighted_jitter_wd` | 55.6 ± 14.2 | 81.3 ± 2.2 | 81.3 ± 4.3 | 88.2 ± 0.0 | 53.9 ± 6.2 | 79.2 ± 4.7 | 90.1 ± 1.8 | 83.1 ± 3.8 |
| `F_finetune2_weighted` | 77.8 ± 14.2 | 90.1 ± 1.7 | 72.2 ± 7.9 | 67.6 ± 4.3 | 44.5 ± 7.5 | 67.5 ± 3.4 | 94.6 ± 6.2 | 85.5 ± 3.1 |
| `G_finetune2_weighted_jitter` | 75.0 ± 6.8 | 88.9 ± 3.0 | 72.7 ± 10.8 | 60.3 ± 2.4 | 48.2 ± 3.2 | 60.2 ± 5.8 | 91.9 ± 4.2 | 88.5 ± 1.9 |
| `ref:majority_class` | 0.0 ± 0.0 | 0.0 ± 0.0 | 44.4 ± 15.7 | 55.4 ± 16.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 32.5 ± 0.2 | 33.3 ± 0.4 |
| `ref:color_hist_logreg_unweighted` | 41.7 ± 13.6 | 75.4 ± 1.4 | 76.8 ± 0.7 | 63.7 ± 7.8 | 87.5 ± 10.2 | 60.7 ± 3.2 | 83.2 ± 7.4 | 66.6 ± 1.2 |
| `ref:color_hist_logreg_balanced` | 63.9 ± 3.9 | 73.7 ± 6.2 | 76.3 ± 1.4 | 58.8 ± 11.6 | 75.6 ± 8.7 | 60.5 ± 1.3 | 81.2 ± 10.6 | 66.2 ± 2.6 |

## Confusion matrices (summed over repeats; rows = true, columns = predicted)
Order: Healthy, Nitrogen_Deficiency, Phosphorus_Deficiency, Potassium_Deficiency

`A_legacy_frozen`
```
  26    10     0     0
  12   153     2     4
  10    34   109    45
   6    63    15   120
```
`B_frozen_bn_eval`
```
  14    22     0     0
   7   146     3    15
   2     9   170    17
   2    15    12   175
```
`C_frozen_weighted`
```
  20    16     0     0
  15   138     3    15
   3    11   169    15
   3    10    15   176
```
`D_frozen_weighted_lr3e-4`
```
  24    12     0     0
  13   141     5    12
   7    13   161    17
   3    21    17   163
```
`E_frozen_weighted_jitter_wd`
```
  20    16     0     0
  12   139     4    16
   3    13   161    21
   2     8    14   180
```
`F_finetune2_weighted`
```
  28     8     0     0
  14   154     0     3
  11    23   143    21
  12    44    10   138
```
`G_finetune2_weighted_jitter`
```
  27     9     0     0
  16   152     0     3
   7    34   144    13
   6    61    14   123
```
`ref:majority_class`
```
   0     0    16    20
   0     0    76    95
   0     0    88   110
   0     0    91   113
```
`ref:color_hist_logreg_unweighted`
```
  15    14     4     3
   2   129     6    34
   0    18   152    28
   0    52    22   130
```
`ref:color_hist_logreg_balanced`
```
  23     9     3     1
   8   126     5    32
   0    19   151    28
   0    54    30   120
```

## Selection (inner-validation only)
- Highest mean inner-validation macro-F1: C_frozen_weighted (0.898, SE 0.019). Among configurations within one SE of it, the simplest (B_frozen_bn_eval) is chosen. Outer test folds were not used for this choice.

## Notes and limitations
- All images are AI-generated (OpenAI image-generation EXIF on every file). These numbers describe separability of generated images, not accuracy on real plants.
- Selection used inner-validation macro-F1 only. Outer test-fold numbers are descriptive; with 3 independent Healthy sessions in total no class-level claim about Healthy is reliable.
- Std is the spread of the pooled-over-folds metric across repeats (different partitions and seeds); it is not a confidence interval. The bootstrap interval resamples whole sessions.
- Config descriptions: A_legacy_frozen: Replica of the original train_model.py: frozen backbone but BatchNorm left in train mode, plain CE, flip+rotation, early stop on val accuracy.; B_frozen_bn_eval: Same as A with the BatchNorm bug fixed (frozen blocks stay in eval mode) and early stop on val macro-F1.; C_frozen_weighted: B + inverse-frequency class-weighted cross-entropy.; D_frozen_weighted_lr3e-4: C with a 3x lower head learning rate.; E_frozen_weighted_jitter_wd: C + mild brightness/contrast jitter (training only) + weight decay 1e-4.; F_finetune2_weighted: C + fine-tune the last two feature blocks (backbone lr 1e-4) + weight decay 1e-4.; G_finetune2_weighted_jitter: F + mild brightness/contrast jitter.
