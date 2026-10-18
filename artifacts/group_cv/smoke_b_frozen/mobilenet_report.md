# MobileNetV3-Small under group-aware cross-validation

## Protocol (computed)
- Images: 203; independent generation sessions: 80.
- Outer folds: 3 group-disjoint folds, 1 repeat(s) with different partitions; verified leakage-free for every repeat.
- **LIMITATION** Healthy: only 1 independent generation session(s) are tested per fold (threshold 3). Per-fold and pooled metrics for this class are driven by very few independent samples and are NOT statistically reliable.

## Chance baselines (exact, on this label distribution)
- Majority class (Potassium_Deficiency): accuracy 33.5%, macro-F1 12.5%
- Uniform random guess accuracy: 25.0%; frequency-matched random accuracy: 30.0%

## Results (percent; mean ± std over repeats of the pooled-over-folds metric)

| Config | Test acc | Test macro-F1 | Test bal. acc | Macro-F1 95% session-bootstrap CI | Inner-val macro-F1 | Healthy recall | Mean best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B_frozen_bn_eval` | 67.5 ± 0.0 | 55.6 ± 0.0 | 56.3 ± 0.0 | 45.8%–70.2% | 76.0 ± 20.0 | 8.3 ± 0.0 | 1.0 |

## Per-class recall / precision (percent, mean ± std over repeats)

| Config | Healthy R | Nitrogen_Deficiency R | Phosphorus_Deficiency R | Potassium_Deficiency R | Healthy P | Nitrogen_Deficiency P | Phosphorus_Deficiency P | Potassium_Deficiency P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B_frozen_bn_eval` | 8.3 ± 0.0 | 93.0 ± 0.0 | 56.1 ± 0.0 | 67.6 ± 0.0 | 100.0 ± 0.0 | 63.9 ± 0.0 | 80.4 ± 0.0 | 63.0 ± 0.0 |

## Confusion matrices (summed over repeats; rows = true, columns = predicted)
Order: Healthy, Nitrogen_Deficiency, Phosphorus_Deficiency, Potassium_Deficiency

`B_frozen_bn_eval`
```
   1    11     0     0
   0    53     0     4
   0     6    37    23
   0    13     9    46
```

## Selection (inner-validation only)
- Highest mean inner-validation macro-F1: B_frozen_bn_eval (0.760, SE 0.116). Among configurations within one SE of it, the simplest (B_frozen_bn_eval) is chosen. Outer test folds were not used for this choice.

## Notes and limitations
- All images are AI-generated (OpenAI image-generation EXIF on every file). These numbers describe separability of generated images, not accuracy on real plants.
- Selection used inner-validation macro-F1 only. Outer test-fold numbers are descriptive; with 3 independent Healthy sessions in total no class-level claim about Healthy is reliable.
- Std is the spread of the pooled-over-folds metric across repeats (different partitions and seeds); it is not a confidence interval. The bootstrap interval resamples whole sessions.
- Config descriptions: B_frozen_bn_eval: Same as A with the BatchNorm bug fixed (frozen blocks stay in eval mode) and early stop on val macro-F1.
