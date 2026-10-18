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
| `H_robust_sampler` | 71.4 ± 0.0 | 61.9 ± 0.0 | 62.0 ± 0.0 | 51.5%–77.4% | 89.6 ± 2.3 | 25.0 ± 0.0 | 8.7 |

## Per-class recall / precision (percent, mean ± std over repeats)

| Config | Healthy R | Nitrogen_Deficiency R | Phosphorus_Deficiency R | Potassium_Deficiency R | Healthy P | Nitrogen_Deficiency P | Phosphorus_Deficiency P | Potassium_Deficiency P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `H_robust_sampler` | 25.0 ± 0.0 | 75.4 ± 0.0 | 71.2 ± 0.0 | 76.5 ± 0.0 | 21.4 ± 0.0 | 62.3 ± 0.0 | 90.4 ± 0.0 | 76.5 ± 0.0 |

## Confusion matrices (summed over repeats; rows = true, columns = predicted)
Order: Healthy, Nitrogen_Deficiency, Phosphorus_Deficiency, Potassium_Deficiency

`H_robust_sampler`
```
   3     9     0     0
   7    43     0     7
   1     9    47     9
   3     8     5    52
```

## Selection (inner-validation only)
- Highest mean inner-validation macro-F1: H_robust_sampler (0.896, SE 0.013). Among configurations within one SE of it, the simplest (H_robust_sampler) is chosen. Outer test folds were not used for this choice.

## Notes and limitations
- All images are AI-generated (OpenAI image-generation EXIF on every file). These numbers describe separability of generated images, not accuracy on real plants.
- Selection used inner-validation macro-F1 only. Outer test-fold numbers are descriptive; with 3 independent Healthy sessions in total no class-level claim about Healthy is reliable.
- Std is the spread of the pooled-over-folds metric across repeats (different partitions and seeds); it is not a confidence interval. The bootstrap interval resamples whole sessions.
- Config descriptions: H_robust_sampler: D_frozen_weighted_lr3e-4 baseline with three targeted generalisation improvements: (1) WeightedRandomSampler replaces inverse-frequency loss weighting for class balance (sampler='weighted', loss='ce') to avoid the double-balancing interaction; (2) robust augmentation (aug='robust') adds RandomVerticalFlip, ±25° rotation, moderate colour jitter (brightness/contrast/saturation 0.25, hue 0.04), RandomPerspective(0.2) and RandomGrayscale(0.05) — designed to preserve diagnostic leaf colours while reducing background-texture memorisation; (3) label_smoothing=0.1 reduces overconfidence on the ambiguous minority classes. Backbone frozen, bn_eval_frozen=True, weight_decay=1e-4, head lr=3e-4.
