# Controlled Experiment Report: 3-Fold Group-Aware Evaluation

## 1. Protocol & Splitting Design
- **Dataset**: 203 unique images across 80 OpenAI generation sessions.
- **Grouping Key**: `openai_gen_id` (Tag 37500 in EXIF metadata).
- **Partitions**: Exactly 3 folds (`StratifiedGroupKFold` equivalent). Zero session leakage across folds.
- **Fold Sizes**: Fold 0 = 68 images (26 groups); Fold 1 = 68 images (27 groups); Fold 2 = 67 images (27 groups).
- **Inner Validation**: For each fold, inner validation was drawn strictly from training groups (1 Healthy group, ~15% of other groups; ~20 images). Held-out test fold remained 100% unseen.

## 2. Controlled Experiment Results

| Experiment | Description | Mean Val Acc | Mean Test Acc | Mean Test Macro-F1 | Pooled Test Acc | Pooled Test Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| `exp1_baseline_frozen` | Baseline: Pretrained MobileNetV3-Small, ... | 55.99% | 57.73% ± 17.70% | 45.30% ± 20.97% | 57.64% | 50.32% |
| `exp2_finetune_last_block` | Fine-tuning last 2 conv blocks of Mobile... | 60.75% | 60.66% ± 18.98% | 47.06% ± 20.10% | 60.59% | 50.37% |
| `exp3_class_weighted_ce` | Class-weighted Cross-Entropy to penalize... | 65.04% | 63.59% ± 7.65% | 56.22% ± 8.98% | 63.55% | 59.56% |
| `exp4_mild_color_jitter` | Frozen backbone with mild color jitter a... | 55.75% | 66.94% ± 10.59% | 58.01% ± 17.81% | 67.00% | 62.58% |
| `exp5_weighted_finetuned_regularized` | Combined candidate: class-weighted CE + ... | 66.63% | 59.58% ± 7.84% | 50.18% ± 6.03% | 59.61% | 54.21% |

## 3. Analysis & Key Insights

- **Leakage-Free Baseline Reality**: Under strict group-aware evaluation with zero prompt leakage, baseline accuracy is substantially lower than the previously reported 65.71% (which had 88.6% prompt leakage). This proves that the earlier single-split metric was heavily inflated by memorization of DALL-E prompt variations.
- **Per-Class Breakdown for Best Validation Model** (`exp5_weighted_finetuned_regularized`):

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Healthy | 12 | 33.33% | 33.33% | 33.33% |
| Nitrogen_Deficiency | 57 | 44.35% | 96.49% | 60.77% |
| Phosphorus_Deficiency | 66 | 92.31% | 54.55% | 68.57% |
| Potassium_Deficiency | 68 | 92.86% | 38.24% | 54.17% |

### Pooled Confusion Matrix (All 203 Held-Out Samples):
Row order: `Healthy`, `Nitrogen_Deficiency`, `Phosphorus_Deficiency`, `Potassium_Deficiency`
```
[[ 4  8  0  0]
 [ 1 55  0  1]
 [ 3 26 36  1]
 [ 4 35  3 26]]
```

## 4. Checkpoint Policy
- The production checkpoint `artifacts/plant_health_model.pth` was trained on the old split and is maintained to prevent breaking existing demonstrations.
- The best candidate checkpoint under group-aware evaluation was saved to `artifacts/experiments/best_candidate.pth` (from Fold 0 of `exp5_weighted_finetuned_regularized`).
