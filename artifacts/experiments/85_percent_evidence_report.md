# Lettuce Plant-Health Model ? Reproducible Evaluation

Evaluation performed on existing checkpoints. No training was performed.
Device: CPU. Input: RGB, resized to 224x224, ImageNet normalization.

## Important interpretation
Scores describe only these saved datasets. They do not establish guaranteed accuracy on new farm photographs. The deficiency classes have limited source diversity, and the synthetic benchmark is not a substitute for independently labelled real-world field images.

## Leak-free held-out test

Dataset path: `A:\Projects!!!\PlantHealthWebsite\external_data\lettuce_leak_free_split\test`

### Production

- Checkpoint: `artifacts\plant_health_model.pth`
- SHA-256: `ed3e5679380e183732d7cf5b081ffd3215c50a5f4692cd75ab3f5127d5576759`
- Images evaluated: 121
- Accuracy: **42.98%**
- Balanced accuracy: **39.82%**
- Macro-F1: **36.84%**

Confusion matrix (actual rows, predicted columns; class order shown below):

Classes: `['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']`

```text
[[13 18 20  1]
 [ 0  1  8  1]
 [ 0  5  7  0]
 [ 0  7  9 31]]
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 100.00% | 25.00% | 40.00% | 52 |
| Nitrogen_Deficiency | 3.23% | 10.00% | 4.88% | 10 |
| Phosphorus_Deficiency | 15.91% | 58.33% | 25.00% | 12 |
| Potassium_Deficiency | 93.94% | 65.96% | 77.50% | 47 |

### Experiment A

- Checkpoint: `artifacts\experiments\Exp_A_EndToEnd_Weighted.pth`
- SHA-256: `eec3956fc7b6600917b3e11578d1a958027a871d3896eea081e08c2f87f31a96`
- Images evaluated: 121
- Accuracy: **100.00%**
- Balanced accuracy: **100.00%**
- Macro-F1: **100.00%**

Confusion matrix (actual rows, predicted columns; class order shown below):

Classes: `['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']`

```text
[[52  0  0  0]
 [ 0 10  0  0]
 [ 0  0 12  0]
 [ 0  0  0 47]]
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 100.00% | 100.00% | 100.00% | 52 |
| Nitrogen_Deficiency | 100.00% | 100.00% | 100.00% | 10 |
| Phosphorus_Deficiency | 100.00% | 100.00% | 100.00% | 12 |
| Potassium_Deficiency | 100.00% | 100.00% | 100.00% | 47 |

### Experiment B

- Checkpoint: `artifacts\experiments\Exp_B_TwoStage_Frozen.pth`
- SHA-256: `0fdfc4d2458607b3f54c8a6d302ded6bdea43f7575b23a282a0006e4c5485b88`
- Images evaluated: 121
- Accuracy: **97.52%**
- Balanced accuracy: **98.56%**
- Macro-F1: **96.00%**

Confusion matrix (actual rows, predicted columns; class order shown below):

Classes: `['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']`

```text
[[49  3  0  0]
 [ 0 10  0  0]
 [ 0  0 12  0]
 [ 0  0  0 47]]
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 100.00% | 94.23% | 97.03% | 52 |
| Nitrogen_Deficiency | 76.92% | 100.00% | 86.96% | 10 |
| Phosphorus_Deficiency | 100.00% | 100.00% | 100.00% | 12 |
| Potassium_Deficiency | 100.00% | 100.00% | 100.00% | 47 |

### Experiment C

- Checkpoint: `artifacts\experiments\Exp_C_Robust_RandomErasing.pth`
- SHA-256: `8ff6e71b99e95fac2e4aa25129b977af8759d2963228936e17ee0abcde4c13fe`
- Images evaluated: 121
- Accuracy: **100.00%**
- Balanced accuracy: **100.00%**
- Macro-F1: **100.00%**

Confusion matrix (actual rows, predicted columns; class order shown below):

Classes: `['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']`

```text
[[52  0  0  0]
 [ 0 10  0  0]
 [ 0  0 12  0]
 [ 0  0  0 47]]
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 100.00% | 100.00% | 100.00% | 52 |
| Nitrogen_Deficiency | 100.00% | 100.00% | 100.00% | 10 |
| Phosphorus_Deficiency | 100.00% | 100.00% | 100.00% | 12 |
| Potassium_Deficiency | 100.00% | 100.00% | 100.00% | 47 |

## Synthetic benchmark (OOD)

Dataset path: `A:\Projects!!!\PlantHealthWebsite\data\test`

### Production

- Checkpoint: `artifacts\plant_health_model.pth`
- SHA-256: `ed3e5679380e183732d7cf5b081ffd3215c50a5f4692cd75ab3f5127d5576759`
- Images evaluated: 35
- Accuracy: **65.71%**
- Balanced accuracy: **66.44%**
- Macro-F1: **66.33%**

Confusion matrix (actual rows, predicted columns; class order shown below):

Classes: `['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']`

```text
[[2 1 0 0]
 [1 9 0 0]
 [0 3 7 1]
 [0 6 0 5]]
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 66.67% | 66.67% | 66.67% | 3 |
| Nitrogen_Deficiency | 47.37% | 90.00% | 62.07% | 10 |
| Phosphorus_Deficiency | 100.00% | 63.64% | 77.78% | 11 |
| Potassium_Deficiency | 83.33% | 45.45% | 58.82% | 11 |

### Experiment A

- Checkpoint: `artifacts\experiments\Exp_A_EndToEnd_Weighted.pth`
- SHA-256: `eec3956fc7b6600917b3e11578d1a958027a871d3896eea081e08c2f87f31a96`
- Images evaluated: 35
- Accuracy: **8.57%**
- Balanced accuracy: **25.00%**
- Macro-F1: **4.41%**

Confusion matrix (actual rows, predicted columns; class order shown below):

Classes: `['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']`

```text
[[ 3  0  0  0]
 [10  0  0  0]
 [ 7  0  0  4]
 [11  0  0  0]]
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 9.68% | 100.00% | 17.65% | 3 |
| Nitrogen_Deficiency | 0.00% | 0.00% | 0.00% | 10 |
| Phosphorus_Deficiency | 0.00% | 0.00% | 0.00% | 11 |
| Potassium_Deficiency | 0.00% | 0.00% | 0.00% | 11 |

### Experiment B

- Checkpoint: `artifacts\experiments\Exp_B_TwoStage_Frozen.pth`
- SHA-256: `0fdfc4d2458607b3f54c8a6d302ded6bdea43f7575b23a282a0006e4c5485b88`
- Images evaluated: 35
- Accuracy: **8.57%**
- Balanced accuracy: **25.00%**
- Macro-F1: **4.05%**

Confusion matrix (actual rows, predicted columns; class order shown below):

Classes: `['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']`

```text
[[ 3  0  0  0]
 [10  0  0  0]
 [10  0  0  1]
 [11  0  0  0]]
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 8.82% | 100.00% | 16.22% | 3 |
| Nitrogen_Deficiency | 0.00% | 0.00% | 0.00% | 10 |
| Phosphorus_Deficiency | 0.00% | 0.00% | 0.00% | 11 |
| Potassium_Deficiency | 0.00% | 0.00% | 0.00% | 11 |

### Experiment C

- Checkpoint: `artifacts\experiments\Exp_C_Robust_RandomErasing.pth`
- SHA-256: `8ff6e71b99e95fac2e4aa25129b977af8759d2963228936e17ee0abcde4c13fe`
- Images evaluated: 35
- Accuracy: **20.00%**
- Balanced accuracy: **34.09%**
- Macro-F1: **17.67%**

Confusion matrix (actual rows, predicted columns; class order shown below):

Classes: `['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']`

```text
[[ 3  0  0  0]
 [10  0  0  0]
 [ 5  0  4  2]
 [ 8  2  1  0]]
```

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 11.54% | 100.00% | 20.69% | 3 |
| Nitrogen_Deficiency | 0.00% | 0.00% | 0.00% | 10 |
| Phosphorus_Deficiency | 80.00% | 36.36% | 50.00% | 11 |
| Potassium_Deficiency | 0.00% | 0.00% | 0.00% | 11 |

## Claim boundary

A result of 85% or more on the held-out test set demonstrates performance on that test set only. It is not proof of 85% accuracy on arbitrary real-world images. The synthetic benchmark results should be reported separately.
