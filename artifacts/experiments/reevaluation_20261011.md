# Checkpoint re-evaluation

Each dataset is reported separately. Scores describe only the listed dataset.

## heldout_test
Folder: `external_data\lettuce_leak_free_split\test`  |  images: 121

### Production
- SHA-256: `ed3e5679380e183732d7cf5b081ffd3215c50a5f4692cd75ab3f5127d5576759`
- Accuracy **42.98%** (95% Wilson 34.5-51.9%), balanced accuracy **39.82%**, macro-F1 **36.84%**
- Confident (>=90%) and wrong: 0.0% of images; predicted-Healthy share: 10.7%

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 100.00% | 25.00% | 40.00% | 52 |
| Nitrogen_Deficiency | 3.23% | 10.00% | 4.88% | 10 |
| Phosphorus_Deficiency | 15.91% | 58.33% | 25.00% | 12 |
| Potassium_Deficiency | 93.94% | 65.96% | 77.50% | 47 |

Confusion matrix (rows = actual, cols = predicted, class order as in classes.json):
```text
[13, 18, 20, 1]
[0, 1, 8, 1]
[0, 5, 7, 0]
[0, 7, 9, 31]
```

### ExperimentB
- SHA-256: `0fdfc4d2458607b3f54c8a6d302ded6bdea43f7575b23a282a0006e4c5485b88`
- Accuracy **97.52%** (95% Wilson 93.0-99.2%), balanced accuracy **98.56%**, macro-F1 **96.00%**
- Confident (>=90%) and wrong: 0.0% of images; predicted-Healthy share: 40.5%

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 100.00% | 94.23% | 97.03% | 52 |
| Nitrogen_Deficiency | 76.92% | 100.00% | 86.96% | 10 |
| Phosphorus_Deficiency | 100.00% | 100.00% | 100.00% | 12 |
| Potassium_Deficiency | 100.00% | 100.00% | 100.00% | 47 |

Confusion matrix (rows = actual, cols = predicted, class order as in classes.json):
```text
[49, 3, 0, 0]
[0, 10, 0, 0]
[0, 0, 12, 0]
[0, 0, 0, 47]
```

## synthetic_benchmark
Folder: `data\test`  |  images: 35

### Production
- SHA-256: `ed3e5679380e183732d7cf5b081ffd3215c50a5f4692cd75ab3f5127d5576759`
- Accuracy **65.71%** (95% Wilson 49.2-79.2%), balanced accuracy **66.44%**, macro-F1 **66.33%**
- Confident (>=90%) and wrong: 0.0% of images; predicted-Healthy share: 8.6%

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 66.67% | 66.67% | 66.67% | 3 |
| Nitrogen_Deficiency | 47.37% | 90.00% | 62.07% | 10 |
| Phosphorus_Deficiency | 100.00% | 63.64% | 77.78% | 11 |
| Potassium_Deficiency | 83.33% | 45.45% | 58.82% | 11 |

Confusion matrix (rows = actual, cols = predicted, class order as in classes.json):
```text
[2, 1, 0, 0]
[1, 9, 0, 0]
[0, 3, 7, 1]
[0, 6, 0, 5]
```

### ExperimentB
- SHA-256: `0fdfc4d2458607b3f54c8a6d302ded6bdea43f7575b23a282a0006e4c5485b88`
- Accuracy **8.57%** (95% Wilson 3.0-22.4%), balanced accuracy **25.00%**, macro-F1 **4.05%**
- Confident (>=90%) and wrong: 80.0% of images; predicted-Healthy share: 97.1%

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 8.82% | 100.00% | 16.22% | 3 |
| Nitrogen_Deficiency | 0.00% | 0.00% | 0.00% | 10 |
| Phosphorus_Deficiency | 0.00% | 0.00% | 0.00% | 11 |
| Potassium_Deficiency | 0.00% | 0.00% | 0.00% | 11 |

Confusion matrix (rows = actual, cols = predicted, class order as in classes.json):
```text
[3, 0, 0, 0]
[10, 0, 0, 0]
[10, 0, 0, 1]
[11, 0, 0, 0]
```
