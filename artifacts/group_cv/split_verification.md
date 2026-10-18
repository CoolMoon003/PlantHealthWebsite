# Group-aware split verification

- Unique images: 203 in 80 generation sessions (203 files found, 0 exact duplicates ignored).
- All checks passed: every session sits in exactly one fold, no session or image overlaps between train and test, every class is present in every test fold.
- 3 distinct partition(s) among 3 repeat(s).

| Class | Images | Sessions | Session sizes (size: count) |
|---|---:|---:|---|
| Healthy | 12 | 3 | {4: 3} |
| Nitrogen_Deficiency | 57 | 23 | {1: 9, 2: 3, 3: 2, 4: 9} |
| Phosphorus_Deficiency | 66 | 27 | {1: 11, 2: 3, 3: 3, 4: 10} |
| Potassium_Deficiency | 68 | 27 | {1: 8, 2: 7, 3: 2, 4: 10} |

## Test composition per fold, repeat 0 (images / sessions)

| Fold | Healthy | Nitrogen_Deficiency | Phosphorus_Deficiency | Potassium_Deficiency |
|---|---:|---:|---:|---:|
| 0 | 4 / 1 | 19 / 7 | 22 / 9 | 23 / 9 |
| 1 | 4 / 1 | 19 / 8 | 22 / 9 | 23 / 9 |
| 2 | 4 / 1 | 19 / 8 | 22 / 9 | 22 / 9 |

## Limitations found
- Healthy: only 1 independent generation session(s) are tested per fold (threshold 3). Per-fold and pooled metrics for this class are driven by very few independent samples and are NOT statistically reliable.

## Old image-level split (data/train|val|test) for comparison
- Test images sharing a session with a training image: 31 / 35
- Validation images sharing a session with a training image: 23 / 28
- Sessions present in more than one split: 35 / 80
