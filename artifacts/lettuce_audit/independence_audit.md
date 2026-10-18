# Dataset Provenance and Independence Audit: Lettuce NPK

## Executive Summary

An in-depth EXIF metadata, provenance, and perceptual hash analysis was conducted on all 203 unique images in the prepared Lettuce NPK dataset (`data/`).

### Key Findings
1. **Synthetic AI Origin**: **100% (203/203)** of images contain embedded OpenAI EXIF metadata (`Tag 37510: 'Made with OpenAI Labs'`, `Tag 37500: 'OpenAI--ImageGeneration--generation-...'`). Despite the upstream repository claim of being collected from a hydroponic farm, these images are synthetic DALL-E outputs.
2. **Session / Prompt Batching**: The 203 images originate from only **80 distinct OpenAI generation sessions** (prompts), typically producing 4 variations per generation prompt.
3. **Cross-Split Session Leakage**: **35 of the 80 generation sessions (43.8%)** cross partition boundaries.
   - **31 of 35 test images (88.6%)** share an exact generation session with training images.
   - **23 of 28 validation images (82.1%)** share an exact generation session with training images.
   - For the **Healthy** class, **100% (3/3)** of test images share generation sessions with train images.

---

## 1. Class and Generation Session Breakdown

| Class | Image Count | Unique Gen Sessions | Avg Images / Session |
|---|---:|---:|---:|
| Healthy | 12 | 3 | 4.0 |
| Nitrogen_Deficiency | 57 | 23 | 2.48 |
| Phosphorus_Deficiency | 66 | 27 | 2.44 |
| Potassium_Deficiency | 68 | 27 | 2.52 |

> [!IMPORTANT]
> The entire `Healthy` class comprises only **3 unique generation sessions** (prompts) producing 12 images (4 variations each).

---

## 2. Split Leakage Analysis by Generation Session

Because `prepare_image_dataset.py` performed image-level stratified random splitting without awareness of the OpenAI generation session IDs, variations from the same generation prompt were distributed across `train`, `val`, and `test`:

| Class | Test Images | Test Images Leaked from Train | Leakage % |
|---|---:|---:|---:|
| Healthy | 3 | 3 | 100.0% |
| Nitrogen_Deficiency | 10 | 8 | 80.0% |
| Phosphorus_Deficiency | 11 | 10 | 90.9% |
| Potassium_Deficiency | 11 | 10 | 90.9% |

- **Overall Test Leakage**: **31 / 35 (88.6%)**
- **Overall Val Leakage**: **23 / 28 (82.1%)**

---

## 3. Perceptual Similarity Analysis

- **Method**: 64-bit Difference Hash (`dHash`) computed by downsampling to 9×8 grayscale and checking horizontal luminance gradient.
- **Threshold**: Hamming distance $\le 10$ out of 64 bits.
- **Total Near-Duplicate Pairs**: 9 pairs identified.
- **Cross-Split Near-Duplicate Pairs**: 4 pairs cross between `train`, `val`, or `test`.

### Distinction of Similarity Types
1. **Exact Bitwise Duplicates**: 5 files in the raw repository had identical SHA-256 digests and were removed during initial preparation.
2. **Confirmed Same-Generation Variations**: Images sharing the identical OpenAI `generation-...` identifier. These represent 4 visual variants produced simultaneously by DALL-E from the same text prompt.
3. **Perceptually Similar Images**: Images with low dHash distance ($\le 10$) that may share visual backgrounds, textures, or lighting, but do not share identical generation IDs.
4. **Biological Same-Plant Images**: **None.** Because the images are DALL-E generations, they do not depict biological plants.

---

## 4. Evaluation Protocol Recommendation

### Ordinary vs. Group-Aware Stratified Cross-Validation
- **Ordinary Stratified CV is Scientifically Flawed for this Dataset**: Ordinary shuffling randomly separates the 4 variations of each generation session between folds. Because 88.6% of test images share prompt sessions with train images, a standard CV score largely measures the model's ability to recognize variations of memorized prompt batches, rather than general nutrient deficiency features.
- **Group-Aware Stratified Cross-Validation (`GroupKFold` / `StratifiedGroupKFold`)**: Grouping by `openai_gen_id` ensures that all variations from a prompt generation session stay within the same fold.

### Feasibility and Critical Limitation of the Healthy Class
- The `Healthy` class has **only 3 generation sessions** (12 images).
- Consequently, a Group-Aware cross-validation can support at most **3 folds** (1 Healthy session per fold) or **Leave-One-Group-Out (LOGO)**.
- In a 3-fold group-stratified split, each fold evaluates on exactly **1 Healthy session (4 images)**.
- While this small sample size limits statistical power, group-aware evaluation is the only method that prevents severe prompt-variation leakage.
