# Comprehensive Model Audit, Evaluation & Verification Report

**Project**: PlantHealthWebsite (`A:\Projects!!!\PlantHealthWebsite`)  
**Date**: October 2026  
**Status**: Production Hardening Completed & Verified (All 31 Tests Passing)

---

## 1. Executive Summary & Root Cause Analysis

### 1.1 The Repeated Phosphorus Deficiency Issue
Users reported that webcam and indoor camera photos frequently resulted in `Phosphorus_Deficiency` predictions. Our audit identified two compounding technical reasons:

1. **Closed-Set Inductive Bias on Out-of-Domain Data**:
   - The production model checkpoint (`artifacts/plant_health_model.pth`, MD5 hash `ED3E5679...`) is a 4-class closed-set classifier trained exclusively on synthetic DALL-E generative images (`data/train`).
   - In synthetic training data, `Phosphorus_Deficiency` samples exhibit significantly darker RGB mean values (127, 138, 113) compared to `Healthy` and `Nitrogen_Deficiency` images. The model's final classifier layer also carries a positive bias (`+0.0312`) towards Phosphorus Deficiency.
   - Real webcam captures taken in indoor, diffuse, or low-light conditions feature darker foliage and shadowy backgrounds compared to brightly illuminated synthetic studio renders. As a result, the model's feature extractor naturally projects these images closest to the `Phosphorus_Deficiency` hyperplane.

2. **Absence of Leaf & Out-of-Distribution (OOD) Gating**:
   - In a closed-set softmax classifier without an OOD reject option, non-leaf inputs (human faces, pets, cars, furniture, screens, blank walls) are forcibly mapped to one of the 4 nutrient deficiency classes.
   - Non-leaf images disproportionately fell into `Phosphorus_Deficiency` because their overall luminance and non-plant textures mapped farthest from the clean green manifold of `Healthy` synthetic leaves.

### 1.2 Root Cause of the 8.57% Independent Benchmark
When models trained on real hydroponic lettuce datasets were evaluated against the 35 images in [`data/test`](file:///A:/Projects%21%21%21/PlantHealthWebsite/data/test), accuracy plummeted to **8.57% (3/35 correct)**:
- **Synthetic vs. Real Domain Shift**: Every single image in `data/test` contains JPEG EXIF metadata from OpenAI Labs (`Made with OpenAI Labs`, `OpenAI--ImageGeneration...`). They are 100% synthetic DALL-E generated images.
- **Visual Discrepancy**: Synthetic deficiency images depict clean, diffuse studio lettuce heads lacking real cellular necrosis, purpling, or hydroponic PVC channel artifacts. Models trained on genuine agricultural pathology recognized them as clean lettuce foliage, classifying 31 of the 35 images as `Healthy` (yielding 100% recall on Healthy [3/3], but 0% recall on synthetic deficiencies, leading to $3/35 = 8.57\%$).
- **Conclusion**: The 8.57% score stems from a severe **Synthetic-to-Real Domain Gap**, not an incorrect label mapping or corrupted class indexing.

---

## 2. Quantitative Model Performance Benchmarks

### 2.1 Production Model (`artifacts/plant_health_model.pth`)

The production checkpoint was evaluated across both the synthetic test partition and the real hydroponic leak-free test partition:

| Evaluation Benchmark | Sample Count | Accuracy | Balanced Accuracy | Macro-F1 | Key Observations |
|---|---:|---:|---:|---:|---|
| **Synthetic Test Set** (`data/test`) | 35 | **65.71%** | **66.44%** | **66.33%** | Trained domain; reasonable performance on synthetic images. |
| **Real Hydroponic Test Set** (`external_data/.../test`) | 120 | **42.50%** | **39.46%** | **36.37%** | Significant drop due to synthetic-to-real domain gap. |

#### Detailed Confusion Matrix on `data/test` (35 synthetic images):
```
                       Predicted:
                Healthy   Nitrogen   Phosphorus   Potassium
True Healthy          2          1            0           0
True Nitrogen         1          9            0           0
True Phosphorus       0          3            7           1
True Potassium        0          6            0           5
```

#### Detailed Confusion Matrix on Real Hydroponic Test Set (120 images):
```
                       Predicted:
                Healthy   Nitrogen   Phosphorus   Potassium
True Healthy         12         18           20           1
True Nitrogen         0          1            8           1
True Phosphorus       0          5            7           0
True Potassium        0          7            9          31
```
*(Notice that 37 out of 120 real images are erroneously predicted as Phosphorus Deficiency, confirming the low-light/darker texture bias).*

---

## 3. Implemented Engineering Solutions

### 3.1 Two-Stage Out-of-Distribution Leaf Verifier (`leaf_verifier.py`)
To prevent arbitrary non-leaf inputs and invalid images from generating false deficiency diagnoses and unwarranted chemical treatments, we built a zero-download, offline leaf verification pipeline:

1. **Structural Verification**:
   - Minimum dimensions check ($32 \times 32$ pixels).
   - Pixel variance check ($\sigma \ge 3.0$) to reject blank, uniform, or corrupted inputs.
2. **Semantic Verification (Offline ImageNet MobileNetV3-Small)**:
   - Uses the local cached torchvision checkpoint (`mobilenet_v3_small-047dcff4.pth`).
   - Computes aggregated foliage/plant class probability versus non-plant object categories (people, vehicles, electronics, furniture, animals).
3. **Lettuce Feature Manifold Verification**:
   - Projects image embeddings against a precomputed 1024-dimensional lettuce centroid vector ([`artifacts/lettuce_centroid.npy`](file:///A:/Projects%21%21%21/PlantHealthWebsite/artifacts/lettuce_centroid.npy)).
   - Evaluates cosine similarity to ensure input features reside within the botanical manifold.
4. **Structured Status Outputs**:
   - `PREDICTION_AVAILABLE`: Confirmed leaf; inference proceeds normally.
   - `NO_LEAF_DETECTED`: Non-leaf object detected (e.g., face, vehicle, pet). Nutrient prediction and farmer action plan suppressed.
   - `INVALID_IMAGE`: Corrupted, blank, or undersized image.
   - `LEAF_VERIFICATION_UNCERTAIN`: Ambiguous visual features. Asks user for a clearer photograph.

### 3.2 Farmer Recommendation & Conflict Engine Integration (`recommendations.py` & `app.py`)
- **Strict Gating**: Action plans (Check / Act / Monitor) only generate when the leaf verifier explicitly confirms `PREDICTION_AVAILABLE`.
- **Independent Soil Screening**: If an uploaded image is rejected (e.g. non-leaf), soil screening findings remain fully functional and informative.
- **Evidence Consistency Cross-Checking**:
  - `consistent`: Image prediction aligns with soil deficit screening.
  - `conflict`: Image prediction suggests deficiency, but soil measurement does not support it (advises user to check pH, root health, and irrigation before applying fertilizer).
  - `insufficient`: Corresponding soil measurement not provided.
- **Local Ollama Support**: Explains approved guidance in farmer-friendly language without hallucinating unverified dosages or contradictory facts, with automatic fallback to deterministic guidance.

### 3.3 Streamlit Dark Botanical Dashboard Polish
- Single clean image source control via `st.radio("Image input method", ["Upload photo", "Capture with camera"])`.
- Camera is never active on initial load and only starts when explicitly selected.
- No white boxes, borders, or layout clutter against the dark botanical theme (`#0B1210`).
- Strictly compliant widget layout maintaining full backward compatibility with test suites.

---

## 4. Test Verification Summary

Both test suites pass 100%:

1. **Legacy Test Suite** ([`test_streamlit_app.py`](file:///A:/Projects%21%21%21/PlantHealthWebsite/test_streamlit_app.py)):
   - **14 / 14 tests passing** (41.96s)
   - Verifies widget order, default thresholds, crop compatibility, invalid soil inputs, and model metadata integrity.

2. **Hardened Pipeline Test Suite** ([`test_leaf_pipeline.py`](file:///A:/Projects%21%21%21/PlantHealthWebsite/test_leaf_pipeline.py)):
   - **17 / 17 tests passing** (3.98s)
   - Verifies OOD rejection of human faces, vehicles, uniform patches, blank images, and small files.
   - Verifies 100% acceptance of real test leaves (`PREDICTION_AVAILABLE`).
   - Verifies recommendation conflict detection, missing soil handling, Ollama fallback, and Streamlit state switching.

**Total Test Coverage**: **31 / 31 tests passing (100% pass rate)**.

