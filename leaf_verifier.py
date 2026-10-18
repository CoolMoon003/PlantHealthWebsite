"""
leaf_verifier.py
================
Dedicated input validity and leaf verification stage before nutrient classification.

Responsibilities:
1. Low-level image validation (format, dimensions, non-blank, color variance).
2. Semantic plant vs non-plant verification using locally cached ImageNet MobileNetV3-Small:
   - Detects overt non-leaf subjects: people/clothing, pets/animals, vehicles, electronics, furniture.
3. Botanical manifold check:
   - Measures cosine similarity against the precomputed lettuce feature centroid (artifacts/lettuce_centroid.npy).
4. Structured decision states:
   - INVALID_IMAGE: Corrupted, blank, or unusable file.
   - NO_LEAF_DETECTED: Verified to be an unrelated object, human, pet, or man-made article.
   - LEAF_VERIFICATION_UNCERTAIN: Ambiguous visual characteristics; prompts user for a clear leaf close-up.
   - PREDICTION_AVAILABLE: Confirmed leaf image suitable for nutrient deficiency screening.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
CENTROID_PATH = ROOT / "artifacts" / "lettuce_centroid.npy"
PLANT_MODEL_PATH = ROOT / "artifacts" / "plant_health_model.pth"

# Plant / vegetable / agricultural keywords in ImageNet-1K
PLANT_KEYWORDS = {
    "cabbage", "cauliflower", "broccoli", "zucchini", "cucumber", "artichoke",
    "cardoon", "squash", "pepper", "corn", "ear", "rapeseed", "daisy",
    "slipper", "pot", "hay", "greenhouse", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "pomegranate", "apple", "mushroom",
    "agaric", "bolete", "earthstar", "hen-of-the-woods", "stinkhorn", "leaf"
}

# Explicit non-plant keywords in ImageNet-1K (people, pets, vehicles, tech, furniture)
NON_PLANT_KEYWORDS = {
    "dog", "cat", "terrier", "hound", "retriever", "spaniel", "shepherd", "poodle",
    "wolf", "fox", "bear", "lion", "tiger", "leopard", "cheetah", "elephant",
    "horse", "zebra", "cow", "ox", "sheep", "pig", "monkey", "gorilla", "chimpanzee",
    "car", "truck", "bus", "train", "wagon", "cab", "bicycle", "motorcycle", "airplane",
    "boat", "ship", "canoe", "yacht", "convertible", "minivan", "pickup", "trailer",
    "suit", "groom", "wig", "jersey", "trench coat", "jean", "sweatshirt", "kimono",
    "scuba", "mask", "gown", "tie", "hat", "shoe", "boot", "sandal", "brassiere",
    "cellular telephone", "laptop", "computer keyboard", "mouse", "monitor", "television",
    "desk", "dining table", "bookcase", "chair", "wardrobe", "refrigerator", "microwave",
    "building", "church", "bridge", "palace", "fountain", "street sign", "traffic light",
    "pillow", "quilt", "envelope", "plate", "cup", "coffee mug", "water bottle"
}


@dataclass
class LeafVerificationResult:
    status: str
    is_leaf: bool
    reason: str
    top_detected_class: str
    plant_probability: float
    centroid_similarity: float
    details: dict[str, Any] = field(default_factory=dict)


# Cached singletons
_IMAGENET_MODEL: torch.nn.Module | None = None
_IMAGENET_CATEGORIES: list[str] | None = None
_IMAGENET_TRANSFORM: Any = None
_PLANT_INDICES: set[int] | None = None
_NON_PLANT_INDICES: set[int] | None = None
_CENTROID: np.ndarray | None = None
_FEATURE_MODEL: torch.nn.Module | None = None


def _load_imagenet_model():
    global _IMAGENET_MODEL, _IMAGENET_CATEGORIES, _IMAGENET_TRANSFORM, _PLANT_INDICES, _NON_PLANT_INDICES
    if _IMAGENET_MODEL is not None:
        return _IMAGENET_MODEL, _IMAGENET_CATEGORIES, _IMAGENET_TRANSFORM, _PLANT_INDICES, _NON_PLANT_INDICES

    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        model.eval()
        categories = weights.meta["categories"]
        tf = weights.transforms()

        plant_idx = set(i for i, cat in enumerate(categories) if any(w in cat.lower() for w in PLANT_KEYWORDS))
        non_plant_idx = set(i for i, cat in enumerate(categories) if any(w in cat.lower() for w in NON_PLANT_KEYWORDS))

        _IMAGENET_MODEL = model
        _IMAGENET_CATEGORIES = categories
        _IMAGENET_TRANSFORM = tf
        _PLANT_INDICES = plant_idx
        _NON_PLANT_INDICES = non_plant_idx
        return _IMAGENET_MODEL, _IMAGENET_CATEGORIES, _IMAGENET_TRANSFORM, _PLANT_INDICES, _NON_PLANT_INDICES
    except Exception as exc:
        logger.warning(f"Could not initialize ImageNet model for leaf verification: {exc}")
        return None, None, None, None, None


def _load_centroid() -> np.ndarray | None:
    global _CENTROID
    if _CENTROID is not None:
        return _CENTROID
    if CENTROID_PATH.exists():
        try:
            arr = np.load(str(CENTROID_PATH))
            _CENTROID = arr / np.linalg.norm(arr)
            return _CENTROID
        except Exception as exc:
            logger.warning(f"Failed to load centroid: {exc}")
    return None


def _get_embedding_similarity(image: Image.Image) -> float | None:
    centroid = _load_centroid()
    if centroid is None or not PLANT_MODEL_PATH.exists():
        return None

    global _FEATURE_MODEL
    try:
        if _FEATURE_MODEL is None:
            m = models.mobilenet_v3_small(weights=None)
            m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, 4)
            st = torch.load(str(PLANT_MODEL_PATH), map_location="cpu", weights_only=True)
            m.load_state_dict(st)
            m.eval()
            _FEATURE_MODEL = m

        captured_feat = []

        def _hook(module, inp, out):
            captured_feat.append(out.squeeze())

        handle = _FEATURE_MODEL.classifier[0].register_forward_hook(_hook)

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            _ = _FEATURE_MODEL(tf(image.convert("RGB")).unsqueeze(0))
        handle.remove()

        if captured_feat:
            v = captured_feat[0].numpy()
            norm_v = v / (np.linalg.norm(v) + 1e-9)
            return float(np.dot(norm_v, centroid))
    except Exception as exc:
        logger.warning(f"Could not compute embedding similarity: {exc}")
    return None


def verify_leaf_image(image: Image.Image | None) -> LeafVerificationResult:
    """
    Main verification pipeline entry point.
    Returns LeafVerificationResult with explicit status and diagnostic reasons.
    """
    if image is None:
        return LeafVerificationResult(
            status="INVALID_IMAGE",
            is_leaf=False,
            reason="No image was provided or the file could not be read.",
            top_detected_class="none",
            plant_probability=0.0,
            centroid_similarity=0.0,
        )

    # 1. Low-level structural checks
    try:
        rgb_img = image.convert("RGB")
        width, height = rgb_img.size
        if width < 32 or height < 32:
            return LeafVerificationResult(
                status="INVALID_IMAGE",
                is_leaf=False,
                reason=f"Image dimensions ({width}x{height}) are too small for safe inspection (minimum 32x32).",
                top_detected_class="too_small",
                plant_probability=0.0,
                centroid_similarity=0.0,
            )

        arr = np.array(rgb_img, dtype=np.float32)
        std_val = float(arr.std())
        if std_val < 3.0:
            return LeafVerificationResult(
                status="INVALID_IMAGE",
                is_leaf=False,
                reason="The image is uniform or nearly blank without identifiable leaf structure.",
                top_detected_class="blank_uniform",
                plant_probability=0.0,
                centroid_similarity=0.0,
            )
    except Exception as exc:
        return LeafVerificationResult(
            status="INVALID_IMAGE",
            is_leaf=False,
            reason=f"Failed to decode or inspect image pixels: {exc}",
            top_detected_class="decode_error",
            plant_probability=0.0,
            centroid_similarity=0.0,
        )

    # 2. Embedding similarity to lettuce manifold
    similarity = _get_embedding_similarity(rgb_img)
    # Missing similarity is unknown, not evidence of a leaf.
    sim_val = similarity if similarity is not None else 0.0

    # 3. High-level Semantic Check (ImageNet MobileNetV3-Small)
    model, categories, tf, plant_idx, non_plant_idx = _load_imagenet_model()

    if model is not None and categories is not None and tf is not None:
        try:
            tensor_in = tf(rgb_img).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(tensor_in), dim=1)[0]

            top_idx = int(probs.argmax().item())
            top_cat = categories[top_idx]
            top_prob = float(probs[top_idx].item())
            plant_prob = float(sum(probs[i].item() for i in plant_idx))
            non_plant_prob = float(sum(probs[i].item() for i in non_plant_idx))

            details = {
                "top_detected_class": top_cat,
                "top_class_confidence": round(top_prob * 100, 1),
                "plant_probability": round(plant_prob * 100, 1),
                "non_plant_probability": round(non_plant_prob * 100, 1),
                "centroid_similarity": round(sim_val, 3),
            }

            # Decision Logic (multi-step, from most certain to most uncertain):
            # Step 1: Immediately reject if plant probability is extremely low.
            #         This fires BEFORE sim_val is used, so the 0.5 fallback cannot
            #         rescue a clear non-plant image (e.g. car plant_prob=0.4%).
            if plant_prob < 0.05:
                return LeafVerificationResult(
                    status="NO_LEAF_DETECTED",
                    is_leaf=False,
                    reason=f"No plant leaf detected (plant probability: {plant_prob*100:.1f}%, top detected: {top_cat}). The image does not show identifiable foliage.",
                    top_detected_class=top_cat,
                    plant_probability=plant_prob,
                    centroid_similarity=sim_val,
                    details=details,
                )

            # Step 2: Reject if combined signal is weak OR object is an obvious non-plant.
            if (plant_prob < 0.08 and sim_val < 0.20) or (top_idx in non_plant_idx and top_prob >= 0.12 and plant_prob < 0.10):
                return LeafVerificationResult(
                    status="NO_LEAF_DETECTED",
                    is_leaf=False,
                    reason=f"No plant leaf detected. The image appears to contain {top_cat} ({top_prob*100:.1f}%), a pet, person, or non-plant object.",
                    top_detected_class=top_cat,
                    plant_probability=plant_prob,
                    centroid_similarity=sim_val,
                    details=details,
                )

            # Step 3: Reject if manifold similarity is far below the leaf feature space.
            if sim_val < 0.15:
                return LeafVerificationResult(
                    status="NO_LEAF_DETECTED",
                    is_leaf=False,
                    reason=f"Image features are incompatible with lettuce foliage (similarity: {sim_val:.2f}, plant probability: {plant_prob*100:.1f}%).",
                    top_detected_class=top_cat,
                    plant_probability=plant_prob,
                    centroid_similarity=sim_val,
                    details=details,
                )

            # Step 4: Confirm as leaf — BOTH plant_prob AND sim_val must be above threshold,
            #         OR one is very high and the other is at least moderate.
            if (plant_prob >= 0.10 and sim_val >= 0.20) or plant_prob >= 0.30 or (sim_val >= 0.35 and plant_prob >= 0.05):
                return LeafVerificationResult(
                    status="PREDICTION_AVAILABLE",
                    is_leaf=True,
                    reason=f"Leaf structure verified (plant probability: {plant_prob*100:.1f}%, lettuce manifold similarity: {sim_val:.2f}).",
                    top_detected_class=top_cat,
                    plant_probability=plant_prob,
                    centroid_similarity=sim_val,
                    details=details,
                )

            # Case C: Ambiguous visual characteristics
            return LeafVerificationResult(
                status="LEAF_VERIFICATION_UNCERTAIN",
                is_leaf=False,
                reason=f"Leaf verification is inconclusive (top candidate: {top_cat}, plant probability: {plant_prob*100:.1f}%). Please provide a clearer, closer photograph of a single leaf.",
                top_detected_class=top_cat,
                plant_probability=plant_prob,
                centroid_similarity=sim_val,
                details=details,
            )

        except Exception as exc:
            logger.warning(f"Error during semantic leaf verification: {exc}")

    # Fallback when ImageNet is unavailable: only use a real similarity score.
    if similarity is None:
        return LeafVerificationResult(
            status="LEAF_VERIFICATION_UNCERTAIN",
            is_leaf=False,
            reason=(
                "Leaf verification is unavailable because neither semantic "
                "verification nor lettuce similarity could be computed. "
                "Please retry with a clear lettuce-leaf photograph."
            ),
            top_detected_class="verification_unavailable",
            plant_probability=0.0,
            centroid_similarity=0.0,
        )

    if sim_val >= 0.25:
        return LeafVerificationResult(
            status="PREDICTION_AVAILABLE",
            is_leaf=True,
            reason=f"Foliage manifold similarity verified ({sim_val:.2f}).",
            top_detected_class="foliage_pattern",
            plant_probability=0.5,
            centroid_similarity=sim_val,
        )
    elif sim_val < 0.05:
        return LeafVerificationResult(
            status="NO_LEAF_DETECTED",
            is_leaf=False,
            reason="Visual pattern is far outside the plant leaf feature manifold.",
            top_detected_class="unrelated_object",
            plant_probability=0.0,
            centroid_similarity=sim_val,
        )
    else:
        return LeafVerificationResult(
            status="LEAF_VERIFICATION_UNCERTAIN",
            is_leaf=False,
            reason="Ambiguous image pattern. Please provide a clear photograph of a single lettuce leaf.",
            top_detected_class="unknown",
            plant_probability=0.2,
            centroid_similarity=sim_val,
        )

