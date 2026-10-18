"""
app.py - Plant Health Analyzer (Streamlit website)

Two evidence sources are kept strictly separate:
  1. the image model (MobileNetV3-Small) reads the leaf photograph, and
  2. transparent threshold rules (recommendations.interpret_soil) screen the soil readings.
Soil readings never change the image prediction. All thresholds and wording live in
recommendations.py so that the page, the tests and the Ollama prompt always agree.

This file is plain ASCII on purpose, so it cannot be damaged by Windows code pages.
"""
import io
import json
import logging
import math
import os
from numbers import Real
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

from leaf_verifier import LeafVerificationResult, verify_leaf_image
from recommendations import (
    IMAGE_OK,
    IMAGE_WITHHELD_MESSAGES,
    PROVISIONAL_NOTE,
    SOIL_INPUT_LIMITS,
    STATUS_CONFLICT,
    STATUS_CONSISTENT,
    STATUS_INSUFFICIENT,
    build_report,
    call_ollama,
    findings_from_readings,
    interpret_soil,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("PLANT_MODEL_PATH", ROOT / "artifacts" / "plant_health_model.pth"))
CLASSES_PATH = Path(os.environ.get("PLANT_CLASSES_PATH", ROOT / "artifacts" / "classes.json"))
PREPROCESSING_PATH = Path(os.environ.get("PLANT_PREPROCESSING_PATH", CLASSES_PATH.parent / "preprocessing.json"))
MODEL_CARD_PATH = Path(os.environ.get("PLANT_MODEL_CARD_PATH", CLASSES_PATH.parent / "model_card.json"))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
CLOSE_CALL_MARGIN = 0.20
LOW_CONFIDENCE_LIMIT = 0.35
DEFAULT_PREPROCESSING = {"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# Model-specific disclosures. Defaults preserve legacy-model wording.
MODEL_DATASET_DESCRIPTION = os.environ.get(
    "PLANT_DATASET_DESCRIPTION", "AI-generated images; not verified on real plants."
)
MODEL_DATASET_PROVENANCE = os.environ.get(
    "PLANT_DATASET_PROVENANCE", "ai_generated_images_not_verified_real_plants"
)
MODEL_SCOPE = os.environ.get("PLANT_MODEL_SCOPE", "hydroponic_lettuce_only")
MODEL_WARNING = os.environ.get(
    "PLANT_MODEL_WARNING",
    "This model was trained using AI-generated images and has not been "
    "validated on real plants. Treat the result as experimental.",
)
SUPPORTED_IMAGE_CROP = "Lettuce"
SOIL_SCREENING_METHOD = "provisional_rule_based_thresholds"
SOIL_SCREENING_DISCLAIMER = (
    "General screening only; thresholds are provisional, not crop-specific, "
    "and are not a diagnosis."
)

# Leaf-verifier status -> status used by this page and the report
_VERIFIER_STATUS = {
    "INVALID_IMAGE": "invalid_image",
    "NO_LEAF_DETECTED": "no_leaf_detected",
    "LEAF_VERIFICATION_UNCERTAIN": "leaf_verification_uncertain",
}

st.set_page_config(
    page_title="Plant Health Analyzer",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling (dark green theme; the native theme is set in .streamlit/config.toml)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .stApp { background-color: #0B1210 !important; color: #E8ECE9 !important;
             font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif; }
    .block-container { padding-top: 1.75rem !important; padding-bottom: 3rem !important; max-width: 1180px !important; }
    header[data-testid="stHeader"] { background-color: rgba(11, 18, 16, 0.85) !important; }
    h1, h2, h3, h4, h5, h6 { color: #F2F6F3 !important; font-weight: 700 !important; letter-spacing: -0.01em; }
    p, span, label, li { color: #E8ECE9; }

    .hero-card { background: linear-gradient(180deg, #13221C 0%, #0F1A15 100%); border: 1px solid #233B2F;
                 border-radius: 16px; padding: 1.6rem 2rem; margin-bottom: 1.25rem; }
    .eyebrow { font-size: 0.75rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;
               color: #8BE28B; margin-bottom: 0.4rem; }
    .hero-title { font-size: 2.2rem; font-weight: 700; color: #FFFFFF; line-height: 1.2; margin: 0 0 0.5rem 0; }
    .hero-subtitle { font-size: 1.02rem; color: #B4C7BB; line-height: 1.55; max-width: 860px; margin-bottom: 1rem; }
    .pill { display: inline-block; font-size: 0.8rem; font-weight: 600; padding: 0.3rem 0.85rem;
            border-radius: 50px; margin-right: 0.6rem; margin-top: 0.3rem; }
    .pill-green { background: rgba(139, 226, 139, 0.08); color: #8BE28B; border: 1px solid rgba(139, 226, 139, 0.3); }
    .pill-amber { background: rgba(224, 169, 83, 0.08); color: #E0A953; border: 1px solid rgba(224, 169, 83, 0.3); }

    .step-title { font-size: 1.25rem; font-weight: 700; color: #F2F6F3; margin: 1.4rem 0 0.15rem 0; }
    .step-help { font-size: 0.9rem; color: #9EB3A6; margin-bottom: 0.8rem; }

    div[data-baseweb="input"] { background-color: #14221C !important; border: 1px solid #2B4337 !important; border-radius: 8px !important; }
    div[data-baseweb="input"]:focus-within { border-color: #8BE28B !important; box-shadow: 0 0 0 1px #8BE28B !important; }
    div[data-baseweb="select"] { background-color: #14221C !important; border-radius: 8px !important; }
    input, textarea { color: #E8ECE9 !important; background-color: transparent !important; }

    .stButton > button { background: linear-gradient(135deg, #1E3B2B 0%, #2A543D 100%) !important; color: #B9F5B9 !important;
                         border: 1px solid #3E7356 !important; font-weight: 700 !important; font-size: 1.05rem !important;
                         border-radius: 10px !important; padding: 0.7rem 2rem !important; }
    .stButton > button:hover { background: linear-gradient(135deg, #2A543D 0%, #3B7555 100%) !important; color: #FFFFFF !important;
                               border-color: #8BE28B !important; }

    [data-testid="stMetric"] { background: #14221C !important; border: 1px solid #1E3027 !important;
                               border-radius: 12px !important; padding: 0.9rem 1.2rem !important; }
    [data-testid="stMetricValue"] { color: #8BE28B !important; font-size: 1.6rem !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #9EB3A6 !important; font-weight: 600 !important; }
    [data-testid="stExpander"] { background-color: #111D18 !important; border: 1px solid #1E3027 !important; border-radius: 12px !important; }
    [data-testid="stSidebar"] { background-color: #0E1714 !important; border-right: 1px solid #1A2822 !important; }
    [data-testid="stFileUploader"] { background-color: #111D18 !important; border: 1.5px dashed #2E5040 !important;
                                     border-radius: 12px !important; padding: 0.4rem !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #111D18 !important; }
    table { width: 100%; }
    th { background-color: #14221C !important; color: #E8ECE9 !important; }
    td { color: #E8ECE9 !important; }
    .plan-card { background: #111D18; border: 1px solid #1E3027; border-radius: 12px; padding: 0.4rem 1.1rem 0.8rem 1.1rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model, image and soil helpers
# ---------------------------------------------------------------------------
def load_model_card():
    try:
        return json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_preprocessing():
    try:
        data = json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8"))
        return {key: data.get(key, DEFAULT_PREPROCESSING[key]) for key in DEFAULT_PREPROCESSING}
    except (OSError, ValueError):
        return dict(DEFAULT_PREPROCESSING)


def image_transform():
    cfg = load_preprocessing()
    size = int(cfg["image_size"])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])


@st.cache_resource(show_spinner=False)
def _load_model_cached(model_path: str, classes_path: str, signature: tuple):
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    if not isinstance(classes, list) or not classes:
        raise ValueError("classes.json must contain a non-empty list")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(classes))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Model checkpoint must contain a state dictionary")
    model.load_state_dict(state)
    model.eval()
    return model, classes


def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Image model checkpoint not found: {MODEL_PATH.name}")
        return None, None
    if not CLASSES_PATH.exists():
        st.error(f"Class mapping file not found: {CLASSES_PATH.name}")
        return None, None
    try:
        signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (MODEL_PATH, CLASSES_PATH))
        return _load_model_cached(str(MODEL_PATH), str(CLASSES_PATH), signature)
    except Exception as exc:
        logger.warning("Model load failed: %r", exc)
        st.error(f"Unable to load the image model ({type(exc).__name__}). The checkpoint file may be corrupted or incompatible.")
        return None, None


def read_uploaded_image(uploaded_file):
    data = uploaded_file.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        return None, f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > MAX_IMAGE_PIXELS:
                return None, "The image has too many pixels to process safely."
            img = ImageOps.exif_transpose(probe).convert("RGB")
            img.load()
        return img, None
    except Exception:
        return None, "The file could not be read as an image. Upload a valid JPG, PNG or WEBP file."


def validate_soil_values(soil_values):
    """Strict validation (raises ValueError). The page itself uses the per-reading rules in recommendations.py."""
    missing = sorted(set(SOIL_INPUT_LIMITS) - set(soil_values))
    if missing:
        raise ValueError(f"Missing soil field(s): {', '.join(missing)}")
    for key, (minimum, maximum) in SOIL_INPUT_LIMITS.items():
        val = soil_values[key]
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, Real) or not math.isfinite(val):
            raise ValueError(f"{key} must be a finite number or empty")
        if not minimum <= val <= maximum:
            raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}")


def assess_soil(soil_values):
    """Flagged soil readings in the original finding format. Uses the same rules as the page."""
    validate_soil_values(soil_values)
    return findings_from_readings(interpret_soil(soil_values))


def run_image_analysis(image, crop_type):
    """
    Verify the photograph, then (only if it is a verified lettuce leaf) run the image model.

    Fails closed: any status other than a verified leaf, and any model problem, produces a
    status that withholds the nutrient prediction. Soil readings are not used here at all.
    """
    out = {
        "status": "unknown", "label": None, "confidence": None, "class_probabilities": {},
        "close_call": False, "verification": {}, "reason": None,
    }
    if crop_type != SUPPORTED_IMAGE_CROP:
        out["status"] = "not_run_incompatible_crop"
        return out

    try:
        verification: LeafVerificationResult = verify_leaf_image(image)
    except Exception as exc:  # the verifier must never crash the page
        logger.exception("Leaf verification crashed: %r", exc)
        out["status"] = "leaf_verification_uncertain"
        out["reason"] = "Leaf verification could not be completed. Please try another photograph."
        return out

    out["verification"] = {
        "status": verification.status,
        "is_leaf": verification.is_leaf,
        "reason": verification.reason,
        "top_detected_class": verification.top_detected_class,
        "plant_probability": verification.plant_probability,
        "centroid_similarity": verification.centroid_similarity,
    }
    out["reason"] = verification.reason

    mapped = _VERIFIER_STATUS.get(verification.status)
    if mapped:
        out["status"] = mapped
        return out
    if verification.status != "PREDICTION_AVAILABLE" or not verification.is_leaf:
        out["status"] = "leaf_verification_uncertain"  # unknown status: fail closed
        return out

    model, classes = load_model()
    if model is None:
        out["status"] = "unavailable"
        return out
    try:
        x = image_transform()(image).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(model(x), dim=1)[0]
    except Exception as exc:
        logger.exception("Image model inference failed: %r", exc)
        out["status"] = "model_error"
        return out
    if not torch.all(torch.isfinite(probabilities)):
        out["status"] = "model_error"
        return out

    conf_val, index = torch.max(probabilities, dim=0)
    out["confidence"] = float(conf_val)
    out["label"] = classes[int(index)]
    out["class_probabilities"] = {name: round(float(p) * 100, 1) for name, p in zip(classes, probabilities)}
    top_two = torch.topk(probabilities, k=min(2, len(classes))).values
    out["close_call"] = len(top_two) == 2 and float(top_two[0] - top_two[1]) < CLOSE_CALL_MARGIN
    out["status"] = "low_confidence" if out["confidence"] < LOW_CONFIDENCE_LIMIT else IMAGE_OK
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def step_heading(title, help_text=""):
    st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)
    if help_text:
        st.markdown(f'<div class="step-help">{help_text}</div>', unsafe_allow_html=True)


def render_image_section(result, report):
    st.markdown("### 1. Leaf image result")
    status = result["status"]
    reason = result["reason"] or ""
    if status == "not_run_incompatible_crop":
        st.warning(
            f"The selected plant is not lettuce. The current image model supports {SUPPORTED_IMAGE_CROP.lower()} only, "
            "so no image diagnosis was generated."
        )
    elif status == "invalid_image":
        st.error(f"Image validation rejected: {reason}")
    elif status == "no_leaf_detected":
        st.warning(f"No supported leaf detected: {reason}")
        st.info("The model is trained on lettuce leaves. Photos of people, pets, vehicles, screens or household objects cannot be assessed.")
    elif status == "leaf_verification_uncertain":
        st.warning(f"Leaf verification uncertain: {reason}")
    elif status == "unavailable":
        st.warning(f"No usable image model is available ({MODEL_PATH.name}). Only the soil screening below can be shown.")
    elif status == "model_error":
        st.error("The image model could not process this photograph. Please try another one.")
    elif status == "low_confidence":
        st.warning(
            f"Low confidence ({result['confidence'] * 100:.1f}%): the visual pattern is ambiguous, so no image result "
            "or nutrient advice is shown. Try a clearer, closer photograph."
        )
        st.warning(MODEL_WARNING)
    elif report["image"]["verified"]:
        label = result["label"]
        confidence = result["confidence"]
        st.warning(MODEL_WARNING)
        col_a, col_b = st.columns(2)
        col_a.metric("Image model prediction", label)
        col_b.metric("Image model confidence", f"{confidence * 100:.1f}%")
        st.caption(
            "This is what the leaf photograph suggests. It is a screening result, not a confirmed diagnosis, "
            "and soil readings do not change it."
        )
        if result["close_call"]:
            st.warning("The top two classes have similar probabilities. Treat this as a preliminary indication, not a firm result.")
        with st.expander("Probability for each class"):
            st.table({"Class": list(result["class_probabilities"]), "Probability (%)": list(result["class_probabilities"].values())})
    else:
        st.warning(report["image"]["message"])


def render_soil_section(report):
    st.markdown("### 2. Soil screening (separate from the image)")
    readings = report["soil"]["readings"]
    summary = report["soil"]["summary"]
    st.table({
        "Reading": [r["label"] for r in readings],
        "Your value": [r["value_text"] for r in readings],
        "Project rule (provisional)": [r["rule"] for r in readings],
        "Result": [r["result_text"] for r in readings],
    })
    if summary["invalid"]:
        names = ", ".join(r["name"] for r in summary["invalid"])
        st.error(f"Not used because the value is invalid: {names}. Enter a valid number to include it.")
    if summary["flagged"]:
        names = ", ".join(r["name"] for r in summary["flagged"])
        st.warning(f"Outside the project screening rules: {names}.")
    elif summary["any_valid"]:
        st.success("All entered soil readings are within the project screening rules. This does not guarantee plant health.")
    else:
        st.info("No soil readings were entered, so no soil screening was done.")
    if summary["missing"] and summary["any_valid"]:
        st.caption("Not entered (not assessed): " + ", ".join(r["name"] for r in summary["missing"]) + ".")
    st.caption(PROVISIONAL_NOTE)


def render_comparison_section(report):
    st.markdown("### 3. Do the image and soil results agree?")
    cmp_ = report["comparison"]
    if cmp_ is None:
        st.info("Not available. The comparison needs a verified leaf image result, and none was produced.")
        return
    label, note, status = cmp_["status_label"], cmp_["note"], cmp_["status"]
    if status == STATUS_CONFLICT:
        st.error(f"{label}. {note}")
    elif status == STATUS_CONSISTENT:
        st.success(f"{label}. {note}")
    elif status == STATUS_INSUFFICIENT:
        st.warning(f"{label}. {note}")
    else:
        st.info(f"{label}. {note}")
    st.table({
        "Nutrient": [r["nutrient"] for r in cmp_["rows"]],
        "Image result": [r["image_says"] for r in cmp_["rows"]],
        "Soil reading": [r["soil_value"] for r in cmp_["rows"]],
        "Soil rule": [r["soil_result"] for r in cmp_["rows"]],
        "Comparison": [r["status_label"] for r in cmp_["rows"]],
    })
    st.caption("Only the nutrient the image predicted is compared as direct evidence. pH and moisture are never used to confirm a nutrient deficiency.")


def render_plan_section(report):
    st.markdown("### 4. Action plan")
    plan = report["plan"]
    st.markdown(f"> {plan['summary']}")
    col_check, col_act, col_monitor = st.columns(3)
    for column, title, key in ((col_check, "Check", "check"), (col_act, "Do now", "act"), (col_monitor, "Monitor", "monitor")):
        with column:
            st.markdown(f"#### {title}")
            for item in plan[key]:
                st.markdown(f"- {item}")


def render_explanation_section(report, soil_values, use_ai):
    st.markdown("### 5. Plain-language explanation")
    if not use_ai:
        st.caption("The AI explanation is switched off in the sidebar. The report above is complete without it.")
        return
    image = report["image"]
    with st.spinner("Writing a plain-language explanation. This can take up to a minute..."):
        text, error = call_ollama(
            predicted_class=image["predicted_class"],
            confidence_pct=image["confidence_pct"],
            soil_values=soil_values,
            recommendation=report["plan"],
        )
    if text:
        st.markdown(text)
        st.caption("Written by a local AI model from the fixed report above. If it ever differs from the tables above, trust the tables.")
    else:
        st.info("The AI explanation is not available right now. The report above is complete without it.")
        if error:
            st.caption(error)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About this tool")
    st.caption(
        "Two separate checks: a leaf-image model and simple soil threshold rules. "
        "Each one works on its own; soil readings never change the image result."
    )
    use_ai_explanation = st.checkbox(
        "Write an AI explanation (needs Ollama)",
        value=True,
        help="Uses a local Ollama model to explain the report in plain English. Turn it off if Ollama is not running.",
    )
    st.markdown("---")
    st.caption(f"Active checkpoint: `{MODEL_PATH.parent.name}/{MODEL_PATH.name}`")
    card = load_model_card()
    if card:
        with st.expander("Model card and benchmarks", expanded=False):
            if "model_details" in card:
                md = card["model_details"]
                st.markdown(f"**Model:** {md.get('name', 'MobileNetV3-Small')}")
                st.markdown(f"**Selection:** {md.get('selection_protocol', 'Validation-selected')}")
                if "evaluation_summary" in card:
                    es = card["evaluation_summary"]
                    st.markdown("**Reported benchmark metrics:**")
                    if "validation_macro_f1" in es:
                        st.write(f"- Validation Macro-F1: `{es['validation_macro_f1']*100:.1f}%`")
                    if "validation_balanced_accuracy" in es:
                        st.write(f"- Validation Balanced Acc: `{es['validation_balanced_accuracy']*100:.1f}%`")
                    if "held_out_test_macro_f1" in es:
                        st.write(f"- Held-Out Test Macro-F1: `{es['held_out_test_macro_f1']*100:.1f}%`")
                    if "out_of_domain_synthetic_test_accuracy" in es:
                        st.write(f"- Cross-Dataset Benchmark Acc (`data/test`): `{es['out_of_domain_synthetic_test_accuracy']*100:.1f}%`")
                    st.caption("Benchmark metrics reflect specific research splits only and do not guarantee real-world field accuracy.")
            elif "training_data" in card:
                td = card["training_data"]
                st.markdown(f"**Dataset:** {td.get('name', 'Lettuce-NPK-dataset')}")
                st.markdown(f"**Provenance:** {td.get('provenance_note', 'AI-generated images')}")
                st.caption("Legacy checkpoint: no verified real-plant accuracy estimate.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero-card">
    <div class="eyebrow">College project | Preliminary plant health check</div>
    <div class="hero-title">Plant Health Analyzer</div>
    <div class="hero-subtitle">
        Upload a photo of a lettuce leaf and, if you have them, enter your soil measurements.
        You get a leaf-image result, a separate soil check, a comparison of the two, and a simple action plan.
    </div>
    <span class="pill pill-green">Experimental prototype</span>
    <span class="pill pill-amber">Lettuce leaves only</span>
    <span class="pill pill-amber">Not a replacement for a soil laboratory or an agronomist</span>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("College project prototype | Predictions are experimental and should not replace expert agricultural advice.")
st.caption("Image analysis supports lettuce leaves. Crop selection is recorded as metadata and does not affect the image model.")

# ---------------------------------------------------------------------------
# Step 1: leaf photograph
# ---------------------------------------------------------------------------
step_heading("Step 1. Leaf photograph", "Use one clear, in-focus lettuce leaf in daylight. JPG, PNG or WEBP, up to 25 MB.")

source_mode = st.radio(
    "Image input method",
    ["Upload photo", "Capture with camera"],
    horizontal=True,
    label_visibility="collapsed",
    key="image_source_mode_radio",
)

active_image_file = None
if source_mode == "Upload photo":
    active_image_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photograph of a single leaf. Supported formats: JPG, PNG, WEBP up to 25 MB.",
        key="leaf_file_uploader",
    )
else:
    active_image_file = st.camera_input("Take a photo of your leaf", key="leaf_camera_widget")

image = None
if active_image_file:
    image, upload_error = read_uploaded_image(active_image_file)
    if upload_error:
        st.error(upload_error)
    elif image is not None:
        col_preview, col_preview_meta = st.columns([1, 2])
        with col_preview:
            st.image(image, caption="Selected leaf photograph", width="stretch")
        with col_preview_meta:
            source_tag = "Camera capture" if source_mode == "Capture with camera" else "Uploaded file"
            st.markdown(f"**Source:** {source_tag}")
            st.markdown(f"**Size:** {image.width} x {image.height} px")
            st.caption("The photo is resized to 224 x 224 pixels for the model. The photograph is checked to confirm it shows a leaf before any prediction is made.")

# ---------------------------------------------------------------------------
# Step 2: soil readings (optional)
# ---------------------------------------------------------------------------
step_heading(
    "Step 2. Soil readings (optional)",
    "Leave a box empty if you did not measure it. Empty boxes are treated as not measured, never as zero.",
)
row1 = st.columns(3)
row2 = st.columns(3)
with row1[0]:
    soil_n = st.number_input(
        "Nitrogen (mg/kg)", min_value=0.0, max_value=500.0, value=None, step=1.0, placeholder="e.g., 35",
        help="Available soil Nitrogen. Project rule: flagged low if below 40 mg/kg. Allowed 0 to 500.",
    )
with row1[1]:
    soil_p = st.number_input(
        "Phosphorus (mg/kg)", min_value=0.0, max_value=300.0, value=None, step=1.0, placeholder="e.g., 20",
        help="Available soil Phosphorus. Project rule: flagged low if below 15 mg/kg. Allowed 0 to 300.",
    )
with row1[2]:
    soil_k = st.number_input(
        "Potassium (mg/kg)", min_value=0.0, max_value=500.0, value=None, step=1.0, placeholder="e.g., 180",
        help="Available soil Potassium. Project rule: flagged low if below 140 mg/kg. Allowed 0 to 500.",
    )
with row2[0]:
    soil_ph = st.number_input(
        "Soil pH (0-14)", min_value=0.0, max_value=14.0, value=None, step=0.1, placeholder="e.g., 6.2",
        help="Provisional project range: 5.0 to 7.5.",
    )
with row2[1]:
    soil_moisture = st.number_input(
        "Soil moisture (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, placeholder="e.g., 32",
        help="Provisional project range: 15% to 55%. Allowed 0 to 100.",
    )
st.caption("The soil ranges are the project's provisional screening rules. " + PROVISIONAL_NOTE.split(". ", 1)[1])

soil_values = {
    "soil_n": soil_n,
    "soil_p": soil_p,
    "soil_k": soil_k,
    "soil_ph": soil_ph,
    "soil_moisture": soil_moisture,
}

# ---------------------------------------------------------------------------
# Step 3: extra details (optional)
# ---------------------------------------------------------------------------
with st.expander("Step 3. Extra details (optional)", expanded=False):
    crop_type = st.selectbox(
        "Plant species / crop",
        ["Lettuce", "Tomato", "Potato", "Pepper", "Corn", "Grape", "Apple", "Other", "Not sure"],
        index=0,
        help="The image model supports Lettuce only.",
    )
    if crop_type == SUPPORTED_IMAGE_CROP:
        st.caption("Lettuce: compatible with the leaf image model.")
    else:
        st.warning(f"The image model supports Lettuce only. For '{crop_type}' only the soil screening is done.")
    plant_age = st.selectbox(
        "Growth stage", ["Not sure", "Seedling", "Vegetative", "Flowering", "Fruiting"],
        help="Recorded in the report only.",
    )
    location = st.text_input("Location (optional)", placeholder="e.g., Greenhouse B, Hydroponics Bay 3")
    notes = st.text_area("Observed symptoms or notes (optional)", placeholder="e.g., lower leaves showing yellowing at the edge")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if active_image_file and image is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Analyze leaf", type="primary", width="stretch"):
        with st.spinner("Checking the photograph and running the image model..."):
            result = run_image_analysis(image, crop_type)

        confidence_pct = round(result["confidence"] * 100, 1) if result["confidence"] is not None else None
        report = build_report(result["status"], result["label"], confidence_pct, soil_values, result["reason"])
        image_model_status = result["status"]
        prediction_generated = report["image"]["verified"]
        label = result["label"] if prediction_generated else None
        soil_readings = {r["key"]: r for r in report["soil"]["readings"]}
        soil_findings = report["soil"]["findings"]
        soil_invalid = report["soil"]["summary"]["invalid"]

        st.markdown("## Report")
        render_image_section(result, report)
        render_soil_section(report)
        render_comparison_section(report)
        render_plan_section(report)
        if prediction_generated:
            render_explanation_section(report, soil_values, use_ai_explanation)
        else:
            st.info("Nutrient treatment advice and the AI explanation are withheld because no verified leaf prediction was produced.")

        with st.expander("Why this result?", expanded=False):
            st.markdown(
                "**How to read it**\n"
                "- The image model looks for visual patterns such as yellowing, edge scorch and discolouration.\n"
                "- The confidence percentage is the model's own score. It is not a measured accuracy and not a guarantee.\n"
                "- Leaf symptoms can also be caused by disease, root stress, watering or lighting, so confirm with a soil or leaf test before treating.\n"
                "- Soil readings are checked against simple threshold rules and are never fed into the image model."
            )
        with st.expander("Model information", expanded=False):
            st.markdown(
                f"- **Model:** MobileNetV3-Small\n"
                f"- **Weights file:** `{MODEL_PATH.name}`\n"
                f"- **Training data:** {MODEL_DATASET_DESCRIPTION}"
            )
        with st.expander("Extra details you entered", expanded=False):
            cols = st.columns(4)
            cols[0].markdown(f"**Crop:**<br>{crop_type}", unsafe_allow_html=True)
            cols[1].markdown(f"**Growth stage:**<br>{plant_age}", unsafe_allow_html=True)
            cols[2].markdown("**Location:**<br>" + (location or "Not provided").replace("<", "&lt;"), unsafe_allow_html=True)
            cols[3].markdown("**Notes:**<br>" + (notes or "None provided").replace("<", "&lt;"), unsafe_allow_html=True)

        report_json = {
            "crop_type": crop_type,
            "growth_stage": plant_age,
            "location": location or None,
            "observations": notes or None,
            "soil_parameters": soil_values,
            "soil_screening_findings": soil_findings,
            "soil_screening_method": SOIL_SCREENING_METHOD,
            "soil_screening_disclaimer": SOIL_SCREENING_DISCLAIMER,
            "soil_screening_status": "invalid_input" if soil_invalid else "completed",
            "soil_screening_units": {
                "soil_n": "mg/kg", "soil_p": "mg/kg", "soil_k": "mg/kg", "soil_ph": "pH scale", "soil_moisture": "%",
            },
            "soil_reading_results": {
                key: {"value": r["value"], "state": r["state"], "result": r["result_text"]}
                for key, r in soil_readings.items()
            },
            "image_analysis_method": "image_model_only",
            "image_model_status": image_model_status,
        }
        if result["verification"]:
            report_json["leaf_verification"] = result["verification"]
        if prediction_generated and label is not None:
            report_json["image_prediction"] = label
            report_json["image_model_confidence_percent"] = confidence_pct
            report_json["image_model_scope"] = MODEL_SCOPE
            report_json["image_model_checkpoint"] = MODEL_PATH.name
            report_json["image_model_checkpoint_path"] = str(MODEL_PATH.resolve())
            report_json["class_probabilities_percent"] = result["class_probabilities"]
            report_json["training_data_provenance"] = MODEL_DATASET_PROVENANCE
            report_json["validated_on_real_plants"] = False
            report_json["image_vs_soil_evidence"] = {
                "status": report["comparison"]["status"],
                "note": report["comparison"]["note"],
            }
            card_obj = load_model_card()
            if card_obj and card_obj.get("config_name"):
                report_json["model_config"] = card_obj["config_name"]
        elif crop_type != SUPPORTED_IMAGE_CROP:
            report_json["crop_compatibility_notice"] = (
                f"Model supports {SUPPORTED_IMAGE_CROP.lower()} only; "
                f"no image diagnosis was generated for {crop_type}."
            )

        with st.expander("Technical details (report JSON)", expanded=False):
            st.json(report_json)
else:
    st.info(
        "To start: (1) upload a clear photo of one lettuce leaf, or switch to the camera, "
        "(2) optionally enter your soil readings, then (3) press Analyze leaf."
    )
