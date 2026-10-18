"""
recommendations.py
==================
Deterministic soil screening, image-vs-soil evidence comparison, farmer action
plan, and the (optional) Ollama plain-language explanation for the Plant Health
Analyzer.

This module deliberately does NOT import torch or streamlit, so every rule in it
can be unit-tested without a model, a GPU or a running website.

How the pieces fit together
---------------------------
1. ``interpret_soil``            - turns the five raw soil inputs into per-reading
                                   results (missing / invalid / low / ok ...).
                                   This is the ONLY place thresholds are applied.
2. ``compare_image_with_soil``   - compares the image model's predicted class with
                                   the matching soil reading (agreement, conflict
                                   or insufficient evidence). It never changes the
                                   predicted class.
3. ``build_recommendation``      - Check / Do now ("act") / Monitor plan for one
                                   predicted class (kept for backward compatibility).
4. ``build_report``              - one complete deterministic report, including the
                                   cases where the photo was rejected or unverified.
5. ``call_ollama``               - asks a local language model to explain the
                                   deterministic findings in plain English, then
                                   checks the answer against those findings. If
                                   anything goes wrong the caller simply keeps the
                                   deterministic report.

Design rules
------------
- Thresholds are the project's provisional screening rules (N < 40, P < 15,
  K < 140 mg/kg; pH 5.0-7.5; moisture 15-55 %). They are not universal
  agronomic truths and are always described as provisional.
- Soil readings never change the image prediction.
- Missing values stay missing; invalid values are never treated as measurements.
- No fertiliser doses, rates, schedules or guaranteed outcomes are produced.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import socket
import urllib.error
import urllib.request
from numbers import Real
from typing import Any

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment; fall back safely."""
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Ollama configuration - read from environment with sensible defaults
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_S: int = _env_int("OLLAMA_TIMEOUT_S", 90)

# ---------------------------------------------------------------------------
# Soil rules (single source of truth - app.py imports these)
# ---------------------------------------------------------------------------
SOIL_KEYS: tuple[str, ...] = ("soil_n", "soil_p", "soil_k", "soil_ph", "soil_moisture")

_SOIL_THRESHOLDS: dict[str, tuple[float | None, float | None]] = {
    # key: (low_flag_value, high_flag_value); None means no check on that side
    "soil_n": (40.0, None),
    "soil_p": (15.0, None),
    "soil_k": (140.0, None),
    "soil_ph": (5.0, 7.5),
    "soil_moisture": (15.0, 55.0),
}

# Physically plausible input limits. Values outside them are treated as invalid,
# never as real measurements.
SOIL_INPUT_LIMITS: dict[str, tuple[float, float]] = {
    "soil_n": (0.0, 500.0),
    "soil_p": (0.0, 300.0),
    "soil_k": (0.0, 500.0),
    "soil_ph": (0.0, 14.0),
    "soil_moisture": (0.0, 100.0),
}

_SOIL_LABELS: dict[str, str] = {
    "soil_n": "Nitrogen (mg/kg)",
    "soil_p": "Phosphorus (mg/kg)",
    "soil_k": "Potassium (mg/kg)",
    "soil_ph": "Soil pH",
    "soil_moisture": "Soil moisture (%)",
}
SOIL_NAMES: dict[str, str] = {
    "soil_n": "Nitrogen",
    "soil_p": "Phosphorus",
    "soil_k": "Potassium",
    "soil_ph": "Soil pH",
    "soil_moisture": "Soil moisture",
}
SOIL_UNITS: dict[str, str] = {
    "soil_n": "mg/kg",
    "soil_p": "mg/kg",
    "soil_k": "mg/kg",
    "soil_ph": "",
    "soil_moisture": "%",
}

SUPPORTED_CLASSES: tuple[str, ...] = (
    "Healthy",
    "Nitrogen_Deficiency",
    "Phosphorus_Deficiency",
    "Potassium_Deficiency",
)

# Nutrient deficiency class -> (nutrient name, soil key)
_DEFICIENCY_SOIL_KEY: dict[str, str] = {
    "Nitrogen_Deficiency": "soil_n",
    "Phosphorus_Deficiency": "soil_p",
    "Potassium_Deficiency": "soil_k",
}
_NPK_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Nitrogen", "soil_n", "Nitrogen_Deficiency"),
    ("Phosphorus", "soil_p", "Phosphorus_Deficiency"),
    ("Potassium", "soil_k", "Potassium_Deficiency"),
)

# Reading states
STATE_MISSING = "missing"
STATE_INVALID = "invalid"
STATE_LOW = "low"                 # N, P, K below their threshold
STATE_BELOW_RANGE = "below_range"  # pH / moisture below the provisional range
STATE_ABOVE_RANGE = "above_range"  # pH / moisture above the provisional range
STATE_OK = "ok"
FLAGGED_STATES = (STATE_LOW, STATE_BELOW_RANGE, STATE_ABOVE_RANGE)

# Evidence-agreement statuses (names kept from the original module)
STATUS_CONSISTENT = "consistent"
STATUS_CONFLICT = "conflict"
STATUS_INSUFFICIENT = "insufficient"
STATUS_NOT_NEEDED = "not_needed"  # Healthy prediction or no comparison applies

STATUS_LABELS: dict[str, str] = {
    STATUS_CONSISTENT: "AGREEMENT",
    STATUS_CONFLICT: "CONFLICT",
    STATUS_INSUFFICIENT: "INSUFFICIENT EVIDENCE",
    STATUS_NOT_NEEDED: "NOT COMPARED",
}

# Image-analysis statuses used by app.py
IMAGE_OK = "prediction_generated"
IMAGE_WITHHELD_MESSAGES: dict[str, str] = {
    "no_leaf_detected": "No supported lettuce leaf was found in the photograph, so no nutrient prediction is shown.",
    "invalid_image": "The photograph could not be used, so no nutrient prediction is shown.",
    "leaf_verification_uncertain": "The app could not confirm that the photograph shows a lettuce leaf, so no nutrient prediction is shown.",
    "low_confidence": "The image model was not confident enough to give a usable result, so no nutrient advice is shown.",
    "unavailable": "The image model is not available on this computer, so no image result can be shown.",
    "model_error": "The image model produced an unusable output, so no image result can be shown.",
    "not_run_incompatible_crop": "The image model supports lettuce only, so no image result was generated for the selected crop.",
    "unknown": "The photograph was not verified, so no nutrient prediction is shown.",
}

PROVISIONAL_NOTE = (
    "These soil ranges are the project's provisional screening rules, not universal agronomic truths. "
    "Suitable values depend on the crop, soil type, measurement method and local agricultural guidance."
)


# ---------------------------------------------------------------------------
# 1. Soil interpretation (deterministic - the only place thresholds are applied)
# ---------------------------------------------------------------------------

def _fmt(value: float, key: str | None = None) -> str:
    """Format a number without trailing zeros (50.0 -> '50'). pH keeps 1-2 decimals (8.0, 7.25)."""
    if key == "soil_ph":
        return f"{value:.1f}" if round(value, 1) == round(value, 6) else f"{value:.2f}"
    return f"{value:g}"


def _rule_text(key: str) -> str:
    low, high = _SOIL_THRESHOLDS[key]
    unit = SOIL_UNITS[key]
    if high is None:
        return f"Flagged low if below {_fmt(low)} {unit}"
    suffix = f" {unit}" if unit else ""
    return f"Provisional range {_fmt(low, key)} to {_fmt(high, key)}{suffix}"


def interpret_soil_value(key: str, raw: Any) -> dict[str, Any]:
    """Interpret ONE soil input. Returns a dict describing the reading."""
    unit = SOIL_UNITS[key]
    reading: dict[str, Any] = {
        "key": key,
        "name": SOIL_NAMES[key],
        "label": _SOIL_LABELS[key],
        "unit": unit,
        "value": None,
        "state": STATE_MISSING,
        "flagged": False,
        "rule": _rule_text(key),
        "value_text": "Not entered",
        "result_text": "Not entered - not assessed",
    }
    if raw is None:
        return reading

    if isinstance(raw, bool) or not isinstance(raw, Real):
        reading.update(state=STATE_INVALID, value_text="Invalid",
                       result_text="Invalid - not a number, so it was not used")
        return reading
    value = float(raw)
    if not math.isfinite(value):
        reading.update(state=STATE_INVALID, value_text="Invalid",
                       result_text="Invalid - not a finite number, so it was not used")
        return reading
    lo, hi = SOIL_INPUT_LIMITS[key]
    if not lo <= value <= hi:
        reading.update(
            state=STATE_INVALID,
            value_text=f"{_fmt(value, key)} (invalid)",
            result_text=f"Invalid - outside the plausible range {_fmt(lo)} to {_fmt(hi)}, so it was not used",
        )
        return reading

    low, high = _SOIL_THRESHOLDS[key]
    suffix = f" {unit}" if unit else ""
    reading["value"] = value
    reading["value_text"] = f"{_fmt(value, key)}{suffix}"
    if high is None:  # nutrient: only a low-side rule
        if value < low:
            reading.update(state=STATE_LOW, flagged=True,
                           result_text=f"LOW - below {_fmt(low)} {unit}")
        else:
            reading.update(state=STATE_OK,
                           result_text=f"Not low - not below {_fmt(low)} {unit}")
    else:  # pH / moisture: range rule
        if value < low:
            reading.update(state=STATE_BELOW_RANGE, flagged=True,
                           result_text=f"BELOW RANGE - under {_fmt(low, key)}{suffix}")
        elif value > high:
            reading.update(state=STATE_ABOVE_RANGE, flagged=True,
                           result_text=f"ABOVE RANGE - over {_fmt(high, key)}{suffix}")
        else:
            reading.update(state=STATE_OK,
                           result_text=f"Within range {_fmt(low, key)} to {_fmt(high, key)}{suffix}")
    return reading


def interpret_soil(soil_values: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Interpret all five soil inputs. Absent keys count as missing (never zero)."""
    values = soil_values if isinstance(soil_values, dict) else {}
    return {key: interpret_soil_value(key, values.get(key)) for key in SOIL_KEYS}


_LEGACY_FINDING = {
    "soil_n": ("Nitrogen", "Nitrogen may be low"),
    "soil_p": ("Phosphorus", "Phosphorus may be low"),
    "soil_k": ("Potassium", "Potassium may be low"),
    "soil_ph": ("pH", "pH is outside the general screening range 5.0-7.5"),
    "soil_moisture": ("Moisture", "Moisture is outside the general screening range 15-55%"),
}


def findings_from_readings(readings: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flagged readings in the legacy ``assess_soil`` finding format."""
    findings = []
    for key in SOIL_KEYS:
        reading = readings[key]
        if reading["flagged"]:
            nutrient, message = _LEGACY_FINDING[key]
            findings.append({"nutrient": nutrient, "value": reading["value"], "finding": message})
    return findings


def summarize_soil(readings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Group readings into flagged / ok / not-entered / invalid lists."""
    ordered = [readings[k] for k in SOIL_KEYS]
    return {
        "flagged": [r for r in ordered if r["flagged"]],
        "ok": [r for r in ordered if r["state"] == STATE_OK],
        "missing": [r for r in ordered if r["state"] == STATE_MISSING],
        "invalid": [r for r in ordered if r["state"] == STATE_INVALID],
        "any_valid": any(r["value"] is not None for r in ordered),
    }


# ---------------------------------------------------------------------------
# 2. Image prediction vs soil evidence
# ---------------------------------------------------------------------------

def _evidence_for_predicted(nutrient: str, reading: dict[str, Any]) -> tuple[str, str]:
    """(status, plain-language note) for the nutrient the image predicts as deficient."""
    key = reading["key"]
    threshold = _SOIL_THRESHOLDS[key][0]
    unit = SOIL_UNITS[key]
    state = reading["state"]

    if state == STATE_MISSING:
        return (
            STATUS_INSUFFICIENT,
            f"No {nutrient} soil measurement was provided. The image prediction cannot be "
            f"cross-checked against soil data. Consider testing your soil for {nutrient}.",
        )
    if state == STATE_INVALID:
        return (
            STATUS_INSUFFICIENT,
            f"The submitted {nutrient} value is not usable ({reading['result_text'].lower()}). "
            "No soil-versus-image comparison is possible until a valid reading is entered.",
        )
    if state == STATE_LOW:
        return (
            STATUS_CONSISTENT,
            f"Both the leaf image and the submitted {nutrient} reading ({reading['value_text']}, "
            f"below the screening threshold of {_fmt(threshold)} {unit}) point toward a possible "
            f"{nutrient} shortage. The threshold is a provisional screening rule, not a confirmed "
            "diagnostic limit.",
        )
    return (
        STATUS_CONFLICT,
        f"The leaf image suggests a possible {nutrient} deficiency, but the submitted {nutrient} "
        f"reading ({reading['value_text']}) is not below the project screening threshold "
        f"({_fmt(threshold)} {unit}), so the soil rule does not support that specific deficiency. "
        "This does not prove that the image result is wrong: the soil sample, the measurement, "
        "watering and drainage, pH or root health could all explain a difference.",
    )


def compare_image_with_soil(predicted_class: str, readings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Compare the image prediction with the soil readings, nutrient by nutrient.

    Only the nutrient the image predicted is compared as direct evidence. The other
    nutrients are shown as independent soil results. pH and moisture are never used
    as confirmation of a nutrient deficiency.
    """
    rows: list[dict[str, Any]] = []
    overall_status = STATUS_NOT_NEEDED
    overall_note = ""
    for nutrient, key, cls in _NPK_ROWS:
        reading = readings[key]
        predicted = predicted_class == cls
        if predicted:
            status, note = _evidence_for_predicted(nutrient, reading)
            overall_status, overall_note = status, note
            image_says = "Deficiency predicted"
        else:
            status = STATUS_NOT_NEEDED
            image_says = "Not predicted"
            if reading["flagged"]:
                note = (f"Soil flags {nutrient} as low. The image did not predict this, so treat it as a "
                        "separate soil concern, not an image result.")
            elif reading["state"] == STATE_OK:
                note = f"{nutrient} is not flagged as low by the soil rule."
            else:
                note = f"{nutrient} soil reading not available."
        rows.append({
            "nutrient": nutrient,
            "key": key,
            "image_says": image_says,
            "soil_value": reading["value_text"],
            "soil_result": reading["result_text"],
            "status": status,
            "status_label": STATUS_LABELS[status],
            "note": note,
        })
    if predicted_class == "Healthy":
        overall_note = ("The image model predicts Healthy, so no deficiency is compared with the soil. "
                        "The soil readings are assessed on their own.")
    elif predicted_class not in _DEFICIENCY_SOIL_KEY:
        overall_note = "No comparison is available for this prediction class."
    return {
        "status": overall_status,
        "status_label": STATUS_LABELS[overall_status],
        "note": overall_note,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3. Action plan (Check / Do now / Monitor)
# ---------------------------------------------------------------------------

_CLASS_TEXT: dict[str, dict[str, Any]] = {
    "Healthy": {
        "summary": (
            "The leaf image model did not identify visual patterns matching any of its three "
            "deficiency classes (Nitrogen, Phosphorus or Potassium deficiency). This does not "
            "guarantee the plant is fully healthy - early-stage deficiencies, other stress factors "
            "or image quality may prevent detection."
        ),
        "leaf_check": "Inspect the leaves closely for subtle discolouration, spotting or wilting that the photo may not show.",
        "monitor": [
            "Check the leaves again in a few days, especially new growth.",
            "Note any change in colour, texture or growth rate.",
        ],
    },
    "Nitrogen_Deficiency": {
        "summary": (
            "The leaf image shows a pattern the model associates with possible Nitrogen deficiency. "
            "Nitrogen shortage is commonly described as general yellowing of older leaves, but leaf "
            "symptoms can have other causes."
        ),
        "leaf_check": "Look at older leaves for even yellowing and compare them with the newest growth.",
        "monitor": [
            "Watch whether yellowing spreads to newer leaves.",
            "Track the colour of new growth over the next few days.",
        ],
    },
    "Phosphorus_Deficiency": {
        "summary": (
            "The leaf image shows a pattern the model associates with possible Phosphorus deficiency. "
            "Phosphorus shortage is commonly described as dark green or purplish colouring on older "
            "leaves, but leaf symptoms can have other causes."
        ),
        "leaf_check": "Look for dark green, reddish or purple tinting on older leaves, especially the underside and edges.",
        "monitor": [
            "Watch whether purple or reddish colouring spreads to younger leaves.",
            "Monitor overall plant vigour and growth rate.",
        ],
    },
    "Potassium_Deficiency": {
        "summary": (
            "The leaf image shows a pattern the model associates with possible Potassium deficiency. "
            "Potassium shortage is commonly described as browning or scorching along the edges of "
            "older leaves, but leaf symptoms can have other causes."
        ),
        "leaf_check": "Look at the edges of older leaves for browning or scorching.",
        "monitor": [
            "Watch whether edge browning spreads along the leaf or to younger leaves.",
            "Check whether several plants show the same symptom, which may point to a shared cause.",
        ],
    },
}

_ESCALATE_LINE = (
    "Ask a local agricultural adviser, or arrange a laboratory soil or leaf test, if symptoms spread "
    "or worsen, new leaves are affected, or the image and soil evidence still disagree after re-checking."
)


def _other_flag_lines(readings: dict[str, dict[str, Any]], skip_key: str | None) -> list[str]:
    """Check-list lines for flagged / invalid readings other than the compared nutrient."""
    lines = []
    for key in SOIL_KEYS:
        if key == skip_key:
            continue
        r = readings[key]
        name = r["name"]
        unit = f" {r['unit']}" if r["unit"] else ""
        low, high = _SOIL_THRESHOLDS[key]
        if r["state"] == STATE_LOW:
            lines.append(f"{name} ({r['value_text']}) is below the project threshold of {_fmt(low)}{unit}. "
                         "Verify this reading; it is a separate soil concern, not an image result.")
        elif r["state"] == STATE_BELOW_RANGE:
            lines.append(f"{name} ({r['value_text']}) is below the provisional range ({_fmt(low, key)} to {_fmt(high, key)}{unit}). "
                         "Re-check the measurement before changing anything.")
        elif r["state"] == STATE_ABOVE_RANGE:
            lines.append(f"{name} ({r['value_text']}) is above the provisional range ({_fmt(low, key)} to {_fmt(high, key)}{unit}). "
                         "Re-check the measurement before changing anything.")
        elif r["state"] == STATE_INVALID:
            lines.append(f"The {name} value could not be used ({r['result_text'].lower()}). Re-enter it to include it.")
    return lines


def _plan(predicted_class: str, readings: dict[str, dict[str, Any]], comparison: dict[str, Any]) -> dict[str, Any]:
    text = _CLASS_TEXT[predicted_class]
    skip_key = _DEFICIENCY_SOIL_KEY.get(predicted_class)
    status = comparison["status"]
    nutrient = SOIL_NAMES[skip_key] if skip_key else None
    other_lines = _other_flag_lines(readings, skip_key)

    check: list[str] = []
    act: list[str] = []
    if nutrient:
        if status == STATUS_CONSISTENT:
            check.append(f"Confirm the {nutrient} soil reading is accurate (re-test, or send a sample to a laboratory).")
            act.append(f"Do not apply {nutrient} fertiliser until the soil reading is confirmed. If it is confirmed low, "
                       "ask a local agricultural adviser which product and amount suit your crop and growing system.")
        elif status == STATUS_CONFLICT:
            check.append(f"The image and the {nutrient} soil reading disagree. Re-check how the soil was sampled and measured, "
                         "and re-take the leaf photo in even daylight.")
            act.append(f"Do not apply {nutrient} fertiliser on the strength of the image alone. First resolve the "
                       "disagreement by re-checking the soil reading, the photo, watering and drainage, and the roots.")
        else:
            check.append(f"Measure {nutrient} in the soil (or re-enter a valid value) so it can be compared with the image prediction.")
            act.append(f"Do not apply {nutrient} fertiliser based on the image prediction alone. Get a {nutrient} soil reading first.")
    check.append(text["leaf_check"])
    check.extend(other_lines)
    check.append("Check watering, drainage and root health, which can limit nutrient uptake whatever the soil reading shows.")

    if predicted_class == "Healthy":
        act.append("If the plant looks vigorous and no soil reading is flagged, keep your current care routine.")
    if other_lines:
        act.append("Verify each flagged or invalid soil reading listed under Check before changing anything, and follow "
                   "local agricultural guidance for your crop. A soil flag is separate and does not mean the image result is wrong.")
    act.append("This tool gives no fertiliser amounts. Use a laboratory result and local guidance for any treatment decision.")

    monitor = list(text["monitor"])
    monitor.append("Re-photograph the same leaf in a few days in similar light and compare it with today's result.")
    monitor.append(_ESCALATE_LINE)
    return {"summary": text["summary"], "check": check, "act": act, "monitor": monitor}


def build_recommendation(predicted_class: str, soil_values: dict[str, Any]) -> dict[str, Any]:
    """
    Return the deterministic plan for a VERIFIED image prediction.

    Keys: summary, check, act (shown as "Do now"), monitor, evidence_status,
    evidence_note, evidence_rows, soil_findings and, for deficiency classes,
    soil_evidence. The predicted class is only read, never changed.
    """
    if predicted_class not in SUPPORTED_CLASSES:
        return {
            "summary": f"Unrecognised prediction class '{predicted_class}'. No recommendation is available.",
            "check": [],
            "act": ["Consult a qualified agricultural extension adviser."],
            "monitor": [],
            "evidence_status": STATUS_INSUFFICIENT,
            "evidence_note": "Unknown class; no evidence comparison possible.",
            "evidence_rows": [],
            "soil_findings": [],
        }
    readings = interpret_soil(soil_values)
    comparison = compare_image_with_soil(predicted_class, readings)
    plan = _plan(predicted_class, readings, comparison)
    plan.update(
        evidence_status=comparison["status"],
        evidence_note=comparison["note"],
        evidence_rows=comparison["rows"],
        soil_findings=findings_from_readings(readings),
    )
    skip_key = _DEFICIENCY_SOIL_KEY.get(predicted_class)
    if skip_key:
        r = readings[skip_key]
        plan["soil_evidence"] = f"{r['label']}: {r['value_text'] if r['value'] is not None else r['value_text'].lower()}"
    return plan


def _withheld_plan(image_status: str, readings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Plan when there is no verified image prediction: no nutrient-specific advice."""
    check = [
        "Take a new photo of a single lettuce leaf: in focus, in even daylight, filling most of the frame, on a plain background.",
    ]
    check.extend(_other_flag_lines(readings, None))
    act = [
        "Do not apply any nutrient treatment based on this result. No leaf diagnosis was produced.",
        "If you have flagged soil readings, verify them first and follow local agricultural guidance.",
    ]
    monitor = [
        "Upload a clearer photo and run the analysis again.",
        "Watch the plant for spreading or worsening symptoms.",
        _ESCALATE_LINE,
    ]
    return {"summary": IMAGE_WITHHELD_MESSAGES.get(image_status, IMAGE_WITHHELD_MESSAGES["unknown"]),
            "check": check, "act": act, "monitor": monitor}


# ---------------------------------------------------------------------------
# 4. Complete deterministic report
# ---------------------------------------------------------------------------

def build_report(
    image_status: str,
    predicted_class: str | None,
    confidence_pct: float | None,
    soil_values: dict[str, Any] | None,
    verification_reason: str | None = None,
) -> dict[str, Any]:
    """
    Build the full deterministic report. The image section is copied exactly as
    supplied; soil readings can add findings but can never change it.

    A nutrient-specific plan and an image-vs-soil comparison are produced only when
    ``image_status == "prediction_generated"`` and the class is a supported label.
    """
    readings = interpret_soil(soil_values)
    summary = summarize_soil(readings)

    confidence_ok = isinstance(confidence_pct, Real) and not isinstance(confidence_pct, bool) \
        and math.isfinite(float(confidence_pct))
    verified = (image_status == IMAGE_OK and predicted_class in SUPPORTED_CLASSES and confidence_ok)

    if image_status == IMAGE_OK and not verified:
        withheld_message = "The image model returned a result this app cannot interpret, so no nutrient prediction is shown."
    else:
        withheld_message = IMAGE_WITHHELD_MESSAGES.get(image_status, IMAGE_WITHHELD_MESSAGES["unknown"])
    image = {
        "status": image_status if image_status in IMAGE_WITHHELD_MESSAGES or image_status == IMAGE_OK else "unknown",
        "verified": verified,
        "predicted_class": predicted_class if verified else None,
        "confidence_pct": round(float(confidence_pct), 1) if verified else None,
        "message": "" if verified else withheld_message,
        "verification_reason": verification_reason,
    }
    if verified:
        comparison = compare_image_with_soil(predicted_class, readings)
        plan = build_recommendation(predicted_class, soil_values)
        plan = {k: plan[k] for k in ("summary", "check", "act", "monitor")}
    else:
        comparison = None
        plan = _withheld_plan(image["status"], readings)
    return {
        "image": image,
        "soil": {"readings": [readings[k] for k in SOIL_KEYS], "summary": summary,
                 "findings": findings_from_readings(readings), "note": PROVISIONAL_NOTE},
        "comparison": comparison,
        "plan": plan,
    }


def report_to_text(report: dict[str, Any]) -> str:
    """Plain-text version of the deterministic report (used for tests and fallbacks)."""
    lines: list[str] = []
    img = report["image"]
    lines.append("LEAF IMAGE RESULT")
    if img["verified"]:
        lines.append(f"Image model prediction: {img['predicted_class']} (confidence {img['confidence_pct']:.1f}%)")
    else:
        lines.append(img["message"])
    lines.append("")
    lines.append("SOIL READINGS (checked separately from the image)")
    for r in report["soil"]["readings"]:
        lines.append(f"- {r['label']}: {r['value_text']} -> {r['result_text']}")
    lines.append(report["soil"]["note"])
    lines.append("")
    lines.append("DO THE RESULTS AGREE?")
    cmp_ = report["comparison"]
    if cmp_ is None:
        lines.append("Not available - the image was not verified.")
    else:
        lines.append(f"{cmp_['status_label']}: {cmp_['note']}")
        for row in cmp_["rows"]:
            lines.append(f"- {row['nutrient']}: image {row['image_says'].lower()}; soil {row['soil_value']}; {row['status_label']}")
    plan = report["plan"]
    for title, key in (("CHECK", "check"), ("DO NOW", "act"), ("MONITOR", "monitor")):
        lines.append("")
        lines.append(title)
        lines.extend(f"- {item}" for item in plan[key])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Ollama explanation (optional layer on top of the deterministic report)
# ---------------------------------------------------------------------------

def _build_ollama_prompt(
    predicted_class: str,
    confidence_pct: float | None,
    soil_values: dict[str, Any],
    recommendation: dict[str, Any],
) -> str:
    """Build an evidence-grounded prompt from the deterministic findings."""
    readings = interpret_soil(soil_values)
    comparison = compare_image_with_soil(predicted_class, readings)

    soil_lines = []
    for key in SOIL_KEYS:
        r = readings[key]
        if r["state"] == STATE_MISSING:
            soil_lines.append(f"- {r['label']}: NOT PROVIDED. Say it is not available. Do not guess a value.")
        elif r["state"] == STATE_INVALID:
            soil_lines.append(f"- {r['label']}: INVALID, not used. Say it could not be assessed.")
        else:
            soil_lines.append(f"- {r['label']}: {r['value_text']}. Result: {r['result_text']}. Rule: {r['rule']}.")

    evidence_lines = []
    for row in comparison["rows"]:
        evidence_lines.append(
            f"- {row['nutrient']}: image {row['image_says'].lower()}; soil {row['soil_value']}; "
            f"{row['status_label']}. {row['note']}"
        )
    confidence = f"{confidence_pct:.1f}%" if confidence_pct is not None else "not available"

    def bullets(key: str) -> str:
        return "\n".join(f"- {item}" for item in recommendation.get(key, []))

    return f"""You are a plain-English plant-health assistant for a college project. Explain the findings below to a farmer.
You only EXPLAIN findings that the application already worked out. You do not calculate or decide anything.

FACTS FROM THE APPLICATION (authoritative):
Image model prediction: {predicted_class}
Image model confidence: {confidence}
Overall image-versus-soil evidence: {comparison['status_label']}. {comparison['note']}

SOIL READINGS AND THEIR FIXED RESULTS (use these exactly):
{chr(10).join(soil_lines)}

PER-NUTRIENT COMPARISON:
{chr(10).join(evidence_lines)}

APPROVED GUIDANCE:
Summary: {recommendation.get('summary', '')}
Check:
{bullets('check')}
Do now:
{bullets('act')}
Monitor:
{bullets('monitor')}

Write the explanation using exactly these markdown headings, in this order:
## 1. Plant Health Status
## 2. Image Analysis
## 3. Soil Reading Assessment
## 4. Combined Evidence Assessment
## 5. Recommended Next Steps
## 6. What to Monitor
## 7. Important Precautions

STRICT RULES:
- Quote the image prediction ({predicted_class}) and its confidence ({confidence}) exactly as given. It is the image model's result, not a confirmed diagnosis.
- Discuss the image evidence and the soil evidence separately BEFORE explaining how they relate.
- Describe each soil reading exactly as its fixed result says. A reading that is "Not low" must never be called low. A reading that is not provided must be called not available.
- If the evidence is a CONFLICT, say plainly that the soil reading does not support the image prediction and that this does not prove the image model is wrong.
- Other flagged soil readings are separate soil concerns. They are NOT a new image prediction and must not change the predicted class.
- Use only the numbers shown above. Do not invent measurements, thresholds, percentages, confidence scores or diagnoses.
- Do not give fertiliser names, amounts, doses, rates or schedules, and do not promise results. Recommend only the approved guidance.
- Say that the soil ranges are provisional screening rules and that this tool cannot definitively diagnose a plant problem.
- Use plain English. Keep it under 300 words.
"""


# --- checking the model's answer against the deterministic findings ---------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUTRIENT_WORDS = {"soil_n": "nitrogen", "soil_p": "phosphorus", "soil_k": "potassium"}
_LOW_WORDS = re.compile(r"\b(low|lowest|deficien\w*|insufficient|inadequate|lacking|shortage|depleted|below)\b", re.I)
_NEGATION = re.compile(
    r"\b(not|no|never|neither|nor|without|adequate|sufficient|within|above|normal|rather than|instead|unflagged)\b|n't|at or above", re.I)
_IMAGE_CONTEXT = re.compile(r"\b(image|leaf|leaves|photo|photograph|model|predict\w*|visual\w*|symptom\w*|appear\w*)\b", re.I)
_SOIL_CONTEXT = re.compile(r"\b(soil|reading|measurement|measured|mg/kg|level|test)\b", re.I)
_OK_CLAIM = re.compile(r"(not\s+(?:flagged\s+)?(?:as\s+)?low|not below|adequate|sufficient|normal|at or above|healthy level)", re.I)
_DOSE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:kg|kilograms?|g|grams?|mg|ml|millilit(?:re|er)s?|litres?|liters?|lbs?|pounds?|oz|ounces?|"
    r"tsp|tbsp|teaspoons?|tablespoons?|cups?)\s*(?:/|per)\s*(?:ha|hectares?|acres?|plants?|m2|m\^2|sq\.?\s*m|gallons?|"
    r"litres?|liters?|bed|row)\b", re.I)
_GUARANTEE = re.compile(r"\b(guarantee[sd]?|will (?:definitely |certainly )?(?:cure|fix|restore|resolve|solve))\b", re.I)
_MGKG = re.compile(r"(\d+(?:\.\d+)?)\s*mg\s*/\s*kg", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_PH_NUM = re.compile(r"\bpH\s*(?:value\s*)?(?:of|is|was|at|=|:)?\s*(\d+(?:\.\d+)?)", re.I)


def _in_set(number: float, allowed: list[float]) -> bool:
    return any(abs(number - a) < 0.051 for a in allowed)


def _nutrient_word_for_class(cls: str) -> str:
    """Lower-case nutrient word for a deficiency class ('' for Healthy)."""
    return cls.split("_")[0].lower() if cls in _DEFICIENCY_SOIL_KEY else ""


def validate_explanation(
    text: str,
    predicted_class: str,
    confidence_pct: float | None,
    soil_values: dict[str, Any] | None,
) -> list[str]:
    """
    Return a list of problems found in a language-model explanation (empty list = passed).

    This is a safety net, not a proof of correctness: it catches invented numbers,
    fertiliser doses, guarantees, wrong predicted class, and sentences that describe a
    nutrient's soil reading in the opposite way to the deterministic result.
    """
    problems: list[str] = []
    readings = interpret_soil(soil_values)

    # --- invented numbers ---------------------------------------------------
    valid = {k: r["value"] for k, r in readings.items() if r["value"] is not None}
    allowed_mgkg = [40.0, 15.0, 140.0] + [valid[k] for k in ("soil_n", "soil_p", "soil_k") if k in valid]
    for m in _MGKG.finditer(text):
        if not _in_set(float(m.group(1)), allowed_mgkg):
            problems.append(f"invented soil value '{m.group(0)}'")
    allowed_pct = [0.0, 100.0, 15.0, 55.0]
    if confidence_pct is not None:
        allowed_pct += [round(confidence_pct, 1), round(confidence_pct)]
    if "soil_moisture" in valid:
        allowed_pct.append(valid["soil_moisture"])
    for m in _PERCENT.finditer(text):
        if not _in_set(float(m.group(1)), allowed_pct):
            problems.append(f"unexpected percentage '{m.group(0)}'")
    allowed_ph = [5.0, 7.5] + ([valid["soil_ph"]] if "soil_ph" in valid else [])
    for m in _PH_NUM.finditer(text):
        if not _in_set(float(m.group(1)), allowed_ph):
            problems.append(f"unexpected pH value '{m.group(1)}'")

    # --- doses and guarantees -------------------------------------------------
    if _DOSE.search(text):
        problems.append("contains a fertiliser dose or application rate")
    for m in _GUARANTEE.finditer(text):
        before = text[max(0, m.start() - 12):m.start()].lower()
        if not re.search(r"(not|n't|cannot|can't|no)\s*$", before):
            problems.append(f"promises an outcome ('{m.group(0)}')")

    # --- sentence-level checks ------------------------------------------------
    other_classes = {c: c.split("_")[0].lower() for c in SUPPORTED_CLASSES if c != predicted_class and c != "Healthy"}
    for sentence in (s.strip() for s in _SENT_SPLIT.split(text) if s.strip()):
        low_s = sentence.lower()
        mentioned = [k for k, w in _NUTRIENT_WORDS.items() if w in low_s]
        is_image_sentence = bool(_IMAGE_CONTEXT.search(sentence))

        # wrong predicted class stated as the model's result
        if re.search(r"\b(predict\w*|classif\w*|suggests?)\b", low_s) and not _NEGATION.search(sentence):
            for cls, word in other_classes.items():
                if word in low_s and "deficien" in low_s and _nutrient_word_for_class(predicted_class) not in low_s:
                    problems.append(f"names a different image prediction ({cls})")

        if len(mentioned) != 1 or not _SOIL_CONTEXT.search(sentence):
            continue
        key = mentioned[0]
        state = readings[key]["state"]
        if state != STATE_LOW and not is_image_sentence and _LOW_WORDS.search(sentence) and not _NEGATION.search(sentence):
            problems.append(f"calls {SOIL_NAMES[key]} low but the soil rule does not flag it ({state})")
        if state == STATE_LOW and not is_image_sentence and _OK_CLAIM.search(sentence):
            problems.append(f"calls {SOIL_NAMES[key]} adequate but the soil rule flags it as low")
    return problems


def call_ollama(
    predicted_class: str,
    confidence_pct: float | None,
    soil_values: dict[str, Any],
    recommendation: dict[str, Any],
) -> tuple[str | None, str | None]:
    """
    Ask the local Ollama model to explain the deterministic findings.

    Returns (explanation_text, None) on success, or (None, message) on any failure.
    The message is short and safe to show a farmer; technical detail goes to the log.
    The caller must keep showing the deterministic report either way.
    """
    if predicted_class not in SUPPORTED_CLASSES:
        return None, "No verified image prediction is available to explain. Showing fixed recommendations."

    prompt = _build_ollama_prompt(predicted_class, confidence_pct, soil_values, recommendation)
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 700},
    }).encode("utf-8")
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_S) as resp:
            body = resp.read()
    except (TimeoutError, socket.timeout) as exc:
        logger.warning("Ollama timed out after %s s: %r", OLLAMA_TIMEOUT_S, exc)
        return None, "The AI explanation took too long to respond. Showing fixed recommendations."
    except urllib.error.HTTPError as exc:
        logger.warning("Ollama HTTP error %s for model %s", exc.code, OLLAMA_MODEL)
        return None, "The AI explanation service returned an error (is the model installed?). Showing fixed recommendations."
    except urllib.error.URLError as exc:
        logger.warning("Ollama not reachable at %s: %r", OLLAMA_BASE_URL, exc.reason)
        return None, "The AI explanation is unavailable because Ollama is not running. Showing fixed recommendations."
    except Exception as exc:  # noqa: BLE001 - never let the website crash
        logger.warning("Unexpected error contacting Ollama: %r", exc)
        return None, "The AI explanation could not be generated. Showing fixed recommendations."

    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("response is not a JSON object")
        if data.get("error"):
            logger.warning("Ollama returned an error field: %r", data.get("error"))
            return None, "The AI explanation service reported an error. Showing fixed recommendations."
        text = data.get("response")
        if not isinstance(text, str):
            raise ValueError("missing 'response' text")
        text = text.strip()
    except (ValueError, UnicodeDecodeError) as exc:  # JSONDecodeError is a ValueError
        logger.warning("Malformed Ollama response: %r", exc)
        return None, "The AI explanation came back unreadable. Showing fixed recommendations."

    if not text:
        logger.warning("Ollama returned an empty response")
        return None, "The AI explanation came back empty. Showing fixed recommendations."

    problems = validate_explanation(text, predicted_class, confidence_pct, soil_values)
    if problems:
        logger.warning("Ollama explanation rejected by safety check: %s", "; ".join(problems))
        return None, "The AI explanation did not pass the accuracy check, so only the fixed report is shown. Showing fixed recommendations."

    return text, None
