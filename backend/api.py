"""FastAPI bridge for the LesionIQ frontend.

The frontend intentionally speaks to one narrow endpoint:
POST /cases/analyze with multipart image + metadata JSON.  This module
adapts the live inference bundle into the CaseRecord TypeScript shape.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

ARTIFACT_ROOT = Path(
    os.getenv("LESIONIQ_ARTIFACT_ROOT", BACKEND_ROOT / "output" / "inference")
).resolve()
UPLOAD_ROOT = ARTIFACT_ROOT / "_uploads"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

FRONTEND_CLASS_ORDER = ["MEL", "NV", "BCC", "BKL", "AK", "SCC", "VASC", "DF"]
CLASS_FULL = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma",
}
THRESHOLDS = {
    "MEL": 0.57,
    "NV": 0.48,
    "BCC": 0.42,
    "BKL": 0.39,
    "AK": 0.31,
    "SCC": 0.35,
    "VASC": 0.22,
    "DF": 0.24,
}
PREPROCESSING_STEPS = [
    {
        "id": "raw",
        "title": "Raw dermoscopy input",
        "shortTitle": "Raw",
        "description": "Original uploaded dermoscopy frame retained for audit comparison.",
        "technicalNote": "Saved as raw.png before any cleanup.",
        "status": "Complete",
        "previewTone": "raw",
    },
    {
        "id": "dullrazor",
        "title": "DullRazor hair removal",
        "shortTitle": "DullRazor",
        "description": "Linear hair-like artifacts are detected and inpainted before normalization.",
        "technicalNote": "Saved as 01_dullrazor.png.",
        "status": "Complete",
        "previewTone": "hair",
    },
    {
        "id": "shadesofgrey",
        "title": "Shades-of-Gray color normalization",
        "shortTitle": "Shades-of-Gray",
        "description": "Device color cast is normalized to stabilize image branch features.",
        "technicalNote": "Saved as 02_shades_of_grey.png.",
        "status": "Complete",
        "previewTone": "normalized",
    },
    {
        "id": "clahe",
        "title": "LAB CLAHE contrast enhancement",
        "shortTitle": "CLAHE",
        "description": "Local lesion contrast is enhanced in LAB luminance space.",
        "technicalNote": "Saved as 03_clahe.png.",
        "status": "Complete",
        "previewTone": "normalized",
    },
    {
        "id": "borderremoved",
        "title": "Circular border removal and resize",
        "shortTitle": "Border removed",
        "description": "Dermoscope vignette borders are cropped when present, then resized to 384x384.",
        "technicalNote": "Saved as 04_border_removed.png and final_preprocessed.png for model input.",
        "status": "Complete",
        "previewTone": "border",
    },
]


_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Allow custom frontend domain set via env (support both DERMALENS and legacy LESIONIQ prefix)
_frontend_url = os.getenv("DERMALENS_FRONTEND_URL", os.getenv("LESIONIQ_FRONTEND_URL", ""))
if _frontend_url:
    _CORS_ORIGINS.append(_frontend_url.rstrip("/"))

app = FastAPI(title="DermaLens API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.ngrok-free\.(app|dev)|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACT_ROOT)), name="artifacts")


def _frontend_mode(mode: str | None) -> str:
    allowed = {"Full Hybrid", "Image Only", "EffNet Only", "Swin Only"}
    return mode if mode in allowed else "Full Hybrid"


def _backend_mode(mode: str | None) -> str:
    return {
        "Full Hybrid": "full",
        "Image Only": "image_only",
        "EffNet Only": "effnet_only",
        "Swin Only": "swin_only",
    }.get(_frontend_mode(mode), "full")


def _sex(value: Any) -> str:
    label = str(value or "Unknown").strip().lower()
    if label == "female":
        return "Female"
    if label == "male":
        return "Male"
    return "Unknown"


def _site(value: Any) -> str:
    label = str(value or "").strip()
    return label or "Unknown"


def _artifact_url(request: Request, bundle_id: str, name: str | None) -> str:
    if not name:
        return ""
    basename = Path(name).name
    return f"/artifacts/{bundle_id}/{basename}"


def _probability(diagnosis: dict[str, Any], class_code: str) -> float:
    return float(diagnosis.get("probabilities", {}).get(class_code, 0.0))


def _prediction_scores(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for class_code in FRONTEND_CLASS_ORDER:
        probability = _probability(diagnosis, class_code)
        threshold = THRESHOLDS[class_code]
        rows.append(
            {
                "classCode": class_code,
                "classLabel": CLASS_FULL[class_code],
                "probability": probability,
                "threshold": threshold,
                "thresholdMargin": probability - threshold,
            }
        )
    return sorted(rows, key=lambda row: row["probability"], reverse=True)


def _urgency(class_code: str, diagnosis: dict[str, Any]) -> str:
    flags = diagnosis.get("clinical_flags", {})
    if class_code in {"MEL", "SCC"} or flags.get("requires_biopsy"):
        return "High concern"
    if class_code in {"BCC", "AK"}:
        return "Expedited review"
    return "Routine"


def _recommendation(class_code: str, confidence: float, diagnosis: dict[str, Any]) -> str:
    flags = diagnosis.get("clinical_flags", {})
    if flags.get("requires_biopsy"):
        return (
            f"Prioritize dermatologist review for {CLASS_FULL[class_code]}; "
            "biopsy consideration requires clinician confirmation."
        )
    if flags.get("low_confidence") or flags.get("differential_diagnosis"):
        return "Review the differential diagnosis and image evidence before final disposition."
    return (
        f"Routine dermatologist review; model confidence is {round(confidence * 100)}% "
        "with calibrated threshold support."
    )


def _metadata_signals(shap_values: dict[str, float] | None) -> list[dict[str, Any]]:
    if not shap_values:
        return [
            {
                "id": "metadata-unavailable",
                "label": "Metadata attribution",
                "value": 0,
                "direction": "weakens",
                "note": "Metadata attribution was not generated for this mode.",
            }
        ]

    rows = sorted(
        shap_values.items(), key=lambda item: abs(float(item[1])), reverse=True
    )[:6]
    return [
        {
            "id": key,
            "label": key.replace("_", " ").replace("site ", "Site: ").title(),
            "value": float(value),
            "direction": "supports" if float(value) >= 0 else "weakens",
            "note": "Perturbation-based metadata contribution for the predicted class.",
        }
        for key, value in rows
    ]


def _read_image_base64(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return base64.b64encode(candidate.read_bytes()).decode("ascii")


def _build_slm_prompt(diagnosis: dict[str, Any], metadata: dict[str, Any]) -> str:
    predicted = diagnosis.get("prediction", {})
    class_code = str(predicted.get("class", ""))
    probabilities = diagnosis.get("probabilities", {})
    flags = diagnosis.get("clinical_flags", {})
    explainability = diagnosis.get("explainability", {})
    artifact_names = diagnosis.get("artifacts", {})
    evidence_packet = {
        "patient_metadata": metadata,
        "prediction": predicted,
        "probabilities": probabilities,
        "threshold": THRESHOLDS.get(class_code, 0.5),
        "threshold_context": "tuned threshold when available; default comparator is 0.50",
        "clinical_flags": flags,
        "gradcam": {
            "artifact": explainability.get("gradcam") or artifact_names.get("gradcam"),
            "description": explainability.get("gradcam_description"),
            "region": explainability.get("gradcam_region"),
            "area_coverage": explainability.get("gradcam_area_coverage"),
            "peak_activation_pixel": explainability.get("gradcam_peak_activation_pixel"),
        },
        "attention": {
            "artifact": explainability.get("attention") or artifact_names.get("attention"),
            "description": explainability.get("attention_description"),
            "region": explainability.get("attention_region"),
            "peak_attention_patch": explainability.get("attention_peak_patch"),
            "attention_weights": explainability.get("attention_weights"),
        },
        "metadata_attribution": explainability.get("shap_metadata"),
        "artifact_names": artifact_names,
    }
    return (
        "You are a clinical dermatology AI assistant embedded in LesionIQ, a decision-support "
        "tool for skin lesion classification. Sound like a concise senior clinical reviewer "
        "giving preliminary analysis to a physician.\n\n"
        "You will receive structured JSON plus the final preprocessed image, Grad-CAM++ overlay, "
        "and SwinV2 attention overlay. Explain the evidence behind the prediction rather than "
        "restating the prediction.\n\n"
        "Safety and evidence rules:\n"
        "- Do not make an independent diagnosis beyond the supplied prediction.\n"
        "- Do not recommend treatment, biopsy, discharge, or reassurance as a directive.\n"
        "- Use dermoscopic language only when supported by supplied text, heatmap/attention "
        "evidence, or structured metadata.\n"
        "- The explanation must feel like preliminary clinical analysis, not a generic model summary.\n"
        "- Include area coverage and peak activation pixel only if they are present in the JSON.\n"
        "- If Grad-CAM++ and SwinV2 attention appear to focus on different regions, explicitly say "
        "the evidence is not fully aligned and that this reduces certainty. If they agree, say "
        "the maps reinforce the same diagnostic region.\n"
        "- Use age, sex, and anatomical site only to contextualize the hybrid model output; do not "
        "overstate metadata as diagnostic evidence.\n"
        "- Keep the wording concise, physician-facing, and clinically cautious. The EVIDENCE block "
        "should usually be five sentences.\n\n"
        "Return exactly this report structure and headings:\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "LESIONIQ CLINICAL EXPLAINABILITY REPORT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "PREDICTION\n"
        "  Diagnosis   : [Top class label] ([class code])\n"
        "  Confidence  : [confidence]% ([high / moderate / low confidence])\n"
        "  Threshold   : [threshold] (tuned — default 0.50)\n\n"
        "EVIDENCE\n"
        "  [Sentence 1: state the top prediction and why the model leaned toward it using visible evidence.]\n"
        "  [Sentence 2: describe the Grad-CAM++ region in clinical terms and link it to a plausible "
        "dermoscopic feature only if supported.]\n"
        "  [Sentence 3: describe the SwinV2 attention map and explain whether it reinforces the same "
        "region or shifts attention elsewhere.]\n"
        "  [Sentence 4: if Grad-CAM++ and attention differ, explicitly say the evidence is not fully "
        "aligned and that this reduces certainty; otherwise state that the maps reinforce each other.]\n"
        "  [Sentence 5: use age, sex, and site only if they contextualize the hybrid model output; "
        "end with a concise clinician-verification note.]\n\n"
        "Evidence packet JSON:\n"
        f"{json.dumps(evidence_packet, sort_keys=True)}"
    )


async def _generate_slm_summary(
    diagnosis: dict[str, Any],
    artifact_paths: dict[str, str],
    metadata: dict[str, Any],
) -> tuple[str, str]:
    from backend.classifier.inference import (
        build_slm_payload,
        build_slm_prompt,
        validate_and_repair_slm_output,
        _fallback_slm_report,
    )

    model = os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    explainability = diagnosis.get("explainability", {})
    slm_payload = diagnosis.get("slm_payload") or build_slm_payload(
        diagnosis,
        gradcam_summary=explainability.get("gradcam_summary"),
        attention_summary=explainability.get("attention_summary"),
        metadata=metadata,
    )
    images = [
        encoded
        for encoded in [
            _read_image_base64(artifact_paths.get("original")),
            _read_image_base64(artifact_paths.get("gradcam")),
            _read_image_base64(artifact_paths.get("attention")),
        ]
        if encoded
    ]
    prompt = build_slm_prompt(slm_payload)
    try:
        # connect_timeout is short so an offline Ollama fails fast and hits
        # the fallback below before the proxy or UI times out.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=150.0, write=10.0, pool=5.0)
        ) as client:
            message_content = [{"type": "text", "text": prompt}]
            for img_b64 in images:
                message_content.insert(0, {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            response = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt, "images": images},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 1024},
                },
            )
            response.raise_for_status()
            payload = response.json()
            text = str(
                payload.get("message", {}).get("content", "")
            ).strip()
            if text:
                return validate_and_repair_slm_output(text, slm_payload), "Generated (SLM)"
    except Exception as exc:
        fallback = _fallback_slm_report(slm_payload)
        return (
            fallback,
            f"Generated (Clinical Fallback - SLM offline: {type(exc).__name__})",
        )
    fallback = _fallback_slm_report(slm_payload)
    return (
        fallback,
        "Generated (Clinical Fallback - Empty SLM response)",
    )


def _audit_checks(metadata_complete: bool, slm_status: str) -> list[dict[str, str]]:
    return [
        {
            "id": "preprocess",
            "label": "Preprocessing trace",
            "status": "pass",
            "note": "Raw, DullRazor, normalization, CLAHE, and final model input artifacts were generated.",
        },
        {
            "id": "metadata",
            "label": "Hybrid metadata",
            "status": "pass" if metadata_complete else "warning",
            "note": "Age, sex, and anatomical site are required for full hybrid inference.",
        },
        {
            "id": "explainability",
            "label": "Visual explanations",
            "status": "pass",
            "note": "Grad-CAM and attention artifacts are included when the active image branches support them.",
        },
        {
            "id": "slm",
            "label": "Clinical narrative generator",
            "status": "pass" if slm_status.startswith("Generated") else "review-needed",
            "note": "Clinical explainability report produced with decision support and evidence breakdown.",
        },
    ]


def _heatmap_summary_text(
    diagnosis: dict[str, Any], key: str, branch_name: str
) -> str:
    """Build a human-readable spatial summary from heatmap analytics."""
    expl = diagnosis.get("explainability", {})
    summary = expl.get(f"{key}_summary") or {}
    if not isinstance(summary, dict) or not summary.get("available"):
        return f"{branch_name} was not generated for this inference mode."
    region = summary.get("region", "lesion region")
    area = summary.get("area_pct")
    peak = summary.get("peak_pixel")
    desc = f"{branch_name} concentrates on the {region}"
    if area is not None:
        desc += f", activating {area}% of the image area"
    if peak is not None:
        desc += f" (peak at pixel {peak[0]}, {peak[1]})"
    desc += "."
    return desc


def _gradcam_summary_text(diagnosis: dict[str, Any]) -> str:
    return _heatmap_summary_text(diagnosis, "gradcam", "Grad-CAM++")


def _attention_summary_text(diagnosis: dict[str, Any]) -> str:
    return _heatmap_summary_text(diagnosis, "attention", "SwinV2 attention")


def _case_record(
    request: Request,
    bundle_id: str,
    source_name: str,
    metadata: dict[str, Any],
    result: dict[str, Any],
    slm_summary: str,
    slm_status: str,
) -> dict[str, Any]:
    diagnosis = result["diagnosis"]
    artifact_paths = result["artifact_paths"]
    artifacts = diagnosis.get("artifacts", {})
    prediction = diagnosis["prediction"]
    class_code = prediction["class"]
    confidence = float(prediction["confidence"])
    threshold = THRESHOLDS[class_code]
    scores = _prediction_scores(diagnosis)
    now = time.strftime("%Y-%m-%d")
    acquired = time.strftime("%Y-%m-%d %H:%M:%S")
    urgency = _urgency(class_code, diagnosis)
    metadata_complete = all(
        [
            metadata.get("ageYears") is not None,
            _sex(metadata.get("sex")) != "Unknown",
            _site(metadata.get("anatomicalSite")) != "Unknown",
        ]
    )
    case_id = bundle_id

    inference_bundle = {
        "sourceImageName": source_name,
        "outputDirectory": f"/artifacts/{bundle_id}/",
        "rawArtifact": _artifact_url(request, bundle_id, artifacts.get("raw")),
        "dullRazorArtifact": _artifact_url(request, bundle_id, artifacts.get("dullrazor")),
        "normalizedArtifact": _artifact_url(request, bundle_id, artifacts.get("shadesofgrey")),
        "claheArtifact": _artifact_url(request, bundle_id, artifacts.get("clahe")),
        "finalPreprocessedArtifact": _artifact_url(request, bundle_id, artifacts.get("final_preprocessed")),
        "originalArtifact": _artifact_url(request, bundle_id, artifacts.get("original")),
        "gradcamArtifact": _artifact_url(request, bundle_id, artifacts.get("gradcam")),
        "attentionArtifact": _artifact_url(request, bundle_id, artifacts.get("attention")),
        "diagnosisArtifact": _artifact_url(request, bundle_id, artifacts.get("diagnosis")),
        "slmContainer": os.getenv("OLLAMA_CONTAINER", "lesioniq_ollama"),
        "slmModel": os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat"),
        "slmStatus": slm_status,
    }

    return {
        "id": bundle_id,
        "caseId": case_id,
        "maskedPatientId": "PT-LIVE-DEID",
        "visitDate": now,
        "acquisitionTimestamp": acquired,
        "reviewStatus": "Needs review",
        "modelMode": _frontend_mode(metadata.get("modelMode")),
        "decisionSupportBanner": "Decision support only; dermatologist review required before clinical action.",
        "predictedClassCode": class_code,
        "predictedClassLabel": CLASS_FULL[class_code],
        "calibratedConfidence": confidence,
        "thresholdMargin": confidence - threshold,
        "recommendation": _recommendation(class_code, confidence, diagnosis),
        "urgency": urgency,
        "uploadedImageUrl": inference_bundle["rawArtifact"],
        "metadata": {
            "ageYears": int(metadata.get("ageYears") or 0),
            "sex": _sex(metadata.get("sex")),
            "anatomicalSite": _site(metadata.get("anatomicalSite")),
            "FitzpatrickType": "Unknown",
            "metadataCompleteness": "Complete" if metadata_complete else "Partial",
        },
        "lesionMetrics": [
            {"label": "Uploaded file", "value": source_name},
            {"label": "Model mode", "value": _frontend_mode(metadata.get("modelMode"))},
            {"label": "Artifact bundle", "value": "raw, preprocessing, Grad-CAM, attention, diagnosis JSON"},
            {
                "label": "Malignant probability",
                "value": f"{round(diagnosis['clinical_flags']['malignant_total_prob'] * 100)}%",
                "note": "Sum of MEL, BCC, AK, and SCC probabilities.",
            },
        ],
        "predictionScores": scores,
        "explainability": {
            "gradcamSummary": _gradcam_summary_text(diagnosis),
            "attentionSummary": _attention_summary_text(diagnosis),
            "metadataSignals": _metadata_signals(
                diagnosis.get("explainability", {}).get("shap_metadata")
            ),
            "slmSummary": slm_summary,
            "auditChecks": _audit_checks(metadata_complete, slm_status),
            "auditNotes": [
                "Image uploaded and persisted through the backend artifact store.",
                "Preprocessing layers were generated before inference.",
                "Final preprocessed image was fed into the model inference layer.",
                "Local SLM handoff used diagnosis JSON, metadata, Grad-CAM, and attention artifacts.",
            ],
        },
        "preprocessingSteps": PREPROCESSING_STEPS,
        "historyEntries": [
            {
                "id": f"{bundle_id}-current",
                "date": now,
                "predictedClassCode": class_code,
                "predictedClassLabel": CLASS_FULL[class_code],
                "confidence": confidence,
                "status": "Needs review",
                "urgency": urgency,
                "note": "Live backend inference result from uploaded dermoscopy image.",
                "samePatient": True,
            }
        ],
        "compareEntries": [
            {
                "id": f"{bundle_id}-raw",
                "date": now,
                "label": "Raw input",
                "predictedClassCode": class_code,
                "predictedClassLabel": CLASS_FULL[class_code],
                "calibratedConfidence": confidence,
                "reviewStatus": "Needs review",
                "summary": "Raw image retained for preprocessing comparison and audit.",
            },
            {
                "id": f"{bundle_id}-inference",
                "date": now,
                "label": "Final inference bundle",
                "predictedClassCode": class_code,
                "predictedClassLabel": CLASS_FULL[class_code],
                "calibratedConfidence": confidence,
                "reviewStatus": "Needs review",
                "summary": "Final preprocessed image, Grad-CAM, attention, and diagnosis JSON are available.",
            },
        ],
        "clinicianNotesPreview": "Live analysis generated; clinician verification remains required.",
        "inferenceBundle": inference_bundle,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "artifactRoot": str(ARTIFACT_ROOT),
        "ollamaModel": os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat"),
    }


@app.post("/cases/analyze")
async def analyze_case(
    request: Request,
    image: UploadFile = File(...),
    metadata: str = Form(...),
) -> JSONResponse:
    try:
        metadata_payload = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc

    if metadata_payload.get("ageYears") is None:
        raise HTTPException(status_code=422, detail="ageYears is required")
    if _sex(metadata_payload.get("sex")) == "Unknown":
        raise HTTPException(status_code=422, detail="sex is required")
    if _site(metadata_payload.get("anatomicalSite")) == "Unknown":
        raise HTTPException(status_code=422, detail="anatomicalSite is required")

    suffix = Path(image.filename or "lesion.png").suffix.lower() or ".png"
    bundle_id = f"LIQ-{int(time.time() * 1000)}"
    saved_path = UPLOAD_ROOT / f"{bundle_id}{suffix}"
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(await image.read())

    try:
        from backend.classifier.inference import run_inference_pipeline

        result = run_inference_pipeline(
            image_path=str(saved_path),
            age=metadata_payload.get("ageYears"),
            sex=metadata_payload.get("sex"),
            site=metadata_payload.get("anatomicalSite"),
            mode=_backend_mode(metadata_payload.get("modelMode")),
            output_dir=str(ARTIFACT_ROOT),
        )
        slm_summary, slm_status = await _generate_slm_summary(
            result["diagnosis"], result["artifact_paths"], metadata_payload
        )
        record = _case_record(
            request=request,
            bundle_id=bundle_id,
            source_name=image.filename or saved_path.name,
            metadata=metadata_payload,
            result=result,
            slm_summary=slm_summary,
            slm_status=slm_status,
        )
        return JSONResponse(record)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
