
"""Professional browser GUI for the existing P300 EEG authentication pipeline.

Run from the repository root: uvicorn gui_app:app --reload
This app loads the existing early-stopping CNN checkpoints and thresholds. It does
not train a new model or silently substitute the older PCA/SVM pipeline.
"""

from __future__ import annotations

import json, time, threading
import paho.mqtt.client as mqtt

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models.cnn_authentication_early_stopping import (
    ROOT, SUBJECTS, DEVICE, EEG1DCNN, load_recording, standardize_windows
)

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "gui_static"
MODEL_RESULTS = ROOT / "results" / "cnn_authentication_early_stopping"
MODEL_DIR = MODEL_RESULTS / "saved_models"
RESULTS_JSON = MODEL_RESULTS / "cnn_authentication_results.json"
GUI_RESULTS = ROOT / "results" / "gui"
ENROLMENT_FILE = GUI_RESULTS / "enrolled_templates.json"

CHANNELS = [
    "Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8",
    "F3", "F4", "FCz", "C3", "C4", "CP3", "CPz", "CP4"
]

WINDOWS_PER_SUBJECT = 153

# MQTT Smart Lock Configuration
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "aryan/p300eeg/smartlock/control"
MQTT_TIMEOUT = 3

app = FastAPI(title="P300 Secure Access", version="1.0.0", docs_url="/api/docs")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

_cache = {}
_lock = threading.RLock()


# ============================================================
# MQTT SMART LOCK
# ============================================================

def publish_mqtt_command(command: str):
    """Publish an authentication decision to the simulated IoT smart lock."""

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )

    connected = False
    loop_started = False

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, MQTT_TIMEOUT)
        connected = True

        client.loop_start()
        loop_started = True

        message = client.publish(
            MQTT_TOPIC,
            command,
            qos=1
        )

        message.wait_for_publish(timeout=MQTT_TIMEOUT)
        success = message.is_published()

        if success:
            print(f"[MQTT] Published {command} to {MQTT_TOPIC}")
            return {
                "success": True,
                "command": command,
                "broker": MQTT_BROKER,
                "topic": MQTT_TOPIC
            }

        print(f"[MQTT] Failed to publish {command}")

        return {
            "success": False,
            "command": command,
            "broker": MQTT_BROKER,
            "topic": MQTT_TOPIC,
            "error": "Message was not confirmed as published"
        }

    except Exception as e:
        print(f"[MQTT] Error publishing {command}: {e}")

        return {
            "success": False,
            "command": command,
            "broker": MQTT_BROKER,
            "topic": MQTT_TOPIC,
            "error": str(e)
        }

    finally:
        if connected:
            try:
                client.disconnect()
            except Exception:
                pass

        if loop_started:
            try:
                client.loop_stop()
            except Exception:
                pass


# ============================================================
# REQUEST MODELS
# ============================================================

class EnrolRequest(BaseModel):
    subject: str
    name: str = Field(default="", max_length=80)
    recording: int = Field(default=1, ge=1, le=1)
    start_window: int = Field(default=0, ge=0, le=152)
    window_count: int = Field(default=10, ge=1, le=153)


class AuthRequest(BaseModel):
    subject: str
    source_subject: str
    recording: int = Field(default=3, ge=2, le=3)
    window_index: int = Field(default=32, ge=0, le=152)


class ExplainRequest(BaseModel):
    subject: str
    source_subject: str
    recording: int = Field(default=3, ge=1, le=3)
    window_index: int = Field(default=32, ge=0, le=152)


class StreamRequest(BaseModel):
    subject: str
    source_subject: str
    recording: int = Field(default=3, ge=2, le=3)
    start_window: int = Field(default=0, ge=0, le=152)
    count: int = Field(default=12, ge=1, le=60)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    with _lock:
        if "data" in _cache:
            return _cache["data"]

        try:
            X1, y1 = load_recording(1)
            X2, y2 = load_recording(2)
            X3, y3 = load_recording(3)

            X1, X2, X3, mean, std = standardize_windows(X1, X2, X3)

            _cache["data"] = {
                1: (X1, y1),
                2: (X2, y2),
                3: (X3, y3),
                "normalization": "Recording 1 channel-wise mean/std"
            }

            return _cache["data"]

        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Dataset is not ready: {e}"
            )


def load_thresholds():
    if not RESULTS_JSON.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Missing saved threshold file: {RESULTS_JSON}"
        )

    with RESULTS_JSON.open(encoding="utf-8") as f:
        obj = json.load(f)

    thresholds = obj.get("validation_thresholds_by_subject", {})

    if not thresholds:
        raise HTTPException(
            status_code=503,
            detail="No validation thresholds found in CNN results JSON."
        )

    return obj, thresholds


def load_model(subject):
    if subject not in SUBJECTS:
        raise HTTPException(
            status_code=422,
            detail="Choose a valid subject A01–A10."
        )

    path = MODEL_DIR / f"{subject}_cnn_state_dict.pt"

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Missing checkpoint: {path}"
        )

    with _lock:
        key = f"model:{subject}"

        if key not in _cache:
            model = EEG1DCNN().to(DEVICE)

            state = torch.load(
                path,
                map_location=DEVICE,
                weights_only=True
            )

            model.load_state_dict(state)
            model.eval()

            _cache[key] = model

        return _cache[key]


# ============================================================
# CNN AUTHENTICATION
# ============================================================

def global_index(subject, local_window):
    return SUBJECTS.index(subject) * WINDOWS_PER_SUBJECT + local_window


def score_window(auth_subject, source_subject, recording, local_window):
    data = load_data()

    X, y = data[recording]

    idx = global_index(source_subject, local_window)

    if idx >= len(X):
        raise HTTPException(
            status_code=422,
            detail="Window index is outside the available data."
        )

    x = X[idx:idx+1]

    model = load_model(auth_subject)

    with torch.no_grad():
        score = float(
            model(
                torch.as_tensor(
                    x,
                    dtype=torch.float32,
                    device=DEVICE
                )
            ).cpu().item()
        )

    _, thresholds = load_thresholds()

    threshold = float(thresholds[auth_subject])

    decision = "ACCEPT" if score >= threshold else "REJECT"

    genuine = auth_subject == source_subject

    return {
        "subject": auth_subject,
        "source_subject": source_subject,
        "recording": recording,
        "window_index": local_window,
        "global_window_index": idx,
        "score_logit": score,
        "threshold": threshold,
        "decision": decision,
        "case": "genuine sample" if genuine else "impostor sample",
        "expected_outcome": "ACCEPT" if genuine else "REJECT",
        "outcome_match": decision == ("ACCEPT" if genuine else "REJECT"),
        "dataset_label_is_subject": bool(
            int(y[idx]) == SUBJECTS.index(auth_subject)
        ),
        "warning": (
            "Offline public-dataset demonstration; "
            "a decision is not proof of real-world identity."
        )
    }


# ============================================================
# ENROLMENT STORAGE
# ============================================================

def enrolments():
    GUI_RESULTS.mkdir(parents=True, exist_ok=True)

    if not ENROLMENT_FILE.exists():
        return {}

    try:
        return json.loads(
            ENROLMENT_FILE.read_text(encoding="utf-8")
        )

    except Exception:
        return {}


# ============================================================
# GUI ROUTES
# ============================================================

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    models = {
        s: (MODEL_DIR / f"{s}_cnn_state_dict.pt").exists()
        for s in SUBJECTS
    }

    dataset_root = ROOT / "data" / "preprocessed_sessions"

    data_count = sum(
        1
        for s in SUBJECTS
        for r in (1, 2, 3)
        if (
            dataset_root / s / f"recording_{r}_epochs.npy"
        ).exists()
    )

    thresholds_ok = RESULTS_JSON.exists()

    return {
        "project": "P300 EEG Authentication",
        "subjects": SUBJECTS,
        "model_checkpoints": models,
        "models_ready": all(models.values()),
        "dataset_files_found": data_count,
        "dataset_files_expected": 30,
        "dataset_ready": data_count == 30,
        "thresholds_ready": thresholds_ok,
        "ready": all(models.values()) and data_count == 30 and thresholds_ok,
        "enrolled": enrolments(),
        "device": str(DEVICE),
        "mode": "Recorded-data simulation",
        "model": (
            "Early-stopping 1D CNN + "
            "subject-specific frozen threshold"
        )
    }


@app.get("/api/overview")
def overview():
    obj, thresholds = load_thresholds()

    return {
        "results": obj,
        "thresholds": thresholds,
        "enrolled": enrolments(),
        "channels": CHANNELS,
        "global_channel_importance": _global_shap()
    }


def _global_shap():
    p = ROOT / "results" / "shap_explainability" / "channel_importance.csv"

    if not p.exists():
        return []

    import csv

    try:
        with p.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    except Exception:
        return []


# ============================================================
# ENROLMENT ENDPOINT
# ============================================================

@app.post("/api/enrol")
def enrol(req: EnrolRequest):
    if req.subject not in SUBJECTS:
        raise HTTPException(
            422,
            "Choose a valid subject A01–A10."
        )

    if req.start_window + req.window_count > WINDOWS_PER_SUBJECT:
        raise HTTPException(
            422,
            "The selected enrolment range exceeds the 153 available windows."
        )

    data = load_data()

    X, y = data[1]

    indices = [
        global_index(req.subject, i)
        for i in range(
            req.start_window,
            req.start_window + req.window_count
        )
    ]

    if any(i >= len(X) for i in indices):
        raise HTTPException(
            422,
            "Selected windows are not available."
        )

    # Explicitly store a reference cohort, not a newly trained biometric template.
    record = {
        "subject": req.subject,
        "display_name": req.name.strip() or req.subject,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recording": 1,
        "window_indices": list(
            range(
                req.start_window,
                req.start_window + req.window_count
            )
        ),
        "window_count": req.window_count,
        "checkpoint": (
            f"results/cnn_authentication_early_stopping/"
            f"saved_models/{req.subject}_cnn_state_dict.pt"
        ),
        "threshold_source": (
            "Recording 2 calibration; frozen from existing project results"
        ),
        "template_type": (
            "Reference cohort metadata; authentication uses the "
            "existing pretrained subject-specific CNN"
        ),
        "note": (
            "Enrolment does not retrain the CNN or modify its threshold."
        )
    }

    current = enrolments()
    current[req.subject] = record

    GUI_RESULTS.mkdir(parents=True, exist_ok=True)

    ENROLMENT_FILE.write_text(
        json.dumps(current, indent=2),
        encoding="utf-8"
    )

    return {
        "ok": True,
        "enrolment": record,
        "message": (
            f"Reference cohort saved for {req.subject}. "
            "Existing CNN and calibrated threshold remain unchanged."
        )
    }


# ============================================================
# AUTHENTICATION + AUTOMATIC MQTT SMART LOCK
# ============================================================

@app.post("/api/authenticate")
def authenticate(req: AuthRequest):
    if req.subject not in SUBJECTS or req.source_subject not in SUBJECTS:
        raise HTTPException(
            422,
            "Choose valid enrolled and sample subjects."
        )

    if req.subject not in enrolments():
        raise HTTPException(
            409,
            f"{req.subject} has no saved enrolment record. Enrol first."
        )

    # Step 1: CNN performs authentication.
    result = score_window(
        req.subject,
        req.source_subject,
        req.recording,
        req.window_index
    )

    # Step 2: Convert the CNN decision into the IoT lock command.
    command = "ACCEPT" if result["decision"] == "ACCEPT" else "REJECT"

    # Step 3: Automatically publish the decision to MQTT.
    mqtt_result = publish_mqtt_command(command)

    # Step 4: Include MQTT publishing status in the API response.
    result["mqtt"] = mqtt_result
    result["lock_command"] = command

    return result


# ============================================================
# BATCH / STREAM AUTHENTICATION
# ============================================================

@app.post("/api/stream")
def stream(req: StreamRequest):
    if req.subject not in SUBJECTS or req.source_subject not in SUBJECTS:
        raise HTTPException(
            422,
            "Choose valid subjects."
        )

    if req.subject not in enrolments():
        raise HTTPException(
            409,
            f"{req.subject} has no saved enrolment record. Enrol first."
        )

    end = min(
        req.start_window + req.count,
        WINDOWS_PER_SUBJECT
    )

    rows = [
        score_window(
            req.subject,
            req.source_subject,
            req.recording,
            i
        )
        for i in range(req.start_window, end)
    ]

    return {
        "rows": rows,
        "count": len(rows),
        "accepted": sum(
            r["decision"] == "ACCEPT" for r in rows
        ),
        "rejected": sum(
            r["decision"] == "REJECT" for r in rows
        ),
        "genuine": req.subject == req.source_subject
    }


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

@app.post("/api/explain")
def explain(req: ExplainRequest):
    if req.subject not in SUBJECTS or req.source_subject not in SUBJECTS:
        raise HTTPException(
            422,
            "Choose valid subjects."
        )

    if req.subject not in enrolments():
        raise HTTPException(
            409,
            f"{req.subject} has no saved enrolment record. Enrol first."
        )

    try:
        from models.local_shap_explainer import explain_window

        idx = global_index(
            req.source_subject,
            req.window_index
        )

        result = explain_window(
            req.subject,
            idx,
            recording=req.recording
        )

        values = np.asarray(result["shap_values"])
        signal = np.asarray(result["input_window"])

        # Compact channel/time values for plotting while preserving per-channel attribution.
        time_bins = 64
        step = max(1, values.shape[1] // time_bins)

        heat = np.stack(
            [
                values[:, i:min(i+step, values.shape[1])].mean(axis=1)
                for i in range(0, values.shape[1], step)
            ],
            axis=1
        )

        sig = np.stack(
            [
                signal[:, i:min(i+step, signal.shape[1])].mean(axis=1)
                for i in range(0, signal.shape[1], step)
            ],
            axis=1
        )

        return {
            "subject": req.subject,
            "source_subject": req.source_subject,
            "recording": req.recording,
            "window_index": req.window_index,
            "score_logit": result["score_logit"],
            "threshold": result["validation_threshold"],
            "decision": result["decision"],
            "dataset_label_is_subject": result["dataset_label_is_subject"],
            "top_channels": result["top_channels"],
            "plain_english": result["plain_english"],
            "channels": result["channels"],
            "time_bins": int(heat.shape[1]),
            "shap_heatmap": np.round(heat, 7).tolist(),
            "signal_preview": np.round(sig, 5).tolist(),
            "sampling_rate_hz": result["sampling_rate_hz"],
            "interpretation_note": (
                "SHAP explains attribution to the CNN logit, not a "
                "calibrated probability. Attribution is not causal proof."
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explanation failed: {type(e).__name__}: {e}"
        )
