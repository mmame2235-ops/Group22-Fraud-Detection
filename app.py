"""
Group 22 — Credit Card Fraud Detection API
==========================================
Loads the artifact saved by notebook Step 14, which is a dict:
    {
        "model": <trained XGBoost Pipeline>,
        "threshold": <float>,
        "feature_names": <list of 30 feature names>,
        "target_name": "Class",
        "selected_model_name": "XGBoost",
        "random_state": 42
    }
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App and model loading
# ---------------------------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/fraud_detection_model.joblib")
THRESHOLD_ENV = os.environ.get("THRESHOLD")

model = None
threshold = 0.5
feature_names = []
model_loaded = False
model_error = None
artifact_type = None

try:
    if not os.path.exists(MODEL_PATH):
        artifacts_dir = "artifacts" if os.path.exists("artifacts") else "(no artifacts dir)"
        contents = os.listdir("artifacts") if os.path.exists("artifacts") else []
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            f"CWD={os.getcwd()}. "
            f"artifacts/ contents={contents or artifacts_dir}"
        )

    log.info(f"Loading artifact from {MODEL_PATH} ...")
    artifact = joblib.load(MODEL_PATH)
    artifact_type = type(artifact).__name__
    log.info(f"Artifact loaded. Type={artifact_type}")

    # The artifact from Step 14 is a dict. Handle both dict and raw model
    # so this file is robust to future changes.
    if isinstance(artifact, dict):
        if "model" not in artifact:
            raise KeyError(
                f"Artifact is a dict but has no 'model' key. "
                f"Keys present: {list(artifact.keys())}"
            )
        model = artifact["model"]
        threshold = float(artifact.get("threshold", 0.5))
        feature_names = list(artifact.get("feature_names", []))
        log.info(f"Unpacked dict. threshold={threshold}, n_features={len(feature_names)}")
    else:
        # Fallback: raw model. Use env threshold if provided.
        model = artifact
        threshold = float(THRESHOLD_ENV) if THRESHOLD_ENV else 0.5
        log.warning("Artifact was not a dict; threshold/features inferred.")

    # Validate the model has the API we need
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"Loaded object of type {type(model).__name__} has no predict_proba method."
        )

    # Allow env var override for threshold (12-factor style)
    if THRESHOLD_ENV:
        try:
            threshold = float(THRESHOLD_ENV)
            log.info(f"Threshold overridden by env THRESHOLD={threshold}")
        except ValueError:
            log.warning(f"Ignoring invalid THRESHOLD env value: {THRESHOLD_ENV}")

    model_loaded = True
    log.info(f"✅ Model loaded successfully. Features: {len(feature_names)}")
    if feature_names:
        log.info(f"   Feature order: {feature_names[:3]} ... {feature_names[-2:]}")

except Exception as e:
    model_error = f"{type(e).__name__}: {e}"
    log.exception(f"❌ Model load failed: {model_error}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if model_loaded else "degraded",
        "service": "Group 22 Credit Card Fraud Detection API",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "artifact_type": artifact_type,
        "n_features": len(feature_names),
        "threshold": threshold if model_loaded else None,
        "model_error": model_error,        # remove before final public submission
        "endpoints": ["/health", "/predict"]
    })


@app.post("/predict")
def predict():
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded. Check /health for details.",
            "model_error": model_error
        }), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({
            "error": "Send a JSON object with the 30 transaction features."
        }), 400

    # Validate required features (from artifact)
    required = feature_names or list(payload.keys())
    missing = [f for f in required if f not in payload]
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing": missing
        }), 400

    # Reject unexpected extras (helps catch typos early)
    extra = [k for k in payload.keys() if k not in required]
    if extra:
        return jsonify({
            "error": "Unexpected features",
            "extra": extra,
            "expected": required
        }), 400

    try:
        # Build the row in the EXACT order the model expects
        row = pd.DataFrame([[payload[f] for f in required]], columns=required)
        # Coerce to numeric — model pipeline will impute missing values
        row = row.apply(pd.to_numeric, errors="coerce")

        if row.isna().all(axis=1).any():
            return jsonify({
                "error": "All feature values are non-numeric for at least one row."
            }), 400

        probability = float(model.predict_proba(row)[0, 1])
        predicted_class = int(probability >= threshold)
        decision = "FLAG FOR ANALYST REVIEW" if predicted_class == 1 else "NOT FLAGGED"

        return jsonify({
            "fraud_probability": round(probability, 4),
            "predicted_Class": predicted_class,
            "decision": decision,
            "threshold": round(threshold, 4),
            "model_version": "1.0.0"
        })

    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed",
            "detail": f"{type(e).__name__}: {e}"
        }), 500


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
