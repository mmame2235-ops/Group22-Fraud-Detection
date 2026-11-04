import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# The notebook in this project saves this file in Step 14.
MODEL_PATH = Path(os.environ.get(
    "FRAUD_MODEL_PATH",
    "artifacts/fraud_detection_model.joblib"
))

artifact = None
model = None
threshold = None
features = None
load_error = None


def load_model():
    """Load the trained notebook artifact, if it has been deployed."""
    global artifact, model, threshold, features, load_error

    if not MODEL_PATH.is_file():
        load_error = f"Model artifact not found at {MODEL_PATH}"
        return

    try:
        loaded = joblib.load(MODEL_PATH)
        if not isinstance(loaded, dict):
            raise ValueError("Artifact must be a dictionary.")

        required_keys = {"model", "threshold", "feature_names"}
        missing_keys = required_keys - set(loaded)
        if missing_keys:
            raise ValueError(
                f"Artifact is missing keys: {sorted(missing_keys)}"
            )

        loaded_features = list(loaded["feature_names"])
        if not loaded_features:
            raise ValueError("Artifact feature_names is empty.")

        artifact = loaded
        model = loaded["model"]
        threshold = float(loaded["threshold"])
        features = loaded_features
        load_error = None
    except Exception as exc:
        artifact = model = threshold = features = None
        load_error = f"Could not load model artifact: {exc}"


load_model()


@app.get("/")
def index():
    return jsonify({
        "service": "Group 22 Credit Card Fraud Detection API",
        "status": "ok",
        "model_loaded": model is not None,
        "endpoints": ["/health", "/predict"]
    })


@app.get("/health")
def health():
    """Health check for Render. Reports whether the model is ready."""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_error": load_error
    }), 200


@app.post("/predict")
def predict():
    """Predict fraud probability for one transaction.

    Send a JSON object containing all 30 fields:
    Time, V1..V28, Amount.
    """
    if model is None:
        return jsonify({
            "error": "The API is running, but the trained model artifact is not available.",
            "detail": load_error,
            "next_step": (
                "Run Step 14 of the project notebook, then include "
                "artifacts/fraud_detection_model.joblib in the deployment."
            )
        }), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a JSON object containing transaction features."
        }), 400

    missing = [column for column in features if column not in payload]
    extra = [column for column in payload if column not in features]

    if missing or extra:
        return jsonify({
            "error": "Input fields do not match the model's expected features.",
            "missing_features": missing,
            "unexpected_features": extra,
            "expected_feature_count": len(features)
        }), 400

    try:
        row = pd.DataFrame(
            [{column: payload[column] for column in features}],
            columns=features
        )
        for column in features:
            row[column] = pd.to_numeric(row[column], errors="coerce")

        if row.isna().all(axis=1).any():
            return jsonify({
                "error": "The transaction has no usable numeric feature values."
            }), 400

        probability = float(model.predict_proba(row)[0, 1])
        flagged = int(probability >= threshold)

        return jsonify({
            "fraud_probability": probability,
            "predicted_Class": flagged,
            "decision": (
                "FLAG FOR ANALYST REVIEW"
                if flagged
                else "NOT FLAGGED"
            ),
            "threshold": threshold,
            "model_name": artifact.get("selected_model_name", "trained model")
        }), 200

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed for this input.",
            "detail": str(exc)
        }), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
