from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(APP_ROOT / "cattle_weight_B3_B4_MobileNetV3_BASELINE.keras"),
    )
)
IMAGE_SIZE = (224, 224)
MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Cattle Weight Prediction API",
    version="1.0.0",
    description="Predicts cattle weight from an uploaded image.",
)

_model: Any = None


def get_model() -> Any:
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"Model file was not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


def prepare_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from error

    resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32)[np.newaxis, ...]


def prediction_response(filename: str, predicted_weight_kg: float) -> dict[str, Any]:
    # The bundled model is a regressor, not an object detector. These fields
    # keep the response compatible with the existing Flutter client.
    return {
        "valid": True,
        "cattle_detected": True,
        "animal": "cattle",
        "detection_confidence_percent": 100.0,
        "predicted_weight_kg": round(predicted_weight_kg, 2),
        "confidence_level_percent": 0.0,
        "estimated_accuracy_percent": 0.0,
        "confidence_interval_kg": {"lower": 0.0, "upper": 0.0},
        "expected_error_kg": 0.0,
        "model_rmse_kg": 0.0,
        "filename": filename,
        "message": "This model predicts weight only; uncertainty metrics are not included in the saved model.",
        "detections": [],
    }


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "cattle-weight-prediction"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH.name}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded image is larger than 10 MB.")

    inputs = prepare_image(image_bytes)
    try:
        prediction = get_model().predict(inputs, verbose=0)
        predicted_weight_kg = float(np.asarray(prediction).reshape(-1)[0])
    except (RuntimeError, ValueError, TypeError, IndexError) as error:
        raise HTTPException(status_code=500, detail="The model could not process this image.") from error

    if not np.isfinite(predicted_weight_kg) or predicted_weight_kg < 0:
        raise HTTPException(status_code=500, detail="The model returned an invalid weight.")

    return prediction_response(file.filename or "image", predicted_weight_kg)