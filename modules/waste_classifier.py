"""Inference module for the AI Vision Studio nine-class waste model."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "waste_model_v4.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "outputs" / "waste_v4" / "class_names.json"
IMAGE_SIZE = (160, 160)


@lru_cache(maxsize=1)
def _load_model():
    """Load the trained Keras model once and reuse it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"V4 waste model not found: {MODEL_PATH}")
    from tensorflow import keras
    return keras.models.load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_class_names():
    """Load class order saved during training."""
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            names = json.load(file)
        if isinstance(names, dict):
            names = names.get("class_names", names.get("classes"))
        if names:
            return list(names)

    # Fallback order must match the training directory class order.
    return [
        "Battery", "Cardboard", "Organic", "Plastic", "Textile",
        "Metal", "Glass", "Paper", "Trash"
    ]


def predict_waste(uploaded_image):
    """
    Predict one of the nine waste classes.

    Returns (predicted_label, confidence, score_dictionary).
    """
    image = uploaded_image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(image_array, axis=0)

    model = _load_model()
    probabilities = model.predict(batch, verbose=0)[0]
    class_names = _load_class_names()

    if len(class_names) != len(probabilities):
        raise ValueError(
            "Class-name count does not match model output. "
            "Check outputs/waste_v4/class_names.json."
        )

    best_index = int(np.argmax(probabilities))
    label = class_names[best_index]
    confidence = float(probabilities[best_index])
    scores = {
        class_names[index]: float(probabilities[index])
        for index in range(len(class_names))
    }
    return label, confidence, scores
