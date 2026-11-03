
"""Animal classification using MobileNetV2 transfer learning."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "animal_model_v2.keras"
)

IMAGE_SIZE = (160, 160)

# Must match the class order used during model training.
CLASS_NAMES = ["Cat", "Dog"]


@lru_cache(maxsize=1)
def load_animal_model():
    """Load the trained Animal V2 model once."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Animal V2 model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


def predict_animal(uploaded_image):
    """
    Predict whether an uploaded image contains a Cat or Dog.

    Returns:
        predicted_label: "Cat" or "Dog"
        scores: Dictionary containing softmax scores (0 to 1).
    """

    model = load_animal_model()

    # Convert uploaded image to RGB.
    image = uploaded_image.convert("RGB")

    # Resize to the model's expected input dimensions.
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array.
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension.
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict.
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Convert model outputs into scores.
    predicted_index = int(np.argmax(predictions))
    predicted_label = CLASS_NAMES[predicted_index]

    scores = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    return predicted_label, scores
