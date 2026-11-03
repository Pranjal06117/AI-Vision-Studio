
import cv2
import numpy as np
from PIL import Image


def convert_to_sketch(
    uploaded_image,
    line_strength=5
):
    """
    Create a clear, medium-dark pencil/technical sketch.

    line_strength:
        1  = Light
        5  = Medium-dark (recommended)
        10 = Dark
    """

    # Convert PIL image to NumPy array
    image = np.array(
        uploaded_image.convert("RGB")
    )

    # Convert RGB to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Reduce noise while preserving details
    smooth = cv2.bilateralFilter(
        gray,
        9,
        60,
        60
    )

    # Improve visibility of faint details
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(smooth)

    # ----------------------------------
    # PENCIL SHADING
    # ----------------------------------

    inverted = 255 - enhanced

    blurred = cv2.GaussianBlur(
        inverted,
        (21, 21),
        0
    )

    inverted_blur = 255 - blurred

    pencil = cv2.divide(
        enhanced,
        inverted_blur,
        scale=255.0
    )

    # ----------------------------------
    # EDGE DETECTION
    # ----------------------------------

    lower_threshold = int(
        np.interp(line_strength, [1, 10], [75, 35])
    )

    upper_threshold = int(
        np.interp(line_strength, [1, 10], [170, 100])
    )

    edges = cv2.Canny(
        enhanced,
        lower_threshold,
        upper_threshold
    )

    # ----------------------------------
    # MEDIUM-DARK OUTLINES
    # ----------------------------------

    # Slightly strengthen detected edges
    kernel = np.ones((2, 2), np.uint8)

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    # Choose edge darkness
    edge_tone = int(
        np.interp(line_strength, [1, 10], [125, 45])
    )

    # Darken detected edges without
    # turning the entire image black
    sketch = pencil.copy()

    sketch[edges > 0] = np.minimum(
        sketch[edges > 0],
        edge_tone
    )

    # ----------------------------------
    # FINAL CONTRAST
    # ----------------------------------

    sketch = cv2.convertScaleAbs(
        sketch,
        alpha=1.12,
        beta=-7
    )

    # Return final sketch
    return Image.fromarray(sketch)
