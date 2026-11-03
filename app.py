
import io
import streamlit as st
import pandas as pd

from PIL import Image

from modules.animal_classifier import predict_animal
from modules.waste_classifier import predict_waste
from modules.sketch_model import convert_to_sketch


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Vision Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM DESIGN
# ==================================================

st.markdown("""
<style>

.stApp {
    background: #0b1220;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #111c30;
    border-right: 1px solid #26364f;
}

h1, h2, h3 {
    color: #f8fafc;
}

p, label {
    color: #cbd5e1;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(120deg, #172c4a, #164e63);
    border: 1px solid #31516b;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 40px;
}

.hero p {
    color: #dbeafe;
    font-size: 17px;
}

.section-card {
    padding: 22px;
    border-radius: 17px;
    background: #172337;
    border: 1px solid #2c3e59;
    min-height: 190px;
}

.result-card {
    padding: 22px;
    border-radius: 17px;
    background: #142c32;
    border: 1px solid #27766c;
    margin: 15px 0;
}

div[data-testid="stFileUploader"] {
    background: #172337;
    padding: 15px;
    border-radius: 14px;
    border: 1px dashed #54708f;
}

.stButton button, .stDownloadButton button {
    border-radius: 10px;
    font-weight: 600;
    min-height: 43px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def display_scores(scores):
    """Display model class scores as percentages."""

    if not isinstance(scores, dict):
        return

    st.subheader("Class Confidence")

    for class_name, score in scores.items():

        percentage = float(score) * 100

        st.write(
            f"**{class_name}** — {percentage:.2f}%"
        )

        st.progress(
            max(0.0, min(float(score), 1.0))
        )


def display_animal_result(result):
    """Display the Animal classifier's two-part result."""

    label, scores = result

    st.markdown(
        f"""
        <div class="result-card">
            <h3>🐾 Animal Prediction</h3>
            <h2>{label}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    if isinstance(scores, dict):
        confidence = scores.get(label)

        if confidence is not None:
            st.metric(
                "Prediction Confidence",
                f"{float(confidence) * 100:.2f}%"
            )

        display_scores(scores)


def display_waste_result(result):
    """Display the Waste classifier's three-part result."""

    label, confidence, scores = result

    st.markdown(
        f"""
        <div class="result-card">
            <h3>♻️ Waste Prediction</h3>
            <h2>{label}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric(
        "Prediction Confidence",
        f"{float(confidence) * 100:.2f}%"
    )

    display_scores(scores)


def image_to_png_bytes(image):
    """Convert a PIL image to downloadable PNG bytes."""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    output = io.BytesIO()
    image.save(output, format="PNG")

    return output.getvalue()


def read_uploaded_image(uploaded_file):
    """Read uploaded image safely as RGB."""

    image = Image.open(uploaded_file)
    image.load()

    return image.convert("RGB")


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("# 👁️ AI Vision Studio")

    st.caption("Intelligent Image Classification")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "🐾 Animal Classification",
            "♻️ Waste Classification",
            "✏️ Image to Sketch",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.markdown("### Technologies")

    st.caption("Python")
    st.caption("Streamlit")
    st.caption("TensorFlow / Keras")
    st.caption("MobileNetV2")
    st.caption("OpenCV")


# ==================================================
# HOME
# ==================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>👁️ AI Vision Studio</h1>
        <p>
        A multi-module AI application for image classification
        and image processing using Python, TensorFlow and OpenCV.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Explore AI Tools")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h2>🐾</h2>
            <h3>Animal Classification</h3>
            <p>Identify cats and dogs using a trained AI model.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <h2>♻️</h2>
            <h3>Waste Classification</h3>
            <p>Classify waste into nine different categories.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
            <h2>✏️</h2>
            <h3>Image to Sketch</h3>
            <p>Convert photographs into pencil-style sketches.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.info(
        "Select a module from the sidebar to get started."
    )


# ==================================================
# ANIMAL CLASSIFICATION
# ==================================================

elif page == "🐾 Animal Classification":

    st.title("🐾 Animal Classification")

    st.write(
        "Upload an image to predict whether it contains a cat or dog."
    )

    uploaded_file = st.file_uploader(
        "Upload an animal image",
        type=["jpg", "jpeg", "png"],
        key="animal_upload"
    )

    if uploaded_file is not None:

        image = read_uploaded_image(uploaded_file)

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Your uploaded animal image",
            use_container_width=True
        )

        if st.button(
            "🔍 Classify Animal",
            use_container_width=True
        ):

            with st.spinner("Analysing image..."):

                try:
                    result = predict_animal(image)

                    st.divider()

                    display_animal_result(result)

                except Exception as error:
                    st.error(
                        f"Animal prediction failed: {error}"
                    )

    else:
        st.info("Upload a JPG or PNG image to begin.")


# ==================================================
# WASTE CLASSIFICATION
# ==================================================

elif page == "♻️ Waste Classification":

    st.title("♻️ Waste Classification")

    st.write(
        "Upload a waste image to classify it into one of nine categories."
    )

    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png"],
        key="waste_upload"
    )

    if uploaded_file is not None:

        image = read_uploaded_image(uploaded_file)

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Your uploaded waste image",
            use_container_width=True
        )

        if st.button(
            "🔍 Classify Waste",
            use_container_width=True
        ):

            with st.spinner("Analysing waste image..."):

                try:
                    result = predict_waste(image)

                    st.divider()

                    display_waste_result(result)

                except Exception as error:
                    st.error(
                        f"Waste prediction failed: {error}"
                    )

    else:
        st.info("Upload a JPG or PNG image to begin.")


# ==================================================
# IMAGE TO SKETCH
# ==================================================

elif page == "✏️ Image to Sketch":

    st.title("✏️ Image to Sketch Studio")

    st.write(
        "Transform your photograph into a pencil or technical sketch."
    )

    uploaded_file = st.file_uploader(
        "Upload your photograph",
        type=["jpg", "jpeg", "png"],
        key="sketch_upload"
    )

    line_strength = st.slider(
        "Sketch Line Strength",
        min_value=1,
        max_value=10,
        value=5,
        help="1 = Light, 5 = Medium-dark, 10 = Dark"
    )

    if uploaded_file is not None:

        original_image = read_uploaded_image(uploaded_file)

        st.subheader("Original Photograph")

        st.image(
            original_image,
            caption="Original Image",
            use_container_width=True
        )

        if st.button(
            "✨ Generate Sketch",
            use_container_width=True
        ):

            with st.spinner("Creating your sketch..."):

                try:

                    sketch_image = convert_to_sketch(
                        original_image,
                        line_strength=line_strength
                    )

                    st.session_state["generated_sketch"] = sketch_image
                    st.session_state["sketch_source"] = uploaded_file.name

                    st.success("Sketch generated successfully!")

                except Exception as error:

                    st.error(
                        f"Sketch generation failed: {error}"
                    )

        sketch_image = st.session_state.get(
            "generated_sketch"
        )

        sketch_source = st.session_state.get(
            "sketch_source"
        )

        # Only show the saved sketch for the same uploaded file.
        if (
            sketch_image is not None
            and sketch_source == uploaded_file.name
        ):

            st.divider()

            st.subheader("Before and After")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    original_image,
                    caption="Original Photograph",
                    use_container_width=True
                )

            with col2:
                st.image(
                    sketch_image,
                    caption="Generated Sketch",
                    use_container_width=True
                )

            st.divider()

            try:

                png_data = image_to_png_bytes(
                    sketch_image
                )

                st.download_button(
                    label="⬇️ Download Sketch as PNG",
                    data=png_data,
                    file_name="AI_Vision_Studio_Sketch.png",
                    mime="image/png",
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    f"Download preparation failed: {error}"
                )

    else:

        st.info("Upload a photograph to create your sketch.")


# ==================================================
# ABOUT PROJECT
# ==================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About AI Vision Studio")

    st.markdown("""
    ## Project Overview

    AI Vision Studio is a multi-module image classification
    and image processing application developed using Python.

    ### 1. Animal Classification

    - Identifies Cat and Dog images.
    - Uses a trained MobileNetV2-based deep learning model.
    - Built using TensorFlow and Keras.

    ### 2. Waste Classification

    - Identifies nine waste categories.
    - Uses a trained MobileNetV2-based deep learning model.
    - Displays prediction confidence and class scores.

    ### 3. Image-to-Sketch Studio

    - Converts uploaded photographs into pencil sketches.
    - Uses OpenCV image-processing techniques.
    - Includes adjustable line strength.
    - Allows users to download generated sketches.

    ## Technologies Used

    - Python
    - Streamlit
    - TensorFlow
    - Keras
    - MobileNetV2
    - OpenCV
    - NumPy
    - Pillow

    ## Project Objective

    To demonstrate image classification and image processing
    through an interactive web application.
    """)

    st.divider()

    st.caption(
        "AI Vision Studio | NIELIT AI Development Associate Capstone"
    )
