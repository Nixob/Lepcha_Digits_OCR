import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

st.set_page_config(page_title="Lepcha Digit OCR", page_icon="🔢", layout="centered")

MODEL_PATH = "lepcha_ocr.keras"
IMG_SIZE = (28, 28)
CLASS_LABELS = [str(i) for i in range(10)]  # folders '0'-'9', alphabetical == numeric order


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def fix_orientation(pil_img: Image.Image) -> Image.Image:
    """Apply EXIF rotation metadata (phone photos are often stored sideways)."""
    return ImageOps.exif_transpose(pil_img)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Convert an uploaded image into the (1, 28, 28, 1) float array the model expects.

    The training data is dark background / bright digit strokes (like MNIST).
    Photos of pen-on-paper are usually the opposite (light background / dark
    strokes), which the model was never trained on and causes it to lock onto
    one class regardless of the actual shape. We auto-detect that case by
    checking the image border and invert if needed.
    """
    img = pil_img.convert("L")  # grayscale (orientation already fixed by caller)
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0

    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border.mean() > 0.5:  # background is light -> invert to match training polarity
        arr = 1.0 - arr

    arr = arr.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    return arr


st.title("🔢 Lepcha Digit OCR")
st.write(
    "Upload an image of a single handwritten Lepcha digit (0–9) and the model "
    "will predict which digit it is."
)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a digit image", type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    pil_img = fix_orientation(Image.open(uploaded_file))

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    input_arr = preprocess_image(pil_img)

    with col2:
        st.image(
            (input_arr.reshape(IMG_SIZE) * 255).astype("uint8"),
            caption="Model input (28x28 grayscale)",
            use_container_width=True,
        )

    preds = model.predict(input_arr, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_LABELS[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    st.success(f"Predicted digit: **{pred_label}**  (confidence: {confidence:.1f}%)")

    with st.expander("Show full probability breakdown"):
        probs_sorted = sorted(
            zip(CLASS_LABELS, preds), key=lambda x: x[1], reverse=True
        )
        for label, p in probs_sorted:
            st.write(f"{label}: {p * 100:.2f}%")
            st.progress(float(p))
else:
    st.info("Upload an image to get a prediction.")

st.markdown("---")
st.caption(
    "Model: CNN trained on Lepcha digit dataset (28x28 grayscale, 10 classes)."
)
