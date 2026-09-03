import streamlit as st
import numpy as np
import cv2
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


def preprocess_image(pil_img: Image.Image, padding: int = 12):
    """Convert an uploaded image into the (1, 28, 28, 1) float array the model expects.

    Pipeline: denoise -> blur -> auto-polarity Otsu threshold -> crop to the
    digit's bounding box -> center on a square canvas -> resize to 28x28.

    The auto-polarity step handles both cases without assuming either one:
    training data is dark background / bright strokes, but photos of pen on
    paper are usually the opposite. Whichever class (light or dark) covers
    less of the image is treated as the digit, so the output always matches
    the training convention regardless of the input's original polarity.

    Note: deliberately no morphological opening step -- at this resolution it
    erodes away thin digit strokes entirely and causes misclassification.

    Returns None if no content is detected (e.g. a blank image).
    """
    gray = np.array(pil_img.convert("L"))

    img = cv2.fastNlMeansDenoising(gray, h=10)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    _, binary_raw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.count_nonzero(binary_raw) / binary_raw.size
    binary = cv2.bitwise_not(binary_raw) if white_ratio > 0.5 else binary_raw

    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, binary.shape[1] - x)
    h = min(h + 2 * padding, binary.shape[0] - y)
    digit = binary[y : y + h, x : x + w]

    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - w) // 2
    y_off = (size - h) // 2
    square[y_off : y_off + h, x_off : x_off + w] = digit

    final = cv2.resize(square, IMG_SIZE, interpolation=cv2.INTER_AREA)
    arr = (final / 255.0).reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
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

    if input_arr is None:
        st.error(
            "Couldn't detect any digit content in this image — it may be blank "
            "or too low-contrast. Try a clearer, closer photo of a single digit."
        )
    else:
        with col2:
            st.image(
                (input_arr.reshape(IMG_SIZE) * 255).astype("uint8"),
                caption="Model input (28x28, after crop/threshold)",
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
