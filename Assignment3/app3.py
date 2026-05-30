import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Traffic Sign Detection")

st.title("🚦 Traffic Sign Detection")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="traffic_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ FULL 43 CLASS LABELS (GTSRB)
class_names = [str(i) for i in range(43)]

# ✅ Map only important classes
label_map = {
    14: "🛑 Stop Sign",
    17: "⛔ No Entry",
    2: "🚗 Speed Limit"
}

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Traffic Image",
    type=["jpg", "png", "jpeg"]
)

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# -------------------------------
# PREDICTION
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect"):
        try:
            img = preprocess(image)

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()

            pred = interpreter.get_tensor(output_details[0]['index'])[0]

            pred_class = int(np.argmax(pred))
            confidence = float(np.max(pred))

            # ✅ Show real class always
            st.write(f"Raw Class ID: {pred_class}")

            # ✅ Show mapped class if important
            if pred_class in label_map:
                label = label_map[pred_class]
            else:
                label = f"Other Traffic Sign (Class {pred_class})"

            st.success(f"🚦 Detected: {label}")
            st.info(f"Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")