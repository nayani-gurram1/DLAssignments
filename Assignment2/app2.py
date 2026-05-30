import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Pneumonia Detection")

st.title("🩺 Pneumonia Detection from X-Ray")

# -------------------------------
# LOAD TFLITE MODEL
# -------------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

# -------------------------------
# PREPROCESS FUNCTION (FIXED)
# -------------------------------
def preprocess(img):
    # Convert to RGB (VERY IMPORTANT for X-ray images)
    img = img.convert("RGB")

    # Resize
    img = img.resize((224, 224))

    # Normalize
    img = np.array(img) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Convert dtype
    img = img.astype(np.float32)

    return img

# -------------------------------
# PREDICTION
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            img = preprocess(image)

            # Debug (optional)
            st.write("Input shape:", img.shape)

            # Set tensor
            interpreter.set_tensor(input_details[0]['index'], img)

            # Run model
            interpreter.invoke()

            # Get prediction
            pred = interpreter.get_tensor(output_details[0]['index'])[0][0]

            st.subheader("Result")

            if pred > 0.5:
                st.error(f"⚠️ Pneumonia Detected ({pred:.2f})")
            else:
                st.success(f"✅ Normal ({pred:.2f})")

        except Exception as e:
            st.error(f"Error: {e}")