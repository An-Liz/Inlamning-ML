import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST – Sifferigenkänning")

st.title("✍️ MNIST – Sifferigenkänning")

# Ladda modell och scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("mnist_model.pkl")
    scaler = joblib.load("mnist_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.success("Modell och scaler laddade!")

st.write("Rita en siffra i rutan och klicka på Prediktera.")

# ===============================
# RITYTA
# ===============================

canvas_result = st_canvas(
    stroke_width=12,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ===============================
# PREPROCESS
# ===============================

def preprocess(image_data):
    img = Image.fromarray(image_data.astype("uint8")).convert("L")

    # Invertera
    img = ImageOps.invert(img)

    arr = np.array(img).astype(np.uint8)

    # 🔥 HÅRD TRÖSKEL
    arr = np.where(arr > 120, 255, 0).astype(np.uint8)

    img = Image.fromarray(arr)

    # Resize till 28x28
    img_28 = img.resize((28, 28), Image.Resampling.NEAREST)

    st.image(img_28, caption="Efter preprocess (28x28)", width=140)

    flat = np.array(img_28).astype(np.float32).reshape(1, -1)

    flat_scaled = scaler.transform(flat)

    return flat_scaled

# ===============================
# PREDIKTION
# ===============================

if st.button("Prediktera"):

    if canvas_result.image_data is None:
        st.warning("Rita en siffra först 🙂")

    else:
        X = preprocess(canvas_result.image_data)

        prediction = model.predict(X)[0]

        st.subheader("Prediktion:")
        st.write(f"Modellen tror att siffran är: **{prediction}**")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]

            st.subheader("Sannolikheter:")
            fig, ax = plt.subplots()
            ax.bar(range(10), probs)
            ax.set_xticks(range(10))
            ax.set_ylim(0, 1)
            st.pyplot(fig)