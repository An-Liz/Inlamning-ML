import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# ===============================
# PAGE
# ===============================
st.set_page_config(
    page_title="MNIST – Sifferigenkänning",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ MNIST – Sifferigenkänning")
st.caption("Rita en siffra till vänster och klicka på **Prediktera**.")
st.divider()

# ===============================
# SESSION STATE INIT
# ===============================
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "stroke_width" not in st.session_state:
    st.session_state.stroke_width = 12

if "threshold" not in st.session_state:
    st.session_state.threshold = 120

# ===============================
# LOAD MODEL + SCALER
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("mnist_model.pkl")
    scaler = joblib.load("mnist_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ===============================
# SIDEBAR SETTINGS
# ===============================
with st.sidebar:
    st.header("⚙️ Inställningar")

    stroke_width = st.slider(
        "Pennbredd",
        4, 30,
        st.session_state.stroke_width,
        key="stroke_width"
    )

    threshold = st.slider(
        "Tröskel (svart/vitt)",
        0, 255,
        st.session_state.threshold,
        key="threshold"
    )

    show_preprocess = st.checkbox(
        "Visa preprocess-bild (28×28)",
        value=False
    )

    st.divider()
    st.write("Justera tröskel eller pennbredd om modellen gissar fel.")

# ===============================
# LAYOUT
# ===============================
left, right = st.columns([2, 1], gap="large")

# ===============================
# PREPROCESS
# ===============================
def preprocess(image_data, threshold_value, show_preview):
    img = Image.fromarray(image_data.astype("uint8")).convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img).astype(np.uint8)
    arr = np.where(arr > threshold_value, 255, 0).astype(np.uint8)

    img = Image.fromarray(arr)
    img_28 = img.resize((28, 28), Image.Resampling.NEAREST)

    if show_preview:
        st.image(img_28, caption="Efter preprocess (28×28)", width=160)

    flat = np.array(img_28).astype(np.float32).reshape(1, -1)
    flat_scaled = scaler.transform(flat)

    return flat_scaled

def has_ink(image_data, threshold_value):
    if image_data is None:
        return False
    gray = Image.fromarray(image_data.astype("uint8")).convert("L")
    inv = ImageOps.invert(gray)
    arr = np.array(inv)
    return (arr > threshold_value).sum() > 50

# ===============================
# CANVAS + BUTTONS
# ===============================
with left:
    st.subheader("1️⃣ Rita här")

    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        predict_clicked = st.button(
            "🔮 Prediktera",
            type="primary",
            use_container_width=True
        )

    with c2:
        if st.button("🧽 Prova igen", use_container_width=True):
            st.session_state.canvas_key += 1
            st.rerun()

    with c3:
        if st.button("🔄 Återställ inställningar", use_container_width=True):
            st.session_state.canvas_key += 1
            st.session_state.stroke_width = 12
            st.session_state.threshold = 120
            st.rerun()

# ===============================
# RESULTAT
# ===============================
with right:
    st.subheader("2️⃣ Resultat")

    if predict_clicked:
        if canvas_result.image_data is None or not has_ink(canvas_result.image_data, threshold):
            st.warning("Rita en siffra först 🙂")
        else:
            X = preprocess(canvas_result.image_data, threshold, show_preprocess)

            pred = model.predict(X)[0]
            st.success(f"Prediktion: **{pred}**")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                confidence = float(np.max(probs))
                st.metric("Säkerhet", f"{confidence*100:.1f}%")

                with st.expander("Visa sannolikheter (0–9)"):
                    fig, ax = plt.subplots()
                    ax.bar(range(10), probs)
                    ax.set_xticks(range(10))
                    ax.set_ylim(0, 1)
                    ax.set_xlabel("Siffra")
                    ax.set_ylabel("Sannolikhet")
                    st.pyplot(fig)
    else:
        st.info("Tryck på **Prediktera** när du ritat en siffra.")