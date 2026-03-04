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
    page_title="Prova sifferigenkänning med MNIST",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 Prova sifferigenkänning med MNIST")
st.caption("Rita en siffra till vänster och klicka på **Prediktera**. Du kan ändra inställningar till vänster.")
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

if "do_reset_settings" not in st.session_state:
    st.session_state.do_reset_settings = False

# Om vi tryckte "Återställ inställningar" förra körningen
if st.session_state.do_reset_settings:
    st.session_state.canvas_key += 1
    st.session_state["stroke_width"] = 12
    st.session_state["threshold"] = 120
    st.session_state.do_reset_settings = False

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
import numpy as np
from PIL import Image, ImageOps

def has_ink(image_data, min_pixels: int = 50) -> bool:
    """Kollar om användaren faktiskt ritat något."""
    if image_data is None:
        return False
    gray = Image.fromarray(image_data.astype("uint8")).convert("L")
    inv = ImageOps.invert(gray)
    arr = np.array(inv)
    return (arr > 10).sum() > min_pixels  # lågt tröskelvärde bara för 'något finns'

X = preprocess(canvas_result.image_data, show_preview=show_preprocess)
    """
    Gör canvas-bilden mer MNIST-lik:
    - Gråskala + invert
    - Crop runt 'bläck'
    - Resize så max-dimension blir 20 px (som MNIST-ish), sen pad till 28x28
    - Ingen hård threshold (behåll gråskala)
    """
    # 1) Canvas RGBA -> grayscale
    img = Image.fromarray(image_data.astype("uint8")).convert("L")

    # 2) Invertera: svart penna -> vitt "bläck"
    img = ImageOps.invert(img)

    arr = np.array(img).astype(np.uint8)

    # 3) Hitta bounding box runt pixlar som inte är "bakgrund"
    ys, xs = np.where(arr > 10)  # 10 = ganska låg, bara för att hitta stroke
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # lite marginal runt siffran
    pad = 10
    x0 = max(0, x0 - pad); x1 = min(arr.shape[1] - 1, x1 + pad)
    y0 = max(0, y0 - pad); y1 = min(arr.shape[0] - 1, y1 + pad)

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))

    # 4) Resize så att siffran får plats bra i 28x28 (typ 20x20 + padding)
    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20 / h))))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 5) Pad till 28x28 och centrera
    canvas_28 = Image.new("L", (28, 28), 0)  # svart bakgrund
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas_28.paste(resized, (left, top))

    if show_preview:
        st.image(canvas_28, caption="Efter preprocess (centrerad 28×28)", width=160)

    # 6) Flatten + scale (som i träningen)
    flat = np.array(canvas_28).astype(np.float32).reshape(1, -1)
    flat_scaled = scaler.transform(flat)
    return flat_scaled

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
            st.session_state.do_reset_settings = True
            st.rerun()

# ===============================
# RESULTAT
# ===============================
with right:
    st.subheader("2️⃣ Resultat")

    if predict_clicked:
        if canvas_result.image_data is None or not has_ink(canvas_result.image_data):
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