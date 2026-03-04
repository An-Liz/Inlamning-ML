import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageFilter
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
# ===============================
# PREPROCESS (MNIST-lik)
# ===============================

def preprocess(image_data, show_preview: bool = False):
    """
    Preprocess som gör handritat mer MNIST-likt:
    - Gråskala
    - Invertera så siffran blir ljus på mörk bakgrund (som MNIST)
    - Lätt blur (anti-alias)
    - Crop runt bläck
    - Resize så siffran får plats i 20x20
    - Center-of-mass centering i 28x28
    """
    if image_data is None:
        return None

    # 1) RGBA -> grayscale (0=svart, 255=vitt)
    img = Image.fromarray(image_data.astype("uint8")).convert("L")

    # 2) Invertera: svart penna på vit bakgrund -> vit siffra på svart bakgrund
    img = ImageOps.invert(img)

    # 3) Lätt blur för att undvika "kantiga block"
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    arr = np.array(img).astype(np.uint8)

    # 4) Crop runt bläck
    ys, xs = np.where(arr > 20)  # 20 = ignorerar svagt brus
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # lite marginal
    pad = 8
    x0 = max(0, x0 - pad); x1 = min(arr.shape[1] - 1, x1 + pad)
    y0 = max(0, y0 - pad); y1 = min(arr.shape[0] - 1, y1 + pad)

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))

    # 5) Resize så största sidan blir 20 px (MNIST-siffran är typ 20x20 i en 28x28)
    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20 / h))))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6) Lägg in i 28x28
    canvas_28 = Image.new("L", (28, 28), 0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas_28.paste(resized, (left, top))

    # 7) Center-of-mass centering (flytta till mitten)
    a = np.array(canvas_28).astype(np.float32)
    a = a / (a.max() + 1e-6)

    ys2, xs2 = np.nonzero(a > 0.05)
    if len(xs2) > 0 and len(ys2) > 0:
        cx = xs2.mean()
        cy = ys2.mean()
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))

        a_shift = np.zeros_like(a)
        y_from = max(0, -shift_y); y_to = min(28, 28 - shift_y)
        x_from = max(0, -shift_x); x_to = min(28, 28 - shift_x)

        a_shift[y_from + shift_y:y_to + shift_y, x_from + shift_x:x_to + shift_x] = a[y_from:y_to, x_from:x_to]
        a = a_shift

    img_final = Image.fromarray((a * 255).astype(np.uint8))

    if show_preview:
        st.image(img_final, caption="Preprocess (MNIST-lik 28×28)", width=160)

    flat = np.array(img_final).astype(np.float32).reshape(1, -1)
    return scaler.transform(flat)

	def has_ink(image_data) -> bool:
    """
    Kollar om det faktiskt finns något ritat på canvasen.
    Undviker att modellen försöker prediktera tom bild.
    """
    if image_data is None:
        return False

    gray = Image.fromarray(image_data.astype("uint8")).convert("L")
    arr = np.array(gray)

    # Canvas är vit bakgrund (255)
    # Bläck är mörkt (< 250 ungefär)
    ink_pixels = (arr < 250).sum()

    return ink_pixels > 50  # kräver minst lite ritad yta

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
            X = preprocess(canvas_result.image_data, show_preprocess)

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