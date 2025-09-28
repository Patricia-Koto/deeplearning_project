# app.py
# Dépendances :
#   pip install "tensorflow>=2.12" "keras>=3.0" streamlit pillow matplotlib

import io
import pathlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image as kimage
import keras  # pour charger correctement le format .keras (Keras 3)

tfk = tf.keras

st.set_page_config(page_title="Scenes Classifier", page_icon="🖼️", layout="centered")

# ========= CONFIG =========
DEFAULT_MODEL_PATH = "transfer_best.keras"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# ========= UTILS =========
def any_layer_recursive(layer, predicate) -> bool:
    if predicate(layer):
        return True
    for sub in getattr(layer, "layers", []):
        if any_layer_recursive(sub, predicate):
            return True
    return False

def model_has_rescaling(model) -> bool:
    return any_layer_recursive(model, lambda l: isinstance(l, tfk.layers.Rescaling))

def preprocess_for_model(x_uint8: np.ndarray, has_rescaling: bool) -> np.ndarray:
    x = x_uint8.astype("float32")
    if not has_rescaling:
        x = x / 255.0
    return x

# ========= CACHES =========
@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    try:
        # Keras 3 (.keras)
        return keras.models.load_model(model_path, compile=False, safe_mode=False)
    except Exception:
        # tf.keras (SavedModel/H5)
        return tfk.models.load_model(model_path, compile=False)

@st.cache_data(show_spinner=False)
def read_image(file_bytes: bytes, size=(150, 150)):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize(size)
    x = kimage.img_to_array(img_resized)     # (H,W,3) float32 [0..255]
    x = np.expand_dims(x, axis=0)            # (1,H,W,3)
    return img, img_resized, x

# ========= UI =========
st.title("🖼️ Intel Scenes — Prédiction simple (sans Grad-CAM)")
st.caption("Charge un modèle .keras/.h5, envoie une image, et vois la prédiction.")

# Choix modèle + Top-k
colA, colB = st.columns([2, 1])
with colA:
    model_path = st.text_input("Chemin du modèle", DEFAULT_MODEL_PATH)
with colB:
    topk = st.number_input("Top-k", min_value=1, max_value=len(CLASS_NAMES), value=3)
st.caption("""
**ℹ️ Top-k** : affiche les k classes les plus probables, triées de la plus certaine à la moins certaine.  
""")
# Chargement modèle
try:
    model = load_model(model_path)
    HAS_RESCALING = model_has_rescaling(model)
    st.success("Modèle chargé.")
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

# ========= Uploader (multi) + Prédiction =========
uploaded_files = st.file_uploader(
    "Glisse une ou plusieurs images (JPG/PNG)…",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True  # <— la seule option à ajouter
)

if uploaded_files:
    for file in uploaded_files:
        # Lecture + prétraitement
        pil_img, pil_resized, x_uint8 = read_image(file.read(), size=IMG_SIZE)
        x = preprocess_for_model(x_uint8, HAS_RESCALING)

        # Aperçu
        st.subheader(f"Aperçu — {file.name}")
        st.image(pil_img, caption="Image originale", use_container_width=True)

        # Prédiction (appel direct)
        y = model(tf.convert_to_tensor(x, dtype=tf.float32), training=False).numpy()[0]
        # Si le modèle renvoie des logits, applique softmax
        if np.any(y < 0) or not np.isclose(np.sum(y), 1.0, atol=1e-3):
            y = tf.nn.softmax(y).numpy()

        idx_sorted = y.argsort()[::-1][:topk]
        st.subheader("Prédictions")
        for i in idx_sorted:
            st.write(f"- **{CLASS_NAMES[i]}** : {y[i]:.2%}")
        st.progress(float(y[idx_sorted[0]]))

