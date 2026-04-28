import streamlit as st
import json
from PIL import Image
from utils.predict import predict_image

# Daten laden
try:
    with open("database/motorcycles.json") as f:
        db = json.load(f)
except:
    db = {}

st.title("🏍️ Motorrad Erkennung")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild", use_column_width=True)

    with st.spinner("Analysiere Bild..."):
        try:
            label, confidence = predict_image(image)
        except Exception as e:
            st.error(f"Fehler: {e}")
            st.stop()

    st.subheader("Ergebnis")
    st.write(f"{label} ({confidence*100:.1f}%)")

    # DEBUG (kannst du später entfernen)
    st.write("DEBUG Labels:", label)

    if label in db:
        st.subheader("Technische Daten")
        for k, v in db[label].items():
            st.write(f"**{k}:** {v}")
    else:
        st.warning("Keine Daten gefunden.")
