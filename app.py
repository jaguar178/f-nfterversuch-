import streamlit as st
import json
from PIL import Image
from utils.predict import predict_image

# --- Daten laden ---
@st.cache_data
def load_db():
    try:
        with open("database/motorcycles.json") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Datenbank Fehler: {e}")
        return {}

db = load_db()

st.title("Motorrad Erkennung")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild", use_column_width=True)

    with st.spinner("Analysiere Bild..."):
        try:
            predictions = predict_image(image)
        except Exception as e:
            st.error(str(e))
            st.stop()

    if not predictions:
        st.error("Keine Vorhersage möglich.")
        st.stop()

    top_pred = predictions[0]
    label = top_pred["label"]
    confidence = top_pred["confidence"]

    # Confidence Filter
    if confidence < 0.6:
        st.warning("Unsicheres Ergebnis – bitte besseres Bild verwenden.")
        st.stop()

    st.success(f"Erkannt: **{label}** ({confidence*100:.1f}%)")

    # --- Top 3 anzeigen ---
    with st.expander("🔝 Weitere Vorhersagen"):
        for p in predictions[:3]:
            st.write(f"{p['label']}: {p['confidence']*100:.1f}%")

    # --- Technische Daten ---
    if label in db:
        st.subheader("Technische Daten")
        bike = db[label]

        st.write(f"**Name:** {bike.get('name', '-')}")
        st.write(f"**Marke:** {bike.get('brand', '-')}")
        st.write(f"**Typ:** {bike.get('type', '-')}")
        st.write(f"**Motor:** {bike.get('engine', '-')}")
        st.write(f"**Leistung:** {bike.get('hp', '-')}")
        st.write(f"**Gewicht:** {bike.get('weight', '-')}")
    else:
        st.warning("Keine Daten gefunden.")

    # --- Debug ---
    with st.expander("🔧 Debug"):
        st.json(predictions)
