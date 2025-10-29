#!/usr/bin/env python3
"""
app_streamlit.py — Streamlit frontend for wheat disease detection
Run:
    streamlit run src/app_streamlit.py
"""
import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"

# --------------------------------------------------------------------
# Disease information database (editable, extendable)
# --------------------------------------------------------------------
DISEASE_INFO = {
    "BlackPoint": {
        "description": (
            "Black point appears as dark discoloration on the embryo or surrounding "
            "area of the wheat kernel. It is caused by fungi such as Alternaria, "
            "Cladosporium, and Helminthosporium species, often under humid conditions."
        ),
        "treatment": [
            "Use certified, disease-free seeds.",
            "Avoid late irrigation near harvest.",
            "Harvest and store grain under dry conditions.",
            "Rotate crops to reduce fungal inoculum in the soil.",
        ],
    },
    "FusariumFootRot": {
        "description": (
            "Fusarium foot rot causes browning and decay of the wheat stem base, "
            "leading to poor tillering and whiteheads. It thrives in warm, moist soils."
        ),
        "treatment": [
            "Use resistant wheat varieties where available.",
            "Ensure proper drainage and avoid waterlogging.",
            "Apply fungicidal seed treatments containing tebuconazole or prothioconazole.",
            "Rotate with non-cereal crops to break the fungal cycle.",
        ],
    },
    "HealthyLeaf": {
        "description": (
            "This leaf appears healthy with no visible disease symptoms — uniform green color "
            "and no necrotic spots or chlorosis."
        ),
        "treatment": [
            "Maintain regular monitoring for early signs of infection.",
            "Use balanced fertilizer and irrigation practices.",
            "Follow integrated pest management for preventive care.",
        ],
    },
    "LeafBlight": {
        "description": (
            "Leaf blight causes elongated brown or gray lesions on leaves, "
            "reducing photosynthesis and yield. Commonly caused by Bipolaris or Drechslera fungi."
        ),
        "treatment": [
            "Spray fungicides such as mancozeb or azoxystrobin during early infection.",
            "Destroy infected crop residues after harvest.",
            "Practice crop rotation and ensure adequate plant spacing for airflow.",
        ],
    },
    "WheatBlast": {
        "description": (
            "Wheat blast is a severe fungal disease caused by *Magnaporthe oryzae* pathotype Triticum. "
            "It leads to bleaching of spikes and can cause total yield loss under humid, warm conditions."
        ),
        "treatment": [
            "Avoid late sowing and excessive nitrogen use.",
            "Apply triazole fungicides (e.g., tebuconazole) at heading stage.",
            "Use resistant cultivars and destroy infected residues.",
            "Avoid wheat after rice to reduce spore carryover.",
        ],
    },
    "Unknown": {
        "description": (
            "This image does not appear to be a recognizable wheat disease. "
            "The model has low confidence in its prediction. Please ensure you upload "
            "a clear, well-lit image of wheat leaves, stems, or kernels showing disease symptoms."
        ),
        "treatment": [
            "Verify the image shows wheat plant parts (not other crops or objects).",
            "Ensure good lighting and focus quality.",
            "Capture close-up images of affected areas.",
            "Try uploading a different image if available.",
        ],
    },
}
# --------------------------------------------------------------------

st.set_page_config(page_title="🌾 Wheat Disease Detector", page_icon="🌿", layout="centered")

st.title("🌾 Wheat Disease Detector")
st.write("Upload a wheat leaf image to identify possible diseases and get treatment suggestions.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Resize display size
    display_width = 300
    aspect_ratio = img.height / img.width
    img_resized = img.resize((display_width, int(display_width * aspect_ratio)))

    st.image(img_resized, caption="Uploaded Image", use_container_width=False)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing image..."):
            files = {"file": ("leaf.jpg", buf, "image/jpeg")}
            try:
                response = requests.post(API_URL, files=files, timeout=15)
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Could not connect to backend: {e}")
                st.stop()

            if response.status_code == 200:
                result = response.json()
                label = result["prediction"]
                confidence = result["confidence"]
                message = result.get("message", "")

                # Display prediction with appropriate styling
                if label == "Unknown":
                    st.warning(f"⚠️ **Prediction:** {label}")
                    st.warning(f"**Confidence:** {confidence}%")
                    if message:
                        st.info(message)
                else:
                    st.success(f"✅ **Predicted:** {label}")
                    st.success(f"**Confidence:** {confidence}%")

                if label in DISEASE_INFO:
                    info = DISEASE_INFO[label]
                    st.markdown("### 🧬 About the disease")
                    st.info(info["description"])

                    st.markdown("### 🌿 Suggested Actions")
                    for step in info["treatment"]:
                        st.write(f"- {step}")
                else:
                    st.warning("No additional information found for this disease label.")
            else:
                st.error("Server error: " + response.text)
