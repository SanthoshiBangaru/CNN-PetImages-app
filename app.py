import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# ==================================================# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="PetVision AI",
    page_icon="🐾",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg,#6a11cb,#2575fc);
    color: white;
    text-align:center;
}

.metric-card{
    background-color:#f5f5f5;
    padding:10px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "model/cat_dog_cnn.keras"

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Unable to load model\n\n{e}")
    st.stop()

# ==================================================
# HERO SECTION
# ==================================================

st.markdown("""
<div class="hero">
<h1>🐾 PetVision AI</h1>
<h4>CNN-Based Dog vs Cat Image Classification System</h4>
</div>
""", unsafe_allow_html=True)

st.write("")

# ==================================================
# PROJECT OVERVIEW
# ==================================================

col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Model Type","CNN")

with col2:
    st.metric("Classes","2")

with col3:
    st.metric("Framework","TensorFlow")

st.divider()

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📋 Project Details")

    st.info("""
Dataset : PetImages

Algorithm : CNN

Framework : TensorFlow

Deployment : Streamlit

Classes :
• Cat
• Dog
""")

    st.success("Model Loaded Successfully")

# ==================================================
# FILE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload Pet Image",
    type=["jpg","jpeg","png"]
)

# ==================================================
# PREDICTION
# ==================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    left,right = st.columns([1,1])

    with left:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.write("### Image Information")

        st.write(f"Width : {image.size[0]} px")
        st.write(f"Height : {image.size[1]} px")

    # ==========================================
    # PREPROCESS
    # ==========================================

    img = image.resize((128,128))

    img_array = np.array(img)/255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    dog_prob = float(prediction)
    cat_prob = float(1-prediction)

    if prediction > 0.5:

        label = "🐶 Dog"
        confidence = dog_prob*100

    else:

        label = "🐱 Cat"
        confidence = cat_prob*100

    with right:

        st.subheader("Prediction Result")

        st.success(label)

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(
            min(int(confidence),100)
        )

    st.divider()

    # ==========================================
    # PROBABILITY CHART
    # ==========================================

    st.subheader("Class Probability Distribution")

    prob_df = pd.DataFrame({
        "Class":["Cat","Dog"],
        "Probability":[cat_prob,dog_prob]
    })

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        prob_df["Class"],
        prob_df["Probability"]
    )

    ax.set_ylabel("Probability")
    ax.set_ylim([0,1])

    st.pyplot(fig)

    # ==========================================
    # PREDICTION SUMMARY
    # ==========================================

    st.subheader("Prediction Summary")

    st.dataframe(
        prob_df,
        use_container_width=True
    )

# ==================================================
# ABOUT PROJECT
# ==================================================

st.divider()

st.subheader("About This Project")

st.write("""
This application uses a Convolutional Neural Network (CNN)
trained on the PetImages dataset to classify uploaded pet images.

### Workflow

1. Upload Image
2. Image Preprocessing
3. Feature Extraction using CNN
4. Probability Estimation
5. Final Classification

### Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Streamlit
""")

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "PetVision AI • Deep Learning Based Image Classification System"
)