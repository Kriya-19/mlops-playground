import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



import streamlit as st
from app.utils.api_client import upload_model, backend_health

st.set_page_config(page_title="Upload Model", page_icon="📤", layout="wide")

st.title("📤 Upload Model")
st.markdown("Upload a trained scikit-learn model (`.pkl` or `.joblib`) to deploy it as a live API.")
st.markdown("---")

if not backend_health():
    st.error("🔴 Backend is offline. Please start it before uploading.")
    st.stop()

with st.form("upload_form", clear_on_submit=True):
    st.markdown("### Model Details")

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.text_input(
            "Model Name *",
            placeholder="e.g. IrisClassifier, FraudDetector",
            max_chars=100,
        )
    with col2:
        description = st.text_input(
            "Description",
            placeholder="e.g. RandomForest trained on Iris dataset",
            max_chars=255,
        )

    uploaded_file = st.file_uploader(
        "Choose model file *",
        type=["pkl", "joblib"],
        help="Max 100MB. Must be a scikit-learn compatible model with a predict() method.",
    )

    submitted = st.form_submit_button("🚀 Upload & Validate", use_container_width=True)

if submitted:
    if not model_name.strip():
        st.error("Model name is required.")
        st.stop()
    if uploaded_file is None:
        st.error("Please select a model file.")
        st.stop()

    with st.spinner(f"Uploading and validating **{model_name}**..."):
        try:
            result = upload_model(
                file_bytes=uploaded_file.getvalue(),
                filename=uploaded_file.name,
                name=model_name.strip(),
                description=description.strip(),
            )
            st.success(f"✅ {result['message']}")
            st.markdown("#### Upload Result")
            col1, col2, col3 = st.columns(3)
            col1.metric("Model ID", result["model_id"][:8] + "...")
            col2.metric("Status", result["status"].upper())
            col3.metric("Name", result["name"])
            st.info("👉 Go to **Deployed Models** to deploy this model as a live API.")

        except Exception as e:
            err = str(e)
            if "400" in err:
                st.error(f"❌ Validation failed: {err}")
            elif "422" in err:
                st.error("❌ Invalid request format.")
            else:
                st.error(f"❌ Upload failed: {err}")

st.markdown("---")
st.markdown("### ✅ Requirements for Upload")
st.markdown("""
- File format: `.pkl` or `.joblib`
- Must be a **scikit-learn compatible** model
- Must have a `predict()` method
- Maximum file size: **100MB**
- Trained with Python 3.x and scikit-learn
""")

st.markdown("### 🧪 Don't have a model? Create one:")
st.code("""
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(np.random.rand(100, 4), np.random.randint(0, 2, 100))
joblib.dump(clf, "my_model.pkl")
print("my_model.pkl ready to upload!")
""", language="python")