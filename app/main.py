import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.utils.api_client import backend_health

st.set_page_config(
    page_title="MLOps Playground",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("## 🚀 MLOps Playground")
    st.markdown("---")
    status = backend_health()
    if status:
        st.success("● Backend Online", icon="✅")
    else:
        st.error("● Backend Offline", icon="🔴")
        st.warning("Run in another terminal:\n```\nuvicorn backend.main:app --port 8000 --app-dir .\n```")
    st.markdown("---")
    st.caption("Python 3.12 · FastAPI · Docker · Prometheus")

st.title("🚀 AI Model Deployment Playground")
st.markdown("#### End-to-end MLOps platform — upload, deploy, monitor ML models")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Platform", "MLOps v1.0")
col2.metric("Backend", "Online ✅" if status else "Offline 🔴")
col3.metric("Model Format", ".pkl / .joblib")
col4.metric("Deployment", "Docker Containers")

st.markdown("---")
st.markdown("""
### How it works
| Step | Action | Description |
|------|--------|-------------|
| 1️⃣ | **Upload** | Upload a trained `.pkl` or `.joblib` sklearn model |
| 2️⃣ | **Validate** | Backend checks format, integrity, and predict interface |
| 3️⃣ | **Deploy** | Model is containerized with Docker and launched as an API |
| 4️⃣ | **Infer** | Send features to `/predict` and get predictions back |
| 5️⃣ | **Monitor** | Prometheus + Grafana track latency, errors, and traffic |

### Quick Start
👈 Use the sidebar to navigate. Start with **01 Upload Model**.
""")
