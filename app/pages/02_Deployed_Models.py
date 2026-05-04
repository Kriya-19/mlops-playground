import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import streamlit as st
import pandas as pd
from app.utils.api_client import (
    list_models, deploy_model, stop_model,
    delete_model, predict, model_health, backend_health,
)

st.set_page_config(page_title="Deployed Models", page_icon="🖥️", layout="wide")

st.title("🖥️ Deployed Models")
st.markdown("Manage your uploaded and deployed ML models.")
st.markdown("---")

if not backend_health():
    st.error("🔴 Backend is offline.")
    st.stop()

# ── Auto-refresh toggle ────────────────────────────────────────────────────
col_r1, col_r2 = st.columns([6, 1])
with col_r2:
    if st.button("🔄 Refresh"):
        st.rerun()

# ── Fetch models ───────────────────────────────────────────────────────────
try:
    data = list_models()
    models = data.get("models", [])
except Exception as e:
    st.error(f"Failed to fetch models: {e}")
    st.stop()

if not models:
    st.info("No models uploaded yet. Go to **Upload Model** to get started.")
    st.stop()

st.markdown(f"**{data['total']} model(s) found**")

# ── Status badge helper ────────────────────────────────────────────────────
STATUS_COLORS = {
    "uploaded":   "🔵",
    "validating": "🟡",
    "deploying":  "🟠",
    "running":    "🟢",
    "failed":     "🔴",
    "stopped":    "⚫",
}

# ── Model cards ───────────────────────────────────────────────────────────
for model in models:
    status_icon = STATUS_COLORS.get(model["status"], "⚪")
    with st.expander(
        f"{status_icon} **{model['name']}** — `{model['model_id'][:8]}...` — {model['status'].upper()}",
        expanded=model["status"] == "running",
    ):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", f"{status_icon} {model['status'].upper()}")
        col2.metric("Model ID", model["model_id"][:12] + "...")
        col3.metric("Port", model["port"] if model["port"] else "—")
        col4.metric("Endpoint", f"localhost:{model['port']}" if model["port"] else "Not deployed")

        if model.get("description"):
            st.caption(f"📝 {model['description']}")

        st.markdown(f"**Created:** {model['created_at'][:19].replace('T', ' ')}")

        # ── Action buttons ─────────────────────────────────────────────────
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        # Deploy button
        with btn_col1:
            if model["status"] in ["uploaded", "failed", "stopped"]:
                if st.button("🚀 Deploy", key=f"deploy_{model['model_id']}",
                             use_container_width=True):
                    with st.spinner(f"Deploying {model['name']}..."):
                        try:
                            result = deploy_model(model["model_id"])
                            st.success(f"✅ {result['message']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Deployment failed: {e}")

        # Stop button
        with btn_col2:
            if model["status"] == "running":
                if st.button("⏹️ Stop", key=f"stop_{model['model_id']}",
                             use_container_width=True):
                    with st.spinner("Stopping..."):
                        try:
                            stop_model(model["model_id"])
                            st.success("Model stopped.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Stop failed: {e}")

        # Health check
        with btn_col3:
            if model["status"] == "running" and model.get("port"):
                if st.button("💓 Health", key=f"health_{model['model_id']}",
                             use_container_width=True):
                    try:
                        h = model_health(model["port"])
                        st.success(f"Status: {h['status']} | Uptime: {h['uptime_seconds']}s")
                    except Exception:
                        st.error("Health check failed — container may be down.")

        # Delete button
        with btn_col4:
            if model["status"] != "running":
                if st.button("🗑️ Delete", key=f"delete_{model['model_id']}",
                             use_container_width=True, type="secondary"):
                    try:
                        delete_model(model["model_id"])
                        st.success("Deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

        # ── Inference panel (only when running) ───────────────────────────
        if model["status"] == "running" and model.get("port"):
            st.markdown("---")
            st.markdown("#### 🔮 Try Inference")
            features_input = st.text_input(
                "Feature values (comma-separated)",
                placeholder="5.1, 3.5, 1.4, 0.2",
                key=f"features_{model['model_id']}",
                help="Enter comma-separated float values matching your model's input shape.",
            )
            if st.button("⚡ Predict", key=f"predict_{model['model_id']}"):
                if not features_input.strip():
                    st.warning("Enter feature values first.")
                else:
                    try:
                        features = [float(x.strip()) for x in features_input.split(",")]
                        with st.spinner("Running inference..."):
                            result = predict(model["port"], features)
                        rcol1, rcol2, rcol3 = st.columns(3)
                        rcol1.metric("Prediction", result["prediction"])
                        rcol2.metric("Latency", f"{result['latency_ms']} ms")
                        if result.get("predict_proba"):
                            rcol3.metric("Confidence", f"{max(result['predict_proba'][0]):.2%}")
                    except ValueError:
                        st.error("Invalid input — use comma-separated numbers only.")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")