import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from app.utils.api_client import list_models, backend_health

st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")

st.title("📊 Monitoring Dashboard")
st.markdown("Real-time metrics from Prometheus for all deployed models.")
st.markdown("---")

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")


def query_prometheus(query: str) -> list:
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        data = resp.json()
        if data["status"] == "success":
            return data["data"]["result"]
        return []
    except Exception:
        return []


def prometheus_live() -> bool:
    try:
        resp = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ── Backend metrics ────────────────────────────────────────────────────────
st.markdown("### 🔧 Backend API Metrics")

prom_live = bool(query_prometheus("up")) or prometheus_live()

if not prom_live:
    st.warning("⚠️ Prometheus is not running yet. It will be available after Step 5 (Docker Compose).")
    st.markdown("For now, here's what will be tracked:")
    st.markdown("""
    - `mlops_request_total` — total HTTP requests by endpoint and status
    - `mlops_request_latency_seconds` — backend API latency histogram
    - `model_predict_total` — predictions per deployed model
    - `model_predict_latency_seconds` — per-model inference latency
    - `model_uptime_seconds` — how long each model has been running
    """)
    st.info("👉 Come back to this page after **Step 5** when Prometheus is running via Docker Compose.")
else:
    # ── Live metrics ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    total_req = query_prometheus("sum(mlops_request_total)")
    total_errors = query_prometheus('sum(mlops_request_total{status_code=~"5.."})')
    avg_latency = query_prometheus("avg(mlops_request_latency_seconds_sum / mlops_request_latency_seconds_count)")

    col1.metric("Total Requests", total_req[0]["value"][1] if total_req else "0")
    col2.metric("Total Errors", total_errors[0]["value"][1] if total_errors else "0")
    col3.metric("Avg Latency", f"{float(avg_latency[0]['value'][1]):.3f}s" if avg_latency else "—")

    # ── Per-model stats ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Per-Model Metrics")

    predict_counts = query_prometheus("model_predict_total")
    if predict_counts:
        df = pd.DataFrame([
            {
                "model_id": r["metric"].get("model_id", "unknown")[:8],
                "status": r["metric"].get("status", "unknown"),
                "count": float(r["value"][1]),
            }
            for r in predict_counts
        ])
        fig = px.bar(df, x="model_id", y="count", color="status",
                     title="Predictions per Model",
                     color_discrete_map={"success": "#01696f", "error": "#a12c7b"})
        st.plotly_chart(fig, use_container_width=True)

    latency_data = query_prometheus(
        "model_predict_latency_seconds_sum / model_predict_latency_seconds_count"
    )
    if latency_data:
        df2 = pd.DataFrame([
            {
                "model_id": r["metric"].get("model_id", "unknown")[:8],
                "avg_latency_ms": float(r["value"][1]) * 1000,
            }
            for r in latency_data
        ])
        fig2 = px.bar(df2, x="model_id", y="avg_latency_ms",
                      title="Average Prediction Latency (ms)",
                      color_discrete_sequence=["#01696f"])
        st.plotly_chart(fig2, use_container_width=True)

# ── Deployed models summary ───────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Models Status Overview")

if backend_health():
    try:
        data = list_models()
        models = data.get("models", [])
        if models:
            df3 = pd.DataFrame([
                {
                    "Name": m["name"],
                    "Status": m["status"].upper(),
                    "Port": m["port"] or "—",
                    "ID": m["model_id"][:12] + "...",
                    "Created": m["created_at"][:19].replace("T", " "),
                }
                for m in models
            ])
            st.dataframe(df3, use_container_width=True, hide_index=True)

            # Status pie chart
            status_counts = df3["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig3 = px.pie(status_counts, names="Status", values="Count",
                          title="Models by Status",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No models yet.")
    except Exception as e:
        st.error(f"Could not fetch model list: {e}")
else:
    st.warning("Backend offline — cannot fetch model list.")

st.markdown("---")
st.markdown("### 🔗 Quick Links (available after Step 5)")
col1, col2 = st.columns(2)
col1.markdown("- 📈 [Grafana Dashboard](http://localhost:3000) — Visual dashboards")
col2.markdown(f"- 🔥 [Prometheus UI]({PROMETHEUS_URL}) — Raw metrics explorer")