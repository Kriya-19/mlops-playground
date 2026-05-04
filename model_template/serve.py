import os
import time
import joblib
import pickle
import numpy as np
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
    REGISTRY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config from environment ──────────────────────────────────────────────────
MODEL_ID   = os.getenv("MODEL_ID", "unknown")
MODEL_NAME = os.getenv("MODEL_NAME", "unknown")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.pkl")
START_TIME = time.time()

# ── Prometheus metrics ────────────────────────────────────────────────────────
PREDICT_COUNT = Counter(
    "model_predict_total",
    "Total prediction requests",
    ["model_id", "status"],
)
PREDICT_LATENCY = Histogram(
    "model_predict_latency_seconds",
    "Prediction latency in seconds",
    ["model_id"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_UPTIME = Gauge(
    "model_uptime_seconds",
    "Seconds since model container started",
    ["model_id"],
)

# ── Global model holder ───────────────────────────────────────────────────────
model_store: dict = {}


def load_model():
    """Load model from disk into memory."""
    logger.info(f"Loading model from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    logger.info(f"✅ Model loaded: {type(model).__name__}")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_store["model"] = load_model()
    logger.info(f"🚀 Model server ready | ID: {MODEL_ID} | Name: {MODEL_NAME}")
    yield
    model_store.clear()
    logger.info("🛑 Model server shut down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=f"Model: {MODEL_NAME}",
    description=f"Auto-generated inference API for model ID: {MODEL_ID}",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Flat list of feature values. e.g. [5.1, 3.5, 1.4, 0.2]",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )


class PredictResponse(BaseModel):
    model_id: str
    model_name: str
    prediction: list
    predict_proba: list | None = None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_name: str
    model_class: str
    uptime_seconds: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check — confirms model is loaded and server is running."""
    model = model_store.get("model")
    uptime = time.time() - START_TIME
    MODEL_UPTIME.labels(model_id=MODEL_ID).set(uptime)
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        model_class=type(model).__name__ if model else "None",
        uptime_seconds=round(uptime, 2),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Run inference on the deployed model.
    Send a flat list of feature values matching the model's expected input.
    """
    model = model_store.get("model")
    if model is None:
        PREDICT_COUNT.labels(model_id=MODEL_ID, status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start = time.time()
    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X).tolist()

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X).tolist()
            except Exception:
                proba = None

        latency_ms = round((time.time() - start) * 1000, 3)

        PREDICT_COUNT.labels(model_id=MODEL_ID, status="success").inc()
        PREDICT_LATENCY.labels(model_id=MODEL_ID).observe(time.time() - start)

        return PredictResponse(
            model_id=MODEL_ID,
            model_name=MODEL_NAME,
            prediction=prediction,
            predict_proba=proba,
            latency_ms=latency_ms,
        )

    except ValueError as e:
        PREDICT_COUNT.labels(model_id=MODEL_ID, status="error").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Feature shape mismatch: {str(e)}. "
                   f"Check the number of features your model expects.",
        )
    except Exception as e:
        PREDICT_COUNT.labels(model_id=MODEL_ID, status="error").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus scrape endpoint for this model container."""
    MODEL_UPTIME.labels(model_id=MODEL_ID).set(time.time() - START_TIME)
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["Info"])
async def root():
    model = model_store.get("model")
    info: dict = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "status": "running",
        "endpoints": ["/predict", "/health", "/metrics", "/docs"],
    }
    if model and hasattr(model, "n_features_in_"):
        info["expected_features"] = int(model.n_features_in_)
    if model and hasattr(model, "classes_"):
        info["classes"] = model.classes_.tolist()
    return info