import os
import time
import joblib
import pickle
import numpy as np
import logging
from typing import Any
from collections.abc import Sequence
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


def _normalize_features(features: Sequence) -> np.ndarray:
    """Convert nested feature payloads into a single flat numeric row."""

    def _flatten(values: Sequence) -> list:
        flattened: list = []
        for value in values:
            if isinstance(value, (list, tuple, np.ndarray)):
                flattened.extend(_flatten(value))
            else:
                flattened.append(value)
        return flattened

    flat_features = _flatten(features)

    try:
        return np.asarray([float(value) for value in flat_features], dtype=np.float64).reshape(1, -1)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Features must be a flat list of numeric values. "
                f"Received: {features!r}"
            ),
        ) from exc


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
    features: list[Any] = Field(
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
        X = _normalize_features(request.features)

        expected_features = getattr(model, "n_features_in_", None)
        if expected_features is not None and X.shape[1] != expected_features:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Expected {expected_features} features, but received {X.shape[1]}. "
                    "Check the comma-separated input length."
                ),
            )

        prediction = model.predict(X).tolist()

        # Try to map prediction indices/classes to human-readable names
        def _map_class_label(cls):
            """Convert class label to string, handling numpy types and known mappings."""
            # If it's already a string, return it
            if isinstance(cls, str):
                return cls
            # For numeric indices, try known dataset mappings (e.g., Iris)
            if isinstance(cls, (int, float, np.integer, np.floating)):
                idx = int(cls)
                # Iris dataset mapping (common for demos)
                iris_classes = ['setosa', 'versicolor', 'virginica']
                if 0 <= idx < len(iris_classes):
                    return iris_classes[idx]
            # Try to convert numpy types to python types
            try:
                return cls.item()
            except (AttributeError, TypeError):
                return str(cls)

        # Apply mapping to predictions
        if hasattr(model, "classes_"):
            try:
                if len(prediction) == 1:
                    prediction = [_map_class_label(prediction[0])]
                else:
                    prediction = [_map_class_label(c) for c in prediction]
            except Exception:
                pass  # fallback to raw prediction

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
    except HTTPException:
        raise
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