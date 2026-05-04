import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from dotenv import load_dotenv

from backend.database import init_db
from backend.routers import models, deploy

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter(
    "mlops_request_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "mlops_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 MLOps Platform starting up...")
    await init_db()
    os.makedirs(os.getenv("UPLOAD_DIR", "./deployed_models"), exist_ok=True)
    logger.info("✅ Database initialized. Upload dir ready.")
    yield
    logger.info("🛑 MLOps Platform shutting down.")


app = FastAPI(
    title="MLOps Deployment Playground",
    description="Upload, deploy, and monitor ML models as live API services.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows Streamlit frontend on port 8501
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Prometheus middleware ---
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        return response


app.add_middleware(PrometheusMiddleware)

# --- Routers ---
app.include_router(models.router)
app.include_router(deploy.router)


# --- Core endpoints ---
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "MLOps Deployment Playground",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)