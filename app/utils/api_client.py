import requests
import os

BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def upload_model(file_bytes: bytes, filename: str, name: str, description: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/models/upload",
        files={"file": (filename, file_bytes, "application/octet-stream")},
        data={"name": name, "description": description},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def list_models() -> dict:
    resp = requests.get(f"{BASE_URL}/models/", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_model(model_id: str) -> dict:
    resp = requests.get(f"{BASE_URL}/models/{model_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def delete_model(model_id: str) -> None:
    resp = requests.delete(f"{BASE_URL}/models/{model_id}", timeout=10)
    resp.raise_for_status()


def deploy_model(model_id: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/deploy/",
        json={"model_id": model_id},
        timeout=120,  # Docker build can take a while
    )
    resp.raise_for_status()
    return resp.json()


def stop_model(model_id: str) -> dict:
    resp = requests.post(f"{BASE_URL}/deploy/stop/{model_id}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def predict(port: int, features: list[float]) -> dict:
    resp = requests.post(
        f"http://localhost:{port}/predict",
        json={"features": features},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def model_health(port: int) -> dict:
    resp = requests.get(f"http://localhost:{port}/health", timeout=5)
    resp.raise_for_status()
    return resp.json()


def backend_health() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False