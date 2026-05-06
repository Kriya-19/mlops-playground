import requests
import os
import time


def _post_with_retry(url: str, *, json: dict, timeout: int, attempts: int = 3, delay: float = 0.75) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(url, json=json, timeout=timeout)
            if resp.status_code in {502, 503, 504} and attempt < attempts:
                time.sleep(delay * attempt)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(delay * attempt)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Request failed without an exception: {url}")

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
    # Use backend proxy to avoid container-localhost networking issues
    resp = _post_with_retry(
        f"{BASE_URL}/models/proxy/predict",
        json={"port": port, "features": features},
        timeout=15,
    )
    return resp.json()


def model_health(port: int) -> dict:
    # Use backend proxy to avoid container-localhost networking issues
    resp = requests.get(f"{BASE_URL}/models/proxy/health?port={port}", timeout=5)
    resp.raise_for_status()
    return resp.json()


def backend_health() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False