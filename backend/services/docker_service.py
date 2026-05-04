import os
import asyncio
import shutil
import logging
import socket
from pathlib import Path

logger = logging.getLogger(__name__)

DEPLOY_MODE      = os.getenv("DEPLOY_MODE", "stub")
UPLOAD_DIR       = os.getenv("UPLOAD_DIR", "./deployed_models")
TEMPLATE_DIR     = Path(__file__).resolve().parents[2] / "model_template"
PORT_RANGE_START = int(os.getenv("PORT_RANGE_START", "8100"))
PORT_RANGE_END   = int(os.getenv("PORT_RANGE_END", "8199"))


# ── Port helpers ──────────────────────────────────────────────────────────────

def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("localhost", port)) != 0


def _find_free_port() -> int:
    for port in range(PORT_RANGE_START, PORT_RANGE_END + 1):
        if _is_port_free(port):
            return port
    raise RuntimeError(
        f"No free ports available in range {PORT_RANGE_START}–{PORT_RANGE_END}."
    )


# ── Async shell helper ────────────────────────────────────────────────────────

async def _run(cmd: str) -> tuple[int, str, str]:
    """Run a shell command asynchronously, return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode().strip(), stderr.decode().strip()


# ── Scaffold model directory ──────────────────────────────────────────────────

def _scaffold_model_dir(model_id: str, file_path: str) -> Path:
    """
    Copy serve.py, Dockerfile, requirements.txt into the model's deploy folder.
    The model file is already there (saved during upload).
    """
    model_dir = Path(UPLOAD_DIR) / model_id

    for fname in ["serve.py", "Dockerfile", "requirements.txt"]:
        src = TEMPLATE_DIR / fname
        dst = model_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)
            logger.info(f"Copied {fname} → {dst}")

    # Rename model file to model.pkl for consistency inside container
    existing = list(model_dir.glob("model.*"))
    if existing and existing[0].name != "model.pkl":
        existing[0].rename(model_dir / "model.pkl")
        logger.info(f"Renamed {existing[0].name} → model.pkl")

    return model_dir


# ── Public API ────────────────────────────────────────────────────────────────

async def deploy_model_container(
    model_id: str, file_path: str, model_name: str
) -> tuple[int, str]:
    """
    Build and run a Docker container for the given model.
    Returns (host_port, container_id).
    """
    if DEPLOY_MODE == "stub":
        logger.warning("DEPLOY_MODE=stub — skipping real Docker deploy.")
        await asyncio.sleep(1)
        import random
        return random.randint(PORT_RANGE_START, PORT_RANGE_END), f"stub_{model_id[:8]}"

    model_dir = _scaffold_model_dir(model_id, file_path)
    image_tag = f"mlops-model-{model_id[:8]}:latest"
    container_name = f"mlops-{model_id[:8]}"
    port = _find_free_port()

    # ── Build Docker image ───────────────────────────────────────────────────
    logger.info(f"Building Docker image: {image_tag}")
    rc, out, err = await _run(
        f"docker build -t {image_tag} {model_dir}"
    )
    if rc != 0:
        raise RuntimeError(f"Docker build failed:\n{err}")
    logger.info(f"✅ Image built: {image_tag}")

    # ── Run container ────────────────────────────────────────────────────────
    run_cmd = (
        f"docker run -d "
        f"--name {container_name} "
        f"-p {port}:8000 "
        f"-e MODEL_ID={model_id} "
        f"-e MODEL_NAME='{model_name}' "
        f"-e MODEL_PATH=/app/model.pkl "
        f"--restart unless-stopped "
        f"{image_tag}"
    )
    logger.info(f"Starting container on port {port}...")
    rc, container_id, err = await _run(run_cmd)
    if rc != 0:
        raise RuntimeError(f"Docker run failed:\n{err}")

    # ── Wait for health check ────────────────────────────────────────────────
    await _wait_for_health(port)

    logger.info(f"✅ Container running | ID: {container_id[:12]} | Port: {port}")
    return port, container_id


async def stop_model_container(container_id: str) -> None:
    """Stop and remove a running model container."""
    if DEPLOY_MODE == "stub":
        await asyncio.sleep(0.5)
        return

    logger.info(f"Stopping container: {container_id[:12]}")
    rc, _, err = await _run(f"docker stop {container_id}")
    if rc != 0:
        logger.warning(f"docker stop warning: {err}")

    rc, _, err = await _run(f"docker rm {container_id}")
    if rc != 0:
        logger.warning(f"docker rm warning: {err}")
    logger.info(f"✅ Container stopped and removed.")


async def _wait_for_health(port: int, retries: int = 10, delay: float = 2.0):
    """Poll /health until the container responds or retries are exhausted."""
    import urllib.request
    import urllib.error

    for attempt in range(retries):
        await asyncio.sleep(delay)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            logger.info(f"✅ Health check passed on port {port}")
            return
        except Exception:
            logger.info(f"Health check attempt {attempt + 1}/{retries}...")

    raise RuntimeError(
        f"Container on port {port} did not become healthy after {retries * delay}s."
    )