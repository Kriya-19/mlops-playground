import os
import asyncio
import shutil
import logging
import socket
import requests
import subprocess
import time
from pathlib import Path

try:
    import docker
    DOCKER_CLIENT = docker.from_env()
except Exception as e:
    logging.warning(f"Docker client initialization failed: {e}, falling back to subprocess")
    DOCKER_CLIENT = None

logger = logging.getLogger(__name__)

# Read at call time, not import time — fixes the cached "stub" bug
def _get_deploy_mode() -> str:
    return os.environ.get("DEPLOY_MODE", "docker")

UPLOAD_DIR       = os.getenv("UPLOAD_DIR", "./deployed_models")
TEMPLATE_DIR     = Path(__file__).resolve().parents[2] / "model_template"
PORT_RANGE_START = int(os.getenv("PORT_RANGE_START", "8100"))
PORT_RANGE_END   = int(os.getenv("PORT_RANGE_END", "8199"))
WORKSPACE_PATH   = os.getenv("WORKSPACE_PATH", "/workspace")
NETWORK_NAME     = "mlops-playground_mlops-net"
PROMETHEUS_URL   = "http://mlops-prometheus:9090"


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
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode().strip(), stderr.decode().strip()


# ── Scaffold model directory ──────────────────────────────────────────────────

def _scaffold_model_dir(model_id: str, file_path: str) -> Path:
    model_dir = Path(UPLOAD_DIR) / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["serve.py", "Dockerfile", "requirements.txt"]:
        src = TEMPLATE_DIR / fname
        dst = model_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            logger.info(f"Copied {fname} → {dst}")

    # Ensure model file is named model.pkl
    existing = [f for f in model_dir.glob("model.*") if f.suffix != ".py"]
    if existing and existing[0].name != "model.pkl":
        existing[0].rename(model_dir / "model.pkl")
        logger.info(f"Renamed {existing[0].name} → model.pkl")

    return model_dir


# ── Prometheus reload helper ──────────────────────────────────────────────────

async def _reload_prometheus():
    """Hot-reload Prometheus config so it picks up the new container immediately."""
    try:
        resp = requests.post(f"{PROMETHEUS_URL}/-/reload", timeout=5)
        if resp.status_code == 200:
            logger.info("✅ Prometheus reloaded — new model target auto-discovered")
        else:
            logger.warning(f"Prometheus reload returned {resp.status_code}")
    except Exception as e:
        logger.warning(f"Could not reload Prometheus: {e}")


# ── Git commit helper ─────────────────────────────────────────────────────────

async def _git_commit(model_name: str, model_id: str, port: int):
    """Auto-commit a deployment record to git."""
    try:
        short = model_id[:8]
        rc, _, err = await _run(f'git -C "{WORKSPACE_PATH}" add -A')
        if rc != 0:
            logger.warning(f"git add failed: {err}")
            return
        
        msg = f"deploy: {model_name} ({short}) → port {port} [auto]"
        rc, _, err = await _run(f'git -C "{WORKSPACE_PATH}" commit -m "{msg}"')
        if rc == 0:
            logger.info(f"✅ Git commit created for {model_name} deployment")
        else:
            logger.debug(f"git commit skipped: {err}")
    except Exception as e:
        logger.warning(f"Git commit error: {e}")


# ── Public API ────────────────────────────────────────────────────────────────

async def deploy_model_container(
    model_id: str, file_path: str, model_name: str
) -> tuple[int, str]:
    deploy_mode = _get_deploy_mode()  # read fresh every time, not from cache
    logger.info(f"deploy_model_container called | DEPLOY_MODE={deploy_mode}")

    if deploy_mode == "stub":
        logger.warning("DEPLOY_MODE=stub — skipping real Docker deploy.")
        await asyncio.sleep(1)
        import random
        return random.randint(PORT_RANGE_START, PORT_RANGE_END), f"stub_{model_id[:8]}"

    model_dir = _scaffold_model_dir(model_id, file_path)
    image_tag      = f"mlops-model-{model_id[:8]}:latest"
    container_name = f"mlops-model-{model_id[:8]}"

    # Remove old container if it exists (re-deploy case)
    await _run(f"docker rm -f {container_name} 2>/dev/null || true")

    port = _find_free_port()

    # ── Build Docker image ───────────────────────────────────────────────────
    logger.info(f"Building Docker image: {image_tag} from {model_dir}")
    rc, out, err = await _run(f"docker build -t {image_tag} {model_dir}")
    if rc != 0:
        logger.error(f"Docker build failed:\nSTDOUT: {out}\nSTDERR: {err}")
        raise RuntimeError(f"Docker build failed:\n{err}")
    logger.info(f"✅ Image built: {image_tag}")

    # ── Detect project network ────────────────────────────────────────────────
    try:
        rc, out, err = await _run("docker network ls --format '{{.Name}}'")
        network_candidates = [l.strip() for l in out.splitlines() if l.strip()]
        network = None
        # Get the network the backend container is on
        rc_bn, backend_network_out, _ = await _run("docker inspect mlops-backend --format='{{range .NetworkSettings.Networks}}{{.Name}}{{end}}'")
        backend_network = backend_network_out.strip() if backend_network_out.strip() else None
        
        # Use backend network if available
        if backend_network and backend_network in network_candidates:
            network = backend_network
        # Fallback: look for mlops-net
        if network is None:
            for n in network_candidates:
                if n.endswith('_mlops-net'):
                    network = n
                    break
        if network is None and network_candidates:
            network = next((n for n in network_candidates if n != 'bridge'), network_candidates[0])
    except Exception:
        network = 'bridge'

    net_arg = f"--network {network} " if network else ''
    logger.info(f"Using network: {network}")

    # ── Run container with labels for Prometheus SD ───────────────────────────
    labels = " ".join([
        f'--label model_id={model_id}',
        f'--label model_name="{model_name}"',
        f'--label prometheus_job=mlops-models',
    ])

    run_cmd = (
        f"docker run -d "
        f"--name {container_name} "
        f"{net_arg}"
        f"-p {port}:8000 "
        f"-e MODEL_ID={model_id} "
        f"-e MODEL_NAME='{model_name}' "
        f"-e MODEL_PATH=/app/model.pkl "
        f"--restart unless-stopped "
        f"{labels} "
        f"{image_tag}"
    )
    logger.info(f"Starting container on port {port}...")
    rc, container_id, err = await _run(run_cmd)
    if rc != 0:
        logger.error(f"Docker run failed:\nSTDERR: {err}")
        raise RuntimeError(f"Docker run failed:\n{err}")

    # ── Wait for health ──────────────────────────────────────────────────────
    await _wait_for_health(container_name)

    # ── Post-deploy: reload Prometheus and commit to git ─────────────────────
    await _reload_prometheus()
    await _git_commit(model_name, model_id, port)

    logger.info(f"✅ Container running | ID: {container_id[:12]} | Port: {port}")
    return port, container_id


async def stop_model_container(container_id: str) -> None:
    deploy_mode = _get_deploy_mode()
    if deploy_mode == "stub":
        await asyncio.sleep(0.5)
        return

    logger.info(f"Stopping container: {container_id[:12]}")
    rc, _, err = await _run(f"docker stop {container_id}")
    if rc != 0:
        logger.warning(f"docker stop warning: {err}")

    rc, _, err = await _run(f"docker rm {container_id}")
    if rc != 0:
        logger.warning(f"docker rm warning: {err}")
    logger.info("✅ Container stopped and removed.")


async def _wait_for_health(container_name: str, retries: int = 15, delay: float = 3.0):
    import urllib.request

    for attempt in range(retries):
        await asyncio.sleep(delay)
        try:
            urllib.request.urlopen(f"http://{container_name}:8000/health", timeout=3)
            logger.info(f"✅ Health check passed: {container_name}")
            return
        except Exception:
            logger.info(f"Health check attempt {attempt + 1}/{retries} for {container_name}...")

    raise RuntimeError(
        f"Container {container_name} did not become healthy after {retries * delay}s."
    )