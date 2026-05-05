import os
import uuid
import aiofiles
import requests
import socket
import struct
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db, MLModel
from backend.models.schemas import (
    ModelUploadResponse,
    ModelListResponse,
    ModelRecord,
    ModelStatus,
)
from backend.services.validator import validate_model_file, get_model_info

router = APIRouter(prefix="/models", tags=["Models"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./deployed_models")


@router.post("/upload", response_model=ModelUploadResponse, status_code=201)
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Upload and validate a trained ML model (.pkl or .joblib)."""

    # Validate file
    contents = await validate_model_file(file)
    info = get_model_info(contents)

    # Generate unique model ID
    model_id = str(uuid.uuid4())
    model_dir = os.path.join(UPLOAD_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Save file
    ext = os.path.splitext(file.filename or "model.pkl")[-1].lower()
    file_path = os.path.join(model_dir, f"model{ext}")
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)

    # Persist to DB
    now = datetime.now(timezone.utc)
    db_model = MLModel(
        model_id=model_id,
        name=name,
        description=description or f"Model class: {info.get('class_name', 'Unknown')}",
        status=ModelStatus.uploaded,
        file_path=file_path,
        created_at=now,
        updated_at=now,
    )
    db.add(db_model)
    await db.commit()
    await db.refresh(db_model)

    return ModelUploadResponse(
        model_id=model_id,
        name=name,
        status=ModelStatus.uploaded,
        message=f"Model '{name}' uploaded and validated successfully. Class: {info.get('class_name', 'Unknown')}",
        created_at=now,
    )


@router.get("/", response_model=ModelListResponse)
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all uploaded/deployed models."""
    result = await db.execute(select(MLModel).order_by(MLModel.created_at.desc()))
    models = result.scalars().all()
    return ModelListResponse(
        models=[
            ModelRecord(
                model_id=m.model_id,
                name=m.name,
                description=m.description,
                status=m.status,
                port=m.port,
                container_id=m.container_id,
                file_path=m.file_path,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in models
        ],
        total=len(models),
    )


@router.get("/{model_id}", response_model=ModelRecord)
async def get_model(model_id: str, db: AsyncSession = Depends(get_db)):
    """Get details of a specific model."""
    result = await db.execute(
        select(MLModel).where(MLModel.model_id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return ModelRecord(
        model_id=model.model_id,
        name=model.name,
        description=model.description,
        status=model.status,
        port=model.port,
        container_id=model.container_id,
        file_path=model.file_path,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


@router.delete("/{model_id}", status_code=204)
async def delete_model(model_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a model record and its files."""
    result = await db.execute(
        select(MLModel).where(MLModel.model_id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    # Delete files
    import shutil
    model_dir = os.path.join(UPLOAD_DIR, model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    await db.delete(model)
    await db.commit()


@router.post('/proxy/predict')
async def proxy_predict(payload: dict = Body(...)) -> dict:
    """Proxy prediction requests from the frontend to model containers.

    Expects JSON: {"port": 8100, "features": [...]}
    This avoids container localhost networking issues by letting the backend
    forward requests to the host-mapped model ports.
    """
    port = payload.get('port')
    features = payload.get('features')
    if not port or features is None:
        raise HTTPException(status_code=400, detail='Missing port or features.')

    def _get_docker_host_gateway() -> str:
        """Return the host gateway IP for the container's network by reading /proc/net/route.

        Falls back to '172.17.0.1' if detection fails.
        """
        try:
            with open('/proc/net/route') as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) >= 3 and fields[1] == '00000000':
                        gw_hex = fields[2]
                        gw = socket.inet_ntoa(struct.pack('<L', int(gw_hex, 16)))
                        return gw
        except Exception:
            pass
        return '172.17.0.1'

    host_ip = _get_docker_host_gateway()
    url = f'http://{host_ip}:{port}/predict'
    try:
        resp = requests.post(url, json={'features': features}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as first_exc:
        # Fallback: try to locate a container that publishes this port and
        # forward directly to the container's internal port (8000) using its
        # container name on the compose network. This requires Docker socket
        # access from the backend container.
        import subprocess

        try:
            cp = subprocess.run(
                [
                    'docker',
                    'ps',
                    '--filter',
                    f'publish={port}',
                    '--format',
                    '{{.Names}}',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            cname = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else None
            if cname:
                alt_url = f'http://{cname}:8000/predict'
                try:
                    resp2 = requests.post(alt_url, json={'features': features}, timeout=10)
                    resp2.raise_for_status()
                    return resp2.json()
                except requests.RequestException:
                    pass
        except Exception:
            pass

        raise HTTPException(status_code=502, detail=f'Upstream request failed: {first_exc}')


@router.get('/proxy/health')
async def proxy_health(port: int) -> dict:
    """Proxy health check requests from the frontend to model containers.

    Query param: ?port=8100
    This avoids container localhost networking issues by letting the backend
    forward requests to the host-mapped model ports.
    """
    if not port:
        raise HTTPException(status_code=400, detail='Missing port parameter.')

    def _get_docker_host_gateway() -> str:
        """Return the host gateway IP for the container's network by reading /proc/net/route.

        Falls back to '172.17.0.1' if detection fails.
        """
        try:
            with open('/proc/net/route') as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) >= 3 and fields[1] == '00000000':
                        gw_hex = fields[2]
                        gw = socket.inet_ntoa(struct.pack('<L', int(gw_hex, 16)))
                        return gw
        except Exception:
            pass
        return '172.17.0.1'

    host_ip = _get_docker_host_gateway()
    url = f'http://{host_ip}:{port}/health'
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as first_exc:
        # Fallback: try to locate a container that publishes this port and
        # forward directly to the container's internal port (8000) using its
        # container name on the compose network.
        import subprocess

        try:
            cp = subprocess.run(
                [
                    'docker',
                    'ps',
                    '--filter',
                    f'publish={port}',
                    '--format',
                    '{{.Names}}',
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            cname = cp.stdout.strip().splitlines()[0] if cp.stdout.strip() else None
            if cname:
                alt_url = f'http://{cname}:8000/health'
                try:
                    resp2 = requests.get(alt_url, timeout=5)
                    resp2.raise_for_status()
                    return resp2.json()
                except requests.RequestException:
                    pass
        except Exception:
            pass

        raise HTTPException(status_code=502, detail=f'Upstream request failed: {first_exc}')
