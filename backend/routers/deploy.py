from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
import logging

from backend.database import get_db, MLModel
from backend.models.schemas import DeployRequest, DeployResponse, ModelStatus
from backend.services.docker_service import deploy_model_container, stop_model_container
from backend.services.sns_service import send_alert

router = APIRouter(prefix="/deploy", tags=["Deployment"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=DeployResponse)
async def deploy_model(request: DeployRequest, db: AsyncSession = Depends(get_db)):
    """Deploy a validated model as a containerized API service."""
    result = await db.execute(
        select(MLModel).where(MLModel.model_id == request.model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    if model.status == ModelStatus.running:
        return DeployResponse(
            model_id=model.model_id,
            status=ModelStatus.running,
            port=model.port,
            endpoint=f"http://localhost:{model.port}",
            message="Model is already deployed and running.",
        )

    if model.status not in [ModelStatus.uploaded, ModelStatus.failed, ModelStatus.stopped]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot deploy model in '{model.status}' state.",
        )

    # Update status to deploying
    model.status = ModelStatus.deploying
    model.updated_at = datetime.now(timezone.utc)
    await db.commit()

    # Trigger Docker deployment (Step 3 will fill this out fully)
    try:
        port, container_id = await deploy_model_container(
            model_id=model.model_id,
            file_path=model.file_path,
            model_name=model.name,
        )
        model.status = ModelStatus.running
        model.port = port
        model.container_id = container_id
        model.updated_at = datetime.now(timezone.utc)
        await db.commit()

        try:
            await send_alert(
                subject=f"MLOps model deployed: {model.name}",
                message=(
                    f"Model deployed successfully.\n\n"
                    f"Model: {model.name}\n"
                    f"Model ID: {model.model_id}\n"
                    f"Port: {port}\n"
                    f"Endpoint: http://localhost:{port}\n"
                    f"Status: running"
                ),
            )
        except Exception as alert_exc:
            logger.warning(f"SNS deployment alert skipped: {alert_exc}")

        return DeployResponse(
            model_id=model.model_id,
            status=ModelStatus.running,
            port=port,
            endpoint=f"http://localhost:{port}",
            message=f"Model '{model.name}' deployed successfully on port {port}.",
        )
    except Exception as e:
        model.status = ModelStatus.failed
        model.updated_at = datetime.now(timezone.utc)
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")


@router.post("/stop/{model_id}", response_model=DeployResponse)
async def stop_model(model_id: str, db: AsyncSession = Depends(get_db)):
    """Stop a running deployed model container."""
    result = await db.execute(select(MLModel).where(MLModel.model_id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")

    if model.status != ModelStatus.running:
        raise HTTPException(
            status_code=400, detail=f"Model is not running (current: {model.status})."
        )

    try:
        await stop_model_container(model.container_id)
        model.status = ModelStatus.stopped
        model.updated_at = datetime.now(timezone.utc)
        await db.commit()
        return DeployResponse(
            model_id=model.model_id,
            status=ModelStatus.stopped,
            message=f"Model '{model.name}' stopped successfully.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop: {str(e)}")