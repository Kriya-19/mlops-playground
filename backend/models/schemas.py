from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class ModelStatus(str, Enum):
    uploaded = "uploaded"
    validating = "validating"
    deploying = "deploying"
    running = "running"
    failed = "failed"
    stopped = "stopped"


class ModelUploadResponse(BaseModel):
    model_id: str
    name: str
    status: ModelStatus
    message: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelRecord(BaseModel):
    model_id: str
    name: str
    description: Optional[str] = None
    status: ModelStatus
    port: Optional[int] = None
    container_id: Optional[str] = None
    file_path: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelListResponse(BaseModel):
    models: list[ModelRecord]
    total: int


class DeployRequest(BaseModel):
    model_id: str


class DeployResponse(BaseModel):
    model_id: str
    status: ModelStatus
    port: Optional[int] = None
    endpoint: Optional[str] = None
    message: str


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="List of feature values for inference")


class HealthResponse(BaseModel):
    status: str
    model_id: str
    uptime: float