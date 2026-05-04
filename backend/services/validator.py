import pickle
import joblib
import io
import os
from fastapi import UploadFile, HTTPException
import numpy as np

ALLOWED_EXTENSIONS = {".pkl", ".joblib"}
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


async def validate_model_file(file: UploadFile) -> bytes:
    """
    Validates uploaded model file:
    - Extension check (.pkl, .joblib)
    - File size check (max 100MB)
    - Pickle integrity check (can it be loaded?)
    - Basic sklearn interface check (has predict method)
    Returns raw file bytes on success.
    """
    # 1. Extension check
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Only .pkl and .joblib are allowed.",
        )

    # 2. Read file into memory (chunked to avoid RAM issues)
    contents = b""
    while chunk := await file.read(1024 * 1024):  # 1MB chunks
        contents += chunk
        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB.",
            )

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 3. Pickle integrity check
    try:
        buf = io.BytesIO(contents)
        model = joblib.load(buf)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Model file is corrupt or invalid: {str(e)}",
        )

    # 4. Interface check — must have a predict method
    if not hasattr(model, "predict"):
        raise HTTPException(
            status_code=400,
            detail="Model does not have a 'predict' method. Only scikit-learn compatible models are supported.",
        )

    # 5. Quick smoke test with dummy data
    try:
        dummy = np.zeros((1, 4))  # works for most sklearn models
        model.predict(dummy)
    except Exception:
        # Non-fatal — model might need specific input shape, that's okay
        pass

    return contents


def get_model_info(contents: bytes) -> dict:
    """Extract basic info from model bytes."""
    try:
        buf = io.BytesIO(contents)
        model = joblib.load(buf)
        info = {
            "class_name": type(model).__name__,
            "module": type(model).__module__,
            "has_predict": hasattr(model, "predict"),
            "has_predict_proba": hasattr(model, "predict_proba"),
        }
        if hasattr(model, "n_features_in_"):
            info["n_features"] = int(model.n_features_in_)
        if hasattr(model, "classes_"):
            info["n_classes"] = len(model.classes_)
        return info
    except Exception:
        return {}