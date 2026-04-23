"""
API Routes
"""
import io
import time
import base64
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from .dependencies import get_pipeline

router = APIRouter()


# --- Schemas ---

class RecognizeResponse(BaseModel):
    plate: str
    confidence: float
    region: str
    bbox: list[int]
    valid: bool
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    regions: list[str]


# --- Endpoints ---

@router.get("/health", response_model=HealthResponse)
async def health():
    pipeline = get_pipeline()
    regions = pipeline["postprocessor"]._registry.list_regions()
    return {"status": "ok", "regions": regions}


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    image: UploadFile = File(...),
    region: str = "UZB",
):
    t0 = time.perf_counter()
    pipeline = get_pipeline()

    # 1. Читаем изображение
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, "Cannot decode image")

    # 2. Детекция
    detections = pipeline["detector"].detect(frame)
    if not detections:
        raise HTTPException(422, "No license plate detected")

    best = detections[0]

    # 3. Препроцессинг
    prep = pipeline["preprocessor"].process(best.crop)

    # 4. Распознавание
    rec = pipeline["recognizer"].process(prep.image)

    # 5. Постобработка
    validated_text, is_valid = pipeline["postprocessor"].process(rec.text, region)

    # 6. Логирование низкого confidence
    cfg = pipeline.get("config", {})
    threshold = cfg.get("logging", {}).get("low_confidence_threshold", 0.6)
    if rec.confidence < threshold:
        _log_unknown(frame, best.bbox, validated_text, rec.confidence)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Recognized: {validated_text!r} conf={rec.confidence:.2f} time={elapsed_ms:.1f}ms")

    return RecognizeResponse(
        plate=validated_text,
        confidence=round(rec.confidence, 4),
        region=region.upper(),
        bbox=best.bbox,
        valid=is_valid,
        processing_time_ms=round(elapsed_ms, 2),
    )


def _log_unknown(image, bbox, text, confidence):
    """Сохранить кроп с низким confidence для дообучения."""
    import os
    from datetime import datetime
    os.makedirs("logs/unknowns", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"logs/unknowns/{ts}_{text}_conf{confidence:.2f}.jpg"
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(fname, crop)
