"""
Plate Detection Module — YOLOv10
Вход: numpy BGR image
Выход: List[DetectionResult]
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from loguru import logger


@dataclass
class DetectionResult:
    bbox: List[int]          # [x1, y1, x2, y2]
    confidence: float
    crop: np.ndarray         # кроп номера


class PlateDetector:
    """YOLOv10-based license plate detector."""

    def __init__(self, weights_path: str, confidence: float = 0.5, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence = confidence
        self._model = None
        self._weights_path = weights_path
        logger.info(f"PlateDetector initialized | device={self.device}")

    def _load_model(self):
        """Ленивая загрузка модели."""
        if self._model is None:
            from ultralytics import YOLO
            if not Path(self._weights_path).exists():
                raise FileNotFoundError(f"Weights not found: {self._weights_path}")
            self._model = YOLO(self._weights_path)
            self._model.to(self.device)
            logger.info(f"YOLOv10 loaded from {self._weights_path}")

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Args:
            image: BGR numpy array (H, W, 3)
        Returns:
            List of DetectionResult, sorted by confidence desc
        """
        self._load_model()
        results = self._model(image, conf=self.confidence, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                crop = image[y1:y2, x1:x2]
                detections.append(DetectionResult(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    crop=crop,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
