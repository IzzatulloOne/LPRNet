"""
Recognition Module — LPRNet
Stub: легко заменить на CRNN или любую другую модель
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from loguru import logger


@dataclass
class RecognitionResult:
    text: str
    confidence: float
    raw_text: str   # до постобработки


class PlateRecognizer:
    """
    LPRNet-based OCR для номерных знаков.
    Реализует интерфейс: process(image) -> RecognitionResult

    ЗАМЕНА МОДЕЛИ: достаточно изменить _load_model() и _infer()
    """

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model = None
        self._weights_path = weights_path
        logger.info(f"PlateRecognizer initialized | device={self.device}")

    def _load_model(self):
        if self._model is None:
            if not Path(self._weights_path).exists():
                logger.warning(f"LPRNet weights not found: {self._weights_path} — using stub")
                self._model = "stub"
                return
            # TODO: загрузить LPRNet
            # from .lprnet_model import LPRNet
            # self._model = LPRNet(...)
            # self._model.load_state_dict(torch.load(self._weights_path))
            # self._model.eval().to(self.device)
            self._model = "stub"

    def process(self, image: np.ndarray) -> RecognitionResult:
        """
        Args:
            image: preprocessed BGR numpy array (94x24)
        Returns:
            RecognitionResult
        """
        self._load_model()
        text, conf = self._infer(image)
        return RecognitionResult(text=text, confidence=conf, raw_text=text)

    def _infer(self, image: np.ndarray) -> tuple[str, float]:
        """Инференс. Заменить на реальный при наличии весов."""
        if self._model == "stub":
            # STUB — вернёт заглушку
            logger.debug("Using recognition stub — no weights loaded")
            return "00A000BC", 0.0
        # TODO: реальный инференс
        raise NotImplementedError("LPRNet inference not implemented yet")
