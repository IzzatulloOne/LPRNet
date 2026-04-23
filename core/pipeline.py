"""
ANPRPipeline — удобная обёртка для использования вне API
(тестирование, скрипты, batch обработка)
"""
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

from .detection import PlateDetector
from .preprocessing import Preprocessor
from .recognition import PlateRecognizer, RecognitionResult
from .postprocessing import Postprocessor
from .regions.registry import RegionRegistry


@dataclass
class PipelineResult:
    plate: str
    confidence: float
    region: str
    bbox: list[int]
    valid: bool
    processing_time_ms: float


class ANPRPipeline:
    """Полный пайплайн в одном классе."""

    def __init__(
        self,
        detector: PlateDetector,
        preprocessor: Preprocessor,
        recognizer: PlateRecognizer,
        postprocessor: Postprocessor,
        default_region: str = "UZB",
    ):
        self.detector = detector
        self.preprocessor = preprocessor
        self.recognizer = recognizer
        self.postprocessor = postprocessor
        self.default_region = default_region

    def process_image(self, image: np.ndarray, region: Optional[str] = None) -> PipelineResult:
        t0 = time.perf_counter()
        region = (region or self.default_region).upper()

        detections = self.detector.detect(image)
        if not detections:
            return PipelineResult("", 0.0, region, [], False, 0.0)

        best = detections[0]
        prep = self.preprocessor.process(best.crop)
        rec = self.recognizer.process(prep.image)
        validated, is_valid = self.postprocessor.process(rec.text, region)
        elapsed = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            plate=validated,
            confidence=rec.confidence,
            region=region,
            bbox=best.bbox,
            valid=is_valid,
            processing_time_ms=round(elapsed, 2),
        )

    def process_file(self, path: str, region: Optional[str] = None) -> PipelineResult:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        return self.process_image(image, region)
