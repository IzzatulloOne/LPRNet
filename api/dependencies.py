"""
Singleton pipeline — инициализируется один раз при старте.
"""
import yaml
from functools import lru_cache
from pathlib import Path

from core.detection import PlateDetector
from core.preprocessing import Preprocessor
from core.recognition import PlateRecognizer
from core.postprocessing import Postprocessor
from core.regions.registry import RegionRegistry


_pipeline = None


def init_pipeline():
    global _pipeline
    config = _load_config()
    registry = RegionRegistry.default()

    _pipeline = {
        "detector":     PlateDetector(
            weights_path=config["model"]["detector"]["weights"],
            confidence=config["model"]["detector"]["confidence_threshold"],
            device=config["model"]["detector"]["device"],
        ),
        "preprocessor": Preprocessor(),
        "recognizer":   PlateRecognizer(
            weights_path=config["model"]["recognizer"]["weights"],
            device=config["model"]["recognizer"]["device"],
        ),
        "postprocessor": Postprocessor(registry),
        "config": config,
    }
    return _pipeline


def get_pipeline() -> dict:
    global _pipeline
    if _pipeline is None:
        init_pipeline()
    return _pipeline


def _load_config() -> dict:
    path = Path("configs/config.yaml")
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}
