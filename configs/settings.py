from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Модели
    yolo_weights: str = "models/yolov10_plates.pt"
    lprnet_weights: Optional[str] = None   # None → MockRecognizer

    # Железо
    device: str = "cuda"   # "cuda" | "cpu"

    # Детекция
    detection_conf: float = 0.5

    # Регион по умолчанию
    default_region: str = "UZB"

    # API
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Логирование
    log_level: str = "INFO"
    log_unrecognized: bool = True
    unrecognized_dir: str = "logs/unrecognized"

    model_config = {"env_file": ".env", "env_prefix": "ANPR_"}


settings = Settings()
