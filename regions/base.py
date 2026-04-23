from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RegionResult:
    raw: str
    text: str
    valid: bool
    region: str
    confidence_penalty: float = 0.0


class BaseRegionHandler(ABC):
    region_code: str = "UNKNOWN"

    @abstractmethod
    def validate(self, text: str) -> bool: ...

    @abstractmethod
    def correct(self, text: str) -> str: ...

    @abstractmethod
    def process(self, raw_text: str) -> RegionResult: ...
