"""
Postprocessing — региональная валидация и коррекция
"""

from loguru import logger
from ..regions.base import RegionHandler
from ..regions.registry import RegionRegistry


class Postprocessor:
    """Применяет региональные правила к raw тексту от OCR."""

    def __init__(self, registry: RegionRegistry):
        self._registry = registry

    def process(self, raw_text: str, region_code: str) -> tuple[str, bool]:
        """
        Args:
            raw_text: строка от OCR
            region_code: напр. 'UZB'
        Returns:
            (validated_text, is_valid)
        """
        handler = self._registry.get(region_code)
        if handler is None:
            logger.warning(f"No handler for region: {region_code}")
            return raw_text, False

        corrected = handler.correct(raw_text)
        is_valid = handler.validate(corrected)
        logger.debug(f"Postprocess [{region_code}]: {raw_text!r} → {corrected!r} valid={is_valid}")
        return corrected, is_valid
