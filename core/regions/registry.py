"""
Реестр регионов — добавление нового региона без изменения ядра.
"""

from typing import Dict, Optional, Type
from .base import RegionHandler
from loguru import logger


class RegionRegistry:
    """Singleton-реестр всех региональных обработчиков."""

    def __init__(self):
        self._handlers: Dict[str, RegionHandler] = {}

    def register(self, handler: RegionHandler) -> None:
        code = handler.code.upper()
        self._handlers[code] = handler
        logger.info(f"Region registered: {code} ({handler.name})")

    def get(self, code: str) -> Optional[RegionHandler]:
        return self._handlers.get(code.upper())

    def list_regions(self) -> list[str]:
        return list(self._handlers.keys())

    @classmethod
    def default(cls) -> "RegionRegistry":
        """Создаёт реестр со всеми доступными регионами."""
        from regions.uzb.rules import UZBHandler
        registry = cls()
        registry.register(UZBHandler())
        # registry.register(KZHandler())  # добавить по мере готовности
        return registry
