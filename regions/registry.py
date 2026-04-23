"""
Реестр регионов.
Добавление новой страны: реализуй BaseRegionHandler и зарегистрируй здесь.
Ядро системы при этом не меняется.
"""
from typing import Dict, Optional
from .base import BaseRegionHandler, RegionResult


class RegionRegistry:
    _handlers: Dict[str, BaseRegionHandler] = {}

    @classmethod
    def register(cls, handler: BaseRegionHandler) -> None:
        cls._handlers[handler.region_code] = handler

    @classmethod
    def get(cls, region_code: str) -> Optional[BaseRegionHandler]:
        return cls._handlers.get(region_code.upper())

    @classmethod
    def process(cls, raw_text: str, region_code: str = "UZB") -> RegionResult:
        handler = cls.get(region_code)
        if handler is None:
            return RegionResult(raw=raw_text, text=raw_text, valid=False, region=region_code)
        return handler.process(raw_text)

    @classmethod
    def available(cls):
        return list(cls._handlers.keys())


def _autoregister() -> None:
    try:
        from .uzb.rules import UZBRegionHandler
        RegionRegistry.register(UZBRegionHandler())
    except ImportError:
        pass


_autoregister()
