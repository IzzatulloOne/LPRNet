"""
Базовый класс для всех региональных обработчиков.
Наследуй, переопределяй validate() и correct().
"""

from abc import ABC, abstractmethod
import re


class RegionHandler(ABC):
    """Интерфейс региональных правил."""

    code: str = ""          # напр. "UZB"
    name: str = ""          # напр. "Uzbekistan"
    alphabet: str = ""      # разрешённые символы

    @abstractmethod
    def validate(self, text: str) -> bool:
        """True если текст соответствует формату региона."""
        ...

    @abstractmethod
    def correct(self, text: str) -> str:
        """Исправление OCR-ошибок (O↔0, B↔8 и т.д.)"""
        ...

    def normalize(self, text: str) -> str:
        """Привести к верхнему регистру, убрать пробелы."""
        return text.upper().strip().replace(" ", "")
