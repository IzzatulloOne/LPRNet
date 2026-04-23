"""
UZB — правила для узбекских номерных знаков

Форматы:
  DDLDDDLL  — напр. 01A123BC  (регион + буква + цифры + серия)
  DDDDDLLL  — напр. 12345ABC  (специальные серии)
  LLDDDDDL  — напр. BA12345C  (альтернативный)

D = цифра, L = буква
"""

import re
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.regions.base import RegionHandler


class UZBHandler(RegionHandler):
    code = "UZB"
    name = "Uzbekistan"
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Форматы узбекских номеров
    PATTERNS = [
        re.compile(r"^\d{2}[A-Z]\d{3}[A-Z]{2}$"),   # 01A123BC
        re.compile(r"^\d{5}[A-Z]{3}$"),               # 12345ABC
        re.compile(r"^[A-Z]{2}\d{5}[A-Z]$"),          # AB12345C
    ]

    # Таблица OCR-коррекции: что путает модель
    CORRECTIONS = {
        # В позиции буквы: цифра → буква
        "0": "O",
        "8": "B",
        "1": "I",
        # В позиции цифры: буква → цифра
        "O": "0",
        "B": "8",
        "I": "1",
        "S": "5",
        "Z": "2",
    }

    def validate(self, text: str) -> bool:
        t = self.normalize(text)
        return any(p.match(t) for p in self.PATTERNS)

    def correct(self, text: str) -> str:
        """
        Умная коррекция с учётом позиции символа.
        Формат 01A123BC: pos 0,1 → цифра; pos 2 → буква; 3,4,5 → цифра; 6,7 → буква
        """
        t = self.normalize(text)
        if len(t) == 8:
            return self._correct_format1(t)
        return t  # Для других форматов — без коррекции пока

    def _correct_format1(self, text: str) -> str:
        """DDLDDDLL — позиционная коррекция."""
        result = list(text)
        digit_positions  = {0, 1, 3, 4, 5}
        letter_positions = {2, 6, 7}

        for i, ch in enumerate(result):
            if i in digit_positions and not ch.isdigit():
                result[i] = self.CORRECTIONS.get(ch, ch)
            elif i in letter_positions and not ch.isalpha():
                result[i] = self.CORRECTIONS.get(ch, ch)

        return "".join(result)
