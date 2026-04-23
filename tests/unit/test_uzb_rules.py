"""Unit тесты для UZB правил — запускать: pytest tests/"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from regions.uzb.rules import UZBHandler


handler = UZBHandler()


class TestUZBValidation:
    def test_valid_format1(self):
        assert handler.validate("01A123BC") is True

    def test_valid_format2(self):
        assert handler.validate("12345ABC") is True

    def test_invalid_too_short(self):
        assert handler.validate("01A12") is False

    def test_invalid_wrong_structure(self):
        assert handler.validate("AAAAAAAA") is False


class TestUZBCorrection:
    def test_correct_O_to_0_in_digit_position(self):
        # "O1A123BC" → "01A123BC"
        result = handler.correct("O1A123BC")
        assert result[0] == "0"

    def test_correct_0_to_O_in_letter_position(self):
        # "01A1230C" → "01A123OC" (позиция 6 — буква)
        result = handler.correct("01A1230C")
        assert result[6] == "O"

    def test_normalize_lowercase(self):
        result = handler.normalize("01a123bc")
        assert result == "01A123BC"
