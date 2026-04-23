"""Unit-тесты для UZB региона."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from anpr_system.regions.uzb.rules import UZBRegionHandler

handler = UZBRegionHandler()


@pytest.mark.parametrize("raw,expected_text,expected_valid", [
    ("01A123BC",  "01A123BC",  True),
    ("O1A123BC",  "01A123BC",  True),   # O→0
    ("12345ABC",  "12345ABC",  True),
    ("1234SABC",  "12345ABC",  True),   # S→5
    ("ABCDEFGH",  "AB8DEFGH",  False),  # B→8, не совпадает формат
])
def test_uzb_handler(raw, expected_text, expected_valid):
    result = handler.process(raw)
    assert result.text == expected_text
    assert result.valid == expected_valid
    assert result.region == "UZB"
