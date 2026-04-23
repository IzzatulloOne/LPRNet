"""
KZ region handler — stub, not implemented yet.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.regions.base import RegionHandler

class KZHandler(RegionHandler):
    code = "KZ"
    name = "KZ"
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def validate(self, text: str) -> bool:
        raise NotImplementedError("KZ validation not implemented")

    def correct(self, text: str) -> str:
        return self.normalize(text)
