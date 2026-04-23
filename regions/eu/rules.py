"""
EU region handler — stub, not implemented yet.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.regions.base import RegionHandler

class EUHandler(RegionHandler):
    code = "EU"
    name = "EU"
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def validate(self, text: str) -> bool:
        raise NotImplementedError("EU validation not implemented")

    def correct(self, text: str) -> str:
        return self.normalize(text)
