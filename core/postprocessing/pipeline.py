from anpr_system.regions.registry import RegionRegistry
from anpr_system.regions.base import RegionResult


class PostProcessor:
    def __init__(self, default_region: str = "UZB"):
        self.default_region = default_region

    def process(self, raw_text: str, region: str = None) -> RegionResult:
        return RegionRegistry.process(raw_text, region or self.default_region)
