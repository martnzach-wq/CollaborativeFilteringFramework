"""NUTS and LAU schemes from Eurostat GISCO (versions 2016, 2021, 2024)."""
from __future__ import annotations


class NutsScheme:
    def __init__(self, version: str = "2021", cache_dir: str = "data/raw/gisco"):
        self.name = f"NUTS{version}"
        self.version = version
        self.cache_dir = cache_dir

    def units(self, level=None):
        raise NotImplementedError("M1")

    def edges(self):
        raise NotImplementedError("M1")

    def harmonise(self, codes, source_version):
        raise NotImplementedError("M1")
