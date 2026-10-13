"""Run configuration (pydantic). Every run writes its resolved config and hash next to its outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    indicator_registry: str | list[str]
    geo_scheme: str = "NUTS2024"  # decision D1
    levels: list[int] = Field(default_factory=lambda: [0, 2, 3])
    periods: list[str] = Field(default_factory=lambda: ["2013/2017", "2018/2022"])  # decision D3
    store: str = "data/store"
    raw_dir: str = "data/raw"
    sources: list[str] = Field(default_factory=lambda: ["eurostat"])  # decision D5: eurostat, d36, ardeco
    d36_gpkg: str | None = None
    lau: bool = False

    @field_validator("periods")
    @classmethod
    def _windows(cls, v: list[str]) -> list[str]:
        for p in v:
            start, _, end = p.partition("/")
            if not (start.isdigit() and end.isdigit() and int(start) <= int(end)):
                raise ValueError(f"period window must be 'YYYY/YYYY', got {p!r}")
        return v

    @property
    def registries(self) -> list[str]:
        return (
            [self.indicator_registry] if isinstance(self.indicator_registry, str) else self.indicator_registry
        )

    @property
    def nuts_version(self) -> str:
        return self.geo_scheme.removeprefix("NUTS")

    @property
    def span(self) -> tuple[int, int]:
        years = [int(x) for p in self.periods for x in p.split("/")]
        return min(years), max(years)


class ModelConfig(BaseModel):
    estimator: str = "HierTypedMF"
    rank: int = 12
    lam_u: float = 0.03
    lam_v: float = 0.1
    lam_agg: float = 1.0
    coupling: str = "prior_and_constraint"  # broadcast | prior_only | prior_and_constraint
    seed: int = 0


class RunConfig(BaseModel):
    data: DataConfig
    model: ModelConfig = ModelConfig()
    output_dir: str = "runs"

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def write_resolved(self, folder: str | Path, name: str = "config") -> Path:
        """Write the resolved config and its hash to ``folder``; return the config path."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.resolved.yaml"
        path.write_text(yaml.safe_dump(self.model_dump(), sort_keys=False), encoding="utf-8")
        (folder / f"{name}.hash").write_text(self.config_hash() + "\n", encoding="utf-8")
        return path
