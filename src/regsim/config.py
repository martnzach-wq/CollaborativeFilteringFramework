"""Run configuration (pydantic). Every run writes its resolved config and hash next to its outputs."""
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    indicator_registry: str
    geo_scheme: str = "NUTS2021"           # decision D1
    levels: list[int] = Field(default_factory=lambda: [0, 2, 3])
    periods: list[str] = Field(default_factory=lambda: ["2018/2022"])   # decision D3
    store: str = "data/store"


class ModelConfig(BaseModel):
    estimator: str = "HierTypedMF"
    rank: int = 12
    lam_u: float = 0.03
    lam_v: float = 0.1
    lam_agg: float = 1.0
    coupling: str = "prior_and_constraint"   # broadcast | prior_only | prior_and_constraint
    seed: int = 0


class RunConfig(BaseModel):
    data: DataConfig
    model: ModelConfig = ModelConfig()
    output_dir: str = "runs"

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:12]
