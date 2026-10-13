"""Masking schemes M1-M5 (design §10)."""
from __future__ import annotations


def random_cells(mask, frac=0.1, seed=0): raise NotImplementedError("M2")
def entity_cold_start(mask, levels, level, frac=0.15, k_keep=(1, 3, 6, 10), seed=0): raise NotImplementedError("M2")
def country_blocks(mask, levels, countries, n_blocks=10, seed=0): raise NotImplementedError("M2")
def level_transfer(mask, levels, indicator, level): raise NotImplementedError("M2")
def temporal(mask, periods, last=1): raise NotImplementedError("M6")
