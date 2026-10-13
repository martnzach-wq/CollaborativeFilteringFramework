"""Recommender tasks P1-P7 (design §7)."""

from __future__ import annotations


def fill_gaps(model, geo_id=None, indicator_id=None):
    raise NotImplementedError("M3")  # P1


def new_entity(model, values: dict, parent=None, side=None):
    raise NotImplementedError("M3")  # P2


def new_indicator(model, values: dict, meta=None):
    raise NotImplementedError("M3")  # P3


def vision_entity(model, base_geo, period, targets: dict):
    raise NotImplementedError("M6")  # P4


def analogues(model, profile, lens, k: int = 10):
    raise NotImplementedError("M4")  # P5


def next_indicator(model, geo_id, targets, candidates=None):
    raise NotImplementedError("M5")  # P6


def level_transfer(model, indicator_id, to_level):
    raise NotImplementedError("M3")  # P7
