"""Declarative rules (D3.6 gates) evaluated on observations; soft flags by default (decision D9)."""
from __future__ import annotations


def load_rules(path: str):
    raise NotImplementedError("M5")


def evaluate(rules, pair, observations):
    raise NotImplementedError("M5")
