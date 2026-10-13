"""Transfer-specific similarity (design §5.3): learned non-negative indicator weights for a target quantity."""

from __future__ import annotations


def learn_transfer_weights(model, target_values: dict, lens, n_boot: int = 200):
    raise NotImplementedError("M4")
