import pytest


@pytest.mark.skip(reason="M5 not started")
def test_m5_acceptance():
    """Acceptance: conformal coverage 0.87-0.93; reproducible typology (HANDOVER.md §3)."""
