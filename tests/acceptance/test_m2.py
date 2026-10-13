import pytest


@pytest.mark.skip(reason="M2 not started")
def test_m2_acceptance():
    """Acceptance: baseline ranking and RMSE match docs/prototype.md (HANDOVER.md §3)."""
