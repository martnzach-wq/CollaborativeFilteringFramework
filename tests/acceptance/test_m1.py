import pytest


@pytest.mark.skip(reason="M1 not started")
def test_m1_acceptance():
    """Acceptance: prototype matrix reproduced within 2 % (HANDOVER.md §3)."""
