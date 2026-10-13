import pytest


@pytest.mark.skip(reason="M6 not started")
def test_m6_acceptance():
    """Acceptance: report regenerates all figures (HANDOVER.md §3)."""
