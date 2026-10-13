import pytest


@pytest.mark.skip(reason="M3 not started")
def test_m3_acceptance():
    """Acceptance: synthetic recovery, coherence and fold-in criteria (HANDOVER.md §3)."""
