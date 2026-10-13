import pytest


@pytest.mark.skip(reason="M4 not started")
def test_m4_acceptance():
    """Acceptance: D3.6 urban T2 reproduced; S-profile skill >= S-gower (HANDOVER.md §3)."""
