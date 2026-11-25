"""SD adapter helpers that need no SD checkout."""

import dataclasses

from regsim.causal import sd_knowing as K


@dataclasses.dataclass
class _Mobility:
    policy_cycling_attractiveness: dict
    cycling_resistance_reduction: dict
    policy_capacity_adaptation: dict
    traffic_calming_time_bonus: dict


@dataclasses.dataclass
class _City:
    mobility: _Mobility
    energy_supply: object = None


def test_removed_lever_holds_its_companion_series():
    city = _City(
        _Mobility(
            policy_cycling_attractiveness={2019: 0.0, 2050: 1.0},
            cycling_resistance_reduction={2019: 0.0, 2030: 0.15, 2050: 0.575},
            policy_capacity_adaptation={2019: 1.0, 2050: 0.8},
            traffic_calming_time_bonus={
                "walk": {2019: 0.0, 2050: 0.075},
                "bicycle": {2019: 0.01, 2050: 0.04},
            },
        )
    )
    held = K.configure(city, {}, removed=["policy_cycling_attractiveness", "policy_capacity_adaptation"])
    m = held.mobility
    assert m.policy_cycling_attractiveness == {2019: 0.0, 2050: 0.0}
    assert m.cycling_resistance_reduction == {2019: 0.0, 2030: 0.0, 2050: 0.0}
    assert m.policy_capacity_adaptation == {2019: 1.0, 2050: 1.0}
    assert m.traffic_calming_time_bonus == {
        "walk": {2019: 0.0, 2050: 0.0},
        "bicycle": {2019: 0.01, 2050: 0.01},
    }
    # the original configuration is untouched
    assert city.mobility.cycling_resistance_reduction[2050] == 0.575
