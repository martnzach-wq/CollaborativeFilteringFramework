from regsim.schema import Aggregation, DType, Indicator, Lens


def test_indicator_defaults():
    ind = Indicator("x", DType.CONTINUOUS)
    assert ind.aggregation is Aggregation.NONE
    assert ind.polarity is None
    assert Lens().name == "all"
