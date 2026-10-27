"""Tests reproduce the worked examples printed in Yu (2005), Physica A 349, 609-624."""
import numpy as np
import pytest

from src.fuzzy.fts import (
    FuzzyTimeSeries,
    Universe,
    average_based_length,
    distribution_based_length,
    map_table_base,
)


@pytest.mark.parametrize(
    "length,base",
    [(14, 10), (0.5, 0.1), (1.0, 0.1), (10, 1), (28, 10), (100, 10), (101, 100), (0.05, 0.01)],
)
def test_map_table(length, base):
    assert map_table_base(length) == pytest.approx(base)


def test_huarng_lengths_match_yu_section_2_2():
    x = np.array([30, 50, 80, 120, 100, 70])
    assert average_based_length(x) == pytest.approx(10)  # Yu: "10 is chosen"
    assert distribution_based_length(x) == pytest.approx(20)  # Yu: "20 is chosen"


def _taiex_like_model(method):
    # Produces FLRs A11->A9, A9->A11, A11->A12, A12->A11, A11->A10, A10->A11, A11->A9
    # on 100-point intervals, so the A11 group is (A9, A12, A10, A9) as in Yu Sect. 5.
    x = [3550, 3350, 3550, 3650, 3550, 3450, 3550, 3350]
    return FuzzyTimeSeries(method=method, interval=100).fit(x)


def test_yu_weighted_taiex_example_equals_3440():
    m = _taiex_like_model("yu_weighted")
    state = int(m.universe.fuzzify(3506.8))  # fuzzify(3506.8) = A11
    assert m.universe.midpoints[state] == pytest.approx(3550)
    assert m.groups[state] == [int(m.universe.fuzzify(v)) for v in (3333.46, 3600.75, 3480.47, 3316.36)]
    assert m.predict_next(state) == pytest.approx(3440)  # Yu Step 9


def test_chen_counts_repeats_once_and_equal_recurrence_counts_them():
    state = int(_taiex_like_model("chen").universe.fuzzify(3550))
    assert _taiex_like_model("chen").predict_next(state) == pytest.approx((3350 + 3450 + 3650) / 3)
    assert _taiex_like_model("yu_equal").predict_next(state) == pytest.approx((3350 + 3650 + 3450 + 3350) / 4)


def test_yu_equation_1_weights():
    # FLRs of Eq. (1): A1->A1, A1->A2, A2->A1, A1->A1, A1->A1 ; group A1 = (A1, A2, A1, A1)
    x = [5, 5, 15, 5, 5, 5]
    m = FuzzyTimeSeries(method="yu_weighted", interval=10).fit(x)
    m1, m2 = m.universe.midpoints[:2]
    assert m.groups[0] == [0, 1, 0, 0]
    assert m.predict_next(0) == pytest.approx((1 * m1 + 2 * m2 + 3 * m1 + 4 * m1) / 10)


def test_empty_group_forecasts_own_midpoint_and_out_of_range_is_clipped():
    m = FuzzyTimeSeries(method="yu_weighted", interval=1.0).fit([0.2, 0.4, 3.5])
    last = m.universe.n_intervals - 1
    assert last not in m.groups
    assert m.predict_next(last) == pytest.approx(m.universe.midpoints[last])
    assert m.universe.fuzzify(99.0) == last and m.universe.fuzzify(-99.0) == 0


def test_universe_covers_data_and_handles_negatives():
    x = np.array([-0.83, 0.1, 2.71, 1.4])
    u = Universe.from_data(x, 0.2)
    assert u.start <= x.min() and u.start + u.n_intervals * u.length > x.max()
    assert (u.fuzzify(x) >= 0).all() and (u.fuzzify(x) < u.n_intervals).all()


def test_recursive_forecast_length_and_finite():
    rng = np.random.default_rng(0)
    m = FuzzyTimeSeries().fit(rng.normal(0.5, 0.6, 120))
    f = m.forecast(12)
    assert f.shape == (12,) and np.isfinite(f).all()
