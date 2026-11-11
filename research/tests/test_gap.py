"""Direct gap benchmarks: AR(1) estimation, iteration and the use of information only through each origin."""
import numpy as np
import pandas as pd
import pytest

from src.forecasting.gap import ar1_path, direct_gap_forecasts, fit_ar1


def test_ar1_recovers_exact_coefficients_and_iterates():
    x = [1.0]
    for _ in range(30):
        x.append(0.4 + 0.7 * x[-1])
    a, b = fit_ar1(pd.Series(x))
    assert a == pytest.approx(0.4) and b == pytest.approx(0.7)
    assert ar1_path(2.0, 0.4, 0.7, 3) == pytest.approx([1.8, 1.66, 1.562])


def test_direct_forecasts_use_only_data_through_origin():
    dates = pd.date_range("2010-01-01", periods=60, freq="MS")
    gap = pd.Series(np.random.default_rng(0).normal(size=60), index=dates)
    T = dates[40]
    origins = pd.DataFrame({"origin": [T, T], "h": [1, 3], "target_date": [dates[41], dates[43]], "actual_gap": [gap.iloc[41], gap.iloc[43]]})
    f = direct_gap_forecasts(gap, origins).set_index(["model", "h"])["forecast_gap"]
    later = gap.copy()
    later.iloc[41:] += 100  # changing post-origin data must not change any forecast
    g = direct_gap_forecasts(later, origins.assign(actual_gap=0.0)).set_index(["model", "h"])["forecast_gap"]
    assert f.equals(g)
    assert f[("zero_gap", 1)] == 0 and f[("last_gap", 3)] == gap.iloc[40]
    a, b = fit_ar1(gap.loc[:T])
    assert f[("direct_ar1", 3)] == pytest.approx(ar1_path(gap.iloc[40], a, b, 3)[2])
