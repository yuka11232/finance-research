import numpy as np
import pandas as pd
import pytest

from src.inflation.household import SURVEY_MAP_PRIMARY, laspeyres, price_relatives, survey_to_coicop_shares, window_inflation
from src.inflation.transform import implied_yoy_from_path


def _prices():
    dates = pd.date_range("2024-01-01", "2026-03-01", freq="MS")
    rng = np.random.default_rng(0)
    data = {f"CP{i:02d}": 100 * np.cumprod(1 + rng.normal(0.004, 0.01, len(dates))) for i in range(1, 13)}
    data["ALL"] = np.mean(list(data.values()), axis=0)
    return pd.DataFrame(data, index=dates)


def test_contributions_sum_to_inflation_and_index_is_one_in_base_year():
    rel = price_relatives(_prices(), 2025)
    shares = pd.DataFrame([[0.5, 0.1] + [0.4 / 10] * 10 + [0.0]], columns=[f"CP{i:02d}" for i in range(1, 13)] + ["ALL"])
    idx = laspeyres(shares, rel)
    assert idx[idx.index.year == 2025].mean().iloc[0] == pytest.approx(1.0)
    w = window_inflation(shares, rel, pd.Timestamp("2025-03-01"), pd.Timestamp("2026-03-01"))
    contrib = w.filter(like="contrib_").sum(axis=1).iloc[0]
    assert contrib == pytest.approx(w["inflation_pct"].iloc[0])
    direct = 100 * (idx.loc["2026-03-01"].iloc[0] / idx.loc["2025-03-01"].iloc[0] - 1)
    assert w["inflation_pct"].iloc[0] == pytest.approx(direct)


def test_single_category_household_tracks_that_category():
    p = _prices()
    rel = price_relatives(p, 2025)
    spend = pd.DataFrame({k: [0.0] for k in SURVEY_MAP_PRIMARY})
    spend.loc[0, "food"] = 300
    sh = survey_to_coicop_shares(spend, SURVEY_MAP_PRIMARY)
    assert sh["CP01"].iloc[0] == pytest.approx(1.0)
    w = window_inflation(sh, rel, pd.Timestamp("2025-03-01"), pd.Timestamp("2026-03-01"))
    assert w["inflation_pct"].iloc[0] == pytest.approx(100 * (p.loc["2026-03-01", "CP01"] / p.loc["2025-03-01", "CP01"] - 1))


def test_split_mapping_and_shares_sum_to_one():
    spend = pd.DataFrame({k: [10.0] for k in SURVEY_MAP_PRIMARY})
    sh = survey_to_coicop_shares(spend, SURVEY_MAP_PRIMARY)
    assert sh.sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert sh["CP05"].iloc[0] == pytest.approx(5 / 140)


def test_implied_yoy_constant_monthly_rate():
    out = implied_yoy_from_path(np.full(11, 1.0), np.full(12, 1.0))
    assert out == pytest.approx(np.full(12, 100 * (1.01**12 - 1)))
    mixed = implied_yoy_from_path(np.zeros(11), np.array([2.0]))
    assert mixed[0] == pytest.approx(2.0)
