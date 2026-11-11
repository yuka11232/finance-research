import numpy as np
import pytest

from src.statistics.forecast_tests import bootstrap_skill, diebold_mariano, holm, long_run_variance, moving_block_indices
from src.statistics.group_tests import cliffs_delta, compare, hedges_g, mde_two_sample


def test_holm_matches_hand_calculation():
    p = [0.01, 0.04, 0.03, np.nan]
    adj = holm(p)
    # sorted: .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[3])


def test_dm_identical_forecasts_not_tested_and_clear_difference_detected():
    rng = np.random.default_rng(1)
    e = rng.normal(size=200)
    assert np.isnan(diebold_mariano(e, e, 1)["p_value"])
    r = diebold_mariano(0.5 * e, 2.0 * e, 1)
    assert r["mean_loss_diff"] < 0 and r["p_value"] < 1e-6


def test_dm_size_under_null_is_reasonable():
    rng = np.random.default_rng(2)
    rej = 0
    for _ in range(400):
        e1, e2 = rng.normal(size=80), rng.normal(size=80)
        rej += diebold_mariano(e1, e2, 1)["p_value"] < 0.05
    assert 0.02 < rej / 400 < 0.09


def test_long_run_variance_positive_and_hln_shrinks_stat():
    rng = np.random.default_rng(3)
    d = np.convolve(rng.normal(size=300), np.ones(6) / 6, mode="valid")
    lrv, _ = long_run_variance(d, 6)
    assert lrv > 0
    r = diebold_mariano(d + 0.1, np.zeros_like(d), 6)
    assert abs(r["hln_stat"]) < abs(r["dm_stat"])


def test_block_indices_and_skill_sign():
    rng = np.random.default_rng(4)
    idx = moving_block_indices(50, 7, rng)
    assert len(idx) == 50 and idx.min() >= 0 and idx.max() < 50
    e = rng.normal(size=100)
    s = bootstrap_skill(0.8 * e, e, 1, 300, rng)
    assert s["skill_rmse"] == pytest.approx(0.2)
    assert s["skill_ci95_low"] <= 0.2 <= s["skill_ci95_high"]


def test_effect_sizes_and_mde():
    a, b = np.array([1.0, 2, 3, 4, 5]), np.array([3.0, 4, 5, 6, 7])
    assert cliffs_delta(a, a) == 0
    assert hedges_g(a, b) < 0
    assert mde_two_sample(1, 1, 16, 16) == pytest.approx(2.8016 * np.sqrt(2 / 16), rel=1e-3)
    r = compare(a, b, np.random.default_rng(0), 500)
    assert r["tested"] and r["diff_mean"] == -2
    assert compare(a[:3], b, np.random.default_rng(0), 500)["tested"] is False
