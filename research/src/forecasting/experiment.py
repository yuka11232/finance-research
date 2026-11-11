"""Model selection on validation targets and full rolling backtests for one series."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from src.evaluation.backtest import rolling_forecasts
from src.forecasting.models import AutoAR, Ets, Fts, Naive, Sarima, SeasonalNaive
from src.paths import config

EVAL_H = (1, 3, 6, 12)


def sarima_candidates(target: str):
    c = config()["sarima"]
    d_grid = [0, 1] if target == "yoy" else c["d"]
    for p, d, q, P, Q in itertools.product(c["p"], d_grid, c["q"], c["P"], c["Q"]):
        if p == q == P == Q == 0 and d == 0:
            continue
        yield (p, d, q), (P, 0, Q, 12)


def ets_candidates():
    for trend, damped in ((None, False), ("add", False), ("add", True)):
        for seasonal in (None, "add"):
            yield trend, damped, seasonal


def _validation_score(df: pd.DataFrame, val_end: pd.Timestamp) -> float:
    v = df[(df["target_date"] <= val_end) & df["h"].isin(EVAL_H)].dropna(subset=["actual"])
    per_h = v.groupby("h").apply(lambda g: np.sqrt(np.mean((g["actual"] - g["forecast"]) ** 2)))
    return float(per_h.mean())


def run_series(y: pd.Series, target: str, yoy_actual: pd.Series | None, label: dict) -> dict:
    """Select hyper-parameters on validation targets, then backtest every model.

    y: the modelled series (m/m or y/y inflation, %), monthly DatetimeIndex.
    Returns dict with 'forecasts' (long DataFrame) and 'selection' (DataFrame).
    """
    cfg = config()["forecast"]
    val_start = pd.Timestamp(cfg["validation_start"])
    val_end = pd.Timestamp(cfg["validation_end"])
    first_origin = val_start - pd.DateOffset(months=1)
    sel_last_origin = val_end - pd.DateOffset(months=1)
    last_origin = y.dropna().index.max()
    H = max(EVAL_H)
    sel_rows = []

    def backtest(factory, last=last_origin, yoy=yoy_actual):
        return rolling_forecasts(y, factory, first_origin, last, H=H, yoy_actual=yoy)

    # SARIMA
    best = (np.inf, None)
    for order, sorder in sarima_candidates(target):
        t0 = time.time()
        try:
            df = backtest(lambda o=order, s=sorder: Sarima(o, s), last=sel_last_origin, yoy=None)
            score = _validation_score(df, val_end)
        except Exception as exc:  # numerical failure of a candidate is recorded, not hidden
            score = np.nan
            sel_rows.append({**label, "family": "sarima", "spec": f"SARIMA{order}x{sorder}", "val_rmse_mean_h": np.nan, "note": repr(exc)[:120]})
            continue
        sel_rows.append({**label, "family": "sarima", "spec": f"SARIMA{order}x{sorder}", "val_rmse_mean_h": score, "seconds": round(time.time() - t0, 1)})
        if score < best[0]:
            best = (score, (order, sorder))
    sarima_spec = best[1]

    # ETS
    best = (np.inf, None)
    for trend, damped, seasonal in ets_candidates():
        try:
            df = backtest(lambda t=trend, d=damped, s=seasonal: Ets(t, d, s), last=sel_last_origin, yoy=None)
            score = _validation_score(df, val_end)
        except Exception as exc:
            sel_rows.append({**label, "family": "ets", "spec": Ets(trend, damped, seasonal).label(), "val_rmse_mean_h": np.nan, "note": repr(exc)[:120]})
            continue
        sel_rows.append({**label, "family": "ets", "spec": Ets(trend, damped, seasonal).label(), "val_rmse_mean_h": score})
        if score < best[0]:
            best = (score, (trend, damped, seasonal))
    ets_spec = best[1]

    # Fuzzy: interval method chosen for Yu weighted; same partition used for Chen
    # and equal-weight recurrence so that only recurrence/weighting differs.
    fts_scores = {}
    for method in ("chen", "yu_equal", "yu_weighted"):
        for interval in config()["fuzzy"]["interval_methods"]:
            df = backtest(lambda m=method, i=interval: Fts(m, i), last=sel_last_origin, yoy=None)
            s = _validation_score(df, val_end)
            fts_scores[(method, interval)] = s
            sel_rows.append({**label, "family": "fts", "spec": f"{method}[{interval}]", "val_rmse_mean_h": s})
    fts_interval = min(config()["fuzzy"]["interval_methods"], key=lambda i: fts_scores[("yu_weighted", i)])

    models = {
        "seasonal_naive": (lambda: SeasonalNaive(), "SNaive(m=12)"),
        "naive": (lambda: Naive(), "RandomWalk"),
        "sarima": (lambda: Sarima(*sarima_spec), f"SARIMA{sarima_spec[0]}x{sarima_spec[1]}"),
        "ets": (lambda: Ets(*ets_spec), Ets(*ets_spec).label()),
        "yu_weighted_fts": (lambda: Fts("yu_weighted", fts_interval), f"yu_weighted[{fts_interval}]"),
        "yu_equal_fts": (lambda: Fts("yu_equal", fts_interval), f"yu_equal[{fts_interval}]"),
        "chen_fts": (lambda: Fts("chen", fts_interval), f"chen[{fts_interval}]"),
        "ar_aic": (lambda: AutoAR(), "AR(p by AIC, max 13)"),
    }
    out = []
    for name, (factory, spec) in models.items():
        df = backtest(factory)
        df.insert(0, "model", name)
        df.insert(1, "spec", spec)
        out.append(df)
    fc = pd.concat(out, ignore_index=True)
    for k, v in label.items():
        fc.insert(0, k, v)
    test_origin = pd.Timestamp(cfg["test_start"]) - pd.DateOffset(months=1)
    fc["split"] = np.where(
        fc["actual"].isna(),
        "future",
        np.where(fc["origin"] >= test_origin, "test", np.where(fc["target_date"] <= val_end, "validation", "gap")),
    )
    return {"forecasts": fc, "selection": pd.DataFrame(sel_rows)}
