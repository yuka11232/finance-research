"""Direct forecasts of a group inflation gap (paper Section 2.3).

The three benchmarks forecast the gap Delta_{T+h} from the gap's own history through origin T:
zero (no gap), persistence (Delta_T) and an AR(1), Delta_t = a + b Delta_{t-1} + u_t, fitted by OLS
to all available observations and iterated forward from Delta_T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_ar1(gap: pd.Series) -> tuple[float, float]:
    """OLS intercept and slope of gap_t on gap_{t-1}."""
    b, a = np.polyfit(gap.iloc[:-1].to_numpy(), gap.iloc[1:].to_numpy(), 1)
    return float(a), float(b)


def ar1_path(last: float, a: float, b: float, horizon: int) -> np.ndarray:
    """Iterated AR(1) forecasts for h = 1..horizon."""
    out, x = np.empty(horizon), last
    for i in range(horizon):
        x = a + b * x
        out[i] = x
    return out


def direct_gap_forecasts(gap: pd.Series, origins: pd.DataFrame) -> pd.DataFrame:
    """Zero, persistence and AR(1) forecasts for every (origin, h, target_date, actual_gap) row of `origins`.

    `gap` is the monthly gap series indexed by date; each origin uses only observations dated <= origin.
    """
    rows = []
    for T, g in origins.groupby("origin"):
        hist = gap.loc[:T]
        a, b = fit_ar1(hist)
        path = ar1_path(hist.iloc[-1], a, b, int(g["h"].max()))
        for _, r in g.iterrows():
            for m, f in (("zero_gap", 0.0), ("last_gap", hist.iloc[-1]), ("direct_ar1", path[int(r["h"]) - 1])):
                rows.append({"model": m, "origin": T, "h": r["h"], "target_date": r["target_date"], "actual_gap": r["actual_gap"],
                             "forecast_gap": f, "ar1_intercept": a if m == "direct_ar1" else np.nan, "ar1_slope": b if m == "direct_ar1" else np.nan})
    return pd.DataFrame(rows)
