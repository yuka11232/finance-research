"""Rolling-origin (expanding-window) backtests.

At each origin T the model sees only y[..T]. Parameters of SARIMA/ETS are
re-estimated when the origin is a December (and at the first origin); in
between, the state is updated with the new observations and fixed
parameters. Forecasts for h = 1..H are stored for every origin; targets after
the end of the data are recorded with actual = NaN and never scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.inflation.transform import implied_yoy_from_path


def rolling_forecasts(
    y: pd.Series,
    factory,
    first_origin: pd.Timestamp,
    last_origin: pd.Timestamp,
    H: int = 12,
    refit_every_december: bool = True,
    yoy_actual: pd.Series | None = None,
) -> pd.DataFrame:
    """Return one row per (origin, horizon).

    If yoy_actual is supplied, y is treated as a month-on-month rate and the
    implied year-on-year forecast is added (columns yoy_forecast/yoy_actual).
    """
    y = y.dropna()
    dates = y.index
    values = y.to_numpy()
    origins = dates[(dates >= first_origin) & (dates <= last_origin)]
    rows = []
    model = None
    for T in origins:
        i = dates.get_loc(T)
        hist = values[: i + 1]
        if model is None or (refit_every_december and T.month == 12):
            model = factory().fit(hist)
        else:
            model.update(hist)
        out = model.forecast(H)
        yoy_fc = implied_yoy_from_path(hist, out["mean"]) if yoy_actual is not None else None
        for h in range(1, H + 1):
            target = T + pd.DateOffset(months=h)
            row = {
                "origin": T,
                "target_date": target,
                "h": h,
                "forecast": out["mean"][h - 1],
                "actual": y.get(target, np.nan),
            }
            for k in ("lo80", "hi80", "lo95", "hi95"):
                row[f"model_{k}"] = out[k][h - 1] if out[k] is not None else np.nan
            if yoy_fc is not None:
                row["yoy_forecast"] = yoy_fc[h - 1]
                row["yoy_actual"] = yoy_actual.get(target, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def add_empirical_intervals(
    df: pd.DataFrame,
    fc_col: str = "forecast",
    act_col: str = "actual",
    prefix: str = "emp",
    min_errors: int = 24,
    levels=(0.80, 0.95),
) -> pd.DataFrame:
    """Intervals from quantiles of the model's own past out-of-sample errors at the same horizon.

    At origin T only errors whose targets are <= T are used (they were
    observable at T), so the intervals are genuinely ex ante.
    """
    df = df.sort_values(["h", "origin"]).copy()
    for lvl in levels:
        tag = int(round(lvl * 100))
        df[f"{prefix}_lo{tag}"] = np.nan
        df[f"{prefix}_hi{tag}"] = np.nan
    for h, g in df.groupby("h"):
        err = (g[act_col] - g[fc_col]).to_numpy()
        tgt = g["target_date"].to_numpy()
        org = g["origin"].to_numpy()
        lo = {lvl: np.full(len(g), np.nan) for lvl in levels}
        hi = {lvl: np.full(len(g), np.nan) for lvl in levels}
        for j in range(len(g)):
            past = err[(tgt <= org[j]) & np.isfinite(err)]
            if len(past) >= min_errors:
                for lvl in levels:
                    a = (1 - lvl) / 2
                    lo[lvl][j] = g[fc_col].iloc[j] + np.quantile(past, a)
                    hi[lvl][j] = g[fc_col].iloc[j] + np.quantile(past, 1 - a)
        for lvl in levels:
            tag = int(round(lvl * 100))
            df.loc[g.index, f"{prefix}_lo{tag}"] = lo[lvl]
            df.loc[g.index, f"{prefix}_hi{tag}"] = hi[lvl]
    return df.sort_values(["origin", "h"])
