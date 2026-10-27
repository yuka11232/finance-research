"""CPI index <-> inflation-rate transformations.

Conventions (all rates in percent):
  month-on-month  m_t   = 100 * (P_t / P_{t-1} - 1)
  year-on-year    y_t   = 100 * (P_t / P_{t-12} - 1)
  annual average  a_Y   = 100 * (mean_{t in Y} P_t / mean_{t in Y-1} P_t - 1)
These are different quantities and are never substituted for one another.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mom(index: pd.Series) -> pd.Series:
    return 100.0 * (index / index.shift(1) - 1.0)


def yoy(index: pd.Series) -> pd.Series:
    return 100.0 * (index / index.shift(12) - 1.0)


def annual_average_inflation(index: pd.Series) -> pd.Series:
    g = index.groupby(index.index.year)
    means = g.mean()[g.size() == 12]
    return 100.0 * (means / means.shift(1) - 1.0)


def index_from_mom(m: pd.Series, base_value: float = 100.0) -> pd.Series:
    """Chain month-on-month rates into an index whose value before the first rate is base_value."""
    return base_value * np.cumprod(1.0 + m.to_numpy() / 100.0)


def implied_yoy_from_path(history_mom: np.ndarray, forecast_mom: np.ndarray) -> np.ndarray:
    """Year-on-year inflation implied at each forecast horizon.

    history_mom: observed m/m rates up to and including the forecast origin T
                 (at least 11 values; only the last 11 are used).
    forecast_mom: forecast m/m rates for T+1..T+H.
    Returns y_{T+h} for h=1..H, compounding the 12 monthly relatives ending at
    T+h, of which months <= T are observed and months > T are forecast.
    """
    hist = np.asarray(history_mom, dtype=float)[-11:]
    path = np.concatenate([hist, np.asarray(forecast_mom, dtype=float)])
    rel = 1.0 + path / 100.0
    out = np.empty(len(forecast_mom))
    for h in range(1, len(forecast_mom) + 1):
        window = rel[h - 1 : h + 11]
        out[h - 1] = 100.0 * (np.prod(window) - 1.0)
    return out
