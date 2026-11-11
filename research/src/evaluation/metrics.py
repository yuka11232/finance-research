"""Point-forecast accuracy measures.

Errors are e = actual - forecast. Bias is reported as mean(forecast - actual),
so a positive bias means the model over-predicted on average.
MASE scales MAE by the in-sample MAE of the one-step seasonal-naive forecast
over the training data available before the test period (Hyndman & Koehler
2006). sMAPE is unstable when actual and forecast are both near zero (common
for month-on-month inflation) and is reported only for completeness.
"""
from __future__ import annotations

import numpy as np


def mae(a, f):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(f))))


def rmse(a, f):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(f)) ** 2)))


def bias(a, f):
    return float(np.mean(np.asarray(f) - np.asarray(a)))


def smape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    ok = denom > 0
    return float(100.0 * np.mean(np.abs(a[ok] - f[ok]) / denom[ok]))


def mase_scale(train: np.ndarray, m: int = 12) -> float:
    train = np.asarray(train, float)
    return float(np.mean(np.abs(train[m:] - train[:-m])))


def coverage(a, lo, hi):
    a, lo, hi = map(lambda v: np.asarray(v, float), (a, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(a)
    if not ok.any():
        return np.nan, 0
    return float(np.mean((a[ok] >= lo[ok]) & (a[ok] <= hi[ok]))), int(ok.sum())
