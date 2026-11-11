"""Forecasting models behind one interface.

Each model implements
    fit(y)       estimate parameters on the 1-D array y (expanding window)
    update(y)    incorporate new observations; parameters stay fixed for
                 SARIMA/ETS (state re-filtered), cheap models simply refit
    forecast(H)  -> dict(mean=array(H), lo80, hi80, lo95, hi95) where the
                 interval arrays are None when the model has no analytic
                 intervals (empirical intervals are added by the evaluator).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from src.fuzzy.fts import FuzzyTimeSeries

warnings.filterwarnings("ignore")


def _no_intervals(mean):
    return {"mean": np.asarray(mean, float), "lo80": None, "hi80": None, "lo95": None, "hi95": None}


@dataclass
class SeasonalNaive:
    m: int = 12
    name: str = "seasonal_naive"

    def fit(self, y):
        self.y = np.asarray(y, float)
        return self

    update = fit

    def forecast(self, H):
        idx = [len(self.y) - self.m + ((h - 1) % self.m) for h in range(1, H + 1)]
        return _no_intervals(self.y[idx])


@dataclass
class Naive:
    name: str = "naive"

    def fit(self, y):
        self.y = np.asarray(y, float)
        return self

    update = fit

    def forecast(self, H):
        return _no_intervals(np.repeat(self.y[-1], H))


@dataclass
class Sarima:
    order: tuple = (1, 0, 0)
    seasonal_order: tuple = (0, 0, 0, 12)
    trend: str | None = "c"
    name: str = "sarima"
    res: object = field(default=None, repr=False)

    def _model(self, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        d = self.order[1] + self.seasonal_order[1]
        return SARIMAX(
            np.asarray(y, float),
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend if d == 0 else None,
        )

    def fit(self, y):
        self.res = self._model(y).fit(disp=False, maxiter=200)
        return self

    def update(self, y):
        self.res = self.res.apply(np.asarray(y, float), refit=False)
        return self

    def forecast(self, H):
        fc = self.res.get_forecast(H)
        mean = np.asarray(fc.predicted_mean)
        ci80 = np.asarray(fc.conf_int(alpha=0.20))
        ci95 = np.asarray(fc.conf_int(alpha=0.05))
        return {"mean": mean, "lo80": ci80[:, 0], "hi80": ci80[:, 1], "lo95": ci95[:, 0], "hi95": ci95[:, 1]}

    @property
    def aic(self):
        return float(self.res.aic)

    def label(self):
        return f"SARIMA{self.order}x{self.seasonal_order}"


@dataclass
class Ets:
    trend: str | None = None
    damped: bool = False
    seasonal: str | None = None
    m: int = 12
    name: str = "ets"
    res: object = field(default=None, repr=False)

    def _model(self, y):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        import pandas as pd

        return ETSModel(
            pd.Series(np.asarray(y, float)),  # ndarray endog breaks get_prediction in statsmodels 0.15
            error="add",
            trend=self.trend,
            damped_trend=self.damped if self.trend else False,
            seasonal=self.seasonal,
            seasonal_periods=self.m if self.seasonal else None,
        )

    def fit(self, y):
        self.res = self._model(y).fit(disp=False, maxiter=500)
        return self

    def update(self, y):
        self.res = self._model(y).smooth(self.res.params)
        return self

    def forecast(self, H):
        n = self.res.nobs
        pr = self.res.get_prediction(start=n, end=n + H - 1)
        f80 = pr.summary_frame(alpha=0.20)
        f95 = pr.summary_frame(alpha=0.05)
        return {
            "mean": f80["mean"].to_numpy(),
            "lo80": f80["pi_lower"].to_numpy(),
            "hi80": f80["pi_upper"].to_numpy(),
            "lo95": f95["pi_lower"].to_numpy(),
            "hi95": f95["pi_upper"].to_numpy(),
        }

    @property
    def aic(self):
        return float(self.res.aic)

    def label(self):
        t = {None: "N", "add": "A"}[self.trend] + ("d" if self.damped and self.trend else "")
        s = {None: "N", "add": "A"}[self.seasonal]
        return f"ETS(A,{t},{s})"


@dataclass
class Fts:
    method: str = "yu_weighted"
    interval: str = "average"
    name: str = "yu_weighted_fts"

    def fit(self, y):
        self.model = FuzzyTimeSeries(method=self.method, interval=self.interval).fit(y)
        return self

    update = fit  # fuzzification and FLR groups are rebuilt from all data at every origin

    def forecast(self, H):
        return _no_intervals(self.model.forecast(H))

    def label(self):
        return f"{self.method}[{self.interval}]"


@dataclass
class AutoAR:
    """Supplementary benchmark: AR(p) with constant, p chosen by AIC (max 13) at each re-estimation."""

    maxlag: int = 13
    name: str = "ar_aic"

    def fit(self, y):
        from statsmodels.tsa.ar_model import ar_select_order

        sel = ar_select_order(np.asarray(y, float), maxlag=self.maxlag, ic="aic", trend="c")
        self.lags = sel.ar_lags or [1]
        return self.update(y)

    def update(self, y):
        from statsmodels.tsa.ar_model import AutoReg

        self.y = np.asarray(y, float)
        self.res = AutoReg(self.y, lags=self.lags, trend="c").fit()
        return self

    def forecast(self, H):
        n = len(self.y)
        return _no_intervals(self.res.predict(start=n, end=n + H - 1))

    def label(self):
        return f"AR(lags={max(self.lags)})"
