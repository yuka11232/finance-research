"""Phase 3 diagnostics: stationarity, seasonality and residual checks for the selected models.

Writes outputs/tables/diagnostics_stationarity.csv, diagnostics_residuals.csv and
diagnostics_sarima_parameter_stability.csv. Training sample = data before the test period (to 2018-12).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from src.forecasting.models import Ets, Sarima
from src.paths import PROCESSED, TABLES, config

TRAIN_END = pd.Timestamp("2018-12-01")


def parse_sarima(spec: str):
    inner = spec.replace("SARIMA", "")
    a, b = inner.split("x")
    return tuple(int(v) for v in a.strip("()").split(",")), tuple(int(v) for v in b.strip("()").split(","))


def parse_ets(spec: str):
    t = spec.split(",")[1]
    s = spec.split(",")[2].rstrip(")")
    return ({"N": None, "A": "add", "Ad": "add"}[t], t == "Ad", {"N": None, "A": "add"}[s])


def main():
    cfg = config()
    h = pd.read_csv(PROCESSED / "cpi_headline_monthly.csv", parse_dates=["date"])
    acc = pd.read_csv(TABLES / "INTERNATIONAL_FORECAST_COMPARISON.csv")
    st, res, stab = [], [], []
    for c in cfg["project"]["countries"]:
        g = h[(h["country"] == c) & (h["date"] >= cfg["data"]["analysis_a_start"])].set_index("date")
        for var in ("mom", "yoy"):
            for sample, y in (("train_to_2018-12", g.loc[:TRAIN_END, var].dropna()), ("full", g[var].dropna())):
                adf = adfuller(y, autolag="AIC", regression="c")
                kp = kpss(y, regression="c", nlags="auto")
                row = {"country": c, "variable": var, "sample": sample, "n": len(y), "adf_stat": adf[0], "adf_p": adf[1],
                       "kpss_stat": kp[0], "kpss_p_bounded": kp[1]}
                if var == "mom":  # deterministic seasonality: joint F-test of month dummies
                    X = pd.get_dummies(y.index.month, drop_first=True, dtype=float)
                    X.index = y.index
                    ols = sm.OLS(y, sm.add_constant(X)).fit()
                    row.update(seasonal_dummy_F=float(ols.fvalue), seasonal_dummy_p=float(ols.f_pvalue),
                               acf_lag12=float(sm.tsa.acf(y, nlags=12)[12]))
                row["reading"] = ("stationary (ADF rejects, KPSS does not)" if adf[1] < 0.05 and kp[1] > 0.05 else
                                  "unit root not rejected and KPSS rejects" if adf[1] >= 0.05 and kp[1] <= 0.05 else "mixed evidence")
                st.append(row)
        # residual diagnostics of the validation-selected specifications (analysis A), fitted to the training sample
        for target, var in (("mom", "mom"), ("yoy_direct", "yoy")):
            a = acc[(acc["variable"] == target) & (acc["analysis"] == "A") & (acc["country"] == c) & (acc["horizon"] == 1)]
            y = g.loc[:TRAIN_END, var].dropna().to_numpy()
            for fam in ("sarima", "ets"):
                spec = a[a["model"] == fam]["spec"].iloc[0]
                m = Sarima(*parse_sarima(spec)).fit(y) if fam == "sarima" else Ets(*parse_ets(spec)).fit(y)
                r = np.asarray(m.res.resid)[13:]
                lb = acorr_ljungbox(r, lags=[12, 24], return_df=True)
                res.append({"country": c, "target": target, "model": fam, "spec": spec, "n_resid": len(r),
                            "ljungbox_12_p": float(lb["lb_pvalue"].iloc[0]), "ljungbox_24_p": float(lb["lb_pvalue"].iloc[1]),
                            "resid_sd": float(np.std(r)), "jarque_bera_p": float(sm.stats.jarque_bera(r)[1])})
            # SARIMA parameter stability across the annual re-estimations in the test period
            spec = a[a["model"] == "sarima"]["spec"].iloc[0]
            for end in pd.date_range("2018-12-01", "2025-12-01", freq="12MS"):
                y = g.loc[:end, var].dropna().to_numpy()
                m = Sarima(*parse_sarima(spec)).fit(y)
                for name, val in zip(m.res.model.param_names, m.res.params):
                    stab.append({"country": c, "target": target, "spec": spec, "estimation_end": end.date(), "param": name, "value": float(val)})
    pd.DataFrame(st).to_csv(TABLES / "diagnostics_stationarity.csv", index=False)
    pd.DataFrame(res).to_csv(TABLES / "diagnostics_residuals.csv", index=False)
    s = pd.DataFrame(stab)
    s.to_csv(TABLES / "diagnostics_sarima_parameter_stability.csv", index=False)
    print(pd.DataFrame(st).round(3).to_string())
    print(pd.DataFrame(res).round(3).to_string())
    print(s.groupby(["country", "target", "param"])["value"].agg(["min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
