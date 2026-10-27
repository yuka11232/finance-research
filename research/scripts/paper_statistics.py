"""Statistics added for the paper draft (H5, H6 regression, global multiple-testing check).

Writes
  outputs/tables/paper_pooled_dm.csv        pooled five-country DM/HLN on Analysis B common dates (H5)
  outputs/tables/paper_global_holm.csv      Holm across all 140 confirmatory m/m tests (robustness of H1-H4)
  outputs/tables/paper_gap_regression.csv   Newey-West regression of the D1-D10 y/y gap on relative food inflation (H6)
  outputs/tables/paper_gap_series.csv       the monthly D1-D10 gap and relative food inflation series
  outputs/tables/paper_sign_counts.csv      cell counts and two-sided sign tests (descriptive; cells are not independent)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.paths import FORECASTS, PROCESSED, TABLES
from src.statistics.forecast_tests import holm, long_run_variance

PAIRS = [("sarima", "seasonal_naive"), ("ets", "seasonal_naive"), ("yu_weighted_fts", "seasonal_naive"),
         ("yu_weighted_fts", "sarima"), ("yu_weighted_fts", "ets"), ("yu_weighted_fts", "yu_equal_fts"),
         ("yu_weighted_fts", "chen_fts")]


def pooled_dm() -> pd.DataFrame:
    fc = pd.read_csv(FORECASTS / "international_backtest_forecasts.csv.gz", parse_dates=["origin", "target_date"])
    fc = fc[(fc["analysis"] == "B") & (fc["split"] == "test") & (fc["target"] == "mom")]
    rows = []
    for variable, fcol, acol in (("mom", "forecast", "actual"), ("yoy_implied", "yoy_forecast", "yoy_actual")):
        for loss in ("se", "ae"):
            for h in (1, 3, 6, 12):
                g = fc[fc["h"] == h]
                fam = []
                for m1, m2 in PAIRS:
                    e = {}
                    for m in (m1, m2):
                        x = g[g["model"] == m]
                        e[m] = (x[acol] - x[fcol]).groupby([x["origin"], x["country"]]).first().unstack("country")
                    L = (lambda v: v**2) if loss == "se" else np.abs
                    d = (L(e[m1]) - L(e[m2])).dropna()  # rows = common origins, all 5 countries present
                    dbar = d.mean(axis=1).to_numpy()  # cross-country mean loss differential per origin
                    n = len(dbar)
                    lrv, kernel = long_run_variance(dbar, h)
                    se = np.sqrt(lrv / n)
                    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
                    stat = dbar.mean() / se * adj
                    p = 2 * stats.t.sf(abs(stat), df=n - 1)
                    rows.append({"variable": variable, "loss": loss, "horizon": h, "model": m1, "comparator": m2,
                                 "n_months": n, "n_countries": d.shape[1], "mean_loss_diff": dbar.mean(),
                                 "hln_stat": stat, "p_value": p, "kernel": kernel,
                                 "first_target": f"{(d.index.min() + pd.DateOffset(months=h)):%Y-%m}",
                                 "last_target": f"{(d.index.max() + pd.DateOffset(months=h)):%Y-%m}"})
                    fam.append(len(rows) - 1)
                for i, v in zip(fam, holm([rows[i]["p_value"] for i in fam])):
                    rows[i]["p_holm"] = v
    out = pd.DataFrame(rows)
    out["conclusion"] = np.where(out["p_holm"] < 0.05, np.where(out["mean_loss_diff"] < 0, "model significantly more accurate", "comparator significantly more accurate"),
                                 "not statistically distinguishable at 5%")
    return out


def global_holm() -> pd.DataFrame:
    s = pd.read_csv(TABLES / "FORECAST_SIGNIFICANCE.csv")
    c = s[(s["family"] == "confirmatory") & (s["variable"] == "mom") & (s["analysis"] == "A") & (s["loss"] == "se")].copy()
    c["p_holm_all140"] = holm(c["p_value"].to_numpy())
    c["significant_within_cell"] = c["p_holm"] < 0.05
    c["significant_all140"] = c["p_holm_all140"] < 0.05
    return c[["country", "horizon", "model", "comparator", "mean_loss_diff", "p_value", "p_holm", "p_holm_all140",
              "significant_within_cell", "significant_all140"]]


def gap_regression() -> tuple[pd.DataFrame, pd.DataFrame]:
    P = pd.read_csv(PROCESSED / "aze_cpi_coicop_monthly_2010_100.csv", index_col=0, parse_dates=True).dropna()
    w = pd.read_csv(PROCESSED / "official_shares_2025_coicop.csv")
    cps = [f"CP{i:02d}" for i in range(1, 13)]
    rel = P[cps] / P[cps][P.index.year == 2025].mean()

    def idx(grp):
        s = w[(w["dimension"] == "expenditure_decile") & (w["group"] == grp)][cps].iloc[0]
        return rel @ (s / s.sum())

    d1, d10 = idx("D1"), idx("D10")
    ser = pd.DataFrame({
        "gap_d1_minus_d10_pp": 100 * (d1 / d1.shift(12) - 1) - 100 * (d10 / d10.shift(12) - 1),
        "food_yoy": 100 * (P["CP01"] / P["CP01"].shift(12) - 1),
        "all_items_yoy": 100 * (P["ALL"] / P["ALL"].shift(12) - 1),
    }).dropna()
    ser["relative_food_inflation_pp"] = ser["food_yoy"] - ser["all_items_yoy"]
    rows = []
    for name, sub in (("2006-01..2026-03", ser), ("2016-01..2026-03", ser.loc["2016-01-01":])):
        X = sm.add_constant(sub["relative_food_inflation_pp"])
        res = sm.OLS(sub["gap_d1_minus_d10_pp"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        rows.append({"sample": name, "n_months": int(res.nobs), "slope": res.params.iloc[1], "slope_se_nw": res.bse.iloc[1],
                     "slope_ci95_low": res.conf_int().iloc[1, 0], "slope_ci95_high": res.conf_int().iloc[1, 1],
                     "slope_p": res.pvalues.iloc[1], "intercept": res.params.iloc[0], "intercept_p": res.pvalues.iloc[0],
                     "r2": res.rsquared, "share_months_gap_positive": float((sub["gap_d1_minus_d10_pp"] > 0).mean()),
                     "gap_max": sub["gap_d1_minus_d10_pp"].max(), "gap_max_month": f"{sub['gap_d1_minus_d10_pp'].idxmax():%Y-%m}",
                     "gap_min": sub["gap_d1_minus_d10_pp"].min(), "gap_min_month": f"{sub['gap_d1_minus_d10_pp'].idxmin():%Y-%m}"})
    return pd.DataFrame(rows), ser.reset_index(names="date")


def sign_counts() -> pd.DataFrame:
    a = pd.read_csv(TABLES / "INTERNATIONAL_FORECAST_COMPARISON.csv")
    a = a[(a["variable"] == "mom") & (a["analysis"] == "A")]
    p = a.pivot_table(index=["country", "horizon"], columns="model", values="RMSE")
    rows = []
    for m1, m2 in PAIRS:
        wins = int((p[m1] < p[m2]).sum())
        rows.append({"model": m1, "comparator": m2, "cells": len(p), "model_lower_rmse": wins,
                     "sign_test_p_two_sided": stats.binomtest(wins, len(p), 0.5).pvalue,
                     "note": "descriptive: the 20 cells share countries and overlapping periods, so they are not independent"})
    return pd.DataFrame(rows)


def main():
    pooled_dm().to_csv(TABLES / "paper_pooled_dm.csv", index=False)
    global_holm().to_csv(TABLES / "paper_global_holm.csv", index=False)
    reg, ser = gap_regression()
    reg.to_csv(TABLES / "paper_gap_regression.csv", index=False)
    ser.to_csv(TABLES / "paper_gap_series.csv", index=False)
    sign_counts().to_csv(TABLES / "paper_sign_counts.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
