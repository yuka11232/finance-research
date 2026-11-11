"""Phase 5: accuracy, skill, significance, interval coverage and instability.

Reads outputs/forecasts/international_backtest_forecasts.csv.gz and writes
  outputs/tables/INTERNATIONAL_FORECAST_COMPARISON.csv
  outputs/tables/FORECAST_SIGNIFICANCE.csv
  outputs/tables/FORECAST_RESULTS.csv.gz           (test and future forecasts)
  outputs/tables/forecast_skill_bootstrap.csv
  outputs/tables/forecast_interval_coverage.csv
  outputs/tables/forecast_subperiod_accuracy.csv
  outputs/tables/fuzzy_weighting_comparison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.backtest import add_empirical_intervals
from src.evaluation.metrics import bias, coverage, mae, mase_scale, rmse, smape
from src.paths import FORECASTS, PROCESSED, TABLES, config
from src.statistics.forecast_tests import bootstrap_skill, diebold_mariano, holm

H_EVAL = [1, 3, 6, 12]
PRIMARY = ["seasonal_naive", "sarima", "ets", "yu_weighted_fts"]
COMPARISONS = [  # (model, comparator) -- confirmatory family, Holm-adjusted jointly
    ("sarima", "seasonal_naive"),
    ("ets", "seasonal_naive"),
    ("yu_weighted_fts", "seasonal_naive"),
    ("yu_weighted_fts", "sarima"),
    ("yu_weighted_fts", "ets"),
    ("yu_weighted_fts", "yu_equal_fts"),
    ("yu_weighted_fts", "chen_fts"),
]
EXPLORATORY = [("naive", "seasonal_naive"), ("ar_aic", "seasonal_naive"), ("yu_equal_fts", "chen_fts")]
MIN_N_TEST = 30


def long_variables(fc: pd.DataFrame) -> pd.DataFrame:
    """Stack the evaluated variables: m/m, implied y/y (from m/m models) and direct y/y."""
    keep = ["analysis", "country", "model", "spec", "origin", "target_date", "h", "split"]
    ints = ["model_lo80", "model_hi80", "model_lo95", "model_hi95"]
    mom = fc[fc["target"] == "mom"]
    a = mom[keep + ["forecast", "actual"] + ints].assign(variable="mom")
    b = mom[keep + ["yoy_forecast", "yoy_actual"]].rename(columns={"yoy_forecast": "forecast", "yoy_actual": "actual"})
    b = b.assign(variable="yoy_implied")
    c = fc[fc["target"] == "yoy"][keep + ["forecast", "actual"] + ints].assign(variable="yoy_direct")
    out = pd.concat([a, b, c], ignore_index=True)
    out["error"] = out["actual"] - out["forecast"]
    return out


def mase_scales() -> dict:
    cfg = config()
    h = pd.read_csv(PROCESSED / "cpi_headline_monthly.csv", parse_dates=["date"])
    test_start = pd.Timestamp(cfg["forecast"]["test_start"])
    out = {}
    for analysis in ("A", "B"):
        start = pd.Timestamp(cfg["data"]["analysis_a_start" if analysis == "A" else "analysis_b_start"])
        for c, g in h.groupby("country"):
            g = g[(g["date"] >= start) & (g["date"] < test_start)]
            out[(analysis, c, "mom")] = mase_scale(g["mom"].dropna().to_numpy())
            yy = mase_scale(g["yoy"].dropna().to_numpy())
            out[(analysis, c, "yoy_implied")] = out[(analysis, c, "yoy_direct")] = yy
    return out


def accuracy_table(lv: pd.DataFrame, scales: dict) -> pd.DataFrame:
    test = lv[(lv["split"] == "test") & lv["h"].isin(H_EVAL)]
    rows = []
    for key, g in test.groupby(["variable", "analysis", "country", "model", "h"]):
        variable, analysis, country, model, h = key
        rows.append({
            "variable": variable, "analysis": analysis, "country": country, "model": model, "spec": g["spec"].iloc[0],
            "horizon": h,
            "evaluation_period": f"{g['target_date'].min():%Y-%m}..{g['target_date'].max():%Y-%m}",
            "first_origin": f"{g['origin'].min():%Y-%m}", "last_origin": f"{g['origin'].max():%Y-%m}",
            "MAE": mae(g["actual"], g["forecast"]), "RMSE": rmse(g["actual"], g["forecast"]),
            "MASE": mae(g["actual"], g["forecast"]) / scales[(analysis, country, variable)],
            "sMAPE": smape(g["actual"], g["forecast"]), "Bias": bias(g["actual"], g["forecast"]),
            "n_forecasts": len(g),
        })
    acc = pd.DataFrame(rows)
    sn = acc[acc["model"] == "seasonal_naive"].set_index(["variable", "analysis", "country", "horizon"])["RMSE"]
    rw = acc[acc["model"] == "naive"].set_index(["variable", "analysis", "country", "horizon"])["RMSE"]
    k = acc.set_index(["variable", "analysis", "country", "horizon"]).index
    acc["skill_vs_seasonal_naive"] = 1 - acc["RMSE"].to_numpy() / sn.reindex(k).to_numpy()
    acc["skill_vs_random_walk"] = 1 - acc["RMSE"].to_numpy() / rw.reindex(k).to_numpy()
    acc["model_role"] = np.where(acc["model"].isin(PRIMARY), "primary", np.where(acc["model"].isin(["yu_equal_fts", "chen_fts"]), "fuzzy_benchmark", "supplementary"))
    return acc


def paired(test: pd.DataFrame, m1: str, m2: str) -> pd.DataFrame:
    a = test[test["model"] == m1].set_index(["origin", "h"])["error"]
    b = test[test["model"] == m2].set_index(["origin", "h"])["error"]
    j = pd.concat([a.rename("e1"), b.rename("e2")], axis=1).dropna()
    return j


def significance(lv: pd.DataFrame, rng: np.random.Generator, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = lv[(lv["split"] == "test") & lv["h"].isin(H_EVAL)]
    rows, boot = [], []
    for (variable, analysis, country, h), g in test.groupby(["variable", "analysis", "country", "h"]):
        for family, pairs in (("confirmatory", COMPARISONS), ("exploratory", EXPLORATORY)):
            for m1, m2 in pairs:
                j = paired(g, m1, m2)
                base = {"variable": variable, "analysis": analysis, "country": country, "horizon": h,
                        "model": m1, "comparator": m2, "family": family, "n_pairs": len(j),
                        "evaluation_period": f"{j.index.get_level_values(0).min() + pd.DateOffset(months=h):%Y-%m}.."
                                             f"{j.index.get_level_values(0).max() + pd.DateOffset(months=h):%Y-%m}" if len(j) else ""}
                for loss in ("se", "ae"):
                    if len(j) < MIN_N_TEST:
                        rows.append({**base, "loss": loss, "tested": False, "omission_reason": f"n < {MIN_N_TEST}"})
                        continue
                    r = diebold_mariano(j["e1"], j["e2"], h, loss)
                    tested = np.isfinite(r["p_value"])
                    rows.append({**base, "loss": loss, "tested": bool(tested),
                                 "omission_reason": "" if tested else "identical forecasts / zero variance", **r})
                if m2 in ("seasonal_naive",) or (m1, m2) == ("yu_weighted_fts", "yu_equal_fts"):
                    if len(j) >= MIN_N_TEST:
                        boot.append({**base, **bootstrap_skill(j["e1"], j["e2"], h, reps, rng)})
    sig = pd.DataFrame(rows)
    sig["p_holm"] = np.nan
    fam = sig["family"] == "confirmatory"
    for _, g in sig[fam].groupby(["variable", "analysis", "country", "horizon", "loss"]):
        sig.loc[g.index, "p_holm"] = holm(g["p_value"].to_numpy())
    sig["direction"] = np.where(sig["mean_loss_diff"] < 0, "model lower loss", np.where(sig["mean_loss_diff"] > 0, "comparator lower loss", "equal"))

    def verdict(r):
        if not r.get("tested", False):
            return "not tested"
        p = r["p_holm"] if r["family"] == "confirmatory" else r["p_value"]
        if p < 0.05:
            return "model significantly more accurate" if r["mean_loss_diff"] < 0 else "comparator significantly more accurate"
        return "not statistically distinguishable at 5%"

    sig["conclusion"] = sig.apply(verdict, axis=1)
    return sig, pd.DataFrame(boot)


def coverage_table(lv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for _, g in lv[lv["split"] != "future"].groupby(["variable", "analysis", "country", "model"]):
        parts.append(add_empirical_intervals(g))
    withint = pd.concat(parts, ignore_index=True)
    test = withint[(withint["split"] == "test") & withint["h"].isin(H_EVAL)]
    rows = []
    for key, g in test.groupby(["variable", "analysis", "country", "model", "h"]):
        for kind in ("model", "emp"):
            for lvl in (80, 95):
                cov, n = coverage(g["actual"], g[f"{kind}_lo{lvl}"], g[f"{kind}_hi{lvl}"])
                if n:
                    rows.append(dict(zip(["variable", "analysis", "country", "model", "horizon"], key),
                                     interval_type="model-based" if kind == "model" else "empirical (past OOS errors)",
                                     nominal=lvl / 100, empirical_coverage=cov, n=n,
                                     mean_width=float((g[f"{kind}_hi{lvl}"] - g[f"{kind}_lo{lvl}"]).mean())))
    return pd.DataFrame(rows), withint


def subperiod_table(lv: pd.DataFrame) -> pd.DataFrame:
    sub = config()["forecast"]["subperiods"]
    test = lv[(lv["split"] == "test") & lv["h"].isin(H_EVAL)]
    rows = []
    for name, (a, b) in sub.items():
        t = test[(test["target_date"] >= a) & (test["target_date"] <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))]
        for key, g in t.groupby(["variable", "analysis", "country", "model", "h"]):
            rows.append(dict(zip(["variable", "analysis", "country", "model", "horizon"], key), subperiod=name,
                             RMSE=rmse(g["actual"], g["forecast"]), MAE=mae(g["actual"], g["forecast"]),
                             n=len(g), sd_actual=float(g["actual"].std())))
    out = pd.DataFrame(rows)
    sn = out[out["model"] == "seasonal_naive"].set_index(["variable", "analysis", "country", "horizon", "subperiod"])["RMSE"]
    out["skill_vs_seasonal_naive"] = 1 - out["RMSE"].to_numpy() / sn.reindex(out.set_index(["variable", "analysis", "country", "horizon", "subperiod"]).index).to_numpy()
    return out


def main():
    cfg = config()
    rng = np.random.default_rng(cfg["project"]["seed"])
    reps = cfg["project"]["bootstrap_reps"]
    fc = pd.read_csv(FORECASTS / "international_backtest_forecasts.csv.gz", parse_dates=["origin", "target_date"])
    lv = long_variables(fc)
    acc = accuracy_table(lv, mase_scales())
    acc.to_csv(TABLES / "INTERNATIONAL_FORECAST_COMPARISON.csv", index=False)
    sig, boot = significance(lv, rng, reps)
    sig.to_csv(TABLES / "FORECAST_SIGNIFICANCE.csv", index=False)
    boot.to_csv(TABLES / "forecast_skill_bootstrap.csv", index=False)
    cov, withint = coverage_table(lv)
    cov.to_csv(TABLES / "forecast_interval_coverage.csv", index=False)
    subperiod_table(lv).to_csv(TABLES / "forecast_subperiod_accuracy.csv", index=False)
    # forecast-level results: test + genuine future forecasts from the last origin
    fut = lv[lv["split"] == "future"]
    fut = fut[fut["origin"] == fut.groupby(["variable", "analysis", "country", "model"])["origin"].transform("max")]
    res = pd.concat([withint[withint["split"] == "test"], fut], ignore_index=True)
    res.sort_values(["variable", "analysis", "country", "model", "origin", "h"]).to_csv(
        TABLES / "FORECAST_RESULTS.csv.gz", index=False)
    # weighted vs unweighted fuzzy summary
    f = acc[acc["model"].isin(["yu_weighted_fts", "yu_equal_fts", "chen_fts"])]
    piv = f.pivot_table(index=["variable", "analysis", "country", "horizon"], columns="model", values="RMSE").reset_index()
    piv["weighted_minus_equal_rmse"] = piv["yu_weighted_fts"] - piv["yu_equal_fts"]
    piv["weighted_minus_chen_rmse"] = piv["yu_weighted_fts"] - piv["chen_fts"]
    piv.to_csv(TABLES / "fuzzy_weighting_comparison.csv", index=False)
    print("accuracy rows", len(acc), "| significance rows", len(sig), "| bootstrap rows", len(boot), "| coverage rows", len(cov))


if __name__ == "__main__":
    main()
