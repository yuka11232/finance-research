"""Phase 8c: direct benchmarks for the D1-D10 annual inflation gap and the paired gap tests (paper Tables 4-5).

The gap is Delta_t = pi_D1,t - pi_D10,t, the difference between the two deciles' year-on-year
fixed-2025-basket inflation rates. Three direct forecasts of Delta_{T+h} are made at every origin T:
  zero_gap          0
  last_gap          Delta_T (persistence)
  direct_ar1        Delta_t = a + b Delta_{t-1} + u_t fitted by OLS to all gap observations through T,
                    iterated forward from Delta_T
They use the same origins and targets as the bottom-up group backtest (run_integration.py), whose
D1-D10 forecasts supply the four division-level models. Bottom-up SARIMA is compared with the six
alternatives at h = 1, 3, 6, 12 by two-sided DM tests with the HLN correction and squared loss,
Holm-adjusted jointly over all 24 tests.

Outputs (outputs/tables):
  gap_forecasts.csv          every gap forecast: model, origin, h, target, actual and forecast gap
  GAP_FORECAST_ACCURACY.csv  RMSE / MAE by model and horizon (paper Table 4, gap rows)
  GAP_FORECAST_TESTS.csv     the 24 DM/HLN tests with Holm-adjusted p-values (paper Table 5)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, rmse
from src.forecasting.gap import direct_gap_forecasts
from src.inflation.household import COICOP
from src.paths import FORECASTS, PROCESSED, TABLES, config
from src.statistics.forecast_tests import diebold_mariano, holm

H_EVAL = [1, 3, 6, 12]
BOTTOM_UP = ["seasonal_naive", "sarima", "ets", "yu_weighted_fts"]
DIRECT = ["zero_gap", "last_gap", "direct_ar1"]
COMPARATORS = ["zero_gap", "last_gap", "direct_ar1", "seasonal_naive", "ets", "yu_weighted_fts"]
D1, D10 = "official|expenditure_decile|D1", "official|expenditure_decile|D10"


def actual_gap() -> pd.Series:
    """Monthly D1-D10 year-on-year gap under fixed 2025 baskets (same construction as the group backtest)."""
    P = pd.read_csv(PROCESSED / "aze_cpi_coicop_monthly_2010_100.csv", index_col=0, parse_dates=True).dropna()
    rel = P[COICOP] / P[COICOP][P.index.year == config()["household"]["price_reference_year"]].mean()
    w = pd.read_csv(PROCESSED / "official_shares_2025_coicop.csv")
    w = w[w["dimension"] == "expenditure_decile"].set_index("group")[COICOP]
    w = w.div(w.sum(axis=1), axis=0)
    yoy = 100 * ((rel @ w.T) / (rel @ w.T).shift(12) - 1)
    return (yoy["D1"] - yoy["D10"]).dropna()


def main():
    g = pd.read_csv(FORECASTS / "group_backtest_forecasts.csv.gz", parse_dates=["origin", "target_date"])
    g = g[(g["split"] == "test") & g["h"].isin(H_EVAL) & g["model"].isin(BOTTOM_UP) & g["unit"].isin([D1, D10])]
    w = g.pivot_table(index=["model", "origin", "h", "target_date"], columns="unit", values=["yoy_forecast", "yoy_actual"])
    bu = pd.DataFrame({"forecast_gap": w[("yoy_forecast", D1)] - w[("yoy_forecast", D10)],
                       "actual_gap": w[("yoy_actual", D1)] - w[("yoy_actual", D10)]}).reset_index()
    gap = actual_gap()
    origins = bu[bu["model"] == "sarima"][["origin", "h", "target_date", "actual_gap"]]
    worst = (origins.set_index("target_date")["actual_gap"] - gap.reindex(origins["target_date"]).to_numpy()).abs().max()
    assert worst < 1e-9, f"backtest gap targets differ from the gap series by {worst}"
    fc = pd.concat([bu, direct_gap_forecasts(gap, origins)], ignore_index=True)
    fc["error"] = fc["actual_gap"] - fc["forecast_gap"]
    fc.sort_values(["model", "h", "origin"]).to_csv(TABLES / "gap_forecasts.csv", index=False)

    acc = [{"model": m, "horizon": h, "n": len(t), "RMSE_gap_pp": rmse(t["actual_gap"], t["forecast_gap"]),
            "MAE_gap_pp": mae(t["actual_gap"], t["forecast_gap"])} for (m, h), t in fc.groupby(["model", "h"])]
    acc = pd.DataFrame(acc)
    acc["model"] = pd.Categorical(acc["model"], BOTTOM_UP + DIRECT, ordered=True)
    acc.sort_values(["model", "horizon"]).to_csv(TABLES / "GAP_FORECAST_ACCURACY.csv", index=False)

    tests = []
    e = fc.set_index(["model", "h", "origin"])["error"].sort_index()
    for comp in COMPARATORS:
        for h in H_EVAL:
            j = pd.concat([e.loc["sarima", h].rename("e1"), e.loc[comp, h].rename("e2")], axis=1).dropna()
            tests.append({"model": "sarima", "comparator": comp, "horizon": h, "loss": "se", **diebold_mariano(j["e1"], j["e2"], h, "se")})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p_value"])
    tests["sarima_lower_mse"] = tests["mean_loss_diff"] < 0
    tests.to_csv(TABLES / "GAP_FORECAST_TESTS.csv", index=False)
    print(acc.pivot_table(index="model", columns="horizon", values="RMSE_gap_pp", observed=True).round(3).to_string())
    print(tests.pivot_table(index="comparator", columns="horizon", values="p_holm").round(3).to_string())


if __name__ == "__main__":
    main()
