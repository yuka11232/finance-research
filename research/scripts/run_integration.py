"""Phase 8b: combine COICOP-division forecasts with household/group baskets.

For every model and origin T, division m/m forecasts are chained into index
paths, aggregated with each unit's 2025 basket (official HBS groups and
quality-controlled survey households) and converted into year-on-year
inflation at T+h. Because h <= 12, the y/y denominator I_g(T+h-12) is always
observed. Backtest accuracy is evaluated at the group level, and the final
origin (2026-03) gives projections for 2026-04..2027-03 with empirical
intervals from the group-level backtest errors.

Outputs (outputs/tables unless noted):
  category_forecast_accuracy.csv, category_forecast_significance.csv
  group_forecast_accuracy.csv, group_forecast_significance.csv, group_gap_forecast_accuracy.csv
  HOUSEHOLD_INFLATION_PROJECTIONS.csv, household_projection_model_agreement.csv
  headline_out_of_sample_check.csv
  outputs/forecasts/group_backtest_forecasts.csv.gz
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import mae, rmse
from src.ingestion.hbs_tables import parse_2025_structures, to_coicop
from src.inflation.household import COICOP, SURVEY_MAP_PRIMARY, survey_to_coicop_shares
from src.paths import FORECASTS, PROCESSED, TABLES, config
from src.statistics.forecast_tests import diebold_mariano, holm
from src.survey.clean import CATEGORIES

MODELS = ["seasonal_naive", "naive", "sarima", "ets", "yu_weighted_fts", "yu_equal_fts", "chen_fts", "ar_aic"]
PRIMARY = ["seasonal_naive", "sarima", "ets", "yu_weighted_fts"]
H_EVAL = [1, 3, 6, 12]
KEY_GROUPS = [("area", "national"), ("area", "urban"), ("area", "rural"), ("expenditure_quintile", "Q1"),
              ("expenditure_quintile", "Q5"), ("expenditure_decile", "D1"), ("expenditure_decile", "D10")]
CODES = COICOP + ["ALL"]


def units() -> pd.DataFrame:
    """Rows = units, columns = CODES, shares summing to one."""
    w = to_coicop(parse_2025_structures()).set_index(["dimension", "group"])[COICOP]
    off = w.div(w.sum(axis=1), axis=0)
    off["ALL"] = 0.0
    off.index = [f"official|{d}|{g}" for d, g in off.index]
    hh = pd.read_csv(TABLES / "HOUSEHOLD_EXPENDITURE.csv")  # anonymised survey file written by the household step
    hh = hh[hh["in_quality_controlled"]]
    sh = survey_to_coicop_shares(hh[CATEGORIES], SURVEY_MAP_PRIMARY)
    sh.index = [f"survey|household|{r}" for r in hh["respondent_id"]]
    return pd.concat([off, sh[CODES]]).fillna(0.0)


def category_tables(fc: pd.DataFrame):
    test = fc[(fc["split"] == "test") & fc["h"].isin(H_EVAL)]
    rows, sig = [], []
    for (code, h), g in test.groupby(["coicop", "h"]):
        sn = g[g["model"] == "seasonal_naive"].set_index("origin")
        for m, gm in g.groupby("model"):
            rows.append({"coicop": code, "horizon": h, "model": m, "spec": gm["spec"].iloc[0],
                         "evaluation_period": f"{gm['target_date'].min():%Y-%m}..{gm['target_date'].max():%Y-%m}",
                         "n_forecasts": len(gm), "MAE": mae(gm["actual"], gm["forecast"]), "RMSE": rmse(gm["actual"], gm["forecast"])})
        fam = []
        for m in ("sarima", "ets", "yu_weighted_fts"):
            gm = g[g["model"] == m].set_index("origin")
            j = pd.concat([(gm["actual"] - gm["forecast"]).rename("e1"), (sn["actual"] - sn["forecast"]).rename("e2")], axis=1).dropna()
            sig.append({"coicop": code, "horizon": h, "model": m, "comparator": "seasonal_naive", "loss": "se",
                        **diebold_mariano(j["e1"], j["e2"], h, "se")})
            fam.append(len(sig) - 1)
        adj = holm([sig[i]["p_value"] for i in fam])
        for i, v in zip(fam, adj):
            sig[i]["p_holm"] = v
    acc = pd.DataFrame(rows)
    snr = acc[acc["model"] == "seasonal_naive"].set_index(["coicop", "horizon"])["RMSE"]
    acc["skill_vs_seasonal_naive"] = 1 - acc["RMSE"].to_numpy() / snr.reindex(acc.set_index(["coicop", "horizon"]).index).to_numpy()
    return acc, pd.DataFrame(sig)


def group_backtest(fc: pd.DataFrame, P: pd.DataFrame, U: pd.DataFrame) -> pd.DataFrame:
    rel = P[CODES] / P[CODES][P.index.year == config()["household"]["price_reference_year"]].mean()
    actual_I = rel[CODES].to_numpy() @ U[CODES].to_numpy().T  # dates x units
    actual_I = pd.DataFrame(actual_I, index=rel.index, columns=U.index)
    out = []
    for m in MODELS:
        f = fc[fc["model"] == m]
        wide = f.pivot_table(index=["origin", "h"], columns="coicop", values="forecast")[CODES]
        origins = wide.index.get_level_values(0).unique()
        for T in origins:
            path = wide.loc[T].sort_index()  # h x codes (m/m %)
            rel_hat = rel.loc[T, CODES].to_numpy() * np.cumprod(1 + path.to_numpy() / 100.0, axis=0)
            I_hat = rel_hat @ U[CODES].to_numpy().T  # h x units
            hs = path.index.to_numpy()
            tgt = [T + pd.DateOffset(months=int(h)) for h in hs]
            base = actual_I.loc[[t - pd.DateOffset(months=12) for t in tgt]].to_numpy()
            yoy_hat = 100 * (I_hat / base - 1)
            act = actual_I.reindex(tgt).to_numpy()
            yoy_act = 100 * (act / base - 1)
            keep = np.isin(hs, H_EVAL) | (T == origins.max())
            for i in np.where(keep)[0]:
                out.append(pd.DataFrame({"model": m, "origin": T, "h": int(hs[i]), "target_date": tgt[i], "unit": U.index,
                                         "yoy_forecast": yoy_hat[i], "yoy_actual": yoy_act[i]}))
    g = pd.concat(out, ignore_index=True)
    test_origin = pd.Timestamp(config()["forecast"]["test_start"]) - pd.DateOffset(months=1)
    g["split"] = np.where(g["yoy_actual"].isna(), "future", np.where(g["origin"] >= test_origin, "test", "pre-test"))
    return g


def group_tables(g: pd.DataFrame):
    g = g.assign(error=g["yoy_actual"] - g["yoy_forecast"])
    test = g[(g["split"] == "test") & g["h"].isin(H_EVAL)]
    acc = test.groupby(["unit", "model", "h"]).agg(n=("error", "size"), RMSE=("error", lambda e: float(np.sqrt(np.mean(e**2)))),
                                                   MAE=("error", lambda e: float(np.mean(np.abs(e)))),
                                                   Bias=("error", lambda e: float(-np.mean(e)))).reset_index()
    sn = acc[acc["model"] == "seasonal_naive"].set_index(["unit", "h"])["RMSE"]
    acc["skill_vs_seasonal_naive"] = 1 - acc["RMSE"].to_numpy() / sn.reindex(acc.set_index(["unit", "h"]).index).to_numpy()
    sig = []
    keys = [f"official|{d}|{x}" for d, x in KEY_GROUPS]
    for (unit, h), t in test[test["unit"].isin(keys)].groupby(["unit", "h"]):
        e2 = t[t["model"] == "seasonal_naive"].set_index("origin")["error"]
        fam = []
        for m in ("sarima", "ets", "yu_weighted_fts"):
            e1 = t[t["model"] == m].set_index("origin")["error"]
            j = pd.concat([e1.rename("e1"), e2.rename("e2")], axis=1).dropna()
            sig.append({"unit": unit, "horizon": h, "model": m, "comparator": "seasonal_naive", **diebold_mariano(j["e1"], j["e2"], h)})
            fam.append(len(sig) - 1)
        for i, v in zip(fam, holm([sig[i]["p_value"] for i in fam])):
            sig[i]["p_holm"] = v
    # gap forecasts (poorest minus richest)
    gaps = []
    for a, b, name in (("official|expenditure_decile|D1", "official|expenditure_decile|D10", "D1-D10"),
                       ("official|expenditure_quintile|Q1", "official|expenditure_quintile|Q5", "Q1-Q5")):
        x = test[test["unit"] == a].set_index(["model", "origin", "h"])
        y = test[test["unit"] == b].set_index(["model", "origin", "h"])
        d = pd.DataFrame({"fc": x["yoy_forecast"] - y["yoy_forecast"], "act": x["yoy_actual"] - y["yoy_actual"]}).reset_index()
        for (m, h), t in d.groupby(["model", "h"]):
            gaps.append({"gap": name, "model": m, "horizon": h, "n": len(t), "RMSE_gap_pp": rmse(t["act"], t["fc"]),
                         "MAE_gap_pp": mae(t["act"], t["fc"]), "mean_actual_gap": t["act"].mean(), "mean_forecast_gap": t["fc"].mean(),
                         "sign_agreement": float(np.mean(np.sign(t["act"]) == np.sign(t["fc"])))})
    return acc, pd.DataFrame(sig), pd.DataFrame(gaps)


def projections(g: pd.DataFrame) -> pd.DataFrame:
    g = g.assign(error=g["yoy_actual"] - g["yoy_forecast"])
    last = g["origin"].max()
    fut = g[(g["origin"] == last)].copy()
    hist = g[(g["split"] != "future")]
    rows = []
    for (unit, m, h), t in fut.groupby(["unit", "model", "h"]):
        e = hist[(hist["unit"] == unit) & (hist["model"] == m) & (hist["h"] == h)]["error"].dropna().to_numpy()
        f = float(t["yoy_forecast"].iloc[0])
        r = {"unit": unit, "model": m, "origin": last, "h": h, "target_date": t["target_date"].iloc[0], "yoy_forecast": f,
             "n_past_errors": len(e)}
        if h in H_EVAL and len(e) >= 24:
            for lvl in (80, 95):
                a = (1 - lvl / 100) / 2
                r[f"lo{lvl}"], r[f"hi{lvl}"] = f + np.quantile(e, a), f + np.quantile(e, 1 - a)
        rows.append(r)
    out = pd.DataFrame(rows)
    out[["unit_type", "dimension", "group"]] = out["unit"].str.split("|", expand=True)
    return out


def agreement(proj: pd.DataFrame) -> pd.DataFrame:
    """Do model choices change which households/groups face the highest projected inflation?"""
    rows = []
    for ut in ("official", "survey"):
        p = proj[(proj["unit_type"] == ut) & (proj["h"] == 12) & proj["model"].isin(PRIMARY)]
        wide = p.pivot_table(index="unit", columns="model", values="yoy_forecast")
        for i, a in enumerate(PRIMARY):
            for b in PRIMARY[i + 1:]:
                rho = stats.spearmanr(wide[a], wide[b]).statistic
                top_a = set(wide[a].nlargest(max(1, len(wide) // 5)).index)
                top_b = set(wide[b].nlargest(max(1, len(wide) // 5)).index)
                rows.append({"unit_type": ut, "model_a": a, "model_b": b, "n_units": len(wide), "spearman_rank_corr": rho,
                             "top20pct_overlap": len(top_a & top_b) / len(top_a),
                             "mean_abs_level_diff_pp": float((wide[a] - wide[b]).abs().mean()),
                             "range_a_pp": float(wide[a].max() - wide[a].min()), "range_b_pp": float(wide[b].max() - wide[b].min())})
    return pd.DataFrame(rows)


def headline_check(fc: pd.DataFrame) -> pd.DataFrame:
    """Forecasts of all-items inflation from 2026-03 vs AZSTAT outcomes published for 2026-04..2026-09."""
    h = pd.read_csv(PROCESSED / "cpi_headline_monthly.csv", parse_dates=["date"])
    az = h[h["country"] == "AZE"].set_index("date")
    f = fc[(fc["coicop"] == "ALL") & (fc["split"] == "future")]
    f = f[f["origin"] == f["origin"].max()]
    out = f[["model", "h", "target_date", "forecast", "yoy_forecast"]].copy()
    out["actual_mom_azstat"] = out["target_date"].map(az["mom"])
    out["actual_yoy_azstat"] = out["target_date"].map(az["yoy"])
    out["source_of_actual"] = out["target_date"].map(az["source"])
    out = out.dropna(subset=["actual_yoy_azstat"])
    out["yoy_error"] = out["actual_yoy_azstat"] - out["yoy_forecast"]
    return out


def main():
    fc = pd.read_csv(FORECASTS / "aze_category_backtest_forecasts.csv.gz", parse_dates=["origin", "target_date"])
    P = pd.read_csv(PROCESSED / "aze_cpi_coicop_monthly_2010_100.csv", index_col=0, parse_dates=True).dropna()
    acc, sig = category_tables(fc)
    acc.to_csv(TABLES / "category_forecast_accuracy.csv", index=False)
    sig.to_csv(TABLES / "category_forecast_significance.csv", index=False)
    U = units()
    g = group_backtest(fc, P, U)
    g.to_csv(FORECASTS / "group_backtest_forecasts.csv.gz", index=False, compression="gzip")
    gacc, gsig, gaps = group_tables(g)
    gacc.to_csv(TABLES / "group_forecast_accuracy.csv", index=False)
    gsig.to_csv(TABLES / "group_forecast_significance.csv", index=False)
    gaps.to_csv(TABLES / "group_gap_forecast_accuracy.csv", index=False)
    proj = projections(g)
    proj.to_csv(TABLES / "HOUSEHOLD_INFLATION_PROJECTIONS.csv", index=False)
    agreement(proj).to_csv(TABLES / "household_projection_model_agreement.csv", index=False)
    headline_check(fc).to_csv(TABLES / "headline_out_of_sample_check.csv", index=False)
    print("units", len(U), "| group rows", len(g), "| projections", len(proj))


if __name__ == "__main__":
    main()
