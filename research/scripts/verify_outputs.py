"""Phase 10: independent numerical consistency checks and checks of numbers quoted in the reports.

Every check recomputes from the most primitive saved output it can (raw backtest forecasts,
processed price indices, survey file) rather than re-reading the summary table it verifies.
Exit code 1 if any check fails; results are written to outputs/logs/verification_report.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.paths import FORECASTS, LOGS, PROCESSED, TABLES

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "passed": bool(ok), "detail": detail})


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def main():
    fc = pd.read_csv(FORECASTS / "international_backtest_forecasts.csv.gz", parse_dates=["origin", "target_date"])
    acc = pd.read_csv(TABLES / "INTERNATIONAL_FORECAST_COMPARISON.csv")
    sig = pd.read_csv(TABLES / "FORECAST_SIGNIFICANCE.csv")

    # 1. RMSE recomputed from raw forecasts (no look at the summary computation path)
    t = fc[(fc["target"] == "mom") & (fc["split"] == "test")]
    worst = 0.0
    for (an, c, m, h), g in t[t["h"].isin([1, 3, 6, 12])].groupby(["analysis", "country", "model", "h"]):
        r = np.sqrt(np.mean((g["actual"] - g["forecast"]) ** 2))
        rep = acc[(acc["variable"] == "mom") & (acc["analysis"] == an) & (acc["country"] == c) & (acc["model"] == m) & (acc["horizon"] == h)]["RMSE"].iloc[0]
        worst = max(worst, abs(r - rep))
    check("RMSE in comparison table equals recomputation from raw forecasts", worst < 1e-10, f"max diff {worst:.2e}")

    # 2. no test forecast uses an origin before the test period, and no target is beyond the data
    check("test origins >= 2018-12", (t["origin"] >= "2018-12-01").all())
    check("no NaN actuals in test split", t["actual"].notna().all())
    # 3. selection never saw test targets: selection files only score validation targets (by construction) - check spec constant over origins
    nspec = fc.groupby(["target", "analysis", "country", "model"])["spec"].nunique().max()
    check("one frozen specification per model/series", nspec == 1)

    # 4. DM pairs: n equals paired test forecasts
    s1 = sig[(sig["variable"] == "mom") & (sig["analysis"] == "A") & (sig["country"] == "AZE") & (sig["horizon"] == 12) & (sig["model"] == "sarima") & (sig["comparator"] == "seasonal_naive") & (sig["loss"] == "se")]
    n_raw = len(t[(t["analysis"] == "A") & (t["country"] == "AZE") & (t["model"] == "sarima") & (t["h"] == 12)])
    check("AZE h12 DM n equals number of test forecasts (82 quoted)", int(s1["n_pairs"].iloc[0]) == n_raw == 82, f"n={n_raw}")
    n1 = len(t[(t["analysis"] == "A") & (t["country"] == "AZE") & (t["model"] == "sarima") & (t["h"] == 1)])
    check("AZE h1 test forecasts = 93", n1 == 93, f"{n1}")

    # 5. quoted numbers - forecasting
    def A(c, m, h, var="mom", col="RMSE", an="A"):
        return acc[(acc["variable"] == var) & (acc["analysis"] == an) & (acc["country"] == c) & (acc["model"] == m) & (acc["horizon"] == h)][col].iloc[0]

    quoted = {("AZE", "seasonal_naive", 1): 0.689, ("AZE", "sarima", 1): 0.448, ("AZE", "ets", 1): 0.487, ("AZE", "yu_weighted_fts", 1): 0.526,
              ("AZE", "sarima", 12): 0.539, ("AZE", "yu_weighted_fts", 6): 0.845, ("DZA", "ets", 1): 0.813, ("KAZ", "sarima", 1): 0.387,
              ("NOR", "ets", 1): 0.440, ("RUS", "ets", 1): 0.941, ("RUS", "seasonal_naive", 1): 1.150}
    for (c, m, h), v in quoted.items():
        check(f"quoted RMSE {c} {m} h{h} = {v}", close(A(c, m, h), v, 0.0006), f"{A(c, m, h):.4f}")
    check("quoted implied y/y h12 SARIMA AZE 4.38", close(A("AZE", "sarima", 12, "yoy_implied"), 4.38, 0.006))

    def P(c, h, m, comp, var="mom", loss="se"):
        return sig[(sig["variable"] == var) & (sig["analysis"] == "A") & (sig["country"] == c) & (sig["horizon"] == h) & (sig["model"] == m) & (sig["comparator"] == comp) & (sig["loss"] == loss)]["p_holm"].iloc[0]

    check("Holm p AZE h1 Yu vs SN = 0.204", close(P("AZE", 1, "yu_weighted_fts", "seasonal_naive"), 0.204, 0.0006))
    check("Holm p AZE h3 Yu vs SARIMA = 0.041", close(P("AZE", 3, "yu_weighted_fts", "sarima"), 0.041, 0.0006))
    check("Holm p AZE h6 Yu vs SARIMA < 0.001", P("AZE", 6, "yu_weighted_fts", "sarima") < 0.001)
    v = P("AZE", 1, "sarima", "seasonal_naive")
    check("Holm p AZE h1 SARIMA vs SN quoted 0.0497 (< 0.05)", v < 0.05 and close(v, 0.0497, 0.00051), f"{v:.5f}")
    check("Holm p AZE implied y/y h6/h12 Yu vs SARIMA 0.049/0.006",
          close(P("AZE", 6, "yu_weighted_fts", "sarima", "yoy_implied"), 0.049, 0.0006) and close(P("AZE", 12, "yu_weighted_fts", "sarima", "yoy_implied"), 0.006, 0.0006))
    conf = sig[(sig["family"] == "confirmatory") & (sig["loss"] == "se") & (sig["variable"] == "mom") & (sig["analysis"] == "A")]
    nsig = int((conf["p_holm"] < 0.05).sum())
    check("27 of 140 confirmatory tests significant (m/m, A, SE)", nsig == 27 and len(conf) == 140, f"{nsig}/{len(conf)}")

    fz = pd.read_csv(TABLES / "fuzzy_weighting_comparison.csv")
    f = fz[(fz["variable"] == "mom") & (fz["analysis"] == "A")]
    check("weighted < equal in 9/20 and < Chen in 18/20", (f["weighted_minus_equal_rmse"] < 0).sum() == 9 and (f["weighted_minus_chen_rmse"] < 0).sum() == 18)

    # 6. household inflation: recompute D1/D10 cumulative 2020-23 directly from prices and published shares
    P_ = pd.read_csv(PROCESSED / "aze_cpi_coicop_monthly_2010_100.csv", index_col=0, parse_dates=True)
    w = pd.read_csv(PROCESSED / "official_shares_2025_coicop.csv")
    cps = [f"CP{i:02d}" for i in range(1, 13)]
    ref = P_[P_.index.year == 2025][cps].mean()

    def lasp(dim, grp, t0, t1):
        s = w[(w["dimension"] == dim) & (w["group"] == grp)][cps].iloc[0]
        s = s / s.sum()
        return 100 * ((s * P_.loc[t1, cps] / ref).sum() / (s * P_.loc[t0, cps] / ref).sum() - 1)

    d1 = lasp("expenditure_decile", "D1", "2020-12-01", "2023-12-01")
    d10 = lasp("expenditure_decile", "D10", "2020-12-01", "2023-12-01")
    check("D1 31.17 / D10 28.73 / gap 2.44 (cumulative 2020-23)", close(d1, 31.17, 0.006) and close(d10, 28.73, 0.006) and close(d1 - d10, 2.44, 0.006), f"{d1:.3f} {d10:.3f}")
    g1 = lasp("expenditure_decile", "D1", "2025-03-01", "2026-03-01") - lasp("expenditure_decile", "D10", "2025-03-01", "2026-03-01")
    check("D1-D10 latest 12 months 0.09", close(g1, 0.09, 0.006), f"{g1:.4f}")
    nat = lasp("area", "national", "2025-03-01", "2026-03-01")
    check("national HBS basket 12m to Mar 2026 = 5.93", close(nat, 5.93, 0.006), f"{nat:.3f}")
    infl = pd.read_csv(TABLES / "HOUSEHOLD_INFLATION.csv")
    off = infl[infl["unit"] == "official_group_2025_basket"]
    cc = off[[c for c in off.columns if c.startswith("contrib_CP")]].sum(axis=1)
    m = off["window"] != "W2_annual_avg_2025"
    check("official contributions sum to group inflation", np.allclose(cc[m], off.loc[m, "inflation_pct"], atol=1e-9))
    hh = infl[infl["unit"].str.startswith("survey")]
    for wn in ("W1_yoy_2026_03", "W3_cum_2020_12_to_2023_12"):
        cs = hh[[c for c in hh.columns if c.startswith(f"{wn}__contrib_")]].sum(axis=1)
        check(f"survey contributions sum to household inflation ({wn})", np.allclose(cs, hh[f"infl_{wn}"], atol=1e-9))
    food = 100 * (P_.loc["2023-12-01", "CP01"] / P_.loc["2020-12-01", "CP01"] - 1)
    check("food prices +40.0% Dec2020-Dec2023", close(food, 40.0, 0.051), f"{food:.2f}")
    check("official shares sum to 100", np.allclose(w[cps].sum(axis=1), 100, atol=0.01))

    # 7. survey numbers
    e = pd.read_csv(TABLES / "HOUSEHOLD_EXPENDITURE.csv")
    check("32 households, 26 QC, 25 consistency", len(e) == 32 and e["in_quality_controlled"].sum() == 26 and e["in_consistency_subgroup"].sum() == 25)
    q = e[e["in_quality_controlled"]]
    check("QC mean food share 23.9 / full 21.6", close(q["share_food"].mean(), 23.85, 0.01) and close(e["share_food"].mean(), 21.63, 0.01))
    shares = e[[c for c in e.columns if c.startswith("share_")]].sum(axis=1)
    check("survey shares sum to 100", np.allclose(shares, 100))
    qi = hh[hh["in_quality_controlled"]]["infl_W1_yoy_2026_03"]
    check("QC 12m exposure mean 5.42, SD 0.40, range 4.55-6.08",
          close(qi.mean(), 5.42, 0.005) and close(qi.std(), 0.40, 0.005) and close(qi.min(), 4.55, 0.005) and close(qi.max(), 6.08, 0.005),
          f"{qi.mean():.3f} {qi.std():.3f} {qi.min():.3f} {qi.max():.3f}")
    check("full-sample mean total burden 393, without R02 110",
          close(e["burden_total"].mean(), 393.4, 0.05) and close(e.loc[e["respondent_id"] != "R02", "burden_total"].mean(), 110.2, 0.05))
    hs = pd.read_csv(TABLES / "HOUSEHOLD_STATISTICS.csv")
    r = hs[(hs["row_type"] == "comparison") & (hs["sample"] == "quality_controlled") & (hs["grouping"] == "income_split") & (hs["outcome"] == "infl_W1_yoy_2026_03")].iloc[0]
    check("QC income split: +0.24 pp, MDE 0.47", close(r["diff_mean"], 0.236, 0.001) and close(r["mde_80pct_power"], 0.471, 0.001))
    qt = hs[(hs["row_type"] == "comparison") & (hs["sample"] == "quality_controlled") & (hs["grouping"] == "tenure_group") & (hs["outcome"] == "infl_W1_yoy_2026_03")]["permutation_p_holm"].iloc[0]
    check("QC tenure exposure Holm p quoted 0.24", close(qt, 0.24, 0.006), f"{qt:.3f}")
    nqc = int(((hs["row_type"] == "comparison") & (hs["sample"] == "quality_controlled") & (hs["conclusion"] == "statistically significant after Holm")).sum())
    check("no QC comparison significant after Holm", nqc == 0, f"{nqc}")

    # 8. integration numbers
    pr = pd.read_csv(TABLES / "HOUSEHOLD_INFLATION_PROJECTIONS.csv")
    x = pr[(pr["h"] == 12)].pivot_table(index="unit", columns="model", values="yoy_forecast")
    nat12 = x.loc["official|area|national"]
    gap = x.loc["official|expenditure_decile|D1"] - x.loc["official|expenditure_decile|D10"]
    check("Mar-2027 national projections 5.95/6.63/4.20/10.26",
          all(close(nat12[m], v, 0.006) for m, v in (("seasonal_naive", 5.95), ("sarima", 6.63), ("ets", 4.20), ("yu_weighted_fts", 10.26))))
    check("Mar-2027 D1-D10 gap +0.09/+0.48/-0.74/+1.86",
          all(close(gap[m], v, 0.006) for m, v in (("seasonal_naive", 0.09), ("sarima", 0.48), ("ets", -0.74), ("yu_weighted_fts", 1.86))), gap.round(3).to_dict().__str__())
    hc = pd.read_csv(TABLES / "headline_out_of_sample_check.csv")
    err = hc.groupby("model")["yoy_error"].agg(lambda s: s.abs().max())
    check("headline OOS max |error|: SARIMA 0.35, SN 0.34, ETS 2.70, Yu 2.74",
          close(err["sarima"], 0.35, 0.006) and close(err["seasonal_naive"], 0.34, 0.006) and close(err["ets"], 2.70, 0.006) and close(err["yu_weighted_fts"], 2.74, 0.006),
          err.round(3).to_dict().__str__())
    gs = pd.read_csv(TABLES / "group_forecast_significance.csv")
    h1 = gs[(gs["horizon"] == 1) & gs["model"].isin(["sarima", "ets"])]
    check("h1 SARIMA/ETS vs SN Holm p <= 0.03 for all key groups", (h1["p_holm"] <= 0.0305).all(), f"max {h1['p_holm'].max():.4f}")

    # 9. paper statistics (H5 pooled test, global Holm, H6 regression, sign counts) - recomputed independently
    pdm = pd.read_csv(TABLES / "paper_pooled_dm.csv")
    q = pdm[(pdm["variable"] == "mom") & (pdm["loss"] == "se")]
    yu_sn = q[(q["model"] == "yu_weighted_fts") & (q["comparator"] == "seasonal_naive")]
    yu_ets = q[(q["model"] == "yu_weighted_fts") & (q["comparator"] == "ets")]
    check("H5: Yu never significantly better than SN in pooled test (SE)", (yu_sn["p_holm"] >= 0.05).all())
    check("H5: ETS significantly better than Yu at all 4 horizons (pooled, SE)", ((yu_ets["p_holm"] < 0.05) & (yu_ets["mean_loss_diff"] > 0)).all())
    # independent recomputation of one pooled mean differential (h=1, Yu vs SN) from raw forecasts
    b = fc[(fc["analysis"] == "B") & (fc["split"] == "test") & (fc["target"] == "mom") & (fc["h"] == 1)]
    e = {m: (b[b["model"] == m].set_index(["origin", "country"])["actual"] - b[b["model"] == m].set_index(["origin", "country"])["forecast"]) ** 2
         for m in ("yu_weighted_fts", "seasonal_naive")}
    dbar = (e["yu_weighted_fts"] - e["seasonal_naive"]).unstack("country").dropna().mean(axis=1).mean()
    check("pooled mean loss diff h1 Yu vs SN = -0.105 (recomputed)", close(dbar, -0.105, 0.0006) and close(yu_sn[yu_sn["horizon"] == 1]["mean_loss_diff"].iloc[0], dbar, 1e-9), f"{dbar:.4f}")
    gh = pd.read_csv(TABLES / "paper_global_holm.csv")
    check("27 within-cell vs 9 after Holm over all 140", gh["significant_within_cell"].sum() == 27 and gh["significant_all140"].sum() == 9)
    reg = pd.read_csv(TABLES / "paper_gap_regression.csv").iloc[0]
    ser = pd.read_csv(TABLES / "paper_gap_series.csv")
    slope = np.polyfit(ser["relative_food_inflation_pp"], ser["gap_d1_minus_d10_pp"], 1)[0]
    check("H6 slope 0.29 [0.25, 0.32], R2 0.86, n 243 (slope recomputed)",
          close(reg["slope"], 0.288, 0.001) and close(slope, reg["slope"], 1e-9) and close(reg["slope_ci95_low"], 0.252, 0.001)
          and close(reg["slope_ci95_high"], 0.324, 0.001) and close(reg["r2"], 0.857, 0.001) and int(reg["n_months"]) == 243)
    check("H6 gap positive 74% of months; max 3.60 (2008-06); min -1.85 (2009-10)",
          close(reg["share_months_gap_positive"], 0.745, 0.001) and close(reg["gap_max"], 3.60, 0.006) and reg["gap_max_month"] == "2008-06"
          and close(reg["gap_min"], -1.85, 0.006) and reg["gap_min_month"] == "2009-10")
    sc = pd.read_csv(TABLES / "paper_sign_counts.csv").set_index(["model", "comparator"])["model_lower_rmse"]
    check("sign counts SARIMA<SN 20, ETS<SN 18, Yu<SN 12, Yu<SARIMA 0, Yu<ETS 4, Yu<Chen 18",
          sc[("sarima", "seasonal_naive")] == 20 and sc[("ets", "seasonal_naive")] == 18 and sc[("yu_weighted_fts", "seasonal_naive")] == 12
          and sc[("yu_weighted_fts", "sarima")] == 0 and sc[("yu_weighted_fts", "ets")] == 4 and sc[("yu_weighted_fts", "chen_fts")] == 18)
    ca = pd.read_csv(TABLES / "category_forecast_accuracy.csv")
    cp = ca[ca["model"].isin(["seasonal_naive", "sarima", "ets", "yu_weighted_fts"])]
    best = cp.loc[cp.groupby(["coicop", "horizon"])["RMSE"].idxmin(), "model"].value_counts()
    check("category best-model counts SARIMA 33, ETS 18, Yu 1", best.get("sarima") == 33 and best.get("ets") == 18 and best.get("yu_weighted_fts") == 1)

    # 10. Kazakhstan household data (BNS bulletins + BNS division CPI)
    ks = pd.read_csv(TABLES / "KAZ_HBS_STRUCTURE.csv", dtype={"period": str})
    check("KAZ structure covers 2010..2025 annual + 2026Q1/Q2 for total/urban/rural",
          ks.groupby("area")["period"].nunique().eq(18).all() and set(ks["area"]) == {"total", "urban", "rural"})
    check("KAZ structure rows add to 100 within 0.05 pp", ks["total_check"].abs().max() < 0.05)
    kp = pd.read_csv(PROCESSED / "kaz_cpi_coicop_monthly.csv", index_col=0, parse_dates=True)
    check("KAZ division CPI complete 2008-01..2026-09", kp.notna().all().all() and kp.index.min() == pd.Timestamp("2008-01-01") and kp.index.max() == pd.Timestamp("2026-09-01"))
    kc = pd.read_csv(TABLES / "kaz_data_checks.csv").set_index("check")["value"]
    check("KAZ chained all-items y/y within 0.5 pp of IMF in every month", kc.filter(like="max |diff|").filter(like="IMF").iloc[0] < 0.5)
    ki = pd.read_csv(TABLES / "KAZ_GROUP_INFLATION.csv")
    ksh = pd.read_csv(TABLES / "KAZ_HBS_2025_SHARES.csv")
    cps = [f"CP{i:02d}" for i in range(1, 13)]
    rel = kp[cps] / kp[cps][kp.index.year == 2025].mean()
    s = ksh[(ksh["area"] == "total") & (ksh["region"] == "Kazakhstan")][cps].iloc[0] / 100
    idx = rel @ s
    w4 = 100 * (idx["2026-09-01"] / idx["2025-09-01"] - 1)
    rep_w4 = ki[(ki["window"] == "W4_yoy_2026_09") & (ki["area"] == "total") & (ki["region_en"] == "Kazakhstan")]["inflation_pct"].iloc[0]
    check("KAZ national exposure 12m to Sep 2026 = 9.03% (recomputed)", close(w4, rep_w4, 1e-9) and close(w4, 9.03, 0.006), f"{w4:.3f}")
    kr = pd.read_csv(TABLES / "kaz_gap_regression.csv").iloc[0]
    check("KAZ rural-urban gap within +/-0.35 pp; slope 0.06, R2 0.77",
          max(abs(kr["gap_max"]), abs(kr["gap_min"])) < 0.35 and close(kr["slope"], 0.059, 0.001) and close(kr["r2"], 0.768, 0.001))
    lay = pd.read_csv(LOGS / "figure_layout_check.csv")
    check("figures: no text/label collisions in the saved PNGs", len(lay) == 0, f"{len(lay)} issues")

    rep = pd.DataFrame(results)
    LOGS.mkdir(parents=True, exist_ok=True)
    rep.to_csv(LOGS / "verification_report.csv", index=False)
    print(rep.to_string())
    print(f"\n{rep['passed'].sum()} / {len(rep)} checks passed")
    sys.exit(0 if rep["passed"].all() else 1)


if __name__ == "__main__":
    main()
