"""Phase 9: publication figures from saved outputs only (no re-estimation).

All figures are drawn with matplotlib using constrained layout; legends sit outside the
plotting area and every figure passes the automatic text-collision check in save().
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.inflation.household import COICOP_NAMES
from src.paths import FORECASTS, LOGS, PROCESSED, TABLES
from src.survey.clean import CATEGORIES
from src.visualization.figures import (LAYOUT_ISSUES, COUNTRY_NAME, COUNTRY_STYLE, GRID, INK, INK2, MODEL_STYLE, MUTED,
                                       SLOTS, end_labels, save, setup)

PRIMARY = ["seasonal_naive", "sarima", "ets", "yu_weighted_fts"]
CAT_LABEL = {"food": "Food at home", "rent_repair": "Rent & repairs", "utilities": "Utilities", "transport": "Transport",
             "health": "Health", "education": "Education", "clothing": "Clothing", "communication": "Phone & internet",
             "restaurants": "Restaurants & cafes", "hygiene_household": "Hygiene & household goods", "recreation": "Recreation",
             "tobacco_alcohol": "Tobacco & alcohol", "family_gifts": "Family events & gifts", "other_consumption": "Other"}
AREA_STYLE = {"total": (INK, "-", "All households"), "urban": (SLOTS[0], (0, (6, 2)), "Urban"), "rural": (SLOTS[1], "-", "Rural")}
CPS = [f"CP{i:02d}" for i in range(1, 13)]


def fig(w, h, **kw):
    return plt.subplots(figsize=(w, h), layout="constrained", **kw)


def model_handles(with_actual=True):
    hs = [Line2D([], [], color=INK, lw=1.2, label="Actual")] if with_actual else []
    for m in PRIMARY:
        col, ls, lab = MODEL_STYLE[m]
        hs.append(Line2D([], [], color=col, ls=ls, lw=1.8, label=lab))
    return hs


def year_axis(ax, step=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def fig01_azerbaijan_history(head):
    az = head[head["country"] == "AZE"].set_index("date")
    f, (a1, a2) = fig(9, 5.4, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(az.index, az["yoy"], color=SLOTS[0], lw=1.8, label="IMF CPI")
    spl = az[az["source"] == "AZSTAT_chained"]
    a1.plot(spl.index, spl["yoy"], color=SLOTS[1], lw=2.2, label="Apr-Sep 2026 chained from AZSTAT relatives")
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_ylabel("Year-on-year, %")
    a1.set_title("Year-on-year inflation")
    for lo, hi, txt in (("2008-01-01", "2008-12-01", "2008 boom"), ("2016-01-01", "2017-12-01", "2015 devaluations"),
                        ("2022-01-01", "2023-06-01", "2022 surge")):
        d = az.loc[lo:hi, "yoy"].idxmax()  # label each episode centred above its peak
        a1.annotate(txt, (d, az.loc[d, "yoy"]), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK)
    a1.set_ylim(top=az["yoy"].max() * 1.18)
    a2.bar(az.index, az["mom"], width=25, color=SLOTS[0])
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_ylabel("Month-on-month, %")
    a2.set_title("Month-on-month inflation")
    year_axis(a2)
    a2.set_xlim(az.index.min() - pd.Timedelta(days=45), az.index.max() + pd.Timedelta(days=45))
    f.legend(handles=a1.get_legend_handles_labels()[0], loc="outside lower left", ncol=2)
    f.suptitle("Azerbaijan: consumer price inflation, 2005-2026", x=0.01, ha="left", fontweight="bold")
    save(f, "fig01_azerbaijan_inflation_history",
         "Source: IMF CPI dataset (2010 = 100), Azerbaijan 2005-01..2026-03, extended to 2026-09 with AZSTAT previous-month = 100 relatives.")


def fig02_countries(head):
    f, ax = fig(9.5, 4.6)
    items = []
    for c, col in COUNTRY_STYLE.items():
        s = head[(head["country"] == c) & (head["date"] >= "2001-01-01")].set_index("date")["yoy"].dropna()
        ax.plot(s.index, s, color=col, lw=1.5)
        items.append((s.index[-1], s.iloc[-1], f"{COUNTRY_NAME[c]} {s.iloc[-1]:.1f}%", col))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Year-on-year CPI inflation, %")
    ax.set_xlim(pd.Timestamp("2000-07-01"), pd.Timestamp("2029-03-01"))  # room on the right for the end labels
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2002, 2027, 2)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    end_labels(ax, items)
    f.suptitle("Year-on-year CPI inflation in the five study countries, 2001-2026", x=0.01, ha="left", fontweight="bold")
    save(f, "fig02_inflation_all_countries",
         "Source: IMF CPI dataset (headline index), y/y computed from index levels. Labels give the latest observation.")


def forecast_panel(fc, country, name, title):
    t = fc[(fc["target"] == "mom") & (fc["analysis"] == "A") & (fc["country"] == country) & (fc["split"] == "test")]
    acc = pd.read_csv(TABLES / "INTERNATIONAL_FORECAST_COMPARISON.csv")
    acc = acc[(acc["variable"] == "mom") & (acc["analysis"] == "A") & (acc["country"] == country)]
    f = plt.figure(figsize=(10, 7.2), layout="constrained")
    gs = f.add_gridspec(3, 2, height_ratios=[1, 1, 1.1])
    act = t[(t["model"] == "seasonal_naive") & (t["h"] == 1)].set_index("target_date")["actual"]
    for i, m in enumerate(PRIMARY):
        ax = f.add_subplot(gs[i // 2, i % 2])
        fcs = t[(t["model"] == m) & (t["h"] == 1)].set_index("target_date")["forecast"]
        col, ls, lab = MODEL_STYLE[m]
        ax.plot(act.index, act, color=INK, lw=1.1)
        ax.plot(fcs.index, fcs, color=col, ls=ls, lw=1.5)
        r = acc[(acc["model"] == m) & (acc["horizon"] == 1)]["RMSE"].iloc[0]
        ax.set_title(f"{lab} (h = 1): RMSE {r:.2f} pp", fontsize=9)
        year_axis(ax)
        if i % 2 == 0:
            ax.set_ylabel("m/m inflation, %")
    ax = f.add_subplot(gs[2, :])
    hs, w = [1, 3, 6, 12], 0.2
    for j, m in enumerate(PRIMARY):
        col, _, lab = MODEL_STYLE[m]
        vals = [acc[(acc["model"] == m) & (acc["horizon"] == h)]["RMSE"].iloc[0] for h in hs]
        ax.bar(np.arange(4) + (j - 1.5) * w, vals, w * 0.92, color=col)
    ax.set_xticks(range(4), [f"h = {h}" for h in hs])
    ax.set_xlabel("Forecast horizon, months")
    ax.set_ylabel("RMSE, pp (m/m)")
    ax.set_title("Test-period RMSE by forecast horizon")
    f.suptitle(title, x=0.01, ha="left", fontweight="bold")
    f.legend(handles=model_handles(), loc="outside lower center", ncol=5)
    save(f, name, f"Rolling-origin expanding-window backtest, Analysis A; test targets {act.index.min():%Y-%m}..{act.index.max():%Y-%m} "
                  f"(n = {len(act)} one-step forecasts). Hyper-parameters selected on 2016-2018 validation targets only.")


def fig05_skill(boot):
    b = boot[(boot["variable"] == "mom") & (boot["analysis"] == "A") & (boot["comparator"] == "seasonal_naive") &
             boot["model"].isin(["sarima", "ets", "yu_weighted_fts"])]
    f, axes = fig(10.5, 4.0, ncols=4, sharey=True)
    cs = list(COUNTRY_STYLE)
    for ax, h in zip(axes, [1, 3, 6, 12]):
        for j, m in enumerate(["sarima", "ets", "yu_weighted_fts"]):
            col, _, lab = MODEL_STYLE[m]
            r = b[(b["horizon"] == h) & (b["model"] == m)].set_index("country").reindex(cs)
            y = np.arange(len(cs)) + (j - 1) * 0.22
            ax.errorbar(r["skill_rmse"], y, xerr=[r["skill_rmse"] - r["skill_ci95_low"], r["skill_ci95_high"] - r["skill_rmse"]],
                        fmt="o", ms=4.5, color=col, ecolor=col, elinewidth=1.2, capsize=0, label=lab)
        ax.axvline(0, color=INK2, lw=0.9)
        ax.set_title(f"h = {h}")
        ax.set_xlabel("RMSE skill")
        ax.set_yticks(range(len(cs)), [COUNTRY_NAME[c] for c in cs])
    f.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    f.suptitle("Forecast skill relative to seasonal naive (m/m inflation), with 95% block-bootstrap intervals", x=0.01, ha="left", fontweight="bold")
    save(f, "fig05_cross_country_skill", "Skill = 1 - RMSE(model) / RMSE(seasonal naive). Circular moving-block bootstrap, 2,000 replications, "
                                         "block length max(h, n^(1/3)). Skill > 0: more accurate than seasonal naive. Analysis A test period; panels have different x scales.")


def fig06_fuzzy(acc):
    a = acc[(acc["variable"] == "mom") & (acc["analysis"] == "A")]
    p = a.pivot_table(index=["country", "horizon"], columns="model", values="RMSE")
    f, axes = fig(10, 4.0, ncols=2, sharey=True)
    cs = list(COUNTRY_STYLE)
    for ax, other, ttl in ((axes[0], "yu_equal_fts", "vs equal-weight recurrence (weights only)"),
                           (axes[1], "chen_fts", "vs Chen (1996) (recurrence and weights)")):
        for j, h in enumerate([1, 3, 6, 12]):
            r = (p["yu_weighted_fts"] / p[other]).xs(h, level="horizon").reindex(cs)
            ax.plot(r.values, np.arange(len(cs)) + (j - 1.5) * 0.17, "o", ms=5, color=SLOTS[j], label=f"h = {h}")
        ax.axvline(1, color=INK2, lw=0.9)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("RMSE ratio, Yu weighted / comparator (< 1: Yu better)")
        ax.set_yticks(range(len(cs)), [COUNTRY_NAME[c] for c in cs])
    f.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="outside lower center")
    f.suptitle("Does Yu's chronological weighting improve fuzzy forecasts?", x=0.01, ha="left", fontweight="bold")
    save(f, "fig06_yu_weighted_vs_unweighted", "Same interval partition for all three fuzzy models (chosen on validation for Yu weighted). Test-period m/m inflation, Analysis A.")


def fig07_shares(hh):
    f, (a1, a2) = fig(12, 5.2, ncols=2, gridspec_kw={"width_ratios": [1.1, 1]})
    for k, (sample, col) in enumerate((("in_full_sample", MUTED), ("in_quality_controlled", SLOTS[0]))):
        d = hh[hh[sample]]
        means = np.array([d[f"share_{c}"].mean() for c in CATEGORIES])
        rng = np.random.default_rng(1)
        ci = np.array([np.quantile(rng.choice(d[f"share_{c}"].to_numpy(), (2000, len(d))).mean(1), [0.025, 0.975]) for c in CATEGORIES])
        y = np.arange(len(CATEGORIES)) + (k - 0.5) * 0.36
        a1.barh(y, means, 0.34, color=col, label=f"{'Full' if k == 0 else 'Quality-controlled'} survey sample (n = {len(d)})")
        a1.errorbar(means, y, xerr=[means - ci[:, 0], ci[:, 1] - means], fmt="none", ecolor=INK2, elinewidth=0.8)
    a1.set_yticks(range(len(CATEGORIES)), [CAT_LABEL[c] for c in CATEGORIES])
    a1.invert_yaxis()
    a1.set_xlabel("Mean expenditure share, % of recorded spending")
    a1.set_title("Pilot survey households (Baku, Oct 2026)")
    w = pd.read_csv(PROCESSED / "official_shares_2025_coicop.csv")
    nat = w[(w["dimension"] == "area") & (w["group"] == "national")].iloc[0]
    a2.barh(range(12), [nat[c] for c in CPS], 0.6, color=SLOTS[2], label="Official HBS 2025, all households (AZSTAT)")
    for i, c in enumerate(CPS):
        a2.annotate(f"{nat[c]:.1f}", (nat[c], i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
    a2.set_xlim(0, max(nat[c] for c in CPS) * 1.12)
    a2.set_yticks(range(12), [COICOP_NAMES[c] for c in CPS])
    a2.invert_yaxis()
    a2.set_xlabel("Share of consumption expenditure, %")
    a2.set_title("Official household budget survey 2025 (AZSTAT)")
    hs = a1.get_legend_handles_labels()[0] + a2.get_legend_handles_labels()[0]
    f.legend(handles=hs, loc="outside lower center", ncol=3)
    f.suptitle("Household expenditure shares: pilot survey vs official household budget survey, Azerbaijan", x=0.01, ha="left", fontweight="bold")
    save(f, "fig07_household_expenditure_shares", "Survey categories and COICOP divisions are not identical, so bars are not comparable item by item. Whiskers: 95% bootstrap intervals for the mean.")


def fig09_exposure(infl):
    hh = infl[infl["unit"].str.startswith("survey")]
    off = infl[infl["unit"] == "official_group_2025_basket"]
    f, axes = fig(10.5, 4.6, ncols=2)
    for ax, (win, col_off, ttl) in zip(axes, (("W1_yoy_2026_03", "infl_W1_yoy_2026_03", "12 months to March 2026"),
                                              ("W3_cum_2020_12_to_2023_12", "infl_W3_cum_2020_12_to_2023_12", "Cumulative, Dec 2020 to Dec 2023"))):
        rng = np.random.default_rng(3)
        for k, (sample, col, lab) in enumerate((("in_full_sample", MUTED, "Survey household, full sample"),
                                                ("in_quality_controlled", SLOTS[0], "Survey household, quality-controlled"))):
            v = hh[hh[sample]][col_off]
            ax.plot(np.full(len(v), k) + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=4.5, color=col, mec="white", mew=0.5, label=lab)
            ax.plot([k - 0.2, k + 0.2], [v.median()] * 2, color=INK, lw=2, label="Median" if k == 0 else None)
        dec = off[(off["dimension"] == "expenditure_decile") & (off["window"] == win)].set_index("group")["inflation_pct"]
        ax.plot(np.full(10, 2.0), dec.values, "_", ms=16, mew=2, color=SLOTS[1], label="Official expenditure decile (D1..D10)")
        top, bot = dec.idxmax(), dec.idxmin()  # label the extreme deciles above and below the stack of markers
        ax.annotate(f"{top} {dec[top]:.2f}%", (2.0, dec[top]), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.annotate(f"{bot} {dec[bot]:.2f}%", (2.0, dec[bot]), xytext=(0, -6), textcoords="offset points", ha="center", va="top", fontsize=7.5, color=INK)
        nat = off[(off["dimension"] == "area") & (off["group"] == "national") & (off["window"] == win)]["inflation_pct"].iloc[0]
        ax.axhline(nat, color=SLOTS[1], lw=0.9, ls=(0, (3, 2)), label="Official national basket")
        n_full, n_qc = int(hh["in_full_sample"].sum()), int(hh["in_quality_controlled"].sum())
        ax.set_xticks([0, 1, 2], [f"Survey full\n(n = {n_full})", f"Survey QC\n(n = {n_qc})", "Official\ndeciles"])
        ax.set_xlim(-0.5, 2.9)
        ax.set_ylabel("Fixed-basket inflation, %")
        ax.set_title(f"{ttl} (national basket {nat:.2f}%)", fontsize=9)
    f.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    f.suptitle("Household-specific inflation exposure, Azerbaijan (Laspeyres, 2025 baskets, COICOP prices)", x=0.01, ha="left", fontweight="bold")
    save(f, "fig09_household_inflation_exposure", "Survey households valued with their reported 2026 basket as 2025 weights; official groups with AZSTAT 2025 HBS shares. Periods before 2025 are fixed-basket counterfactuals.")


def fig10_sensitivity(sens):
    vars_ = [("share_food", "Food share, %"), ("share_education", "Education share, %"), ("burden_food", "Food burden, % of income"),
             ("burden_total", "Total spending / income, % (log)"), ("infl_W1_yoy_2026_03", "Exposure, 12 months to Mar 2026, %"),
             ("infl_W3_cum_2020_12_to_2023_12", "Exposure, Dec 2020 to Dec 2023, %")]
    f, axes = fig(11, 5.8, nrows=2, ncols=3, sharey=True)
    samples = [("full", MUTED, "Full"), ("quality_controlled", SLOTS[0], "Quality-controlled"), ("consistency_subgroup", SLOTS[1], "Income-expenditure consistent")]
    for ax, (v, lab) in zip(axes.ravel(), vars_):
        for k, (s, col, sl) in enumerate(samples):
            r = sens[(sens["sample"] == s) & (sens["variable"] == v)].iloc[0]
            ax.errorbar([r["mean"]], [k], xerr=[[r["mean"] - r["mean_ci95_low"]], [r["mean_ci95_high"] - r["mean"]]], fmt="o", ms=5,
                        color=col, ecolor=col, label=f"{sl} sample, mean with 95% CI")
            ax.plot([r["median"]], [k], "D", ms=4, color=INK, mfc="white", label="Median" if k == 0 else None)
        ns = [int(sens[(sens["sample"] == s) & (sens["variable"] == v)]["n"].iloc[0]) for s, _, _ in samples]
        ax.set_yticks(range(3), [f"{sl}\n(n = {n})" for (_, _, sl), n in zip(samples, ns)], fontsize=7.5)
        ax.set_title(lab, fontsize=8.5)
        ax.set_ylim(2.5, -0.5)
        if v == "burden_total":
            ax.set_xscale("log")  # one contradictory record (R02) drives the full-sample mean
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    hs, ls = axes[0, 0].get_legend_handles_labels()
    f.legend(hs, ls, loc="outside lower center", ncol=4)
    f.suptitle("Pilot survey results under three data-quality samples", x=0.01, ha="left", fontweight="bold")
    save(f, "fig10_survey_sample_sensitivity", "Full: all 32 responses. Quality-controlled: documented exclusion rules. Consistent: itemised spending <= upper bound of the reported income bracket (a sensitivity check, not the 'correct' sample).")


def fig12_projections(proj, groupfc):
    units = [("official|expenditure_decile|D1", "Poorest decile (D1)"), ("official|expenditure_decile|D10", "Richest decile (D10)")]
    f, axes = fig(10.5, 4.6, ncols=2, sharey=True)
    hist = groupfc[(groupfc["model"] == "seasonal_naive") & (groupfc["h"] == 1) & (groupfc["origin"] >= "2023-12-01")]
    for ax, (u, lab) in zip(axes, units):
        hst = hist[hist["unit"] == u].set_index("target_date")["yoy_actual"].dropna()
        ax.plot(hst.index, hst, color=INK, lw=1.4)
        for k, m in enumerate(PRIMARY):
            col, ls, _ = MODEL_STYLE[m]
            p = proj[(proj["unit"] == u) & (proj["model"] == m)].sort_values("h")
            ax.plot(pd.to_datetime(p["target_date"]), p["yoy_forecast"], color=col, ls=ls, lw=1.6)
            q = p.dropna(subset=["lo80"])
            x = pd.to_datetime(q["target_date"]) + pd.Timedelta(days=(k - 1.5) * 9)
            ax.vlines(x, q["lo80"], q["hi80"], color=col, lw=1.6, alpha=0.75)
        ax.axvline(pd.Timestamp("2026-03-01"), color=INK2, lw=0.8, ls=(0, (2, 2)))
        ax.set_title(lab)
        ax.set_ylabel("Year-on-year inflation, %")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    hs = model_handles()
    hs[0].set_label("Actual (fixed 2025 basket)")
    f.legend(handles=hs, loc="outside lower center", ncol=5)
    f.suptitle("Projected inflation exposure to March 2027 from division-level forecasts (origin March 2026), Azerbaijan", x=0.01, ha="left", fontweight="bold")
    save(f, "fig12_household_exposure_forecasts", "Each COICOP division forecast separately and aggregated with AZSTAT 2025 HBS shares. Vertical bars: empirical 80% intervals from group-level backtest errors at h = 1, 3, 6, 12 (offset slightly so they do not overlap). Dotted line: forecast origin.")


def fig13_contributions(infl):
    off = infl[(infl["unit"] == "official_group_2025_basket") & (infl["dimension"] == "expenditure_decile")]
    order = ["CP01", "CP04", "CP07", "CP11", "CP05", "CP06"]
    rest = [c for c in CPS if c not in order]
    f, axes = fig(10.5, 4.6, ncols=2)
    for ax, win, ttl in zip(axes, ["W3_cum_2020_12_to_2023_12", "W1_yoy_2026_03"], ["Cumulative, Dec 2020 to Dec 2023", "12 months to March 2026"]):
        d = off[off["window"] == win].set_index("group")
        bottom = np.zeros(len(d))
        for j, c in enumerate(order):
            v = d[f"contrib_{c}"].to_numpy()
            ax.bar(range(len(d)), v, 0.72, bottom=bottom, color=SLOTS[j], label=COICOP_NAMES[c], edgecolor="white", linewidth=0.8)
            bottom += v
        v = d[[f"contrib_{c}" for c in rest]].sum(axis=1).to_numpy()
        ax.bar(range(len(d)), v, 0.72, bottom=bottom, color=MUTED, label="Other divisions", edgecolor="white", linewidth=0.8)
        for i, tot in enumerate(d["inflation_pct"]):
            ax.annotate(f"{tot:.1f}", (i, tot), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_ylim(0, d["inflation_pct"].max() * 1.1)
        ax.set_xticks(range(len(d)), d.index)
        ax.set_ylabel("Contribution to inflation, pp")
        ax.set_title(ttl)
    f.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    f.suptitle("Which categories drive inflation exposure across expenditure deciles? (Azerbaijan)", x=0.01, ha="left", fontweight="bold")
    save(f, "fig13_category_contributions_by_decile", "Exact additive Laspeyres contributions with AZSTAT 2025 HBS shares; 2020-2023 is a fixed-2025-basket counterfactual. Bar labels: total, %.")


def gap_figure(ser, reg, gap_col, name, title, gap_label, note):
    f, (a1, a2) = fig(10.5, 4.3, ncols=2, gridspec_kw={"width_ratios": [1.5, 1]})
    a1.plot(ser["date"], ser[gap_col], color=SLOTS[0], lw=1.6)
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_ylabel(f"{gap_label}, pp")
    a1.set_title("Gap over time (fixed 2025 baskets)")
    year_axis(a1, 4)
    x = ser["relative_food_inflation_pp"]
    a2.plot(x, ser[gap_col], "o", ms=3, color=MUTED, alpha=0.6, mec="none", label="Month")
    xs = np.linspace(x.min(), x.max(), 50)
    a2.plot(xs, reg["intercept"] + reg["slope"] * xs, color=SLOTS[0], lw=1.8,
            label=f"OLS fit: slope {reg['slope']:.2f} (95% CI {reg['slope_ci95_low']:.2f} to {reg['slope_ci95_high']:.2f}), R² {reg['r2']:.2f}")
    a2.axhline(0, color=INK2, lw=0.8)
    a2.axvline(0, color=INK2, lw=0.8)
    a2.set_xlabel("Food minus all-items y/y inflation, pp")
    a2.set_title("Gap vs relative food inflation")
    f.legend(*a2.get_legend_handles_labels(), loc="outside lower right", ncol=2)
    f.suptitle(title, x=0.01, ha="left", fontweight="bold")
    save(f, name, note)


def fig14_gap_vs_food():
    ser = pd.read_csv(TABLES / "paper_gap_series.csv", parse_dates=["date"])
    reg = pd.read_csv(TABLES / "paper_gap_regression.csv").iloc[0]
    gap_figure(ser, reg, "gap_d1_minus_d10_pp", "fig14_gap_vs_relative_food_inflation",
               "Azerbaijan: the poorest-minus-richest decile gap moves with relative food inflation", "D1 minus D10 y/y inflation",
               f"Monthly, {ser['date'].min():%Y-%m}..{ser['date'].max():%Y-%m} (n = {len(ser)}). OLS with Newey-West (12 lags) standard errors. "
               "With fixed baskets the relationship is largely an accounting identity, not a behavioural estimate.")


def fig15_kaz_structure():
    s = pd.read_csv(TABLES / "KAZ_HBS_STRUCTURE.csv", dtype={"period": str})
    f, (a1, a2) = fig(10.5, 4.4, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
    for area, (col, ls, lab) in AREA_STYLE.items():
        d = s[(s["area"] == area) & ~s["period"].str.contains("Q")]
        a1.plot(d["year"], d["food_nonalc"], color=col, ls=ls, lw=1.8, marker="o", ms=3, label=lab)
        q = s[(s["area"] == area) & s["period"].str.contains("Q")]
        xq = 2026 + (q["period"].str[-1].astype(int) - 1) * 0.25
        a1.plot(xq, q["food_nonalc"], ls="none", marker="D", ms=4.5, color=col, mfc="white")
    a1.plot([], [], ls="none", marker="D", ms=4.5, color=INK, mfc="white", label="2026 Q1 and Q2 (quarterly)")
    a1.set_xticks(range(2010, 2027, 2))
    a1.set_ylabel("Food & non-alcoholic beverages, % of consumption")
    a1.set_title("Food share by area, 2010-2026")
    nat = s[(s["area"] == "total") & s["period"].isin(["2010", "2015", "2020", "2025"])]
    parts = [("food_goods", "Food goods (incl. alcohol, tobacco)", SLOTS[1]), ("nonfood_goods", "Non-food goods", SLOTS[0]), ("services", "Paid services", SLOTS[2])]
    bottom = np.zeros(len(nat))
    for col, lab, c in parts:
        v = nat[col].to_numpy()
        a2.bar(nat["period"], v, 0.62, bottom=bottom, color=c, edgecolor="white", linewidth=0.8, label=lab)
        for i, (b, val) in enumerate(zip(bottom, v)):
            a2.text(i, b + val / 2, f"{val:.0f}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", gid="inside-bar")
        bottom += v
    a2.set_ylim(0, 100)
    a2.set_ylabel("% of consumer expenditure")
    a2.set_title("Composition, all households")
    hs = a1.get_legend_handles_labels()[0] + a2.get_legend_handles_labels()[0]
    f.legend(handles=hs, loc="outside lower center", ncol=4)
    f.suptitle("Kazakhstan: household consumption structure, 2010-2026 (BNS household budget survey)", x=0.01, ha="left", fontweight="bold")
    save(f, "fig15_kazakhstan_spending_structure",
         "Source: BNS bulletins 'Expenditures and incomes of the population' 2010-2025 and Q1/Q2 2026 (Google Drive download, 25 Oct 2026). "
         "From 2026 the bulletins follow COICOP 2018, so quarterly 2026 values are shown separately. White bar labels: %.")


def fig16_kaz_regions():
    infl = pd.read_csv(TABLES / "KAZ_GROUP_INFLATION.csv")
    f, axes = fig(10.5, 4.8, ncols=2)
    for ax, (win, ttl) in zip(axes, (("W3_cum_2020_12_to_2023_12", "Cumulative, Dec 2020 to Dec 2023"),
                                     ("W4_yoy_2026_09", "12 months to September 2026"))):
        d = infl[infl["window"] == win]
        reg = d[(d["area"] == "total") & (d["region_en"] != "Kazakhstan")]
        ax.plot(reg["food_share_pct"], reg["inflation_pct"], "o", ms=5.5, color=SLOTS[2], mec="white", mew=0.5, label="Region (all households)")
        for area, mk in (("urban", "s"), ("rural", "^")):
            r = d[(d["area"] == area) & (d["region_en"] == "Kazakhstan")]
            ax.plot(r["food_share_pct"], r["inflation_pct"], mk, ms=7, color=AREA_STYLE[area][0], label=f"National {area}")
        nat = d[(d["area"] == "total") & (d["region_en"] == "Kazakhstan")]
        ax.plot(nat["food_share_pct"], nat["inflation_pct"], "*", ms=11, color=INK, label="National, all households")
        lo, hi = reg.loc[reg["inflation_pct"].idxmin()], reg.loc[reg["inflation_pct"].idxmax()]
        for r, va, dy in ((lo, "top", -8), (hi, "bottom", 8)):
            ax.annotate(f"{r['region_en']} {r['inflation_pct']:.2f}%", (r["food_share_pct"], r["inflation_pct"]), xytext=(0, dy),
                        textcoords="offset points", ha="center", va=va, fontsize=7.5, color=INK)
        pad = (reg["inflation_pct"].max() - reg["inflation_pct"].min()) * 0.25
        ax.set_ylim(reg["inflation_pct"].min() - pad, reg["inflation_pct"].max() + pad)
        xpad = (reg["food_share_pct"].max() - reg["food_share_pct"].min()) * 0.12
        ax.set_xlim(reg["food_share_pct"].min() - xpad, reg["food_share_pct"].max() + xpad)
        ax.set_xlabel("Food & non-alcoholic beverages share, 2025, %")
        ax.set_ylabel("Fixed-basket inflation, %")
        ax.set_title(ttl)
    f.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    f.suptitle("Kazakhstan: regional inflation exposure with 2025 household baskets", x=0.01, ha="left", fontweight="bold")
    save(f, "fig16_kazakhstan_regional_exposure",
         "Laspeyres indices: 2025 BNS household budget survey baskets for 20 regions, valued with national COICOP-division CPI (BNS). "
         "Differences across regions therefore come from baskets only, not from regional prices.")


def fig17_kaz_gap():
    ser = pd.read_csv(TABLES / "kaz_urban_rural_gap_series.csv", parse_dates=["date"])
    reg = pd.read_csv(TABLES / "kaz_gap_regression.csv").iloc[0]
    gap_figure(ser, reg, "gap_rural_minus_urban_pp", "fig17_kazakhstan_rural_urban_gap",
               "Kazakhstan: the rural-minus-urban inflation gap is small and follows relative food inflation", "Rural minus urban y/y inflation",
               f"Monthly, {ser['date'].min():%Y-%m}..{ser['date'].max():%Y-%m} (n = {len(ser)}). 2025 urban and rural BNS baskets, national division prices. "
               "OLS with Newey-West (12 lags) standard errors.")


def fig18_basket_comparison():
    b = pd.read_csv(TABLES / "kaz_aze_basket_comparison.csv")
    f, ax = fig(9.5, 5.0)
    y = np.arange(len(b))
    for k, (col, c, lab) in enumerate((("azerbaijan_pct", COUNTRY_STYLE["AZE"], "Azerbaijan (AZSTAT HBS 2025)"),
                                       ("kazakhstan_pct", COUNTRY_STYLE["KAZ"], "Kazakhstan (BNS HBS 2025)"))):
        yy = y + (k - 0.5) * 0.38
        ax.barh(yy, b[col], 0.36, color=c, label=lab)
        for yi, v in zip(yy, b[col]):
            ax.annotate(f"{v:.1f}", (v, yi), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, [COICOP_NAMES[c] for c in b["coicop"]])
    ax.invert_yaxis()
    ax.set_xlim(0, b[["azerbaijan_pct", "kazakhstan_pct"]].max().max() * 1.1)
    ax.set_xlabel("Share of consumption expenditure, %")
    f.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2)
    f.suptitle("National household baskets in 2025: Azerbaijan vs Kazakhstan", x=0.01, ha="left", fontweight="bold")
    save(f, "fig18_azerbaijan_kazakhstan_baskets",
         "Kazakhstan shares mapped from BNS bulletin items to COICOP divisions (eating out and cafes to restaurants & hotels; "
         "hygiene and other services to miscellaneous). Azerbaijan: AZSTAT 2025 household budget survey.")


def main():
    setup()
    head = pd.read_csv(PROCESSED / "cpi_headline_monthly.csv", parse_dates=["date"])
    fc = pd.read_csv(FORECASTS / "international_backtest_forecasts.csv.gz", parse_dates=["origin", "target_date"])
    acc = pd.read_csv(TABLES / "INTERNATIONAL_FORECAST_COMPARISON.csv")
    boot = pd.read_csv(TABLES / "forecast_skill_bootstrap.csv")
    hh = pd.read_csv(TABLES / "HOUSEHOLD_EXPENDITURE.csv")
    infl = pd.read_csv(TABLES / "HOUSEHOLD_INFLATION.csv")
    sens = pd.read_csv(TABLES / "SURVEY_SENSITIVITY.csv")
    proj = pd.read_csv(TABLES / "HOUSEHOLD_INFLATION_PROJECTIONS.csv")
    gfc = pd.read_csv(FORECASTS / "group_backtest_forecasts.csv.gz", parse_dates=["origin", "target_date"])
    fig01_azerbaijan_history(head)
    fig02_countries(head)
    forecast_panel(fc, "AZE", "fig03_azerbaijan_four_model_forecasts", "Azerbaijan: four-model forecasts of monthly CPI inflation (test period)")
    fig05_skill(boot)
    fig06_fuzzy(acc)
    fig07_shares(hh)
    fig09_exposure(infl)
    fig10_sensitivity(sens)
    fig12_projections(proj, gfc)
    fig13_contributions(infl)
    fig14_gap_vs_food()
    fig15_kaz_structure()
    fig16_kaz_regions()
    fig17_kaz_gap()
    fig18_basket_comparison()
    issues = pd.DataFrame(LAYOUT_ISSUES, columns=["figure", "kind", "a", "b"])
    LOGS.mkdir(parents=True, exist_ok=True)
    issues.to_csv(LOGS / "figure_layout_check.csv", index=False)
    if len(issues):
        print(issues.to_string())
        sys.exit(f"{len(issues)} text/label collisions found (outputs/logs/figure_layout_check.csv)")
    print("figures written; layout check: no text collisions")


if __name__ == "__main__":
    main()
