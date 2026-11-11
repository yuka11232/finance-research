"""Phases 6-7: official-group inflation exposure, and aggregate survey expenditure analysis.

Official outputs (always written; outputs/tables):
  HOUSEHOLD_INFLATION.csv          official-group inflation exposure, with contributions
  official_group_gap_rounding.csv  sensitivity of official gaps to 0.1-pp rounding of published shares

Survey outputs (written only when the private export data/raw/survey/ is present). Every published
survey table is aggregate: no respondent ids and no respondent-level rows, and any summary of fewer
than MIN_CELL households is suppressed.
  SURVEY_SENSITIVITY.csv           means, medians and CIs by sample definition (full / QC / consistency)
  SURVEY_SUMMARY_BY_INCOME.csv     the same by income band within each sample
  survey_share_intervals.csv       mean category shares with bootstrap CIs (input to the shares figure)
  HOUSEHOLD_STATISTICS.csv         group summaries and comparisons with uncertainty, effect sizes, tests and MDEs
  survey_exclusion_log.csv         number of excluded records per exclusion rule
  survey_influence.csv             leave-one-out influence on key means, summarised
  survey_mapping_sensitivity.csv   exposure under alternative category->COICOP mappings
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.ingestion.hbs_tables import parse_2025_structures, to_coicop
from src.ingestion.series import coicop_monthly
from src.inflation.household import (COICOP, SURVEY_MAP_VARIANTS, annual_average_inflation, price_relatives,
                                     survey_to_coicop_shares, window_inflation)
from src.paths import PROCESSED, TABLES, config, ensure_dirs
from src.statistics.forecast_tests import holm
from src.statistics.group_tests import compare, describe
from src.survey.analysis import SAMPLES, add_groupings, shares_and_burdens, trimmed_mean
from src.survey.clean import CATEGORIES, EXCLUSION_RULES, SURVEY_FILE, build

T = pd.Timestamp
WINDOWS = {  # name: (t0, t1, label)
    "W1_yoy_2026_03": (T("2025-03-01"), T("2026-03-01"), "12-month change to March 2026 (latest COICOP month)"),
    "W3_cum_2020_12_to_2023_12": (T("2020-12-01"), T("2023-12-01"), "cumulative Dec 2020 - Dec 2023 (fixed 2025 basket counterfactual)"),
}
ANNUAL = {"W2_annual_avg_2025": 2025}
MIN_CELL = 5  # smallest number of households for which a summary is published
STAT_COLS = ["mean", "median", "sd", "mean_ci95_low", "mean_ci95_high", "median_ci95_low", "median_ci95_high"]
OUTCOMES = ["infl_W1_yoy_2026_03", "infl_W2_annual_avg_2025", "infl_W3_cum_2020_12_to_2023_12", "share_food", "burden_food", "burden_total", "essentials_share"]
GROUPINGS = {
    "income_band": [("low (<2000 AZN)", "high (>=5000 AZN)"), ("low (<2000 AZN)", "middle (2000-4999 AZN)"), ("middle (2000-4999 AZN)", "high (>=5000 AZN)")],
    "income_split": [("<5000 AZN", ">=5000 AZN")],
    "eq_income_tercile": [("T1 (lowest)", "T3 (highest)")],
    "employment_group": [("full-time employed", "not full-time")],
    "size_group": [("4+ persons", "2-3 persons")],
    "children_group": [("1+ children", "no children")],
    "tenure_group": [("mortgage/rent/free", "owner, no mortgage")],
    "confidence_group": [("rough estimate", "fairly/very confident")],
}


def household_exposure(df: pd.DataFrame, rel: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    sh = survey_to_coicop_shares(df[CATEGORIES], mapping)
    sh.index = df["respondent_id"].values
    out = pd.DataFrame(index=sh.index)
    contribs = []
    for name, (t0, t1, _) in WINDOWS.items():
        w = window_inflation(sh, rel, t0, t1)
        out[f"infl_{name}"] = w["inflation_pct"]
        c = w.filter(like="contrib_").add_prefix(f"{name}__")
        contribs.append(c)
    for name, year in ANNUAL.items():
        out[f"infl_{name}"] = annual_average_inflation(sh, rel, year)
    return pd.concat([out, sh.add_prefix("coicop_share_")] + contribs, axis=1)


def official_exposure(rel: pd.DataFrame) -> pd.DataFrame:
    w = to_coicop(parse_2025_structures())
    sh = w.set_index(["dimension", "group"])[COICOP] / 100.0
    sh = sh.div(sh.sum(axis=1), axis=0)  # remove 0.1-pp rounding residue
    rows = []
    for name, (t0, t1, label) in WINDOWS.items():
        r = window_inflation(sh, rel, t0, t1)
        r["window"], r["window_label"] = name, label
        rows.append(r)
    for name, year in ANNUAL.items():
        r = pd.DataFrame({"inflation_pct": annual_average_inflation(sh, rel, year)})
        r["window"], r["window_label"] = name, f"annual average {year} vs {year - 1}"
        rows.append(r)
    out = pd.concat(rows).reset_index()
    out.insert(0, "unit", "official_group_2025_basket")
    return out


def rounding_sensitivity(rel: pd.DataFrame, rng: np.random.Generator, reps: int = 2000) -> pd.DataFrame:
    """Published shares are rounded to 0.1 pp: perturb uniformly within +-0.05 pp and recompute key gaps."""
    items = parse_2025_structures().pivot_table(index=["dimension", "group"], columns="item", values="share_pct")
    direct = {"CP03": "clothing", "CP04": "housing_utilities", "CP05": "furnishings", "CP06": "health", "CP07": "transport",
              "CP08": "communication", "CP09": "recreation", "CP10": "education", "CP11": "restaurants_hotels", "CP12": "other"}

    def coicop_from_items(it: pd.DataFrame) -> pd.DataFrame:
        sh = pd.DataFrame({"CP01": it["food_incl_alcohol"] - it["alcohol"], "CP02": it["alcohol"] + it["tobacco"],
                           **{cp: it[item] for cp, item in direct.items()}})[COICOP]
        return sh.div(sh.sum(axis=1), axis=0)

    pairs = [("expenditure_decile", "D1", "D10"), ("expenditure_quintile", "Q1", "Q5"), ("area", "rural", "urban"),
             ("household_size", "6+ persons", "1 person"), ("children_count", "4+ children", "1 child")]
    rows = []
    t0, t1, _ = WINDOWS["W1_yoy_2026_03"]
    for dim, a, b in pairs:
        base = items.loc[[(dim, a), (dim, b)]]
        gaps = []
        for _ in range(reps):
            r = window_inflation(coicop_from_items(base + rng.uniform(-0.05, 0.05, size=base.shape)), rel, t0, t1)["inflation_pct"]
            gaps.append(r.iloc[0] - r.iloc[1])
        r0 = window_inflation(coicop_from_items(base), rel, t0, t1)["inflation_pct"]
        rows.append({"dimension": dim, "group_a": a, "group_b": b, "window": "W1_yoy_2026_03",
                     "gap_pp": r0.iloc[0] - r0.iloc[1], "gap_min": min(gaps), "gap_max": max(gaps),
                     "gap_p2_5": np.quantile(gaps, 0.025), "gap_p97_5": np.quantile(gaps, 0.975),
                     "note": "each of the 13 published items perturbed uniformly within +-0.05 pp, mapped to COICOP, renormalised"})
    return pd.DataFrame(rows)


def group_statistics(df: pd.DataFrame, rng, reps) -> pd.DataFrame:
    rows = []
    for sample, col in SAMPLES.items():
        d = df[df[col]]
        for outcome in OUTCOMES:
            fam = []
            for grouping, pairs in GROUPINGS.items():
                for lvl in d[grouping].dropna().unique():
                    x = d.loc[d[grouping] == lvl, outcome]
                    rows.append({"sample": sample, "outcome": outcome, "grouping": grouping, "group": str(lvl),
                                 "row_type": "group_summary", **describe(x, rng, reps)})
                for a, b in pairs:
                    xa, xb = d.loc[d[grouping] == a, outcome], d.loc[d[grouping] == b, outcome]
                    r = {"sample": sample, "outcome": outcome, "grouping": grouping, "group": f"{a} minus {b}",
                         "row_type": "comparison", **compare(xa, xb, rng, reps)}
                    rows.append(r)
                    fam.append(len(rows) - 1)
            p = [rows[i].get("permutation_p", np.nan) for i in fam]
            adj = holm(p)
            for i, v in zip(fam, adj):
                rows[i]["permutation_p_holm"] = v
    out = pd.DataFrame(rows)
    out["suppressed"] = (out["row_type"] == "group_summary") & (out["n"] < MIN_CELL)
    out.loc[out["suppressed"], STAT_COLS] = np.nan
    out["conclusion"] = np.where(out["row_type"] != "comparison", "",
                         np.where(~out["tested"].fillna(False).astype(bool), "not tested (group too small)",
                         np.where(out["permutation_p_holm"] < 0.05, "statistically significant after Holm",
                                  "not statistically distinguishable (Holm-adjusted p >= 0.05)")))
    return out


def sensitivity_table(df: pd.DataFrame, rng, reps) -> pd.DataFrame:
    rows = []
    vars_ = [f"share_{c}" for c in CATEGORIES] + [f"burden_{c}" for c in ("food", "rent_repair", "utilities", "transport")] + \
            ["burden_total", "hhi_concentration", "essentials_share", "infl_W1_yoy_2026_03", "infl_W2_annual_avg_2025", "infl_W3_cum_2020_12_to_2023_12"]
    for sample, col in SAMPLES.items():
        d = df[df[col]]
        for v in vars_:
            s = describe(d[v], rng, reps)
            rows.append({"sample": sample, "variable": v, "trimmed_mean_10pct": trimmed_mean(d[v]), **s})
    out = pd.DataFrame(rows)
    full = out[out["sample"] == "full"].set_index("variable")[["mean", "median"]]
    out["mean_minus_full"] = out["mean"] - full["mean"].reindex(out["variable"]).to_numpy()
    out["median_minus_full"] = out["median"] - full["median"].reindex(out["variable"]).to_numpy()
    return out


def influence(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out influence on full-sample means, summarised per variable (per-record values would reveal responses)."""
    d = df[df["in_full_sample"]].reset_index(drop=True)
    rows = []
    for v in ["share_food", "infl_W1_yoy_2026_03", "burden_total"]:
        base = d[v].mean()
        change = pd.Series([d[v].drop(i).mean() - base for i in d.index])
        top = change.abs().idxmax()
        rows.append({"variable": v, "n": len(d), "full_mean": base, "mean_without_most_influential": base + change[top],
                     "largest_abs_change": abs(change[top]), "median_abs_change": change.abs().median(),
                     "n_records_abs_change_gt_1pct_of_mean": int((change.abs() > 0.01 * abs(base)).sum())})
    return pd.DataFrame(rows)


def exclusion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Number of records excluded from the quality-controlled sample under each combination of rules."""
    ex = df.loc[~df["in_quality_controlled"], "exclusion_reasons"]
    out = ex.value_counts().rename_axis("exclusion_reasons").reset_index(name="n_records")
    out["rule_text"] = out["exclusion_reasons"].map(lambda s: " | ".join(EXCLUSION_RULES[k] for k in s.split("; ") if k))
    return out


def income_summary(df: pd.DataFrame, rng, reps) -> pd.DataFrame:
    """Means, medians and bootstrap CIs by income band within each sample; cells below MIN_CELL are suppressed."""
    vars_ = [f"share_{c}" for c in CATEGORIES] + ["burden_food", "burden_total", "essentials_share",
             "infl_W1_yoy_2026_03", "infl_W2_annual_avg_2025", "infl_W3_cum_2020_12_to_2023_12"]
    rows = []
    for sample, col in SAMPLES.items():
        d = df[df[col]]
        for grouping in ("income_split", "income_band"):
            for lvl in sorted(d[grouping].dropna().unique(), key=str):
                g = d[d[grouping] == lvl]
                for v in vars_:
                    r = {"sample": sample, "grouping": grouping, "group": str(lvl), "variable": v, "n": len(g)}
                    if len(g) >= MIN_CELL:
                        r.update(describe(g[v], rng, reps))
                    r["suppressed"] = len(g) < MIN_CELL
                    rows.append(r)
    return pd.DataFrame(rows)


def share_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Mean category shares with 95% bootstrap intervals for the full and QC samples (drawn by the shares figure)."""
    rows = []
    for sample in ("full", "quality_controlled"):
        d = df[df[SAMPLES[sample]]]
        rng = np.random.default_rng(1)
        for c in CATEGORIES:
            x = d[f"share_{c}"].to_numpy()
            lo, hi = np.quantile(rng.choice(x, (2000, len(d))).mean(1), [0.025, 0.975])
            rows.append({"sample": sample, "category": c, "n": len(d), "mean_share": x.mean(), "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    cfg = config()
    rng = np.random.default_rng(cfg["project"]["seed"])
    reps = cfg["project"]["bootstrap_reps"]
    rel = price_relatives(coicop_monthly("AZE").dropna(), cfg["household"]["price_reference_year"])
    structures = parse_2025_structures()
    structures.to_csv(PROCESSED / "official_hbs_2025_structures_long.csv", index=False)
    to_coicop(structures).to_csv(PROCESSED / "official_shares_2025_coicop.csv", index=False)
    official_exposure(rel).to_csv(TABLES / "HOUSEHOLD_INFLATION.csv", index=False)
    rounding_sensitivity(rel, rng).to_csv(TABLES / "official_group_gap_rounding.csv", index=False)

    if not SURVEY_FILE.exists():
        print("official groups done; survey export not found, committed aggregate survey tables kept")
        return
    df = add_groupings(shares_and_burdens(build()))
    exp = household_exposure(df, rel, SURVEY_MAP_VARIANTS["primary"])
    df = df.merge(exp, left_on="respondent_id", right_index=True)
    df.to_csv(PROCESSED / "survey_households_processed.csv", index=False)  # local only (.gitignore): respondent-level
    exclusion_summary(df).to_csv(TABLES / "survey_exclusion_log.csv", index=False)

    # mapping sensitivity for survey households
    ms = []
    for name, mp in SURVEY_MAP_VARIANTS.items():
        e = household_exposure(df, rel, mp)[["infl_W1_yoy_2026_03", "infl_W3_cum_2020_12_to_2023_12"]]
        for sample, col in SAMPLES.items():
            ids = df.loc[df[col], "respondent_id"]
            ms.append({"mapping": name, "sample": sample, "n": len(ids),
                       "mean_W1": e.loc[ids, "infl_W1_yoy_2026_03"].mean(), "sd_W1": e.loc[ids, "infl_W1_yoy_2026_03"].std(),
                       "mean_W3": e.loc[ids, "infl_W3_cum_2020_12_to_2023_12"].mean()})
    pd.DataFrame(ms).to_csv(TABLES / "survey_mapping_sensitivity.csv", index=False)

    group_statistics(df, rng, reps).to_csv(TABLES / "HOUSEHOLD_STATISTICS.csv", index=False)
    sensitivity_table(df, rng, reps).to_csv(TABLES / "SURVEY_SENSITIVITY.csv", index=False)
    influence(df).to_csv(TABLES / "survey_influence.csv", index=False)
    income_summary(df, rng, reps).to_csv(TABLES / "SURVEY_SUMMARY_BY_INCOME.csv", index=False)
    share_intervals(df).to_csv(TABLES / "survey_share_intervals.csv", index=False)
    print("households:", len(df), "| QC:", int(df["in_quality_controlled"].sum()), "| consistency:", int(df["in_consistency_subgroup"].sum()))


if __name__ == "__main__":
    main()
