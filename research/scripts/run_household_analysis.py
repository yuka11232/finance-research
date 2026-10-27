"""Phases 6-7: survey expenditure analysis, data-quality samples and household inflation exposure.

Outputs (outputs/tables):
  HOUSEHOLD_EXPENDITURE.csv      anonymised household shares, burdens and flags
  SURVEY_SENSITIVITY.csv         full vs quality-controlled vs consistency subgroup
  HOUSEHOLD_INFLATION.csv        household (survey) and official-group inflation exposure, with contributions
  HOUSEHOLD_STATISTICS.csv       group comparisons with uncertainty, effect sizes, tests and MDEs
  survey_exclusion_log.csv       every excluded record and the rule applied
  survey_influence.csv           leave-one-out influence on key statistics
  official_group_gap_rounding.csv  sensitivity of official gaps to 0.1-pp rounding of published shares
  survey_mapping_sensitivity.csv household exposure under alternative category->COICOP mappings
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
from src.survey.clean import CATEGORIES, EXCLUSION_RULES, build

T = pd.Timestamp
WINDOWS = {  # name: (t0, t1, label)
    "W1_yoy_2026_03": (T("2025-03-01"), T("2026-03-01"), "12-month change to March 2026 (latest COICOP month)"),
    "W3_cum_2020_12_to_2023_12": (T("2020-12-01"), T("2023-12-01"), "cumulative Dec 2020 - Dec 2023 (fixed 2025 basket counterfactual)"),
}
ANNUAL = {"W2_annual_avg_2025": 2025}
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
    d = df[df["in_full_sample"]].reset_index(drop=True)
    rows = []
    for v in ["share_food", "infl_W1_yoy_2026_03", "burden_total"]:
        base = d[v].mean()
        for i, r in d.iterrows():
            rows.append({"variable": v, "respondent_id": r["respondent_id"], "full_mean": base,
                         "mean_without": d[v].drop(i).mean(), "change": d[v].drop(i).mean() - base})
    return pd.DataFrame(rows).sort_values(["variable", "change"], key=lambda s: s.abs() if s.name == "change" else s, ascending=[True, False])


def main():
    ensure_dirs()
    cfg = config()
    rng = np.random.default_rng(cfg["project"]["seed"])
    reps = cfg["project"]["bootstrap_reps"]
    rel = price_relatives(coicop_monthly("AZE").dropna(), cfg["household"]["price_reference_year"])
    structures = parse_2025_structures()
    structures.to_csv(PROCESSED / "official_hbs_2025_structures_long.csv", index=False)
    to_coicop(structures).to_csv(PROCESSED / "official_shares_2025_coicop.csv", index=False)

    df = add_groupings(shares_and_burdens(build()))
    exp = household_exposure(df, rel, SURVEY_MAP_VARIANTS["primary"])
    df = df.merge(exp, left_on="respondent_id", right_index=True)
    df.to_csv(PROCESSED / "survey_households_processed.csv", index=False)  # local, includes sector detail

    # ---- anonymised household file (no timestamps, no occupation detail) ----
    flags = [c for c in df.columns if c.startswith("flag_")]
    keep = (["respondent_id", "in_full_sample", "in_quality_controlled", "in_consistency_subgroup", "exclusion_reasons",
             "demographics_usable", "household_size", "household_size_topcoded", "children_u18", "earners", "tenure",
             "employment_group", "income_bracket", "income_band", "income_bracket_mid", "expenditure_bracket", "confidence",
             "typical_month", "total_recorded_spending"] + CATEGORIES + [f"share_{c}" for c in CATEGORIES] +
            [f"burden_{c}" for c in CATEGORIES] + ["burden_total", "burden_total_at_income_low", "burden_total_at_income_high",
             "hhi_concentration", "effective_n_categories", "essentials_share"] + flags)
    df[keep].to_csv(TABLES / "HOUSEHOLD_EXPENDITURE.csv", index=False)
    df.loc[~df["in_quality_controlled"], ["respondent_id", "exclusion_reasons"]].assign(
        rule_text=lambda x: x["exclusion_reasons"].map(lambda s: " | ".join(EXCLUSION_RULES[k] for k in s.split("; ") if k))
    ).to_csv(TABLES / "survey_exclusion_log.csv", index=False)

    # ---- household inflation (survey + official groups) ----
    hh_cols = ["respondent_id", "in_full_sample", "in_quality_controlled", "in_consistency_subgroup", "income_band",
               "income_split", "employment_group", "size_group", "children_group", "tenure_group"]
    hh = df[hh_cols + [c for c in exp.columns]].copy()
    hh.insert(0, "unit", "survey_household_2026_basket_as_2025_weights")
    off = official_exposure(rel)
    pd.concat([hh, off], ignore_index=True).to_csv(TABLES / "HOUSEHOLD_INFLATION.csv", index=False)
    rounding_sensitivity(rel, rng).to_csv(TABLES / "official_group_gap_rounding.csv", index=False)

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
    print("households:", len(df), "| QC:", int(df["in_quality_controlled"].sum()), "| consistency:", int(df["in_consistency_subgroup"].sum()))


if __name__ == "__main__":
    main()
