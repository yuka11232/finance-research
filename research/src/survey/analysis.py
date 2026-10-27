"""Expenditure shares, income burdens, groupings and sample definitions for the survey."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.survey.clean import CATEGORIES

SAMPLES = {
    "full": "in_full_sample",
    "quality_controlled": "in_quality_controlled",
    "consistency_subgroup": "in_consistency_subgroup",
}


def shares_and_burdens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tot = df["total_recorded_spending"]
    for c in CATEGORIES:
        df[f"share_{c}"] = 100.0 * df[c] / tot.where(tot > 0)
        df[f"burden_{c}"] = 100.0 * df[c] / df["income_bracket_mid"]
    df["burden_total"] = 100.0 * tot / df["income_bracket_mid"]
    df["burden_total_at_income_low"] = 100.0 * tot / df["income_bracket_low"].replace(0, np.nan)
    df["burden_total_at_income_high"] = 100.0 * tot / df["income_bracket_high"].replace(np.inf, np.nan)
    df["burden_food_at_income_low"] = 100.0 * df["food"] / df["income_bracket_low"].replace(0, np.nan)
    df["burden_food_at_income_high"] = 100.0 * df["food"] / df["income_bracket_high"].replace(np.inf, np.nan)
    sh = df[[f"share_{c}" for c in CATEGORIES]] / 100.0
    df["hhi_concentration"] = (sh**2).sum(axis=1)
    df["effective_n_categories"] = 1.0 / df["hhi_concentration"]
    df["essentials_share"] = df[["share_food", "share_rent_repair", "share_utilities", "share_health"]].sum(axis=1)
    return df


def add_groupings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mid = df["income_bracket_mid"]
    df["income_band"] = pd.cut(mid, [-1, 1999, 4999, np.inf], labels=["low (<2000 AZN)", "middle (2000-4999 AZN)", "high (>=5000 AZN)"])
    df["income_split"] = np.where(mid >= 5000, ">=5000 AZN", "<5000 AZN")
    adults = (df["household_size"] - df["children_u18"]).clip(lower=1)
    scale = 1 + 0.5 * (adults - 1) + 0.3 * df["children_u18"]  # OECD-modified equivalence scale
    df["equivalised_income"] = (mid / scale).where(df["demographics_usable"])
    df["eq_income_tercile"] = pd.qcut(df["equivalised_income"], 3, labels=["T1 (lowest)", "T2", "T3 (highest)"])
    df["employment_group"] = np.where(df["employment"] == "full_time", "full-time employed", "not full-time")
    hs = df["household_size"].where(df["demographics_usable"])
    df["size_group"] = pd.cut(hs, [0, 3, 99], labels=["2-3 persons", "4+ persons"])
    ch = df["children_u18"].where(df["demographics_usable"])
    df["children_group"] = pd.cut(ch, [-1, 0, 99], labels=["no children", "1+ children"])
    df["tenure_group"] = np.where(df["tenure"] == "own_no_mortgage", "owner, no mortgage", "mortgage/rent/free")
    df["confidence_group"] = np.where(df["confidence"] == "rough_estimate", "rough estimate", "fairly/very confident")
    return df


def trimmed_mean(x, prop=0.1):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(stats.trim_mean(x, prop)) if len(x) else np.nan
