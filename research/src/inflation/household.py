"""Household- and group-specific fixed-basket (Laspeyres) price indices.

For a household or group g with expenditure shares s_gk (sum 1) in the weight
reference period b (2025) and COICOP division indices P_k(t):

    I_g(t) = sum_k s_gk * P_k(t) / Pbar_k(b)          (I_g = 1 in period b)

Because shares and price reference refer to the same period this is a true
Laspeyres index. Inflation between t0 and t1 is pi_g = 100 (I_g(t1)/I_g(t0) - 1)
and decomposes exactly into category contributions

    c_gk = 100 * s_gk * (P_k(t1) - P_k(t0)) / Pbar_k(b) / I_g(t0),  sum_k c_gk = pi_g.

Periods other than 2025 apply the 2025 basket counterfactually (fixed-basket
historical exposure); they are labelled as such in every output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COICOP = [f"CP{i:02d}" for i in range(1, 13)]
COICOP_NAMES = {
    "CP01": "Food & non-alcoholic beverages", "CP02": "Alcohol & tobacco", "CP03": "Clothing & footwear",
    "CP04": "Housing, water, electricity, gas", "CP05": "Furnishings & household maintenance", "CP06": "Health",
    "CP07": "Transport", "CP08": "Communication", "CP09": "Recreation & culture", "CP10": "Education",
    "CP11": "Restaurants & hotels", "CP12": "Miscellaneous goods & services", "ALL": "All items",
}

# Survey category -> COICOP division(s) with fractional allocation.
# 'ALL' = all-items CPI (used where no single division applies).
SURVEY_MAP_PRIMARY = {
    "food": {"CP01": 1.0},
    "rent_repair": {"CP04": 1.0},
    "utilities": {"CP04": 1.0},
    "transport": {"CP07": 1.0},
    "health": {"CP06": 1.0},
    "education": {"CP10": 1.0},
    "clothing": {"CP03": 1.0},
    "communication": {"CP08": 1.0},
    "restaurants": {"CP11": 1.0},
    "hygiene_household": {"CP05": 0.5, "CP12": 0.5},  # household goods (CP05) + personal care (CP12); split documented
    "recreation": {"CP09": 1.0},
    "tobacco_alcohol": {"CP02": 1.0},
    "family_gifts": {"ALL": 1.0},  # transfers/gifts: no specific price index; all-items CPI
    "other_consumption": {"ALL": 1.0},
}
# Sensitivity variants
SURVEY_MAP_VARIANTS = {
    "primary": SURVEY_MAP_PRIMARY,
    "hygiene_all_CP05": {**SURVEY_MAP_PRIMARY, "hygiene_household": {"CP05": 1.0}},
    "hygiene_all_CP12": {**SURVEY_MAP_PRIMARY, "hygiene_household": {"CP12": 1.0}},
    "exclude_gifts": {k: v for k, v in SURVEY_MAP_PRIMARY.items() if k != "family_gifts"},
}


def survey_to_coicop_shares(spend: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Convert household category spending (AZN) to COICOP expenditure shares (fractions)."""
    cols = COICOP + ["ALL"]
    out = pd.DataFrame(0.0, index=spend.index, columns=cols)
    for cat, alloc in mapping.items():
        for cp, frac in alloc.items():
            out[cp] += spend[cat].fillna(0) * frac
    tot = out.sum(axis=1)
    return out.div(tot.where(tot > 0), axis=0)


def price_relatives(coicop_index: pd.DataFrame, ref_year: int) -> pd.DataFrame:
    """P_k(t) / mean_{t in ref_year} P_k(t) for every division (and ALL)."""
    ref = coicop_index[coicop_index.index.year == ref_year]
    if len(ref) != 12:
        raise ValueError(f"reference year {ref_year} incomplete ({len(ref)} months)")
    return coicop_index / ref.mean()


def laspeyres(shares: pd.DataFrame, relatives: pd.DataFrame) -> pd.DataFrame:
    """Index paths (rows: dates, columns: households/groups); shares rows must sum to 1."""
    cols = [c for c in shares.columns if c in relatives.columns]
    s = shares[cols].fillna(0.0)
    return relatives[cols] @ s.T


def window_inflation(shares: pd.DataFrame, relatives: pd.DataFrame, t0, t1) -> pd.DataFrame:
    """Inflation between t0 and t1 for each row of `shares`, with exact category contributions (pp)."""
    cols = [c for c in shares.columns if c in relatives.columns]
    s = shares[cols].fillna(0.0)
    r0, r1 = relatives.loc[t0, cols], relatives.loc[t1, cols]
    i0 = s @ r0
    contrib = 100.0 * s.mul(r1 - r0, axis=1).div(i0, axis=0)
    out = contrib.add_prefix("contrib_")
    out["inflation_pct"] = contrib.sum(axis=1)
    out["index_t0"] = i0
    out["index_t1"] = s @ r1
    return out


def annual_average_inflation(shares: pd.DataFrame, relatives: pd.DataFrame, year: int) -> pd.Series:
    idx = laspeyres(shares, relatives)
    a = idx[idx.index.year == year].mean()
    b = idx[idx.index.year == year - 1].mean()
    return 100.0 * (a / b - 1.0)
