"""Household survey (Google Forms export): variable mapping, anonymisation and quality flags.

Columns are mapped by position and each mapping is checked against a keyword
from the original bilingual question text, so a re-ordered export fails loudly
instead of silently mislabelling variables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.paths import RAW

SURVEY_FILE = RAW / "survey" / "household_survey_responses_2026-10.xlsx"

# position -> (variable, keyword that must occur in the original header)
COLUMNS = {
    0: ("timestamp", "Timestamp"),
    1: ("location", "harada yaşayır"),
    2: ("household_size_raw", "neçə nəfər yaşayır"),
    3: ("dependents_raw", "asılıdır"),
    4: ("earners_raw", "müntəzəm gəliri"),
    5: ("tenure_raw", "Yaşayış şəraitiniz"),
    6: ("employment_raw", "məşğulluq statusunuz"),
    7: ("sectors_raw", "fəaliyyət sahələri"),
    8: ("income_bracket", "aylıq gəliri"),
    9: ("expenditure_bracket", "aylıq xərci"),
    10: ("food", "Ərzaq"),
    11: ("rent_repair", "Kirayə"),
    12: ("utilities", "Kommunal"),
    13: ("transport", "Nəqliyyat"),
    14: ("health", "Səhiyyə"),
    15: ("education", "Təhsil"),
    16: ("clothing", "Geyim"),
    17: ("communication", "Telefon"),
    18: ("restaurants", "Restoran"),
    19: ("hygiene_household", "gigiyena"),
    20: ("other_consumption", "Digər istehlak"),
    21: ("confidence_raw", "əminsiniz"),
    22: ("typical_raw", "adi aylıq"),
    23: ("recreation", "Əyləncə"),
    24: ("consent", "Razılıq"),
    25: ("children_raw", "18 yaşınadək"),
    26: ("tobacco_alcohol", "Siqaret"),
    27: ("family_gifts", "Ailə tədbirləri"),
    28: ("other_consumption_dup", "Digər istehlak"),
}

CATEGORIES = [
    "food", "rent_repair", "utilities", "transport", "health", "education", "clothing",
    "communication", "restaurants", "hygiene_household", "recreation", "tobacco_alcohol",
    "family_gifts", "other_consumption",
]

# AZN brackets: (lower, upper, midpoint used for burdens). Open brackets are
# flagged; for ">= 10 000" the lower bound is used (conservative: burdens are
# upper bounds), for "< 500" the midpoint 250.
BRACKETS = {
    "500 AZN-dən az": (0, 500, 250),
    "500–999": (500, 1000, 750),
    "1 000–1 499": (1000, 1500, 1250),
    "1 500–1 999": (1500, 2000, 1750),
    "2 000–2 499": (2000, 2500, 2250),
    "2 500–2 999": (2500, 3000, 2750),
    "3 000–3 999": (3000, 4000, 3500),
    "4 000–4 999": (4000, 5000, 4500),
    "5 000–6 999": (5000, 7000, 6000),
    "7 000–9 999": (7000, 10000, 8500),
    "10 000 AZN və daha çox": (10000, np.inf, 10000),
}
BRACKET_ORDER = list(BRACKETS)

EMPLOYMENT = {
    "Tam iş vaxtı": "full_time",
    "Natamam iş vaxtı": "part_time",
    "Özünüməşğul": "self_employed",
    "Tələbə": "student",
    "Təqaüdçü": "pensioner",
    "Ev işləri": "homemaker_carer",
    "Digər": "other",
}
TENURE = {
    "ipoteka yoxdur": "own_no_mortgage",
    "ipoteka var": "own_mortgage",
    "Kirayə": "rent",
    "pulsuz": "rent_free",
}
CONFIDENCE = {"Çox əminəm": "very_confident", "Qismən əminəm": "fairly_confident", "Təxmini": "rough_estimate"}
TYPICAL = {"Bəli / Да": "yes", "Təxmini olaraq": "approximately", "Əmin deyiləm": "not_sure"}
SECTORS = {
    "Neft": "oil_gas_energy", "Səhiyyə": "health", "Maliyyə": "finance", "Təhsil və elm": "education_science",
    "Dövlət": "government", "Hərbi": "military_security", "İT": "it", "Mühəndislik": "engineering_manufacturing",
    "Satış": "sales_retail", "Marketinq": "marketing_media", "Tikinti": "construction_trades",
    "Nəqliyyat və logistika": "transport_logistics", "Turizm": "tourism_catering", "Kənd": "agriculture",
    "Ümumi fiziki": "manual_services", "Sahibkarlıq": "entrepreneurship", "heç kim işləmir": "nobody_works",
    "Digər": "other",
}


def _lookup(text, table):
    if pd.isna(text):
        return np.nan
    for key, val in table.items():
        if key in str(text):
            return val
    raise ValueError(f"unmapped survey answer: {text!r}")


def _count(text):
    """'4' -> 4 ; '7 və daha çox / 7 и более' -> 7 (top-coded)."""
    if pd.isna(text):
        return np.nan
    return int(str(text).split()[0])


def load_raw() -> pd.DataFrame:
    raw = pd.read_excel(SURVEY_FILE, sheet_name="Form Responses 1")
    for pos, (_, kw) in COLUMNS.items():
        if kw not in str(raw.columns[pos]):
            raise ValueError(f"column {pos} header does not contain {kw!r}: {raw.columns[pos]!r}")
    raw.columns = [COLUMNS[i][0] for i in range(raw.shape[1])]
    return raw


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values("timestamp").reset_index(drop=True)
    out = pd.DataFrame({"respondent_id": [f"R{i + 1:02d}" for i in range(len(df))]})
    out["survey_date"] = pd.to_datetime(df["timestamp"]).dt.date.astype(str)  # exact time dropped
    out["location"] = np.where(df["location"].str.contains("Bakı"), "Baku", df["location"])
    out["household_size"] = df["household_size_raw"].map(_count)
    out["household_size_topcoded"] = df["household_size_raw"].astype(str).str.contains("daha çox")
    out["children_u18"] = df["children_raw"].map(_count)
    out["children_topcoded"] = df["children_raw"].astype(str).str.contains("daha çox")
    out["earners"] = df["earners_raw"].map(_count)
    out["tenure"] = df["tenure_raw"].map(lambda t: _lookup(t, TENURE))
    out["employment"] = df["employment_raw"].map(lambda t: _lookup(t, EMPLOYMENT))
    out["confidence"] = df["confidence_raw"].map(lambda t: _lookup(t, CONFIDENCE))
    out["typical_month"] = df["typical_raw"].map(lambda t: _lookup(t, TYPICAL))
    for col in ("income_bracket", "expenditure_bracket"):
        key = df[col].map(lambda t: next(k for k in BRACKETS if k in str(t)))
        out[col] = key
        out[f"{col}_rank"] = key.map(BRACKET_ORDER.index)
        out[f"{col}_low"] = key.map(lambda k: BRACKETS[k][0])
        out[f"{col}_high"] = key.map(lambda k: BRACKETS[k][1])
        out[f"{col}_mid"] = key.map(lambda k: BRACKETS[k][2])
    for c in CATEGORIES:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    sectors = df["sectors_raw"].fillna("")
    for key, name in SECTORS.items():
        out[f"sector_{name}"] = sectors.map(lambda s: key in s).astype(int)
    out["n_sectors"] = out.filter(like="sector_").sum(axis=1)
    out["dependents_question_answered"] = df["dependents_raw"].notna()
    return out


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Quality flags. Flags are informative; only those listed in EXCLUSION_RULES remove records in Model B."""
    df = df.copy()
    cats = df[CATEGORIES]
    df["total_recorded_spending"] = cats.sum(axis=1, min_count=1)
    f = pd.DataFrame(index=df.index)
    f["flag_missing_income"] = df["income_bracket"].isna()
    f["flag_missing_any_category"] = cats.isna().any(axis=1)
    f["flag_negative_value"] = (cats < 0).any(axis=1)
    f["flag_duplicate_submission"] = df.drop(columns=["respondent_id", "survey_date"]).duplicated(keep="first")
    f["flag_straightlining"] = (cats.nunique(axis=1) == 1) & (cats.iloc[:, 0] > 0)
    f["flag_children_ge_household_size"] = df["children_u18"] >= df["household_size"]
    f["flag_earners_gt_household_size"] = df["earners"] > df["household_size"]
    # itemised total vs the respondent's own total-spending bracket (factor-2 tolerance)
    lo, hi = df["expenditure_bracket_low"], df["expenditure_bracket_high"]
    tot = df["total_recorded_spending"]
    f["flag_itemised_vs_reported_total"] = (tot > 2 * hi) | (tot < 0.5 * lo)
    # spending above income is NOT invalid (savings, credit, transfers); informational only
    f["flag_spending_exceeds_income_upper"] = tot > df["income_bracket_high"]
    f["flag_spending_exceeds_income_mid"] = tot > df["income_bracket_mid"]
    f["flag_extreme_income_spending_contradiction"] = (df["expenditure_bracket_rank"] - df["income_bracket_rank"]) >= 6
    f["flag_category_ge_5000"] = (cats >= 5000).any(axis=1)
    f["flag_food_below_50"] = df["food"] < 50
    f["flag_all_sectors_selected"] = df["n_sectors"] >= 10
    f["flag_low_confidence"] = (df["confidence"] == "rough_estimate") & (df["typical_month"] == "not_sure")
    f["flag_open_income_bracket"] = df["income_bracket"].isin(["500 AZN-dən az", "10 000 AZN və daha çox"])
    return pd.concat([df, f.astype(bool)], axis=1)


EXCLUSION_RULES = {
    "flag_duplicate_submission": "confirmed duplicate submission",
    "flag_straightlining": "identical amount entered for every category (non-substantive response)",
    "flag_itemised_vs_reported_total": "itemised spending inconsistent with the respondent's own total-spending bracket by more than a factor of 2 (unresolved unit/recall error)",
    "flag_extreme_income_spending_contradiction": "total-spending bracket >= 6 brackets above income bracket (e.g. income <500 AZN with spending >=10 000 AZN); unresolvable contradiction",
    "flag_negative_value": "negative spending",
}
# Variable-specific rule (not a record exclusion): when children >= household
# size (impossible, the respondent is an adult) household size and children are
# set to missing in the quality-controlled demographic comparisons only.


def build() -> pd.DataFrame:
    df = add_flags(tidy(load_raw()))
    excl = df[list(EXCLUSION_RULES)].any(axis=1)
    df["in_full_sample"] = True
    df["in_quality_controlled"] = ~excl
    df["exclusion_reasons"] = df[list(EXCLUSION_RULES)].apply(lambda r: "; ".join(k for k, v in r.items() if v), axis=1)
    # occupation answer unusable when every sector (incl. 'nobody works') was ticked
    sec = [c for c in df.columns if c.startswith("sector_")]
    df.loc[df["flag_all_sectors_selected"], sec] = np.nan
    df["demographics_usable"] = ~df["flag_children_ge_household_size"]
    # income-expenditure consistency subgroup: itemised spending <= income bracket upper bound
    df["in_consistency_subgroup"] = df["total_recorded_spending"] <= df["income_bracket_high"]
    return df
