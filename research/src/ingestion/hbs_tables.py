"""Official household-budget-survey consumption structures (AZSTAT).

Parses the 2025 'composition of consumption expenditure' tables from the
2026 household-survey yearbook (PDF text layer, Azerbaijani labels) and the
2024 Excel table, and maps the 12 published expenditure items to COICOP
divisions:

  food (incl. alcohol)      -> CP01 = food - alcohol ; CP02 = alcohol + tobacco
  clothing & footwear       -> CP03
  water, electricity, gas   -> CP04
  furnishings, maintenance  -> CP05
  health                    -> CP06
  transport                 -> CP07
  communication             -> CP08
  recreation & culture      -> CP09
  education                 -> CP10
  hotels, cafes, restaurants-> CP11
  other goods & services    -> CP12
"""
from __future__ import annotations

import re

import pandas as pd
import pypdf

from src.paths import AZ_RAW

YEARBOOK = AZ_RAW / "03_Yearbooks" / "Household_Survey__Edition_2026__AZSTAT.pdf"

ITEMS = {
    "food_incl_alcohol": r"qida\s+məhsullarına",
    "alcohol": r"alkoqollu\s+içkilərə",
    "tobacco": r"tütün\s+məmulatlarına",
    "clothing": r"paltar\s+və\s+ayaqqabıya",
    "housing_utilities": r"yanacaq\s+növlərinə",
    "furnishings": r"gündəlik\s+qulluğa",
    "health": r"səhiyyə\s+xidmətlərinə",
    "transport": r"nəqliyyat\s+xidmətlərinə",
    "communication": r"rabitə\s+xidmətlərinə",
    "recreation": r"mədəniyyət\s+xidmətlərinə",
    "education": r"(?:təhsilə|təhsil\s+xərclərinə)",
    "restaurants_hotels": r"restoran\s+xidmətlərinə",
    "other": r"digər\s+mal\s+və\s+xidmətlərə",
}

# page (0-based == printed page), table number, dimension, column labels.
TABLES = [
    (48, 37, "area", ["national"], "last"),
    (49, 38, "area", ["urban"], "last"),
    (50, 39, "area", ["rural"], "last"),
    (52, 41, "economic_region", ["Baku city", "Nakhchivan", "Absheron-Khizi", "Mountainous Shirvan", "Ganja-Dashkasan",
                                  "Karabakh", "Gazakh-Tovuz", "Guba-Khachmaz", "Lankaran-Astara", "Central Aran",
                                  "Mil-Mughan", "Shaki-Zagatala", "East Zangezur", "Shirvan-Salyan"], "all"),
    (56, 45, "children_count", ["1 child", "2 children", "3 children", "4+ children"], "all"),
    (58, 47, "children_presence", ["with children", "without children"], "all"),
    (60, 49, "household_size", ["1 person", "2 persons", "3 persons", "4 persons", "5 persons", "6+ persons"], "all"),
    (62, 51, "head_status", ["hired workers", "self-employed", "pensioners & benefit recipients", "other"], "all"),
    (64, 53, "head_education", ["higher", "specialised secondary/vocational", "complete general secondary",
                                 "general secondary", "primary or less"], "all"),
    (66, 55, "head_age", ["18-29", "30-39", "40-49", "50-59", "60+"], "all"),
    (68, 57, "head_sex", ["male", "female"], "all"),
    (70, 59, "expenditure_quintile", ["Q1", "Q2", "Q3", "Q4", "Q5"], "all"),
    (72, 61, "expenditure_decile", [f"D{i}" for i in range(1, 11)], "all"),
]
NUM = r"(\d+,\d)"
# In the 2001..2025 time-series tables a missing year is printed as "-" (tobacco, 2001)
# and must be counted as a column, otherwise later columns shift.
TOKEN = r"(\d+,\d|(?<=\s)-(?=\s))"


def _numbers_after(text: str, pattern: str, k: int, take: str) -> list[float]:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"label not found: {pattern}")
    if take == "last":  # time-series tables 2001..2025 (11 columns): 2025 is the 11th
        vals = re.findall(TOKEN, text[m.end():])[:11]
        if len(vals) < 11 or vals[-1] == "-":
            raise ValueError(f"expected 11 year columns after {pattern}")
        return [float(vals[-1].replace(",", "."))]
    nums = re.findall(NUM, text[m.end():])
    if len(nums) < k:
        raise ValueError(f"expected {k} numbers after {pattern}")
    return [float(v.replace(",", ".")) for v in nums[:k]]


def parse_2025_structures() -> pd.DataFrame:
    reader = pypdf.PdfReader(str(YEARBOOK))
    rows = []
    for page, table, dim, cols, take in TABLES:
        text = reader.pages[page].extract_text()
        if f"Cədvəl {table}." not in text:
            raise ValueError(f"page {page} is not table {table}")
        for item, pat in ITEMS.items():
            vals = _numbers_after(text, pat, len(cols), take)
            for col, v in zip(cols, vals):
                rows.append({"weight_year": 2025, "dimension": dim, "group": col, "item": item, "share_pct": v,
                             "source": f"AZSTAT Household Survey yearbook 2026, Table {table}, p.{page}"})
    return pd.DataFrame(rows)


def parse_2024_national_series() -> pd.DataFrame:
    """2024 column of Table 37 (used only to cross-check against the 2024 Excel table)."""
    text = pypdf.PdfReader(str(YEARBOOK)).pages[48].extract_text()
    out = {}
    for item, pat in ITEMS.items():
        m = re.search(pat, text)
        toks = re.findall(TOKEN, text[m.end():])[:11]
        out[item] = float(toks[9].replace(",", "."))
    return pd.Series(out, name="national_2024_from_pdf")


def to_coicop(df: pd.DataFrame) -> pd.DataFrame:
    """Wide COICOP shares (percent, summing to ~100) per (dimension, group)."""
    w = df.pivot_table(index=["dimension", "group"], columns="item", values="share_pct", sort=False)
    out = pd.DataFrame(index=w.index)
    out["CP01"] = w["food_incl_alcohol"] - w["alcohol"]
    out["CP02"] = w["alcohol"] + w["tobacco"]
    for cp, item in [("CP03", "clothing"), ("CP04", "housing_utilities"), ("CP05", "furnishings"), ("CP06", "health"),
                     ("CP07", "transport"), ("CP08", "communication"), ("CP09", "recreation"), ("CP10", "education"),
                     ("CP11", "restaurants_hotels"), ("CP12", "other")]:
        out[cp] = w[item]
    out["sum_published"] = out.sum(axis=1)
    return out.reset_index()
