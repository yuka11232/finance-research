"""Tests of the survey cleaning rules and the official-table parser on the real inputs."""
import numpy as np
import pandas as pd
import pytest

from src.ingestion.hbs_tables import parse_2024_national_series, parse_2025_structures, to_coicop
from src.paths import AZ_RAW
from src.survey.clean import BRACKETS, CATEGORIES, add_flags, build, tidy, load_raw


def test_hbs_2025_shares_sum_to_100_and_2024_column_matches_excel():
    w = to_coicop(parse_2025_structures())
    assert len(w) == 60
    assert np.allclose(w["sum_published"], 100.0, atol=0.01)
    x = pd.read_csv(AZ_RAW / "05_Model_Ready_Data/household_spending_weights_2024.csv", encoding="utf-8-sig")
    assert parse_2024_national_series().tolist() == x[x["group"] == "national"]["share_pct"].tolist()


def test_survey_mapping_counts_and_no_identifiers():
    d = build()
    assert len(d) == 32 and d["respondent_id"].is_unique
    assert "timestamp" not in d.columns
    assert (d["location"] == "Baku").all()
    assert d["in_quality_controlled"].sum() == 26
    assert set(d["income_bracket"]) <= set(BRACKETS)


def test_spending_above_income_is_flagged_but_not_excluded():
    d = build()
    over = d[d["flag_spending_exceeds_income_upper"] & ~d["flag_itemised_vs_reported_total"]
             & ~d["flag_extreme_income_spending_contradiction"] & ~d["flag_straightlining"]]
    assert len(over) > 0 and over["in_quality_controlled"].all()


def test_straightlining_rule_on_synthetic_record():
    t = tidy(load_raw()).head(2).copy()
    t.loc[0, CATEGORIES] = 56
    f = add_flags(t)
    assert bool(f.loc[0, "flag_straightlining"]) and not bool(f.loc[1, "flag_straightlining"])
