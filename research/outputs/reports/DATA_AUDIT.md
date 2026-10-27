# Data audit

All numbers below come from `scripts/audit_data.py` and are stored in `data/metadata/` (`series_coverage.csv`, `cross_source_checks.csv`, `team_workbook_check.csv`).

## 1. Price data used in the experiments

| Series | Source | Frequency | Coverage actually observed | Obs. | Unit | Missing inside range |
|---|---|---|---|---|---|---|
| Azerbaijan all-items CPI | IMF CPI (= AZSTAT) | monthly | 2005-01 – 2026-03, extended to 2026-09 with AZSTAT relatives | 261 | index 2010=100 | 0 |
| Azerbaijan COICOP divisions CP01–CP12 | IMF CPI | monthly | 2005-01 – 2026-03 | 255 each | index 2010=100 | 0 |
| Algeria all-items CPI | IMF CPI | monthly | 2005-01 – 2026-08 | 260 | index 2001=100 | 0 |
| Kazakhstan all-items CPI | IMF CPI | monthly | 2005-01 – 2026-09 | 261 | index Dec-2020=100 | 0 |
| Norway all-items CPI | IMF CPI | monthly | 1995-01 – 2026-09 | 381 | index 2025=100 | 0 |
| Russia all-items CPI | IMF CPI | monthly | 1995-01 – 2026-08 | 380 | index 2010=100 | 0 |
| Annual CPI inflation, five countries | World Bank WDI | annual | 1995/2000 – 2025 | 26–31 | % change, annual average | 0 |
| AZSTAT broad CPI (national, food-beverages-tobacco, non-food goods, services) | AZSTAT bulletins (model-ready CSV) | monthly | 2016-01 – 2026-09 | 129 each | previous month = 100 | 0 |

The IMF API returns rows with empty values before a series starts: AZE before 2005, KAZ and DZA COICOP before 2005, and every KAZ and NOR COICOP row. These are *not* observations. The coverage table distinguishes `rows_returned` from `non_missing`. Russia's COICOP divisions stop in 2022-02. Kazakhstan's and Norway's divisions are empty in this dataset. Division-level work is done for Azerbaijan only.

**Index conventions:**
- **Reference periods differ** (2010, 2001, Dec-2020, 2025). Index levels are therefore never compared across countries. Only m/m and y/y rates computed within each country are compared.
- **All five series are national all-items CPIs** with monthly price collection, so the rates are comparable in definition. No seasonal adjustment is applied to any series. Each country's index is published monthly.

## 2. Cross-source checks

| Check | n | Result |
|---|---|---|
| IMF AZE CPI level vs AZSTAT official 2010=100 level (2019–2025) | 84 months | max relative difference 0.028% |
| IMF AZE m/m vs AZSTAT published m/m (1 decimal) | 123 months | max abs. difference 0.072 pp; correlation 0.9993 (rounding) |
| IMF annual-average inflation vs World Bank WDI, each country | 20–30 years | max abs. difference < 2×10⁻¹³ pp (identical) |
| Spliced AZE y/y Sep 2026 vs AZSTAT published 105.5 | 1 | 5.468% vs 5.5% (difference 0.03 pp, rounding of monthly relatives) |
| `azstat_tables_51-52_expenditure_quintiles_2024.xls` vs AZSTAT copy | 1 | identical (SHA-256) |

**Splice decision.** IMF and AZSTAT agree within rounding. So the Azerbaijan headline series is extended from 2026-04 to 2026-09 by chaining AZSTAT's published previous-month relatives. These six months are flagged `source = AZSTAT_chained` in `cpi_headline_monthly.csv`.

## 3. Household survey (Google Forms export)

- **Responses and coverage:** 32 responses, submitted 12–18 October 2026; all 32 report living in **Baku**. There are no rural or regional respondents, so the survey cannot support an urban/rural comparison.
- **Identifiers:** there are no names, e-mails or phone numbers. Timestamps are dropped in all outputs, and respondents get sequential ids (R01–R32). The detailed occupation combination is kept only in the local processed file, not in published tables.
- **Structure:**
  - **Income and total spending:** answered as brackets (11 bands; open ends "<500" and "≥10 000 AZN").
  - **Spending:** 14 numeric categories in AZN per month.
  - **Household:** size (top-coded 7+), children under 18 (5+), earners (4+), tenure, and the respondent's employment status.
  - **Occupation:** sector, multi-select.
  - **Data-quality questions:** self-assessed confidence, and whether the month was typical.
- **Empty questions:** "number of dependants" (column 4) and a duplicated "other consumption" column (column 29) were never answered, so they are excluded.
- **Missing values:** none among the answered questions.
- **Quality flags** (counts from `HOUSEHOLD_EXPENDITURE.csv`):

| Flag | n | Treatment |
|---|---|---|
| duplicate submission | 0 | — |
| negative values | 0 | — |
| identical amount in every category (straight-lining) | 1 | excluded in quality-controlled (QC) sample |
| itemised total inconsistent with own total-spending bracket by > ×2 | 5 | excluded in QC |
| spending bracket ≥ 6 brackets above income bracket | 1 | excluded in QC |
| children ≥ household size | 3 | household size/children set missing in QC demographic comparisons only |
| all occupation sectors ticked (incl. "nobody works") | 1 | occupation set missing |
| spending above income-bracket upper bound | 7 | **not** an exclusion (savings, credit, transfers) |
| a category ≥ 5 000 AZN | 3 | flagged |
| food < 50 AZN/month | 2 | flagged |
| "rough estimate" + "not sure month is typical" | 4 | flagged |

The QC sample has 26 households, and the income–expenditure consistency subgroup has 25. The exclusion log is `outputs/tables/survey_exclusion_log.csv`.

## 4. Problems found in existing project files

1. **Team CPI workbook, Azerbaijan CPI levels:** 2008 is stored as `93225` and 2014 as `113175`. World Bank values are 93.225 and 113.175, so a decimal separator was lost. Its inflation rates match the World Bank to 5×10⁻⁸. The workbook was not used for modelling.
2. **Earlier package (`06_Code`, `07_Forecast_Results`):**
   - Its "fuzzy" model is a triangular fuzzy regression, not Yu (2005).
   - Its test set is 12 months.
   - Its own README reports that seasonal naive beat it on the test set.
   - It is superseded here, and its numbers are not reused.
3. **Detailed AZSTAT category labels** change wording in 2019/2020, 2024 and 2025, and AZSTAT does not publish within-category CPI weights. This is why the COICOP-division indices from the IMF CPI dataset are used for household indices, instead of averaging AZSTAT items with invented weights.
4. **Household-survey yearbook 2026, Table 37:** tobacco is printed with "–" for 2001. A naive text parser shifts columns. The parser counts the dash, and all 60 parsed 2025 groups sum to 100.0. The parsed 2024 national column equals AZSTAT's 2024 Excel table exactly (13/13 items).

## 5. What is *not* available

- Household-level HBS microdata or survey design weights for Azerbaijan.
- Group expenditure shares before 2024/2025 in machine-readable form. The 2026 yearbook gives national, urban and rural structures for 2001–2025 but group structures for 2025 only.
- Real-time CPI vintages. All backtests use the current vintage.
- A live Google Sheets connection; the Excel export was used.
- Monthly CPI before 2005 for Azerbaijan, Algeria and Kazakhstan in the IMF dataset.
