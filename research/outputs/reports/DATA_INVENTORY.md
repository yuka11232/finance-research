# Data inventory

Prepared 24 October 2026 and updated for the public release. File sizes and SHA-256 hashes for every file in `data/raw/` are in [`data/metadata/file_inventory.csv`](../../data/metadata/file_inventory.csv).

## 1. Source files

| Path (under `data/raw/`) | What it is | Used? |
|---|---|---|
| `Azerbaijan_Data/` | 97-file AZSTAT collection assembled 21 Oct 2026 (README, SOURCE_INDEX with source URLs and hashes) | Yes |
| `survey/household_survey_responses_2026-10.xlsx` *(not distributed)* | Google Forms export of the household survey: sheet "Form Responses 1", 32 responses, 29 columns. The anonymised derivative is `outputs/tables/HOUSEHOLD_EXPENDITURE.csv` | Yes |
| `project_workbooks/team_cpi_workbook_algeria_norway_azerbaijan_2000_2025.xlsx` | Team workbook: annual inflation 2000–2025 (World Bank) plus a reconstructed CPI index (2010=100) and Brent prices, for Algeria, Norway and Azerbaijan | Cross-check only (see DATA_AUDIT §4) |
| `project_workbooks/azstat_tables_51-52_expenditure_quintiles_2024.xls` | AZSTAT Tables 51–52 (2024 expenditure quintiles). Byte-identical (SHA-256) to `Azerbaijan_Data/01_Household_Income_and_Spending/Spending_Expenditure_Quintiles__2024__AZSTAT.xls` | Duplicate; not used separately |
| `kazakhstan_hbs_bulletins/` | BNS Kazakhstan household budget bulletins, 2010–2025 annual and 2026 Q1–Q2 (2010–2019 unpacked from the original RAR archives) | Yes |
| `kazakhstan_bns/` | BNS Kazakhstan CPI tables (13 sections, main groups, regions) with `.meta.json` retrieval records | Yes |
| `international/` | IMF CPI and World Bank WDI downloads (section 2) | Yes |

Team planning notes, internal reports and copyrighted literature PDFs were used as background only and are not part of the repository; the literature is cited in the top-level README.

Inside the AZSTAT zip (details in its `SOURCE_INDEX.csv`):
- `01_Household_Income_and_Spending` holds 19 Excel tables, mostly 2024 only.
- `02_Prices_and_Inflation` holds 12 workbooks: annual CPI 1993–2025, monthly levels 2019–2025, detailed and regional CPI 2020–2025.
- `03_Yearbooks` holds the household-survey yearbooks (2023 and 2024 editions zipped; 2025 and 2026 as PDF) and the price yearbooks for 2025 and 2026.
- `04_Price_Bulletins` holds monthly bulletins for 2013–2026M09.
- `05_Model_Ready_Data` holds 8 CSVs extracted by an earlier script, which also produced `06_Code` and `07_Forecast_Results` (a preliminary one-step backtest).

## 2. Data acquired during this run (not present before)

| File in `research/data/raw/international/` | Source | Retrieved |
|---|---|---|
| `imf_cpi_monthly_index_coicop.csv.gz` | IMF CPI dataset (`IMF.STA:CPI` v5.0.0, SDMX API): monthly index levels, all items and 12 COICOP divisions, for AZE, DZA, KAZ, RUS, NOR | 2026-10-24 15:08 UTC; SHA-256 of the uncompressed file in `.meta.json` |
| `worldbank_FP_CPI_TOTL_ZG.json`, `worldbank_FP_CPI_TOTL.json` | World Bank WDI annual CPI inflation and CPI 2010=100 | 2026-10-24 |

These downloads were needed because the project contained **no data for Kazakhstan or Russia**, no monthly data for Algeria or Norway, and no COICOP-division price indices for Azerbaijan. The team notes assign the Kazakhstan/Russia download to a member, but no file had arrived. Every downloaded series is official national CPI as compiled by the IMF; nothing is substituted across countries.

## 3. Countries outside the primary comparison

The notes mention Ireland and Saudi Arabia/UAE as possible comparators. No data for them exist in the project, and they were not added to the primary comparison.

## 4. Data dictionary

Variable definitions for every processed dataset and output table are in [`data/metadata/DATA_DICTIONARY.csv`](../../data/metadata/DATA_DICTIONARY.csv). Coverage of every downloaded series is in [`data/metadata/series_coverage.csv`](../../data/metadata/series_coverage.csv).
