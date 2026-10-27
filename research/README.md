# Research package: inflation exposure in Azerbaijan and comparative inflation forecasting

This is the reproducible analysis behind the paper in [`../paper/`](../paper/). It covers four strands:
- **Forecasting:** compares four forecasting approaches (seasonal naive, SARIMA, ETS, and Yu's 2005 weighted fuzzy time series) for monthly CPI inflation in Azerbaijan, Algeria, Kazakhstan, Russia and Norway.
- **Exposure:** estimates inflation exposure for Azerbaijani household groups (official 2025 HBS baskets and a 32-household survey).
- **Projection:** projects group exposure from category-level forecasts.
- **Kazakhstan comparison:** consumption structure 2010–2026 and regional/urban/rural exposure from the BNS household budget bulletins and BNS division CPI.

Start with [`outputs/reports/RESEARCH_FINDINGS.md`](outputs/reports/RESEARCH_FINDINGS.md).

## Reproduce

```bash
cd research
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements-lock.txt   # Windows path; use .venv/bin on Linux/macOS
.venv/Scripts/python scripts/run_all.py                 # full pipeline, about 25 min on 10 cores
.venv/Scripts/python scripts/run_all.py --skip-download # reuse the stored IMF/World Bank vintage
.venv/Scripts/python -m pytest                          # 29 unit tests
```

The household-survey microdata are not distributed (see [`data/raw/survey/README.md`](data/raw/survey/README.md)). Without them, `run_all.py` skips the `household` step and later steps use the committed anonymised tables. Everything else runs from the files in this repository.

- **Step order:** `run_all.py` runs `ingest → audit → forecasts → evaluate → household → kazakhstan → categories → integration → diagnostics → paper → figures → verify`. `--from STEP` restarts from any step.
- **Caching:** the two backtest steps cache per-series results in `outputs/forecasts/parts/` and `outputs/forecasts/category_parts/`. These caches are not committed, so a fresh clone re-estimates every model. The combined results are committed in `outputs/forecasts/*.csv.gz`.
- **Configuration:** settings, the seed (20261024) and evaluation windows are in `config/experiment.toml`. Software versions are in `outputs/logs/environment.txt`.

## Deliverables

| # | Deliverable | File |
|---|---|---|
| 1 | Data audit | `outputs/reports/DATA_AUDIT.md` (+ `DATA_INVENTORY.md`, `data/metadata/DATA_DICTIONARY.csv`, `data/metadata/*.csv`) |
| 2 | Methodology | `outputs/reports/RESEARCH_METHODOLOGY.md` |
| 3 | Yu FTS implementation | `outputs/reports/YU_FTS_IMPLEMENTATION.md` |
| 4 | Four-model international comparison | `outputs/tables/INTERNATIONAL_FORECAST_COMPARISON.csv` |
| 5 | Forecast significance | `outputs/tables/FORECAST_SIGNIFICANCE.csv` (+ `forecast_skill_bootstrap.csv`) |
| 6 | Household expenditure (anonymised) | `outputs/tables/HOUSEHOLD_EXPENDITURE.csv` |
| 7 | Survey sensitivity | `outputs/tables/SURVEY_SENSITIVITY.csv` (+ `survey_exclusion_log.csv`, `survey_influence.csv`, `survey_mapping_sensitivity.csv`) |
| 8 | Household inflation | `outputs/tables/HOUSEHOLD_INFLATION.csv` (+ `official_group_gap_rounding.csv`) |
| 9 | Household statistics | `outputs/tables/HOUSEHOLD_STATISTICS.csv` |
| 10 | Forecast-level results | `outputs/tables/FORECAST_RESULTS.csv.gz` |
| 11 | Findings | `outputs/reports/RESEARCH_FINDINGS.md` |
| 12 | Limitations | `outputs/reports/LIMITATIONS.md` |
| 13 | Paper (LaTeX + PDF) | [`../paper/`](../paper/) |
| 14 | Code and tests | `src/`, `scripts/`, `tests/` |

**Further tables:**
- Kazakhstan: `KAZ_HBS_STRUCTURE.csv`, `KAZ_HBS_2025_SHARES.csv`, `KAZ_GROUP_INFLATION.csv`, `kaz_urban_rural_gap_series.csv`, `kaz_gap_regression.csv`, `kaz_aze_basket_comparison.csv`, `kaz_data_checks.csv`.
- Category forecasts: `category_forecast_accuracy.csv`, `category_forecast_significance.csv`.
- Group-level backtest: `group_forecast_accuracy.csv`, `group_forecast_significance.csv`, `group_gap_forecast_accuracy.csv`.
- Projections: `HOUSEHOLD_INFLATION_PROJECTIONS.csv`, `household_projection_model_agreement.csv`, `headline_out_of_sample_check.csv`.
- Forecast evaluation detail: `forecast_interval_coverage.csv`, `forecast_subperiod_accuracy.csv`, `fuzzy_weighting_comparison.csv`, `model_selection_validation.csv`.
- Diagnostics: `diagnostics_*.csv`.

**Figures** (`outputs/figures/`, PNG + SVG). These are the fifteen figures used in the paper; numbering gaps are kept so that file names stay stable:

| Figure | Content |
|---|---|
| fig01 | Azerbaijan inflation history |
| fig02 | Five-country inflation |
| fig03 | Azerbaijan, four-model forecasts |
| fig05 | Cross-country skill with block-bootstrap CIs |
| fig06 | Yu weighted vs unweighted fuzzy models |
| fig07 | Household expenditure shares |
| fig09 | Household exposure distributions |
| fig10 | Full vs quality-controlled survey samples |
| fig12 | Exposure projections |
| fig13 | Category contributions by decile |
| fig14 | Poor–rich gap vs relative food inflation (Azerbaijan) |
| fig15 | Kazakhstan consumption structure, 2010–2026 |
| fig16 | Kazakhstan regional exposure vs food share |
| fig17 | Kazakhstan rural–urban gap vs relative food inflation |
| fig18 | 2025 national baskets, Azerbaijan vs Kazakhstan |

All figures are drawn with matplotlib (constrained layout). `make_figures.py` checks every saved PNG for overlapping text, legends and labels on data, writes `outputs/logs/figure_layout_check.csv`, and fails if any collision remains.

**Verification:** `outputs/logs/verification_report.csv` holds 60 independent checks, including every key number quoted in the findings.

## Layout

```
config/            experiment.toml (all tunable settings)
data/raw/          untouched inputs:
                     Azerbaijan_Data/           AZSTAT archive (README, SOURCE_INDEX with URLs and hashes)
                     international/             IMF CPI (gzip) and World Bank WDI downloads (+ .meta.json with URL, time, SHA-256)
                     kazakhstan_hbs_bulletins/  BNS household budget bulletins 2010-2026 (2010-2019 unpacked from RAR)
                     kazakhstan_bns/            BNS CPI tables (+ .meta.json)
                     project_workbooks/         team CPI workbook (cross-check only) and a duplicate AZSTAT table
                     survey/                    survey export (not distributed)
data/processed/    harmonised series and parsed official structures
data/metadata/     inventory, coverage, cross-source checks, data dictionary
src/               ingestion, survey, inflation, forecasting, fuzzy, evaluation, statistics, visualization
scripts/           pipeline steps (see run_all.py)
tests/             unit tests (fuzzy model vs Yu 2005 worked examples; statistics; household indices; data pipeline)
outputs/           forecasts, tables, figures, reports, logs
```

## Privacy

The survey export has no names or contact details. It is still not distributed, because exact timestamps and occupation-sector combinations from a 32-household sample could help identify respondents. Published outputs drop timestamps and occupation detail and use pseudonymous ids. The local file `data/processed/survey_households_processed.csv` keeps occupation sectors and is excluded by `.gitignore`.
