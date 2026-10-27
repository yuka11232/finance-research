# Unequal Inflation Exposure in Azerbaijan

**Household baskets and forecast reliability.** This repository holds the paper and the full reproducible analysis behind it: data, code, forecasts, tables and figures.

Households face the same price changes with different consumption baskets, so one national inflation rate can hide unequal exposure. We combine monthly price indices for twelve COICOP consumption divisions with Azerbaijan's official 2025 household expenditure shares to build fixed-basket inflation for each household group. We then test how well four forecasting methods predict aggregate inflation and the gaps between groups. A pilot household survey in Baku and Kazakhstan's household budget data give supplementary evidence.

## Key findings

- **The 2020–2023 surge.** From December 2020 to December 2023, the basket of the lowest expenditure decile rose in cost by **31.17%**, against **28.73%** for the highest decile. That is a counterfactual gap of 2.44 percentage points using 2025 baskets. In the year to March 2026 the gap was only 0.09 points.
- **Drivers of the gap.** Food drives much of the widening, while furnishings, restaurants and transport offset part of it.
- **Aggregate forecasts.** Seasonal ARIMA has the lowest RMSE for Azerbaijan's aggregate inflation at every horizon (1, 3, 6 and 12 months). It is compared with seasonal naive, exponential smoothing (ETS) and Yu's (2005) weighted fuzzy time series.
- **Household-group forecasts.** SARIMA and ETS significantly beat seasonal naive only one month ahead. March 2027 projections even disagree on the sign of the gap between the lowest and highest deciles.

## Repository layout

```
paper/                     The manuscript
  main.tex                   LaTeX source (Overleaf-ready, XeLaTeX)
  main.pdf                   compiled paper
  figures/                   the 15 figures used in the paper
  verification/              manuscript checks and the provenance of every table value
research/                  Reproducible analysis package (Python)
  config/experiment.toml     all settings: seed, evaluation windows, model grids
  data/raw/                  source data: AZSTAT, IMF, World Bank, Kazakhstan BNS
  data/processed/            harmonised series
  data/metadata/             inventory with SHA-256 hashes, coverage, cross-source checks, data dictionary
  src/                       ingestion, survey, inflation, forecasting, fuzzy models, evaluation, statistics, plots
  scripts/                   pipeline steps (run_all.py runs them in order)
  tests/                     unit tests
  outputs/                   forecasts, tables, figures, reports and logs
```

Start with [`paper/main.pdf`](paper/main.pdf). The full findings and methods are in [`research/outputs/reports/`](research/outputs/reports/). [`research/README.md`](research/README.md) lists every output file.

## Reproducing the results

Requires Python 3.11+ (the reported run used 3.13).

```bash
cd research
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt   # on Linux/macOS use .venv/bin/python
.venv/Scripts/python scripts/run_all.py --skip-download        # reuse the stored data vintage
.venv/Scripts/python -m pytest                                 # unit tests
```

- **Run time:** a full run from a fresh clone re-estimates every rolling backtest, about 25 minutes on 10 cores.
- **Restarting:** `--from STEP` restarts the pipeline at any step.
- **Fresh data:** omit `--skip-download` to download a new IMF and World Bank vintage.
- **Checks:** the final step, `verify`, runs 60 independent checks of the outputs and of every key number quoted in the findings.

The paper compiles with XeLaTeX. To build it in Overleaf, upload the `paper/` folder and choose XeLaTeX as the compiler.

## Data sources

| Source | Content | Location |
|---|---|---|
| State Statistical Committee of Azerbaijan (AZSTAT) | household budget survey yearbooks, CPI tables, monthly price bulletins | `research/data/raw/Azerbaijan_Data/` |
| IMF, *Consumer Price Index* (IMF.STA:CPI) | monthly all-items and COICOP division CPI for Azerbaijan, Algeria, Kazakhstan, Russia, Norway | `research/data/raw/international/` |
| World Bank, *World Development Indicators* | annual CPI and inflation (FP.CPI.TOTL, FP.CPI.TOTL.ZG) | `research/data/raw/international/` |
| Bureau of National Statistics of Kazakhstan (BNS) | household budget bulletins 2010–2026 and CPI by section and region | `research/data/raw/kazakhstan_hbs_bulletins/`, `research/data/raw/kazakhstan_bns/` |
| Pilot household survey, Baku (October 2026) | 32 responses; anonymised derivative only | `research/outputs/tables/HOUSEHOLD_EXPENDITURE.csv` |

Downloaded files have a `.meta.json` record with the URL, retrieval time and SHA-256 hash. The analysis uses the vintage retrieved on 24–25 October 2026. Third-party data remain subject to their providers' terms of use.

**Privacy.** The raw survey export is not distributed. The published survey tables use pseudonymous ids and contain no timestamps or occupation detail. See [`research/data/raw/survey/README.md`](research/data/raw/survey/README.md).

## Methods references

- **Seasonal ARIMA and ETS:** Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- **Weighted fuzzy time series:** Yu, H.-K. (2005). Weighted fuzzy time series models for TAIEX forecasting. *Physica A*, 349(3–4), 609–624. https://doi.org/10.1016/j.physa.2004.11.006
- **Conventional fuzzy time series:** Chen, S.-M. (1996). Forecasting enrollments based on fuzzy time series. *Fuzzy Sets and Systems*, 81(3), 311–319. https://doi.org/10.1016/0165-0114(95)00220-0
- **Interval lengths:** Huarng, K. (2001). Effective lengths of intervals to improve forecasting in fuzzy time series. *Fuzzy Sets and Systems*, 123(3), 387–394. https://doi.org/10.1016/S0165-0114(00)00057-9
- **Forecast accuracy tests:** Diebold–Mariano tests with the Harvey–Leybourne–Newbold correction and Holm adjustment.

The paper's bibliography has the full reference list.

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The data in `research/data/raw/` belong to their original publishers and are redistributed for reproducibility under those publishers' terms.
