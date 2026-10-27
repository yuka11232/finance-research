# Research findings

**Inflation exposure across socioeconomic and demographic groups in Azerbaijan: household expenditure heterogeneity, comparative forecasting, and weighted fuzzy time-series models**

Every number here comes from a file in `outputs/tables/`. The file is named next to the result. Key figures are re-checked by `scripts/verify_outputs.py`.

**Verdict labels used throughout:**
- **Supported**: the evidence supports the claim.
- **Not distinguishable**: statistical tests at 5%, after multiple-testing correction where a family applies, do not separate the alternatives. This is *not* evidence of equality.
- **Inconclusive**: there is too little information to decide.
- **Not testable**: the data cannot address the question.

---

## RQ2 – Do Yu's weighted fuzzy time-series forecasts beat traditional methods for Azerbaijani inflation?

**Answer: No. Yu's model is generally more accurate than seasonal naive at h = 1, but not significantly so. It is less accurate than SARIMA, significantly so at 3- and 6-month horizons.**

Azerbaijan, monthly CPI inflation (m/m), Analysis A. Test targets run 2019-01 to 2026-09: 93 one-step forecasts and 82 twelve-step forecasts (INTERNATIONAL_FORECAST_COMPARISON.csv).

| Horizon | Seasonal naive RMSE | SARIMA | ETS | Yu weighted FTS | Yu skill vs SN [95% block-bootstrap CI] |
|---|---|---|---|---|---|
| h = 1 | 0.689 | **0.448** | 0.487 | 0.526 | +0.237 [−0.062, 0.448] |
| h = 3 | 0.695 | **0.525** | 0.620 | 0.750 | −0.080 |
| h = 6 | 0.701 | **0.531** | 0.699 | 0.845 | −0.205 |
| h = 12 | 0.710 | **0.539** | 0.685 | 0.650 | +0.084 |

RMSE is in percentage points of m/m inflation.

- **Yu FTS vs seasonal naive:** at h = 1 Yu's RMSE is 24% lower, but the bootstrap CI includes zero and the DM/HLN test is not significant after Holm adjustment (Holm p = 0.204; FORECAST_SIGNIFICANCE.csv). At h = 3 and h = 6 Yu FTS is *worse* than seasonal naive; this is a point estimate, also not significant. **Verdict: not distinguishable.**
- **Yu FTS vs SARIMA:** SARIMA is significantly more accurate at h = 3 (Holm p = 0.041) and h = 6 (Holm p < 0.001). The difference is not significant at h = 1 (Holm p = 0.061) or h = 12 (Holm p = 0.138). **Verdict: the hypothesis that Yu outperforms is not supported. Where the evidence is conclusive, it favours SARIMA.**
- **SARIMA vs seasonal naive:** SARIMA is better at every horizon in Azerbaijan (skill 0.24–0.35). The difference is significant only at h = 1: skill 0.349 [0.097, 0.511], Holm p = 0.0497, a marginal result.
- **Implied y/y inflation:** the ranking is the same. SARIMA has the lowest RMSE at every horizon (h = 12: 4.38 pp, vs 5.79 seasonal naive and 6.20 Yu FTS), and SARIMA is significantly better than Yu at h = 6 and 12 (Holm p = 0.049 and 0.006).

## RQ3 – How does forecast accuracy in Azerbaijan compare with Algeria, Kazakhstan, Russia and Norway?

Figures 3–5 and 11. One-step RMSE of m/m inflation, Analysis A (pp):

| Country | Test targets | SN | SARIMA | ETS | Yu FTS | Best primary model |
|---|---|---|---|---|---|---|
| Azerbaijan | 2019-01..2026-09 (93) | 0.689 | 0.448 | 0.487 | 0.526 | SARIMA |
| Algeria | 2019-01..2026-08 (92) | 0.973 | 0.836 | 0.813 | 1.039 | ETS |
| Kazakhstan | 2019-01..2026-09 (93) | 0.629 | 0.387 | 0.450 | 0.466 | SARIMA |
| Norway | 2019-01..2026-09 (93) | 0.557 | 0.443 | 0.440 | 0.545 | ETS |
| Russia | 2019-01..2026-08 (92) | 1.150 | 0.949 | 0.941 | 1.002 | ETS |

- **Consistency across countries.** Of the 20 country × horizon cells for m/m in Analysis A, SARIMA has the lowest RMSE among the four primary models in 13 and ETS in 7. Yu FTS has the lowest in none, and seasonal naive in none. The supplementary AR(AIC) model is occasionally lower still (for example Kazakhstan h = 12 and Russia h = 6). The cross-country pattern is the same in the common-period Analysis B (SARIMA 16, ETS 4) and for implied y/y inflation (A: SARIMA 15, ETS 5).
- **Statistical significance, m/m, Analysis A** (7 Holm-adjusted comparisons per country and horizon; 20 cells each):
  - SARIMA significantly beats seasonal naive in 4/20 cells: AZE h1, KAZ h1, NOR h6 and h12. ETS does so in 2/20: NOR h6, h12.
  - Yu FTS significantly beats seasonal naive in 1/20 (NOR h12).
  - SARIMA is significantly more accurate than Yu FTS in 10/20, and ETS in 6/20. Yu is never significantly more accurate than SARIMA or ETS.
  - Absolute-error loss gives the same picture: SARIMA vs SN significant in 4/20, ETS in 4/20, Yu in 1/20.
- **Russia.** No comparison is significant at any horizon, even though SARIMA's point skill is 0.15–0.18. The March-2022 shock (+7.6% m/m) accounts for 53% of SARIMA's one-step squared error. The resulting HAC variance leaves the test without power. Bootstrap skill CIs for Russia span roughly −0.1 to +0.45 for SARIMA and −0.5 to +0.7 for Yu (forecast_skill_bootstrap.csv). **Verdict: inconclusive.**
- **Directly modelled y/y inflation (robustness).** Here the random walk is the relevant benchmark.
  - SARIMA's skill over the random walk is substantial in Azerbaijan (0.33 at h1, 0.26 at h12) and Kazakhstan (0.33, 0.22). It is near zero in Algeria, and negative in Norway (h ≥ 3) and Russia.
  - The selected ETS for AZE/DZA/KAZ is simple exponential smoothing with smoothing parameter ≈ 1, so it reproduces the random walk to three decimals.
  - Yu FTS has negative one-step skill against the random walk in every country.
- **Volatility and instability (descriptive; n = 5 countries, not a formal test).**
  - **Sub-periods.** All models' errors are largest in 2021–2023 (forecast_subperiod_accuracy.csv). SARIMA's advantage over seasonal naive did *not* disappear in the surge: skill in 2021–2023 was 0.38 for AZE, 0.40 for KAZ, 0.24 for NOR and 0.18 for RUS. The exception is Algeria, at −0.03. In calm 2019–2020 it was close to zero for AZE (−0.01) and KAZ (0.09).
  - **Across countries.** Test-period volatility of m/m inflation is ranked negatively with one-step skill (Spearman ρ = −0.9 for SARIMA, −0.7 for ETS and Yu FTS). Algeria and Russia are the most volatile and show the lowest skill. With five countries this is an exploratory association only; no mechanism is claimed.
- **Analysis A vs B.** Extra history (Norway and Russia from 2000 rather than 2005) changes little: Norway SARIMA h1 skill 0.206 (A) vs 0.220 (B), and Russia 0.175 vs 0.201.
  - **What each analysis supports.** Analysis B supports like-for-like cross-country statements on identical evaluation dates. Analysis A supports statements about each country's best attainable accuracy.
- **Interval calibration (h = 1–12, averaged over countries).**
  - SARIMA/ETS model-based 80% intervals contain the outcome 87–89% of the time (too wide); their 95% intervals cover 95–96%.
  - Empirical intervals built from past errors under-cover at long horizons: 80% nominal gives 66–70% at h = 12, and 95% gives 83–86% (forecast_interval_coverage.csv).

## RQ4 – Are model differences statistically significant or explainable by sampling variability?

Across 5 countries × 4 horizons × 7 preselected comparisons (m/m, Analysis A, squared-error loss), **27 of 140 tests are significant after Holm correction**:

| Comparison | Model significantly better | Comparator significantly better | Not distinguishable |
|---|---|---|---|
| SARIMA vs seasonal naive | 4 | 0 | 16 |
| ETS vs seasonal naive | 2 | 0 | 18 |
| Yu FTS vs seasonal naive | 1 | 0 | 19 |
| Yu FTS vs SARIMA | 0 | 10 | 10 |
| Yu FTS vs ETS | 0 | 6 | 14 |
| Yu FTS vs equal-weight recurrence FTS | 1 | 0 | 19 |
| Yu FTS vs Chen (1996) FTS | 3 | 0 | 17 |

- **Supported:** SARIMA and ETS are more accurate than Yu's weighted FTS in a substantial share of cases.
- **Not distinguishable in most cells:** SARIMA, ETS and Yu FTS each have lower point RMSE than seasonal naive in most cells, but the improvement is significant in few. Much of the apparent ranking among the better models is within sampling variability over 81–93 test months.
- **Not testable:** naive vs seasonal naive at h = 12, where the two forecasts are identical by construction (40 rows, documented).

### Weighted vs unweighted fuzzy models (fuzzy_weighting_comparison.csv; Figure 6)

- **Weights alone add nothing.** Holding recurrence fixed, Yu's chronological weights have lower RMSE than equal weights in only **9 of 20** country-horizon cells (median RMSE ratio 1.005). The single significant difference is Algeria h = 12. **Verdict: not distinguishable.**
- **Recurrence plus weights beats Chen.** Yu's model has lower RMSE than Chen's in **18 of 20** cells (median ratio 0.938), significant in 3. Because equal-weight recurrence alone performs about as well as Yu, **the gain over Chen comes from keeping recurrent relationships, not from the chronological weights.**
- **Direct y/y target.** Yu beats Chen in only 7/20 cells, and no difference is significant.

## RQ1 / RQ6 – How does inflation exposure vary across Azerbaijani households, and which categories drive the differences?

### Official AZSTAT 2025 household groups (population survey, published aggregates)

Laspeyres exposure with each group's 2025 basket and COICOP division prices (HOUSEHOLD_INFLATION.csv; Figures 9, 13):

| Window | National HBS basket | Poorest decile D1 | Richest decile D10 | D1 − D10 | Q1 − Q5 | Rural − urban |
|---|---|---|---|---|---|---|
| 12 months to March 2026 | 5.93% | 5.94% | 5.84% | **+0.09 pp** | +0.08 pp | +0.01 pp |
| Annual average 2025 vs 2024 | 6.10% | 6.29% | 5.90% | **+0.39 pp** | +0.31 pp | +0.02 pp |
| Cumulative Dec 2020–Dec 2023 (counterfactual 2025 basket) | 30.19% | 31.17% | 28.73% | **+2.44 pp** | +1.95 pp | +0.40 pp |

- **Gradient by expenditure.** Exposure falls broadly, though not strictly monotonically, from the lowest to the highest expenditure groups, but the size of the gap depends entirely on *which prices rose*. During the 2021–2023 surge food prices rose 40.0% (Dec 2020–Dec 2023) against 30.8% for all items, and the poorest decile's extra exposure was 2.4 pp. In the latest 12 months food (6.4%) rose about as fast as all items (5.6%), and the gap shrank to 0.09 pp. This matches the pattern reported for Ireland by Lydon (2022).
- **Contributions.** Over 2020–2023, food contributed **18.2 pp** to D1's inflation but 11.6 pp to D10's (+6.5 pp). This was partly offset by the richer basket's larger contributions from:
  - furnishings (−1.2 pp)
  - restaurants and hotels (−1.0 pp)
  - transport (−0.5 pp)
  - health and recreation (−0.5 pp each)
- **A large share is not a large contribution.** Communication accounts for 3.3–3.6% of every decile's budget, but its prices rose only 3.9% in 2020–2023, so it contributes almost nothing. Restaurants (37.1% price rise) and furnishings (30.9%) contribute more to richer households.
- **Other dimensions**, cumulative 2020–2023:
  - Households headed by someone with primary education or less: 31.0%, vs higher education 29.9%.
  - Households with 4+ children: 30.9%, vs 1 child 30.1%.
  - Single-person households had the *highest* 2025 annual-average exposure (6.40% vs 6.09% for 6+ persons), driven by their housing/utility share (11.2%).
- **Uncertainty.** Published shares are rounded to 0.1 pp. Perturbing them within ±0.05 pp moves the latest D1 − D10 gap only within [0.086, 0.098] pp (official_group_gap_rounding.csv). Sampling standard errors for the official tables are not published, so no significance test of official group gaps is possible. **Verdict: descriptive. The 2.4 pp surge-period gap and the 0.09 pp current gap are point estimates, and their statistical uncertainty is not testable with published aggregates.**
- **Not the official CPI.** The national HBS-weighted index gives 5.93% for the 12 months to March 2026, whereas official headline CPI inflation was 5.55%. The official CPI uses its own weight structure. The group indices here are internally comparable with each other, but they are not the official CPI.

### Convenience survey (32 Baku households, October 2026)

- **Spending shares differ sharply from the national structure.**
  - Mean food-at-home share: 21.6% (full sample) and 23.9% (QC), against 42.3% for food in the national HBS.
  - Education averages 13.0% and 14.0%, against 1.7% nationally.
  - The sample is urban, Baku-only, and skewed to higher incomes: 11 of 32 report ≥ 5 000 AZN a month. It cannot represent Azerbaijani households.
- **Exposure.** Survey households' 12-month exposure to March 2026 averages 5.42% (QC; SD 0.40) and ranges from 4.55% to 6.08%. Over 2020–2023 it averages 26.2% (QC), below the national HBS basket's 30.2%, because of their lower food shares.
- **Group differences in the QC sample** (households with income below 5 000 AZN vs at or above it; 16 vs 10):

  | Outcome | Difference (below minus above 5 000 AZN) | 95% CI | Permutation p | Holm p | Hedges' g |
  |---|---|---|---|---|---|
  | 12-month exposure | +0.24 pp | [−0.08, 0.56] | 0.15 | 1.0 | 0.59 |
  | Cumulative 2020–2023 exposure | +2.17 pp | [−0.44, 4.62] | 0.12 | 0.81 | 0.63 |
  | Food share | +9.6 pp | [−1.7, 20.3] | 0.13 | 1.0 | 0.62 |
  | Food burden (% of income) | +12.7 pp | [3.8, 21.9] | 0.022 | 0.18 | 0.94 |
  | Essentials share | +16.3 pp | [4.7, 28.4] | 0.025 | 0.20 | 0.95 |

  - **Reading the table.** The directions match the official gradient. Some unadjusted bootstrap intervals exclude zero. None survives the Holm correction.
  - **Power.** The minimum detectable difference in 12-month exposure at 80% power is 0.47 pp, about five times the official D1 − D10 gap for that window. **Verdict: not distinguishable or inconclusive, because power is insufficient.**
- **No other characteristic separates households after Holm correction in the QC sample**: employment status, household size, children, tenure, or estimate confidence (HOUSEHOLD_STATISTICS.csv). Urban/rural is **not testable**, because all respondents live in Baku.

## RQ5 – How sensitive are household estimates to survey errors and cleaning rules?

SURVEY_SENSITIVITY.csv, survey_influence.csv, survey_mapping_sensitivity.csv; Figure 10.

- **Inflation exposure is robust.** Mean 12-month exposure is 5.42% in all three samples. Mapping hygiene goods entirely to CP05 or entirely to CP12, or dropping gifts, moves it between 5.37% and 5.46%. Exposure depends only on *shares*, which are unaffected by the scale errors that dominate the data problems.
- **Income burdens are fragile.** Mean total spending relative to income is 393% in the full sample, 83% in QC and 68% in the consistency subgroup. Dropping the single contradictory record R02 alone moves the full-sample mean from 393% to 110%. Medians are far more stable (83% / 74% / 62%).
- **Group conclusions change with the sample.** In the full sample, low-income households (< 2 000 AZN) have significantly higher food burdens than high-income ones (+61 pp, Holm p = 0.007), and higher total burdens. These results are partly mechanical, since income is the denominator, and partly driven by flagged records. In the QC sample the low-income band has 3 households and cannot be tested, and none of the coarser income comparisons survives Holm correction. In the consistency subgroup one comparison is significant (non-owners +0.39 pp exposure, Holm p = 0.045), but it is not significant in the QC sample (Holm p = 0.24). It should be treated as fragile.
- **Spending above income is common and not treated as an error.** 7 households report itemised spending above the top of their income bracket, and 12 above its midpoint.

## RQ7 – Can category forecasts and household weights project future differences in exposure?

**Feasible: yes. Reliable: only at short horizons. Model choice changes the distributional conclusion.**

- **Category forecasts.** All 12 COICOP divisions plus all-items were forecast separately. Across 13 series × 4 horizons, the best primary model was SARIMA in 33 cells, ETS in 18 and Yu FTS in 1 (category_forecast_accuracy.csv).
- **Group-level backtest** (2019–2026; group_forecast_accuracy.csv and group_forecast_significance.csv):
  - At h = 1, SARIMA and ETS combined forecasts are significantly more accurate than seasonal naive for every key group: national, urban, rural, Q1, Q5, D1, D10 (Holm p ≤ 0.03).
  - At h ≥ 3, no model is significantly better than seasonal naive.
  - At h = 12, Yu FTS is significantly *worse* than seasonal naive for Q1 and D1 (Holm p = 0.034 and 0.032).
- **Gap forecasts.** For the D1 − D10 y/y gap, SARIMA has the lowest error at every horizon: RMSE 0.11 pp at h = 1 and 0.66 pp at h = 12, against 0.16 and 1.04 for seasonal naive and 0.18 and 2.05 for Yu (group_gap_forecast_accuracy.csv).
- **Projections to March 2027** (origin March 2026; HOUSEHOLD_INFLATION_PROJECTIONS.csv; Figure 12):

  | Model | National HBS basket | D1 − D10 gap |
  |---|---|---|
  | Seasonal naive | 5.95% | +0.09 pp |
  | SARIMA | 6.63% | +0.48 pp |
  | ETS | 4.20% | −0.74 pp |
  | Yu FTS | 10.26% | +1.86 pp |

  - **Uncertainty.** Empirical 80% intervals for the national basket at h = 12 are very wide (SARIMA 1.2% to 13.6%).
  - **Model agreement.** Across the 60 official groups, the rank correlation of projected exposure is 0.92 between SARIMA and Yu FTS but −0.74 between SARIMA and ETS (household_projection_model_agreement.csv). Which groups face the most projected pressure therefore depends on the model.
  - **Which projection to trust.** The backtest favours SARIMA for both level and gap. The projected 12-month gap remains a model-dependent scenario, not a finding.
- **Genuine out-of-sample check** (headline_out_of_sample_check.csv). All-items forecasts made in March 2026 were compared with AZSTAT's April–September 2026 outcomes:
  - SARIMA's y/y errors stayed within 0.35 pp, and seasonal naive's within 0.34 pp.
  - ETS under-forecast by up to 2.70 pp.
  - Yu FTS over-forecast by up to 2.74 pp.

---

## Kazakhstan comparison – household baskets and exposure (BNS bulletins 2010–2026)

Source: BNS household budget bulletins (team Google Drive download of 25 Oct 2026; 2010–2019 unpacked from RAR archives) and the BNS monthly CPI by 13 sections (stat.gov.kz, 2008-01..2026-09). Tables: `KAZ_HBS_STRUCTURE.csv`, `KAZ_HBS_2025_SHARES.csv`, `KAZ_GROUP_INFLATION.csv`, `kaz_gap_regression.csv`; Figures 15–18.

- **Structure.** The food and non-alcoholic beverage share rose from 38.6% (2010) to 54.2% (2020) and was 52.4% in 2025 (urban 51.2%, rural 55.1%). Azerbaijan's 2025 national share is 42.3%.
- **Exposure, all households (2025 basket).** 43.36% cumulative Dec 2020–Dec 2023 (counterfactual), 11.05% in the 12 months to March 2026, 9.03% in the 12 months to September 2026.
- **Regions.** Over 2020–2023 exposure rose with the food share (Spearman 0.90; Akmola 42.61% to Turkistan 44.45%). In the 12 months to September 2026, with food inflation (8.2%) below all items (9.4%), the ordering reversed (Spearman −0.76; Almaty region 8.81% to Karaganda 9.30%).
- **Urban and rural.** The rural-minus-urban gap stays within ±0.34 pp (2009–2026); slope on relative food inflation 0.06 (Newey–West 95% CI 0.05 to 0.07, R² 0.77), about a fifth of Azerbaijan's decile slope (0.29).
- **Checks.** Chained BNS all-items y/y matches the IMF series (mean |diff| 0.05 pp, max 0.41 pp). The 2025 rural non-food detail table does not add up to the summary table (up to 2.7 pp) and was rescaled; one blank CPI cell (CP02, 2014-02) was recovered from the year-to-date block.

## Summary of verdicts

| Hypothesis | Verdict |
|---|---|
| Yu weighted FTS outperforms traditional methods for Azerbaijan | **Not supported** (SARIMA significantly better at h = 3, 6) |
| Yu FTS beats the seasonal naive benchmark | Not distinguishable (1 of 20 cells significant internationally) |
| SARIMA/ETS beat seasonal naive | Supported in a minority of country-horizon cells; point estimates favour them in most |
| Chronological weighting improves fuzzy forecasts | **Not supported**; recurrence explains the gain over Chen |
| The same model ranking holds across countries | Supported for "SARIMA/ETS > Yu FTS"; the SARIMA vs ETS ordering varies by country |
| Accuracy advantages disappear in instability | Not supported (SARIMA's skill was largest in 2021–2023, except in Algeria); exploratory |
| Poorer households face higher inflation | Supported descriptively in official data during food-led inflation (+2.4 pp in 2021–23); near zero in the latest 12 months; not testable for statistical significance with published aggregates; not distinguishable in the survey |
| Urban and rural exposure differ | Official: negligible (≤ 0.4 pp); survey: not testable |
| Category forecasts give reliable distributional projections | Inconclusive beyond h = 1; conclusions depend on the model |
