# Research methodology

All parameters are in [`config/experiment.toml`](../../config/experiment.toml). The random seed is 20261024, and every bootstrap uses 2,000 replications.

## 1. Targets and transformations

For a CPI index Pₜ:
- **month-on-month inflation:** mₜ = 100 (Pₜ / Pₜ₋₁ − 1)
- **year-on-year inflation:** yₜ = 100 (Pₜ / Pₜ₋₁₂ − 1)
- **annual-average inflation:** a_Y = 100 (P̄_Y / P̄_{Y−1} − 1)

These are different quantities and are never substituted for one another.

| Experiment | Modelled variable | Evaluated variables |
|---|---|---|
| Primary | mₜ (monthly, has annual seasonality) | (i) m_{T+h}; (ii) the implied y/y |
| Robustness | yₜ modelled directly | y_{T+h} |

The implied y/y in the primary experiment is

ŷ_{T+h} = 100 [ ∏_{j=0}^{11} (1 + x_{T+h−j}/100) − 1 ],

where x = observed m for dates ≤ T and forecast m̂ for dates > T. This conversion is level-consistent: it is the y/y change of the index path implied by the forecasts, not the monthly rate relabelled.

## 2. Forecasting models

**Seasonal naive (principal benchmark):** m̂_{T+h} = m_{T+h−12} for h ≤ 12. For the m/m target this implies that the y/y forecast at h = 12 equals the current y/y. For the directly modelled y/y target, seasonal naive (the y/y of 12 months earlier) is a weak benchmark; the random walk is reported alongside it.

**Random walk (diagnostic):** m̂_{T+h} = m_T.

**SARIMA(p,0,q)(P,0,Q)₁₂ with constant** (statsmodels `SARIMAX`):
- **Candidates:** p ∈ {0,1,2}, q ∈ {0,1}, P,Q ∈ {0,1}; d ∈ {0,1} added for the y/y target (23 or 47 candidates).
- **Stationarity:** m/m inflation is modelled in levels (d = 0). Diagnostics on the pre-test sample (to 2018-12; `diagnostics_stationarity.csv`):
  - ADF rejects a unit root and KPSS does not reject stationarity for Algeria, Kazakhstan and Norway m/m.
  - Azerbaijan m/m is borderline: ADF p = 0.057, KPSS p > 0.10.
  - Russia m/m is mixed: ADF rejects, but KPSS also rejects, at p ≤ 0.01.
  - Monthly seasonal dummies are jointly significant in every country (F-test p ≤ 0.001), which justifies a seasonal benchmark.
- **Residual checks:** Ljung–Box tests on the selected specifications (`diagnostics_residuals.csv`) do not reject white noise for SARIMA m/m in Azerbaijan, Kazakhstan and Norway. They reject at lag 12 for Russia (p = 0.004) and at lag 24 for Algeria (p = 0.002). Residual non-normality (Jarque–Bera) is common, which is one reason the empirical intervals are reported alongside the model-based ones.
- **Parameter stability:** annual re-estimates are in `diagnostics_sarima_parameter_stability.csv`.
- **Prediction intervals:** from the state-space forecast variance.

**ETS** (statsmodels `ETSModel`, additive errors, because inflation can be negative):
- **Trend:** none, additive, or additive damped.
- **Seasonality:** none or additive (period 12).
- **Candidates and intervals:** 6 candidates; analytic prediction intervals.

**Yu (2005) weighted FTS** and its two fuzzy benchmarks: see YU_FTS_IMPLEMENTATION.md.

**AR(p) by AIC** (p ≤ 13, OLS): a supplementary benchmark, not one of the four primary models.

### Selection protocol (no use of test data)

- **Sample split.** Validation targets run 2016-01 to 2018-12. Test forecasts are those with origin ≥ 2018-12, so test targets start in 2019-01.
- **Selection rule.** Each candidate is run through the same rolling procedure over the validation period. The specification with the lowest mean RMSE across h ∈ {1, 3, 6, 12} is chosen and then frozen.
- **Rolling procedure.** The window expands from the first observation. SARIMA and ETS parameters are re-estimated whenever the origin is a December, and the state is updated monthly with fixed parameters in between. FTS and AR are rebuilt at every origin.
- **Forecast horizons.** Forecasts for h = 1..12 are produced at every origin; h = 1, 3, 6 and 12 are evaluated.

## 3. International design

The five headline series all come from the IMF CPI dataset (definitions in DATA_AUDIT §1).

| Analysis | Sample | Test targets |
|---|---|---|
| **A – maximum coverage** | from 2000-01 or the first available month (AZE, DZA, KAZ 2005-01; NOR, RUS 2000-01) to each series' end (AZE and KAZ 2026-09, DZA and RUS 2026-08, NOR 2026-09) | 2019-01 to own end (92–93 one-step forecasts) |
| **B – common period** | 2005-01 to 2026-08 for every country | identical calendar test dates |

A and B differ materially only for Norway and Russia, which gain five extra training years in A; for the other countries they differ by at most one month.

## 4. Accuracy, skill, significance and uncertainty

Errors are e = actual − forecast.

**Point accuracy:**
- MAE, RMSE, and Bias = mean(forecast − actual).
- MASE: MAE divided by the in-sample one-step seasonal-naive MAE over the training data before 2019.
- sMAPE is reported but is unstable near zero.

**Skill:** S = 1 − RMSE_model / RMSE_seasonal-naive. Uncertainty comes from a circular moving-block bootstrap of paired error vectors: block length max(h, ⌈n^{1/3}⌉), 2,000 replications, percentile 95% CI, and the bootstrap probability that S > 0.

**Diebold–Mariano (1995) test with the Harvey–Leybourne–Newbold (1997) correction:**
- Loss differential d_t = L(e₁) − L(e₂), with squared-error loss (primary) and absolute-error loss (robustness).
- Long-run variance uses the rectangular kernel truncated at h − 1, which matches the MA(h−1) structure of h-step errors. It falls back to Bartlett weights if not positive; this happened in 27 of 2,360 tests.
- The statistic is multiplied by √[(n+1−2h+h(h−1)/n)/n] and compared with t_{n−1}. The 95% CI for E[d] uses the same variance.
- Tests are skipped when n < 30 or the two forecasts are identical: 40 rows, all naive vs seasonal naive at h = 12, which coincide by construction.

**Multiple testing:** the confirmatory family per (variable, analysis, country, horizon, loss) is these seven comparisons, Holm-adjusted together:
- SARIMA vs SN
- ETS vs SN
- Yu vs SN
- Yu vs SARIMA
- Yu vs ETS
- Yu vs equal-weight FTS
- Yu vs Chen

Exploratory comparisons (random walk vs SN, AR vs SN, equal-weight vs Chen) report raw p-values only.

**Prediction-interval coverage:**
- **Model-based:** SARIMA and ETS intervals.
- **Empirical, for every model:** at origin T, the forecast plus quantiles of the model's own past out-of-sample h-step errors whose targets are ≤ T, with at least 24 errors required.
- Nominal and empirical coverage are reported separately.

**Instability:** test RMSE and skill are computed by sub-period (2019–2020, 2021–2023, 2024–2026). The cross-country relationship between test-period volatility and skill is descriptive only (n = 5).

## 5. Survey processing

**Shares and burdens:**
- **Expenditure share**ₖ = spendingₖ / Σⱼ spendingⱼ × 100, over the 14 recorded categories.
- **Income burden**ₖ = spendingₖ / income-bracket midpoint × 100.
- **Open brackets:** "<500" uses 250; "≥10 000" uses its lower bound 10 000, so burdens for that bracket are upper bounds.
- **Bracket-bound sensitivity:** burdens at the bracket bounds are also reported.

**Concentration:** Herfindahl index Σ sₖ².

**Samples:**
- **Full:** all 32 responses.
- **Quality-controlled (QC):** excludes the rules listed in `src/survey/clean.py::EXCLUSION_RULES` (duplicates, straight-lining, itemised total inconsistent with own total bracket by more than a factor of 2, spending bracket ≥ 6 brackets above income bracket, negative values), leaving n = 26.
- **Income–expenditure consistency subgroup:** itemised spending ≤ upper bound of the income bracket, n = 25.
- **Variable-specific rules:** household size and children are set missing for contradictory records in demographic comparisons, and occupation for the "all sectors" record.

**Groupings:**
- Income band: < 2 000 / 2 000–4 999 / ≥ 5 000 AZN.
- Binary income split at 5 000 AZN.
- Terciles of OECD-modified equivalised income: scale 1 + 0.5(adults − 1) + 0.3 × children.
- Respondent full-time employed vs not.
- Household size 2–3 vs 4+.
- Children 0 vs 1+.
- Owner without mortgage vs mortgage/rent/free.
- Rough estimate vs confident.

## 6. Household and group inflation exposure

**Laspeyres index** with the weight period equal to the price reference period (2025):

I_g(t) = Σₖ s_{gk} · Pₖ(t) / P̄ₖ(2025)

**Window inflation:** π_g = 100 (I_g(t₁)/I_g(t₀) − 1).

**Exact additive contributions:** c_{gk} = 100 s_{gk} (Pₖ(t₁) − Pₖ(t₀)) / P̄ₖ(2025) / I_g(t₀), with Σₖ c_{gk} = π_g.

**Prices:** IMF COICOP division indices for Azerbaijan (CP01–CP12, plus all-items).

**Weights:**
- **Official groups:** AZSTAT 2025 HBS shares for national, urban/rural, 14 economic regions, number of children, children present, household size, head status/education/age/sex, expenditure quintiles and deciles.
- **Mapping and normalisation:** CP01 = food − alcohol; CP02 = alcohol + tobacco. Shares are renormalised to remove the 0.1-pp rounding residue.
- **Survey households:** shares mapped from the 14 survey categories:
  - Direct one-to-one mappings: food→CP01; rent & repairs→CP04; utilities→CP04; transport→CP07; health→CP06; education→CP10; clothing→CP03; phone/internet→CP08; restaurants→CP11; recreation→CP09; tobacco & alcohol→CP02.
  - Hygiene & household goods → 50% CP05 and 50% CP12.
  - Family events/gifts and other consumption → all-items CPI.
  - Sensitivity variants: hygiene all CP05; hygiene all CP12; gifts excluded.

**Windows:**

| Window | Definition |
|---|---|
| W1 | Mar 2025 → Mar 2026 (latest 12 months with COICOP data) |
| W2 | annual average 2025 vs 2024 |
| W3 | cumulative Dec 2020 → Dec 2023 (the inflation surge) |

W3, and any period before 2025, is a **fixed-2025-basket counterfactual**: historical baskets by group are not available.

**Rounding sensitivity:** each of the 13 published items is perturbed by U(−0.05, +0.05) pp and the group gaps recomputed (2,000 draws).

## 7. Household group statistics

For each sample, outcome and grouping:
- **Descriptives:** n, mean, median and SD, with percentile-bootstrap 95% CIs for the mean and median.
- **Group differences:** the difference in means with a bootstrap 95% CI.
- **Tests:** Welch t, Mann–Whitney U, and a two-sided permutation test on the difference in means (20,000 permutations; primary test).
- **Effect sizes:** Hedges' g and Cliff's δ.
- **Power:** minimum detectable difference at 80% power and α = 0.05 (normal approximation).
- **Multiple testing:** Holm adjustment over all comparisons within (sample, outcome).
- **Small groups:** groups with fewer than 5 households are not tested.
- **Weights:** no survey weights exist, so the results describe the surveyed households only.

## 8. Integrating category forecasts with household baskets

1. Each of the 13 Azerbaijan series (CP01–CP12 and all-items) is forecast **separately** with all models, using the same selection and backtest protocol as §2.
2. **Index paths:** at each origin T, m̂ₖ is chained into index paths P̂ₖ(T+h) = Pₖ(T) ∏_{j≤h}(1 + m̂ₖ,T+j/100).
3. **Group y/y forecast:** Î_g(T+h) / I_g(T+h−12) − 1, where Î_g is built from the forecast division paths with each unit's 2025 basket. For h ≤ 12 the denominator is always observed.
4. **Evaluation units:** 60 official groups and the 26 QC survey households. Accuracy is computed per unit; DM/HLN tests are run against seasonal naive for national, urban, rural, Q1, Q5, D1 and D10, Holm-adjusted over the three models. Forecasts of the D1−D10 and Q1−Q5 gaps are scored separately.
5. **Projections:** the final origin, 2026-03, gives projections to 2027-03. Their 80%/95% intervals are empirical, from each unit's group-level backtest errors at the same horizon.
6. **Model agreement:** across units, measured by the Spearman rank correlation and the overlap of the top 20% of projected exposures between models.
7. **Genuine out-of-sample check:** all-items forecasts from 2026-03 are compared with AZSTAT outcomes for 2026-04 to 2026-09, which were not in the COICOP estimation data.
