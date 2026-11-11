# Limitations

## Data

1. **No Kazakhstan or Russia data were in the project.** Their CPI series, together with monthly CPI for Algeria and Norway and COICOP-division CPI for Azerbaijan, were downloaded from the IMF CPI dataset during this run (vintage retrieved 2026-10-24). The project team has not reviewed these downloads against national statistical-office releases. The Azerbaijan series was cross-checked against AZSTAT; the others were checked only against World Bank annual averages (identical).
2. **Unequal coverage.** Monthly CPI starts in 2005 for Azerbaijan, Algeria and Kazakhstan, but in 1995 for Norway and Russia; 1995–1999 was not used. Azerbaijan's IMF series ends in 2026-03 and is extended to 2026-09 with AZSTAT's one-decimal monthly relatives, which adds up to about 0.03 pp of rounding error in y/y rates.
3. **Current vintage only.** Every backtest uses the current data vintage, not real-time vintages. Revisions that forecasters faced in real time are therefore not reflected.
4. **COICOP division indices for Azerbaijan end in 2026-03.** Household exposure and category forecasts cannot use April–September 2026 prices, except for the headline out-of-sample check.
5. **No historical group baskets.** Official group expenditure structures are available for 2025 (and 2024) only. Every exposure estimate before 2025, including the 2020–2023 surge window, is a **counterfactual with a fixed 2025 basket**. It does not measure what households actually bought in those years, and substitution is ignored (a Laspeyres upper-bound property).
6. **Division-level aggregation.** Group indices use 12 COICOP divisions. Within-division differences (for example, which foods poorer households buy) are invisible, so the gaps are probably understated.
7. **HBS weights are not CPI weights.** The national HBS-weighted index (5.93% for the 12 months to March 2026) differs from the official CPI (5.55%), because AZSTAT's CPI uses its own weight structure, which is not published in the files available. The group indices are internally comparable with each other, but they are not official CPI measures.
8. **Expenditure groups, not income groups.** Official quintiles and deciles rank households by consumption expenditure, not income. No official income-group consumption structures were available in machine-readable form.
9. **No sampling errors.** Published HBS tables have no standard errors. Ersado (2006) reports that wealthy households are under-represented in the Azerbaijani HBS, which would compress measured differences.
9a. **Kazakhstan comparison.** Regional baskets are valued with national division prices (regional CPIs are not used), so regional differences are basket effects only. BNS division indices are chained from one-decimal monthly changes. From 2026 BNS uses COICOP 2018, so 2026 division prices map to the 1999 divisions approximately. The bulletins have no income-ranked groups, and the 2025 rural non-food detail was rescaled to the summary table.

## Survey

10. **Convenience sample.** n = 32, all in Baku, recruited through the team's networks, with 11 of 32 reporting income ≥ 5 000 AZN. Results describe these households only; there are no weights, and the sample cannot represent the population.
11. **Coarse income measure.** Income and total spending were asked as brackets. Burdens use bracket midpoints; the open "≥ 10 000" bracket uses its lower bound, so those burdens are upper bounds. Age, education and sex of the household head were not collected, and the employment question describes the respondent, not the household head.
12. **Measurement error.** 15 of 32 respondents describe their spending as a "very rough estimate". Six records fail documented quality rules, and full-sample burdens are dominated by one contradictory record. Monthly recall of irregular spending (education fees, clothing, gifts) probably mixes annual and monthly amounts.
13. **Category mismatch.** The survey's categories do not match COICOP one-to-one: hygiene/household goods are split 50/50, and gifts and "other" are mapped to all-items CPI. Alternative mappings shift mean exposure by at most 0.05 pp.
14. **Low power.** Group comparisons are severely underpowered. The minimum detectable difference in 12-month exposure (≈ 0.47 pp) is about five times the official decile gap for the same window. A null result is not evidence of no difference.
15. **Timing.** The survey basket (October 2026) is used as if it were the 2025 weight-period basket.

## Models and statistics

16. **Yu (2005) extensions.** Yu's paper specifies one-step forecasting. The recursive multi-step extension, the Map Table generalisation, the universe rule for inflation rates and Chen's Rule 1 for empty groups are documented choices (YU_FTS_IMPLEMENTATION.md), not part of the original method. The recursion repeats unprecedented shocks (Russia 2022).
17. **Limited search spaces.** Hyper-parameter grids are deliberately small (23–47 SARIMA candidates, 6 ETS, 2 FTS interval methods), and selection used one validation window (2016–2018). A different window could change the selected specifications; this is not tested.
18. **Diagnostic failures.** Residual autocorrelation remains for some selected models (Russia SARIMA m/m; every directly modelled y/y ETS). Unit-root evidence for Russian and Azerbaijani inflation is mixed (diagnostics_*.csv).
19. **Power of forecast tests.** The test period has 81–93 forecasts per horizon. Overlapping multi-step errors and the 2022 shocks make DM tests low-powered, especially for Russia. Holm correction is applied within a preselected family of 7 comparisons per cell, not across countries and horizons. Correcting across all 140 tests would make results more conservative.
20. **Interval calibration.** SARIMA/ETS 80% intervals over-cover. Empirical intervals under-cover at 12 months. Group-projection intervals are empirical and very wide.
21. **Exploratory cross-country associations.** The volatility–skill association uses five countries and is exploratory only.

## Integration and generalisability

22. **Model-dependent projections.** The sign of the projected poorest-minus-richest gap for March 2027 differs between ETS and the other models. Beyond h = 1, no model is significantly more accurate than seasonal naive at the group level. The projections are scenarios, not forecasts with demonstrated reliability.
23. **No mechanisms tested.** No causal mechanism (oil prices, exchange rates, fiscal policy) is tested. Cross-country differences in accuracy are not attributed to any mechanism.
24. **Not done:**
    - Yu's original TAIEX RMSE table was not replicated, because the data are not available.
    - A live Google Sheets connection was not used; the Excel export was.
    - No Linear Expenditure System or substitution model was estimated (as in Can et al. 2025).
