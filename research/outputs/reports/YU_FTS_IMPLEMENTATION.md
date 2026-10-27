# Yu (2005) weighted fuzzy time series: implementation and verification

**Source:** Yu, H.-K. (2005). Weighted fuzzy time series models for TAIEX forecasting. *Physica A* 349(3–4), 609–624. doi:10.1016/j.physa.2004.11.006. Every step below was checked against the full text of the paper.

**Code:** [`src/fuzzy/fts.py`](../../src/fuzzy/fts.py). **Tests:** [`tests/test_fuzzy.py`](../../tests/test_fuzzy.py).

## 1. Algorithm as published, and where it lives in the code

| Yu (2005) step | Content in the paper | Implementation |
|---|---|---|
| Step 1: universe and intervals | U = [starting, ending] split into equal intervals uₖ = [s+(k−1)l, s+kl]; midpoint mₖ | `Universe` |
| Interval length (Sect. 2.2, from Huarng 2001) | *Average-based:* half the mean absolute first difference, rounded down to the "base" from a Map Table. *Distribution-based:* the largest multiple of the base that is still smaller than at least half of the absolute first differences | `average_based_length`, `distribution_based_length` |
| Step 2: fuzzy sets | Aᵢ = … + 0/uᵢ₋₂ + 0.5/uᵢ₋₁ + 1/uᵢ + 0.5/uᵢ₊₁ + 0/uᵢ₊₂ … | implicit: maximal membership of a datum is in the set whose core interval contains it |
| Step 3: fuzzify | datum → Aⱼ with maximal membership | `Universe.fuzzify` |
| Step 4: FLRs | Aᵢ (t−1) → Aⱼ (t) | `FuzzyTimeSeries.fit` |
| Step 5 (revised): FLRGs with recurrence | all RHSs kept, repeats included, in chronological order (Eq. 4) | `groups[lhs]` is an ordered list |
| Step 6: forecast | F(t−1)=Aᵢ ⇒ forecast is the RHS list of Aᵢ's group | `_defuzzify(state)` |
| Step 7 (revised): defuzzified matrix | M(t) = [m_{j1}, …, m_{jk}] (Eq. 6) | `mids` |
| Step 8: weights | wᵢ = i (Eq. 10); W(t) = [w₁/Σw, …, w_k/Σw] (Eqs. 8, 12); monotone non-decreasing (Eq. 9) | `w = arange(1, k+1); w /= w.sum()` |
| Step 9: result | final(t) = M(t) × W(t)ᵀ (Eq. 13) | `mids @ w` |
| Step 10: evaluation | RMSE of one-step forecasts (Eq. 14) | backtest engine (plus MAE, MASE, sMAPE, bias) |

**Weighting scheme used:** the chronological linear scheme of Eq. (10). The oldest FLR in a group gets weight 1, and the most recent gets weight k. Weights are normalised to sum to one. Yu mentions expert-elicited weights as an alternative; none are available here, so they are not used.

## 2. Numerical verification against the paper

These tests are in `tests/test_fuzzy.py` and all pass:
- **Huarng lengths (Yu Sect. 2.2):** for the series 30, 50, 80, 120, 100, 70 the code returns an average-based length of **10** and a distribution-based length of **20**, as in the paper.
- **TAIEX worked example (Sect. 5, Steps 5–9):** the A₁₁ group is (A₉, A₁₂, A₁₀, A₉) with midpoints (3350, 3650, 3450, 3350) and weights (1, 2, 3, 4)/10. The code returns **3440**, matching the paper.
- **Eq. (1)–(4):** the group A₁ → (A₁, A₂, A₁, A₁) gives weights 1/10, 2/10, 3/10, 4/10.
- **Map Table:** the generalised base rule reproduces Huarng's table, including the values used by Yu (length 14 → base 10; 28 → 10).

## 3. Components not fully specified in Yu (2005), with documented choices

1. **Map Table.** Yu cites Huarng (2001) without reproducing the table. The implemented rule is base = 10^(⌈log₁₀ L⌉−1). It agrees with Huarng's rows (0.1–1 → 0.1; 1.1–10 → 1; 11–100 → 10; 101–1000 → 100) and extends to lengths below 0.1, which inflation rates need. Huarng (2001) itself was not available in the project, so the table rows are taken from the literature and their equivalence with this rule is asserted, not re-verified from that paper.
2. **Universe start.** Yu rounds the sample minimum down "to the nearest hundred" for TAIEX, where the length base is 10. The code rounds down to a multiple of 10 × base, which reproduces that rule. The universe end is extended until the maximum is covered.
3. **Empty FLR group.** The code applies Chen's (1996) Rule 1: if the current state has no recorded successor, the forecast is its own midpoint. Yu does not restate this rule.
4. **Values outside U at forecast time** are assigned to the nearest end interval.
5. **Multi-step forecasts.** Yu evaluates one-step forecasts only. Here, h-step forecasts are recursive: each defuzzified forecast is fuzzified and becomes the next state, and groups are not updated with forecasts. This is an extension, and its weakness is reported in §5.
6. **Rolling re-estimation.** At every forecast origin the universe, interval length, fuzzification and FLR groups are rebuilt from data up to that origin. Yu used one fixed estimation window per year.

## 4. Benchmarks built from the same code

All three models share the same partition, chosen on validation for Yu weighted, so the comparisons isolate the relationship rules:

- **Chen (1996):** repeated FLRs counted once, plain average of midpoints (Yu Sect. 2.3). This is Yu's own comparison.
- **Equal-weight recurrence:** repeated FLRs kept and weighted equally. Yu (Sect. 3.2) calls unweighted models the equal-weight special case. Comparing it with Yu weighted isolates the effect of the **chronological weights**; comparing it with Chen isolates **recurrence**.

## 5. Hyper-parameter selection and observed behaviour

- **Interval method** (average vs distribution) was selected per series on 2016–2018 validation targets. `outputs/tables/model_selection_validation.csv` lists every candidate's validation score.
- **Target variable.** FTS was applied to monthly m/m inflation (primary) and to y/y inflation (robustness). It was not applied to CPI *levels*: a trending level series leaves the fixed universe of discourse almost immediately, so every out-of-sample level would sit in the top interval.
- **Weakness at unprecedented shocks.** After Russia's March-2022 jump (+7.6% m/m) the state had no recorded successor. Under Rule 1, the recursive forecast therefore repeats the spike's midpoint, and implied 12-month y/y errors become as large as the random walk's (RMSE 16.2 pp vs 16.3 pp; INTERNATIONAL_FORECAST_COMPARISON.csv).
- **Projection behaviour.** In the Azerbaijan category projections from March 2026, Yu FTS projects the highest inflation of the four models (national HBS basket, 10.3% y/y by March 2027). In the April–September 2026 outcomes available so far it over-predicts y/y inflation by 0.3–2.7 pp (`headline_out_of_sample_check.csv`).

## 6. Status

- **Verified** against the printed paper: recurrence, chronological weights, defuzzification, both interval-length rules, and the TAIEX worked number.
- **Not reproducible from the paper alone:** Yu's TAIEX RMSE table. The daily TAIEX 1990–1999 data are not in the project, so that replication was not attempted.
