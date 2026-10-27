# Azerbaijan: analysis and forecasting package

Updated 21 October 2026. Open this file first.

## What you can do now

Use 05_Model_Ready_Data/monthly_broad_CPI_2016-01_to_2026-09.csv for a modest fuzzy time-series forecasting experiment. It has 129 consecutive months for each of four series: national, food/beverages/tobacco, non-food goods, services. There are no missing months in these four extracted series. This is sufficient to run and evaluate simple forecasting models, not evidence that a particular model is accurate or publication-ready.

The bundle also contains household weights and historical income/spending data for descriptive group analysis. It does NOT yet contain a valid historical income-group inflation panel or group-specific forecast. Constructing one requires matching price categories to household categories and choosing historically appropriate weights.

## Files and what each represents

| File in 05_Model_Ready_Data | Contents | Use |
|---|---|---|
| monthly_broad_CPI_2016-01_to_2026-09.csv | 516 observations: 129 months x 4 broad series; month-on-month inflation, original previous-month=100 index, derived chained index and year-on-year changes | Primary forecasting input |
| monthly_detailed_CPI_original_labels_2016-01_to_2026-09.csv | 12,298 observed category-month records, original Azerbaijani labels and source coordinates | Detailed analysis after category harmonization; not a ready balanced panel |
| official_monthly_CPI_levels_2019-2025.csv | Official CPI levels with base 2010=100 | Cross-checks and alternative level-based models |
| annual_broad_inflation_1993-2025.csv | Annual-average inflation for four broad series | Long-run descriptive context; only 33 annual observations per series |
| household_national_income_spending_selected_years.csv | National income/spending components, AZN per person per month, in selected years | Household trends; do not interpolate silently |
| household_spending_weights_2024.csv | Spending shares for national, urban, rural and expenditure quintiles Q1-Q5 | Group basket comparison; not income quintiles |
| forecast_inputs_with_chronological_split.csv | Broad monthly inflation with train/validation/test labels | Reproducible time-ordered experiments |
| broad_category_label_checks.csv | Original broad category labels in each annual bulletin | Check definitions across years |

Every extracted record includes its source file; most include original URL and sheet/row. SOURCE_INDEX.csv inventories the raw downloaded files. Data are CSV for direct use in Python, R or VS Code, without merged cells.

## Field meanings

- date: first day of the represented month, ISO YYYY-MM-DD. Not the publication date.
- series: national / food_beverages_tobacco / nonfood_goods / services.
- previous_month_100: official price index relative to the previous month. 101.5 means prices rose 1.5% that month.
- mom_inflation_pct: previous_month_100 minus 100, in percentage points. Negative numbers mean falling prices.
- chained_index_dec2015_100: derived by multiplying published monthly price relatives, setting December 2015 to 100. This is NOT the official 2010=100 index. Rounding compounds over time.
- yoy_inflation_pct: derived compound change over the latest 12 monthly relatives. First 11 rows per series are blank because prior monthly observations are not included. It may differ slightly from the official published year-on-year rate due to rounding/revisions.
- category_label_az / category_label_en: original source label. Detailed categories can be nested or overlapping and may change names; do not add all rows together.
- value_AZN_per_person_per_month: nominal household income or expenditure. An observation reported for a year is a monthly average, not twelve separate monthly observations.
- weight_year: the household-survey observation year.
- share_pct: percentage of total household spending, NOT a fraction. Divide by 100 when weighting.
- is_nested_component: 1 for alcohol, reported as a component of the food row. Exclude nested components when summing the parent categories. Food plus alcohol would double-count alcohol.
- group: national / urban / rural or expenditure_Q1 through Q5. Quintiles are ranked by expenditure, not income; their membership can change across years.
- source_excel_row / source_excel_column: one-based coordinates in the original worksheet.
- split: train, validation or test according to the dates below.

## Running the code

From a terminal:

    python 06_Code/prepare_data.py
    python 06_Code/run_forecasts.py

Extraction requires Python with openpyxl and xlrd. The forecasting script uses only the Python standard library. The raw Excel data are left unchanged. The scripts locate the package folder relative to their own location.

## Forecast method and evaluation

Target: next month's month-on-month inflation, in percent. No household income variable is used as a predictor in this baseline.

Training period: January 2016-September 2024 (105 observations).
Validation period: October 2024-September 2025 (12 observations).
Test period: October 2025-September 2026 (12 observations).

The fuzzy baseline is a first-order triangular fuzzy regression with lag-1 monthly inflation as its input and constant rule consequents. For each fit, partition the training range into 3, 5 or 7 equally spaced triangular membership functions, padded by 5% of the range (minimum padding 0.05 percentage points). Estimate each rule consequent as the membership-weighted mean of the next month's observed inflation. Combine consequents using the current observation's membership weights. Clip out-of-range inputs to the fitted range; unused rules fall back to the training mean.

Select the fuzzy-set count by validation MAE. Compare fuzzy with last-value, seasonal-naive (same month last year) and expanding historical-mean forecasts. Select the candidate model by validation MAE, never by test MAE. At each backtest origin, fit using only earlier observations. Previous test observations become available for later test origins, which is valid expanding-window one-step evaluation. No random shuffling or fitting on future values. Validation metrics for tuned fuzzy models are selection results, not unbiased performance estimates.

The code is an explicitly specified baseline, not an exact reproduction of Chen (1996), and should not be described as such. Fuzzy time-series background: Chen, S.-M. (1996), Forecasting enrollments based on fuzzy time series, Fuzzy Sets and Systems 81(3), 311-319. https://doi.org/10.1016/0165-0114(95)00220-0

## Forecast outputs

07_Forecast_Results/backtest_metrics.csv: MAE and RMSE in percentage points, separately for validation and test.
07_Forecast_Results/backtest_predictions.csv: predictions and actual values at every validation/test origin.
07_Forecast_Results/forecasts_2026-10_to_2027-03.csv: six experimental monthly point forecasts per series and model. selected_by_validation identifies the prespecified choice; it does not assert that it won the later test.
07_Forecast_Results/model_specifications.json: settings and selection results.

The six-month fuzzy forecast is recursive with parameters fitted once through September 2026; predictions are NOT added to the training observations. These are preliminary point forecasts, not official statistics. There are no calibrated prediction intervals. Only one-step forecasts have been backtested, so the six-month paths have NOT been validated at matching horizons. First-order models can converge toward nearly constant forecasts.

## Actual result: do not claim fuzzy superiority

Seasonal-naive achieved lower test MAE than this fuzzy baseline in ALL FOUR broad series. For national inflation, test MAE was approximately 0.183 percentage points for seasonal-naive versus 0.340 for fuzzy. Fuzzy was selected on validation for national, food and services; last-value was selected for non-food. This reversal on the test period should be reported, not hidden. More complex fuzzy models need new validation, not repeated tuning to this test set.

## What still prevents reliable group-specific forecasts

1. The complete 2001-2025 ASIS household tables were located but the data endpoint returned HTTP 503 during this run. No successful full export is claimed.
2. Most directly downloadable group weights are for 2024. Use them for explicitly labelled fixed-2024-basket scenarios, not as historical observations of every year's household behavior.
3. HBS spending divisions and the detailed CPI categories are not yet mapped one-to-one. Broad non-food/services CPI cannot simply be assigned to every mixed spending division. Do not silently use equal within-category weights.
4. No household-level microdata or survey design weights are included. You cannot infer low-income urban households by combining independent urban and quintile averages.
5. The detailed CPI file preserves original labels and hierarchy. Category changes need review before training one long series per category.
6. Release dates/vintage histories are not included. Backtests use the currently downloaded historical values, not a verified real-time vintage dataset.
7. A 2000 start is not available for every series. Full-year 2026 is not yet available.

## Additional source recovered

03_Yearbooks/Household_Survey__Edition_2023__AZSTAT.zip contains Mecmue_2001_2022.pdf. It adds an older household publication. Its tables are preserved as original data; they have NOT yet been merged into the CSV household panel.

Raw source files are in folders 01-04. Read the original headings and methodology before drawing substantive conclusions.
