# Household survey (not distributed)

The pilot survey of 32 Baku households (Google Forms, October 2026) is not included in this repository, to protect respondents.

- **Published data:** only aggregate survey tables are published: means, medians and bootstrap intervals by sample definition (`outputs/tables/SURVEY_SENSITIVITY.csv`) and by income band (`SURVEY_SUMMARY_BY_INCOME.csv`), plus exclusion counts, influence summaries and group comparisons. No table contains respondent ids or one row per respondent, and summaries of fewer than five households are suppressed.
- **Without the export:** the `household` step recomputes the official-group outputs and keeps the committed aggregate survey tables. No other step reads survey data.
- **To rerun the household step:** place the export at `data/raw/survey/household_survey_responses_2026-10.xlsx` (sheet "Form Responses 1").
