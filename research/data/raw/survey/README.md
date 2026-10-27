# Household survey (not distributed)

The pilot survey of 32 Baku households (Google Forms, October 2026) is not included in this repository, to protect respondents.

- **Anonymised data:** respondent-level data are published in `outputs/tables/HOUSEHOLD_EXPENDITURE.csv`. Respondents have pseudonymous ids, and the file has no timestamps or occupation detail.
- **Without the export:** `scripts/run_all.py` skips the `household` step and later steps use the committed tables.
- **To rerun the household step:** place the export at `data/raw/survey/household_survey_responses_2026-10.xlsx` (sheet "Form Responses 1").
