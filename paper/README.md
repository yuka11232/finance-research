# Paper

*Unequal Inflation Exposure in Azerbaijan: Household Baskets and Forecast Reliability*

| File | Content |
|---|---|
| `main.pdf` | Compiled manuscript |
| `main.tex` | LaTeX source. Tables and references are embedded, so no `.bib` file is needed |
| `figures/` | The 15 figures used in the paper (identical to `research/outputs/figures/`) |
| `verification/MANUSCRIPT_CHECKS.csv` | Independent checks of the manuscript against the saved outputs |
| `verification/TABLE_VALUE_PROVENANCE.csv` | Source file, selector, underlying value and printed value for every numeric table cell |

## Building

- **Overleaf:** use New Project > Upload Project and upload this folder (or a ZIP of it). Set `main.tex` as the main document and XeLaTeX as the compiler.
- **Locally:** run `xelatex main.tex` twice, or use `tectonic main.tex`.

The figures are produced by `research/scripts/make_figures.py`. After rerunning the pipeline, copy the regenerated PNGs into `figures/`.
