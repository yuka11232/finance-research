# Paper

*Unequal Inflation Exposure in Azerbaijan: Household Baskets and Forecast Reliability*

| File | Content |
|---|---|
| `main.pdf` | Compiled manuscript |
| `main.tex` | LaTeX source. Tables and references are embedded, so no `.bib` file is needed |
| `figures/` | The 12 figures used in the paper |

## Building

- **Overleaf:** use New Project > Upload Project and upload this folder (or a ZIP of it). Set `main.tex` as the main document and XeLaTeX as the compiler.
- **Locally:** run `xelatex main.tex` twice, or use `tectonic main.tex`.

All figures are produced by `research/scripts/make_figures.py`. The nine PNG figures are byte-identical to the pipeline output. The pipeline also regenerates the three vector figures (`contributions.pdf`, `gap.pdf`, `projections.pdf`) from the same data; their styling differs slightly from the versions in `figures/`. The numbers in the paper's tables are checked by `research/scripts/verify_outputs.py`: the gap results of Tables 4 and 5 come from `research/scripts/run_gap_benchmarks.py`.
