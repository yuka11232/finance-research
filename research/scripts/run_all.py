"""Run the full pipeline in order. Usage: python scripts/run_all.py [--skip-download] [--from STEP]

Steps
  ingest        download IMF/World Bank data if absent, build processed series
  audit         computed data audit (data/metadata)
  forecasts     international rolling backtests (parallel; ~4 min on 10 workers)
  evaluate      accuracy, DM/HLN + Holm, bootstrap skill, coverage, sub-periods
  household     survey cleaning, shares/burdens, household & official-group exposure, group tests
  kazakhstan    BNS household budget bulletins 2010-2026 + BNS division CPI: structure, regional/urban/rural exposure
  categories    Azerbaijan COICOP-division backtests (parallel)
  integration   category forecasts x baskets, group accuracy, projections
  gaptests      direct D1-D10 gap benchmarks (zero, persistence, AR(1)) and the 24 Holm-adjusted DM/HLN gap tests
  diagnostics   ADF/KPSS, seasonality, Ljung-Box, parameter stability
  paper         pooled five-country DM test, global Holm, food-gap regression, sign counts
  figures       publication figures
  verify        independent numerical checks of outputs and quoted numbers
Backtest steps cache per-series results in outputs/forecasts/*parts/; delete them to force re-estimation.
The survey microdata are not distributed. Without data/raw/survey/ the household step recomputes the
official-group outputs only and keeps the committed aggregate survey tables; no later step reads survey data.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEPS = ["ingest", "audit", "forecasts", "evaluate", "household", "kazakhstan", "categories", "integration", "gaptests", "diagnostics", "paper", "figures", "verify"]
SCRIPTS = {
    "audit": "audit_data.py", "forecasts": "run_international_forecasts.py", "evaluate": "evaluate_forecasts.py",
    "household": "run_household_analysis.py", "kazakhstan": "run_kazakhstan_household.py", "categories": "run_category_forecasts.py", "integration": "run_integration.py", "gaptests": "run_gap_benchmarks.py",
    "diagnostics": "run_diagnostics.py", "paper": "paper_statistics.py", "figures": "make_figures.py", "verify": "verify_outputs.py",
}


def ingest(skip_download: bool):
    from src.ingestion import international
    from src.ingestion.series import build_all
    from src.paths import ensure_dirs

    ensure_dirs()
    if not skip_download:
        international.download_imf()
        international.download_worldbank()
    build_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--from", dest="start", default="ingest", choices=STEPS)
    args = ap.parse_args()
    for step in STEPS[STEPS.index(args.start):]:
        t0 = time.time()
        print(f"== {step}", flush=True)
        if step == "ingest":
            ingest(args.skip_download)
        else:
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / SCRIPTS[step])], cwd=ROOT)
            if r.returncode != 0:
                sys.exit(f"step '{step}' failed with exit code {r.returncode}")
        print(f"   done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
