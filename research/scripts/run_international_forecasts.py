"""Phase 4: four-model (plus diagnostics) rolling backtests for five countries.

Experiments
  target = mom : month-on-month CPI inflation (primary modelled variable);
                 implied year-on-year inflation is derived from each m/m path.
  target = yoy : year-on-year inflation modelled directly (robustness).
Analyses
  A : maximum coverage (from 2000-01 or first available month, to own end)
  B : common period 2005-01 .. 2026-08 for all countries

Usage:  python scripts/run_international_forecasts.py [--workers 10]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.paths import FORECASTS, LOGS, PROCESSED, TABLES, config, ensure_dirs


def series_for(country: str, analysis: str, target: str):
    cfg = config()["data"]
    h = pd.read_csv(PROCESSED / "cpi_headline_monthly.csv", parse_dates=["date"])
    g = h[h["country"] == country].set_index("date").sort_index()
    if analysis == "A":
        g = g[g.index >= pd.Timestamp(cfg["analysis_a_start"])]
    else:
        g = g[(g.index >= pd.Timestamp(cfg["analysis_b_start"])) & (g.index <= pd.Timestamp(cfg["analysis_b_end"]))]
    if target == "mom":
        # first m/m of the window needs the previous index, which exists for every country
        return g["mom"].dropna(), g["yoy"]
    return g["yoy"].dropna(), None


PARTS = FORECASTS / "parts"


def task(country: str, analysis: str, target: str):
    """Run one series; results are checkpointed so finished tasks survive a later failure."""
    from src.forecasting.experiment import run_series

    t0 = time.time()
    y, yoy = series_for(country, analysis, target)
    label = {"target": target, "analysis": analysis, "country": country}
    stem = f"{target}_{analysis}_{country}"
    fpath, spath = PARTS / f"{stem}_forecasts.csv.gz", PARTS / f"{stem}_selection.csv"
    if fpath.exists() and spath.exists():
        res = {"forecasts": pd.read_csv(fpath, parse_dates=["origin", "target_date"]), "selection": pd.read_csv(spath)}
    else:
        res = run_series(y, target, yoy, label)
        res["forecasts"].to_csv(fpath, index=False, compression="gzip")
        res["selection"].to_csv(spath, index=False)
    return label, res, time.time() - t0, (y.index.min(), y.index.max(), len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    ensure_dirs()
    PARTS.mkdir(parents=True, exist_ok=True)
    countries = config()["project"]["countries"]
    jobs = [(c, a, t) for t in ("mom", "yoy") for a in ("A", "B") for c in countries]
    fcs, sels, log = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, *j): j for j in jobs}
        for f in as_completed(futs):
            label, res, secs, (start, end, n) = f.result()
            fcs.append(res["forecasts"])
            sels.append(res["selection"])
            log.append({**label, "series_start": start.date(), "series_end": end.date(), "n_obs": n, "seconds": round(secs, 1)})
            print(label, f"{secs:.0f}s", flush=True)
    fc = pd.concat(fcs, ignore_index=True)
    fc.to_csv(FORECASTS / "international_backtest_forecasts.csv.gz", index=False, compression="gzip")
    pd.concat(sels, ignore_index=True).to_csv(TABLES / "model_selection_validation.csv", index=False)
    pd.DataFrame(log).sort_values(["target", "analysis", "country"]).to_csv(LOGS / "international_run_log.csv", index=False)


if __name__ == "__main__":
    main()
