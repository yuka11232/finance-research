"""Phase 8a: four-model rolling backtests for Azerbaijan's 12 COICOP divisions (+ all items).

Target: month-on-month inflation of each division (IMF CPI, 2010=100,
2005-01..2026-03). Same selection protocol as the international experiments;
the final origin (2026-03) yields genuine forecasts for 2026-04..2027-03.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.paths import FORECASTS, LOGS, PROCESSED, TABLES, ensure_dirs

PARTS = FORECASTS / "category_parts"


def task(code: str):
    from src.forecasting.experiment import run_series
    from src.inflation.transform import mom, yoy

    t0 = time.time()
    fpath, spath = PARTS / f"{code}_forecasts.csv.gz", PARTS / f"{code}_selection.csv"
    if fpath.exists() and spath.exists():
        return code, pd.read_csv(fpath, parse_dates=["origin", "target_date"]), pd.read_csv(spath), 0.0
    idx = pd.read_csv(PROCESSED / "aze_cpi_coicop_monthly_2010_100.csv", index_col=0, parse_dates=True)[code].dropna()
    res = run_series(mom(idx).dropna(), "mom", yoy(idx), {"target": "mom", "analysis": "category", "country": "AZE", "coicop": code})
    res["forecasts"].to_csv(fpath, index=False, compression="gzip")
    res["selection"].to_csv(spath, index=False)
    return code, res["forecasts"], res["selection"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()
    ensure_dirs()
    PARTS.mkdir(parents=True, exist_ok=True)
    codes = [f"CP{i:02d}" for i in range(1, 13)] + ["ALL"]
    fcs, sels, log = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(task, c) for c in codes]):
            code, fc, sel, secs = f.result()
            fcs.append(fc)
            sels.append(sel)
            log.append({"coicop": code, "seconds": round(secs, 1)})
            print(code, f"{secs:.0f}s", flush=True)
    pd.concat(fcs, ignore_index=True).to_csv(FORECASTS / "aze_category_backtest_forecasts.csv.gz", index=False, compression="gzip")
    pd.concat(sels, ignore_index=True).to_csv(TABLES / "category_model_selection_validation.csv", index=False)
    pd.DataFrame(log).to_csv(LOGS / "category_run_log.csv", index=False)


if __name__ == "__main__":
    main()
