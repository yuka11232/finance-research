"""Phase 1: computed data audit (file inventory, hashes, coverage, cross-source checks).

Writes data/metadata/file_inventory.csv, data/metadata/series_coverage.csv,
data/metadata/cross_source_checks.csv and data/metadata/team_workbook_check.csv.
DATA_AUDIT.md / DATA_INVENTORY.md cite these files.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.ingestion.international import load_imf_monthly, load_worldbank_annual
from src.paths import AZ_RAW, METADATA, PROCESSED, RAW, ROOT

WORKBOOKS = RAW / "project_workbooks"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory() -> pd.DataFrame:
    rows = []
    for p in sorted(RAW.rglob("*")):
        if p.is_file():
            rows.append({"location": "data/raw", "path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                         "extension": p.suffix.lower(), "sha256": sha(p)})
    return pd.DataFrame(rows)


def coverage() -> pd.DataFrame:
    imf = load_imf_monthly()
    rows = []
    for (c, k), g in imf.groupby(["country", "coicop"]):
        ok = g.dropna(subset=["index"])
        rows.append({"source": "IMF CPI (IMF.STA:CPI 5.0.0)", "country": c, "series": k, "frequency": "monthly", "unit": "index",
                     "reference_period": g["reference_period"].iloc[0], "rows_returned": len(g), "non_missing": len(ok),
                     "first": ok["date"].min(), "last": ok["date"].max(),
                     "internal_gaps": (len(pd.date_range(ok["date"].min(), ok["date"].max(), freq="MS")) - len(ok)) if len(ok) else np.nan})
    wb = load_worldbank_annual()
    for c, g in wb.groupby("country"):
        ok = g.dropna(subset=["inflation_pct"])
        rows.append({"source": "World Bank WDI FP.CPI.TOTL.ZG", "country": c, "series": "ALL", "frequency": "annual", "unit": "% change, annual average",
                     "reference_period": "", "rows_returned": len(g), "non_missing": len(ok), "first": ok["year"].min(), "last": ok["year"].max(), "internal_gaps": 0})
    return pd.DataFrame(rows)


def cross_checks() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    imf = load_imf_monthly().dropna(subset=["index"])
    az = imf[(imf["country"] == "AZE") & (imf["coicop"] == "ALL")].set_index("date")["index"]
    lev = pd.read_csv(AZ_RAW / "05_Model_Ready_Data/official_monthly_CPI_levels_2019-2025.csv", parse_dates=["date"], encoding="utf-8-sig")
    lev = lev[lev["series"] == "national"].set_index("date")["official_CPI_2010_100"]
    j = pd.concat([az, lev], axis=1, keys=["imf", "azstat"]).dropna()
    rows.append({"check": "IMF AZE CPI level vs AZSTAT official 2010=100 level", "n": len(j),
                 "max_abs_diff": float((j["imf"] - j["azstat"]).abs().max()), "max_rel_diff_pct": float((100 * (j["imf"] / j["azstat"] - 1)).abs().max())})
    b = pd.read_csv(AZ_RAW / "05_Model_Ready_Data/monthly_broad_CPI_2016-01_to_2026-09.csv", parse_dates=["date"], encoding="utf-8-sig")
    nat = b[b["series"] == "national"].set_index("date")["mom_inflation_pct"]
    k = pd.concat([100 * az.pct_change(), nat], axis=1, keys=["imf", "azstat"]).dropna()
    rows.append({"check": "IMF AZE m/m vs AZSTAT published m/m (1 decimal)", "n": len(k), "max_abs_diff": float((k["imf"] - k["azstat"]).abs().max()),
                 "corr": float(k["imf"].corr(k["azstat"]))})
    wb = load_worldbank_annual()
    for c, g in imf[imf["coicop"] == "ALL"].groupby("country"):
        s = g.set_index("date")["index"]
        a = s.groupby(s.index.year).agg(["mean", "size"])
        a = 100 * a[a["size"] == 12]["mean"].pct_change()
        w = wb[wb["country"] == c].set_index("year")["inflation_pct"]
        jj = pd.concat([a, w], axis=1, keys=["imf", "wb"]).dropna()
        rows.append({"check": f"{c}: IMF annual-average inflation vs World Bank WDI", "n": len(jj), "max_abs_diff": float((jj["imf"] - jj["wb"]).abs().max())})
    head = pd.read_csv(PROCESSED / "cpi_headline_monthly.csv", parse_dates=["date"])
    s = head[(head["country"] == "AZE") & (head["date"] == "2026-09-01")]["yoy"].iloc[0]
    rows.append({"check": "Spliced AZE y/y Sep 2026 vs AZSTAT published 105.5 (bulletin 2026_09, sheet 'bash')", "n": 1, "max_abs_diff": abs(s - 5.5)})
    # team CPI workbook vs WB
    v = pd.read_excel(WORKBOOKS / "team_cpi_workbook_algeria_norway_azerbaijan_2000_2025.xlsx", header=1)
    vc = []
    for c, col_i, col_l in (("DZA", "algeria_inflation_pct", "algeria_cpi_2010_100_reconstructed"),
                            ("NOR", "norway_inflation_pct", "norway_cpi_2010_100_reconstructed"),
                            ("AZE", "azerbaijan_inflation_pct", "azerbaijan_cpi_2010_100_reconstructed")):
        w = wb[wb["country"] == c].set_index("year")
        for _, r in v.iterrows():
            y = int(r["year"])
            vc.append({"country": c, "year": y, "workbook_inflation": r[col_i], "wb_inflation": w.loc[y, "inflation_pct"],
                       "workbook_cpi": r[col_l], "wb_cpi": w.loc[y, "cpi_2010_100"],
                       "inflation_diff": r[col_i] - w.loc[y, "inflation_pct"], "cpi_ratio": r[col_l] / w.loc[y, "cpi_2010_100"]})
    vc = pd.DataFrame(vc)
    vc["flag"] = np.where((vc["cpi_ratio"] - 1).abs() > 0.01, "CPI level inconsistent (likely lost decimal separator)", "")
    rows.append({"check": "Team CPI workbook inflation vs World Bank WDI (3 countries x 26 years)", "n": len(vc), "max_abs_diff": float(vc["inflation_diff"].abs().max())})
    rows.append({"check": "Team CPI workbook CPI levels with |ratio-1| > 1% vs World Bank", "n": int((vc["flag"] != "").sum()), "max_abs_diff": np.nan})
    dup = sha(WORKBOOKS / "azstat_tables_51-52_expenditure_quintiles_2024.xls") == sha(AZ_RAW / "01_Household_Income_and_Spending/Spending_Expenditure_Quintiles__2024__AZSTAT.xls")
    rows.append({"check": "azstat_tables_51-52_expenditure_quintiles_2024.xls identical to Spending_Expenditure_Quintiles__2024__AZSTAT.xls (sha256)", "n": 1, "max_abs_diff": 0.0 if dup else np.nan,
                 "result": str(dup)})
    return pd.DataFrame(rows), vc


def main():
    METADATA.mkdir(parents=True, exist_ok=True)
    inventory().to_csv(METADATA / "file_inventory.csv", index=False)
    coverage().to_csv(METADATA / "series_coverage.csv", index=False)
    cc, vc = cross_checks()
    cc.to_csv(METADATA / "cross_source_checks.csv", index=False)
    vc.to_csv(METADATA / "team_workbook_check.csv", index=False)
    print(cc.to_string())
    print(vc[vc["flag"] != ""][["country", "year", "workbook_cpi", "wb_cpi"]])


if __name__ == "__main__":
    main()
