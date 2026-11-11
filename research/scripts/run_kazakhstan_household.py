"""Kazakhstan household groups: spending structure 2010-2026 and fixed-basket inflation exposure.

Uses the BNS household budget bulletins (Google Drive download of 2026-10-25) and the BNS
COICOP-division CPI (stat.gov.kz dynamic table). The exposure method is the one used for
Azerbaijan (src/inflation/household.py): Laspeyres indices with 2025 baskets. Regional baskets
are valued with national division prices, so regional differences come from baskets only.

Outputs
  data/processed/kaz_cpi_coicop_monthly.csv   chained division indices (Dec 2007 = 100)
  outputs/tables/KAZ_HBS_STRUCTURE.csv        consumption structure, total/urban/rural, 2010..2026Q2
  outputs/tables/KAZ_HBS_2025_SHARES.csv      2025 COICOP shares (%) by area x region
  outputs/tables/KAZ_GROUP_INFLATION.csv      exposure by area x region and window, with contributions
  outputs/tables/kaz_urban_rural_gap_series.csv  monthly y/y, rural minus urban gap, relative food inflation
  outputs/tables/kaz_gap_regression.csv       Newey-West regression of the gap on relative food inflation
  outputs/tables/kaz_aze_basket_comparison.csv   2025 national COICOP shares, Azerbaijan vs Kazakhstan
  outputs/tables/kaz_data_checks.csv          reconciliation and cross-source checks
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.ingestion.international import load_imf_monthly
from src.ingestion.kz_hbs import coicop_shares, detailed_shares, kz_coicop_monthly, structure_timeseries
from src.inflation.household import COICOP, price_relatives, window_inflation
from src.paths import PROCESSED, TABLES, ensure_dirs

T = pd.Timestamp
WINDOWS = {
    "W1_yoy_2026_03": (T("2025-03-01"), T("2026-03-01")),
    "W3_cum_2020_12_to_2023_12": (T("2020-12-01"), T("2023-12-01")),
    "W4_yoy_2026_09": (T("2025-09-01"), T("2026-09-01")),
}


def checks(struct, detail, prices) -> pd.DataFrame:
    imf = load_imf_monthly()
    imf = imf[(imf["country"] == "KAZ") & (imf["coicop"] == "ALL")].set_index("date")["index"].dropna()
    a = 100 * (prices["ALL"] / prices["ALL"].shift(12) - 1)
    b = 100 * (imf / imf.shift(12) - 1)
    j = pd.concat([a, b], axis=1, keys=["bns", "imf"], sort=True).dropna()
    rows = [
        ("structure: food goods = sum of food items (max |diff|, pp)", struct["food_share_check"].abs().max()),
        ("structure: food + non-food + services = 100 (max |diff|, pp)", struct["total_check"].abs().max()),
        ("2025 services items = table 1.4 services (max |diff|, pp)", detail["services_item_check"].abs().max()),
        ("2025 non-food items = table 1.4 non-food, total & urban (max |diff|, pp)",
         detail[detail["area"] != "rural"]["nonfood_item_check"].abs().max()),
        ("2025 non-food items = table 1.4 non-food, rural, before rescaling (max |diff|, pp)",
         detail[detail["area"] == "rural"]["nonfood_item_check"].abs().max()),
        ("chained BNS all-items y/y vs IMF KAZ y/y, mean |diff| (pp)", (j["bns"] - j["imf"]).abs().mean()),
        ("chained BNS all-items y/y vs IMF KAZ y/y, max |diff| (pp)", (j["bns"] - j["imf"]).abs().max()),
        ("months compared with IMF", len(j)),
    ]
    return pd.DataFrame(rows, columns=["check", "value"])


def gap_series(shares, rel, prices) -> tuple[pd.DataFrame, pd.DataFrame]:
    nat = shares[shares["region"] == "Kazakhstan"].set_index("area")[COICOP]
    idx = rel[COICOP] @ nat.T
    yoy = 100 * (idx / idx.shift(12) - 1)
    ser = pd.DataFrame({"total_yoy": yoy["total"], "urban_yoy": yoy["urban"], "rural_yoy": yoy["rural"]})
    ser["gap_rural_minus_urban_pp"] = ser["rural_yoy"] - ser["urban_yoy"]
    ser["food_yoy"] = 100 * (prices["CP01"] / prices["CP01"].shift(12) - 1)
    ser["all_items_yoy"] = 100 * (prices["ALL"] / prices["ALL"].shift(12) - 1)
    ser["relative_food_inflation_pp"] = ser["food_yoy"] - ser["all_items_yoy"]
    ser = ser.dropna()
    X = sm.add_constant(ser["relative_food_inflation_pp"])
    res = sm.OLS(ser["gap_rural_minus_urban_pp"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    reg = pd.DataFrame([{
        "sample": f"{ser.index.min():%Y-%m}..{ser.index.max():%Y-%m}", "n_months": int(res.nobs), "slope": res.params.iloc[1],
        "slope_se_nw": res.bse.iloc[1], "slope_ci95_low": res.conf_int().iloc[1, 0], "slope_ci95_high": res.conf_int().iloc[1, 1],
        "slope_p": res.pvalues.iloc[1], "intercept": res.params.iloc[0], "r2": res.rsquared,
        "share_months_gap_positive": float((ser["gap_rural_minus_urban_pp"] > 0).mean()),
        "gap_max": ser["gap_rural_minus_urban_pp"].max(), "gap_max_month": f"{ser['gap_rural_minus_urban_pp'].idxmax():%Y-%m}",
        "gap_min": ser["gap_rural_minus_urban_pp"].min(), "gap_min_month": f"{ser['gap_rural_minus_urban_pp'].idxmin():%Y-%m}",
    }])
    return ser.reset_index(names="date"), reg


def main():
    ensure_dirs()
    prices = kz_coicop_monthly()
    prices.to_csv(PROCESSED / "kaz_cpi_coicop_monthly.csv", index_label="date")
    struct = structure_timeseries()
    struct.to_csv(TABLES / "KAZ_HBS_STRUCTURE.csv", index=False)

    detail = detailed_shares("2025")
    shares = coicop_shares(detail)
    out = shares.copy()
    out[COICOP] = 100 * out[COICOP]
    out["food_share_pct"] = detail["food_nonalc"]
    out.to_csv(TABLES / "KAZ_HBS_2025_SHARES.csv", index=False)

    rel = price_relatives(prices, 2025)
    s = shares.set_index(["area", "region_en"])[COICOP]
    frames = []
    for name, (t0, t1) in WINDOWS.items():
        w = window_inflation(s, rel, t0, t1).reset_index()
        w.insert(0, "window", name)
        frames.append(w)
    infl = pd.concat(frames, ignore_index=True)
    infl = infl.merge(out[["area", "region_en", "food_share_pct"]], on=["area", "region_en"])
    infl.to_csv(TABLES / "KAZ_GROUP_INFLATION.csv", index=False)

    ser, reg = gap_series(shares, rel, prices)
    # cross-region association between food share and exposure (descriptive: 20 regions, one price vector)
    for name in WINDOWS:
        r = infl[(infl["window"] == name) & (infl["area"] == "total") & (infl["region_en"] != "Kazakhstan")]
        rho, p = stats.spearmanr(r["food_share_pct"], r["inflation_pct"])
        reg[f"spearman_food_share_vs_{name}"] = rho
        reg[f"spearman_p_{name}"] = p
    ser.to_csv(TABLES / "kaz_urban_rural_gap_series.csv", index=False)
    reg.to_csv(TABLES / "kaz_gap_regression.csv", index=False)

    aze = pd.read_csv(PROCESSED / "official_shares_2025_coicop.csv")
    aze = aze[(aze["dimension"] == "area") & (aze["group"] == "national")][COICOP].iloc[0]
    kaz = out[(out["area"] == "total") & (out["region"] == "Kazakhstan")][COICOP].iloc[0]
    pd.DataFrame({"coicop": COICOP, "azerbaijan_pct": (100 * aze / aze.sum()).values, "kazakhstan_pct": kaz.values}).to_csv(
        TABLES / "kaz_aze_basket_comparison.csv", index=False)

    chk = checks(struct, detail, prices)
    chk.to_csv(TABLES / "kaz_data_checks.csv", index=False)
    print(chk.to_string())
    print(infl[infl["region_en"] == "Kazakhstan"][["window", "area", "inflation_pct"]].round(2).to_string())
    print(reg.T.to_string())


if __name__ == "__main__":
    main()
