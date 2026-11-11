"""Build the processed CPI datasets used by every experiment."""
from __future__ import annotations

import pandas as pd

from src.ingestion.international import load_imf_monthly
from src.inflation.transform import mom, yoy
from src.paths import AZ_RAW, PROCESSED

AZSTAT_BROAD = AZ_RAW / "05_Model_Ready_Data" / "monthly_broad_CPI_2016-01_to_2026-09.csv"


def load_azstat_broad() -> pd.DataFrame:
    return pd.read_csv(AZSTAT_BROAD, parse_dates=["date"], encoding="utf-8-sig")


def headline_monthly() -> pd.DataFrame:
    """Monthly all-items CPI per country with m/m and y/y inflation.

    Azerbaijan: IMF CPI (identical to AZSTAT within rounding over 2016-2026M03)
    is extended from 2026-04 to 2026-09 by chaining AZSTAT's published
    previous-month = 100 relatives (one decimal). Spliced rows carry
    source = 'AZSTAT_chained'.
    """
    imf = load_imf_monthly().dropna(subset=["index"])
    imf = imf[imf["coicop"] == "ALL"]
    frames = []
    for c, g in imf.groupby("country"):
        s = g.set_index("date")["index"].asfreq("MS")
        src = pd.Series("IMF_CPI", index=s.index)
        if c == "AZE":
            az = load_azstat_broad()
            az = az[az["series"] == "national"].set_index("date")["previous_month_100"]
            ext = az[az.index > s.index.max()]
            level = s.iloc[-1]
            add = {}
            for d, rel in ext.items():
                level = level * rel / 100.0
                add[d] = level
            s = pd.concat([s, pd.Series(add)]).asfreq("MS")
            src = pd.concat([src, pd.Series("AZSTAT_chained", index=list(add))]).reindex(s.index)
        df = pd.DataFrame({"country": c, "date": s.index, "index": s.values, "source": src.values})
        df["mom"] = mom(s).values
        df["yoy"] = yoy(s).values
        df["reference_period"] = g["reference_period"].iloc[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def coicop_monthly(country: str = "AZE") -> pd.DataFrame:
    imf = load_imf_monthly().dropna(subset=["index"])
    imf = imf[imf["country"] == country]
    wide = imf.pivot(index="date", columns="coicop", values="index").asfreq("MS")
    return wide


def build_all() -> dict[str, pd.DataFrame]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    head = headline_monthly()
    head.to_csv(PROCESSED / "cpi_headline_monthly.csv", index=False)
    co = coicop_monthly("AZE")
    co.to_csv(PROCESSED / "aze_cpi_coicop_monthly_2010_100.csv")
    return {"headline": head, "aze_coicop": co}
