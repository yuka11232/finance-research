"""Download official international CPI data (IMF CPI dataset, World Bank WDI).

Raw responses are written unchanged to data/raw/international/ together with a
JSON sidecar recording URL, retrieval time and SHA-256, so that the exact
vintage used in every experiment can be audited.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.paths import INTL_RAW

IMF_BASE = "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,CPI/"
IMF_ACCEPT = "application/vnd.sdmx.data+csv;version=1.0.0"
WB_BASE = "https://api.worldbank.org/v2/country/{c}/indicator/{i}?format=json&per_page=20000&date=1995:2026"

ISO3 = ["AZE", "DZA", "KAZ", "RUS", "NOR"]


def _download(url: str, dest: Path, accept: str | None = None, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    dest.write_bytes(body)
    meta = {
        "url": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    dest.with_suffix(dest.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def download_imf(force: bool = False) -> list[Path]:
    """Headline (_T) and COICOP division CPI index levels, monthly."""
    INTL_RAW.mkdir(parents=True, exist_ok=True)
    key = "+".join(ISO3) + ".CPI..IX.M"
    dest = INTL_RAW / "imf_cpi_monthly_index_coicop.csv"
    packed = dest.with_suffix(".csv.gz")
    if force or not packed.exists():
        _download(f"{IMF_BASE}{key}?startPeriod=1995", dest, IMF_ACCEPT)
        # stored gzip-compressed (~64 MB raw); the .meta.json hash refers to the uncompressed body
        packed.write_bytes(gzip.compress(dest.read_bytes(), mtime=0))
        dest.unlink()
    return [packed]


def download_worldbank(force: bool = False) -> list[Path]:
    """Annual CPI inflation (FP.CPI.TOTL.ZG) and CPI index 2010=100 (FP.CPI.TOTL)."""
    INTL_RAW.mkdir(parents=True, exist_ok=True)
    out = []
    for ind in ("FP.CPI.TOTL.ZG", "FP.CPI.TOTL"):
        dest = INTL_RAW / f"worldbank_{ind.replace('.', '_')}.json"
        if force or not dest.exists():
            _download(WB_BASE.format(c=";".join(ISO3), i=ind), dest)
        out.append(dest)
    return out


def load_imf_monthly() -> pd.DataFrame:
    """Tidy monthly CPI index levels: country, coicop, date, index, reference_period."""
    raw = pd.read_csv(INTL_RAW / "imf_cpi_monthly_index_coicop.csv.gz", low_memory=False)
    df = raw[["COUNTRY", "COICOP_1999", "TIME_PERIOD", "OBS_VALUE", "REFERENCE_PERIOD"]].copy()
    df.columns = ["country", "coicop", "period", "index", "reference_period"]
    df["date"] = pd.to_datetime(df["period"].str.replace("-M", "-") + "-01")
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df["coicop"] = df["coicop"].replace({"_T": "ALL"})
    return df.drop(columns="period").sort_values(["country", "coicop", "date"]).reset_index(drop=True)


def load_worldbank_annual() -> pd.DataFrame:
    frames = []
    for ind, name in (("FP_CPI_TOTL_ZG", "inflation_pct"), ("FP_CPI_TOTL", "cpi_2010_100")):
        payload = json.loads((INTL_RAW / f"worldbank_{ind}.json").read_text())
        rows = [
            {"country": r["countryiso3code"], "year": int(r["date"]), name: r["value"]}
            for r in payload[1]
        ]
        frames.append(pd.DataFrame(rows))
    out = frames[0].merge(frames[1], on=["country", "year"], how="outer")
    return out.sort_values(["country", "year"]).reset_index(drop=True)
