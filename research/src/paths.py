"""Project paths and configuration loading."""
from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
METADATA = DATA / "metadata"
AZ_RAW = RAW / "Azerbaijan_Data"
INTL_RAW = RAW / "international"
OUTPUTS = ROOT / "outputs"
FORECASTS = OUTPUTS / "forecasts"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
REPORTS = OUTPUTS / "reports"
LOGS = OUTPUTS / "logs"


@lru_cache(maxsize=1)
def config() -> dict:
    with open(ROOT / "config" / "experiment.toml", "rb") as fh:
        return tomllib.load(fh)


def ensure_dirs() -> None:
    for p in (PROCESSED, METADATA, INTL_RAW, FORECASTS, FIGURES, TABLES, REPORTS, LOGS):
        p.mkdir(parents=True, exist_ok=True)
