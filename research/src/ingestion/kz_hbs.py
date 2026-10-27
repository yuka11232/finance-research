"""Kazakhstan household budget survey bulletins (BNS) and COICOP-division CPI.

Sources (data/raw):
  kazakhstan_hbs_bulletins/  BNS bulletins "Расходы и доходы населения" 2010-2025 (annual)
                             and I/II quarter 2026 (hbs_bulletin_<period>.xls[x]); 2010-2019
                             arrived as RAR archives, unpacked to annual_2010_2019/<year>/
                             (7-Zip, contents unchanged)
  kazakhstan_bns/            BNS dynamic table "ИПЦ по 13 основным разделам" (monthly,
                             2008-01..2026-09), downloaded with a .meta.json sidecar

The consumption-structure table ("Структура потребительских расходов населения")
has the same eight columns in every year: total, food goods, of which food and
non-alcoholic beverages / eating out / alcohol / tobacco, non-food goods, paid
services (percent of consumer expenditure). Region rows are aligned by position
because continuation blocks label regions in Kazakh.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from src.paths import RAW

KZ_HBS_RAW = RAW / "kazakhstan_hbs_bulletins"
KZ_BNS_RAW = RAW / "kazakhstan_bns"
STRUCT_COLS = ["total", "food_goods", "food_nonalc", "eating_out", "alcohol", "tobacco", "nonfood_goods", "services"]
AREAS = {"total": "Total", "urban": "Urban", "rural": "Rural"}
NATIONAL = re.compile(r"Республика\s+Казахстан|Қазақстан\s+Республикасы")
REGION_EN = {
    "Абай": "Abai", "Акмолинская": "Akmola", "Актюбинская": "Aktobe", "Алматинская": "Almaty region",
    "Атырауская": "Atyrau", "Западно-Казахстанская": "West Kazakhstan", "Жамбылская": "Zhambyl", "Жетісу": "Zhetisu",
    "Карагандинская": "Karaganda", "Костанайская": "Kostanay", "Кызылординская": "Kyzylorda", "Мангистауская": "Mangystau",
    "Павлодарская": "Pavlodar", "Северо-Казахстанская": "North Kazakhstan", "Туркестанская": "Turkistan", "Ұлытау": "Ulytau",
    "Восточно-Казахстанская": "East Kazakhstan", "г. Астана": "Astana city", "г. Алматы": "Almaty city", "г. Шымкент": "Shymkent city",
}
# Detailed 2025 bulletin items (tables 1.4, 1.8, 1.10) -> COICOP 1999 division
ITEM_COICOP = {
    "food_nonalc": "CP01", "eating_out": "CP11", "alcohol": "CP02", "tobacco": "CP02",
    "clothing_footwear": "CP03", "furniture_household": "CP05", "repair_materials": "CP04", "lpg_fuel": "CP04",
    "medicines": "CP06", "vehicles_fuel": "CP07", "audiovisual": "CP09", "phones": "CP08", "other_durables": "CP09",
    "recreation_goods": "CP09", "books_stationery": "CP09", "electric_personal_care": "CP12", "hygiene": "CP12",
    "personal_care_nec": "CP12",
    "housing_utilities": "CP04", "education": "CP10", "health_services": "CP06", "transport_services": "CP07",
    "communication_services": "CP08", "leisure_culture": "CP09", "restaurants_cafes": "CP11", "hotels": "CP11",
    "clothing_repair": "CP03", "shoe_repair": "CP03", "furniture_repair": "CP05", "appliance_repair": "CP05",
    "av_repair": "CP09", "other_services": "CP12",
}
NONFOOD_ITEMS = ["clothing_footwear", "furniture_household", "repair_materials", "lpg_fuel", "medicines", "vehicles_fuel",
                 "audiovisual", "phones", "other_durables", "recreation_goods", "books_stationery", "electric_personal_care",
                 "hygiene", "personal_care_nec"]
SERVICE_ITEMS = ["housing_utilities", "education", "health_services", "transport_services", "communication_services",
                 "leisure_culture", "restaurants_cafes", "hotels", "clothing_repair", "shoe_repair", "furniture_repair",
                 "appliance_repair", "av_repair", "other_services"]
SECTION_RU = {  # BNS CPI section label prefix -> COICOP 1999 division
    "Товары и услуги": "ALL", "Продукты питания": "CP01", "Алкогольные напитки": "CP02", "Одежда и обувь": "CP03",
    "Жилищные услуги": "CP04", "Предметы домашнего": "CP05", "Здравоохранение": "CP06", "Транспорт": "CP07",
    "Связь": "CP08", "Информация и связь": "CP08", "Отдых и культура": "CP09", "Образование": "CP10",
    "Рестораны и гостиницы": "CP11", "Разные товары": "CP12", "Личный уход": "CP12",
}


def _read(path, sheet=0) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_excel(path, sheet_name=sheet, header=None)


def _num(row: pd.Series, n: int) -> list[float]:
    vals = pd.to_numeric(row.iloc[1:1 + n], errors="coerce").to_list()
    return [np.nan if pd.isna(v) else float(v) for v in vals]


def _area_of(text: str) -> str | None:
    t = text.lower()
    if "городск" in t or t.strip() == "город":
        return "urban"
    if "сельск" in t or t.strip() == "село":
        return "rural"
    if "всего" in t or "по регионам" in t:
        return "total"
    return None


def _blocks(df: pd.DataFrame, default_area: str | None) -> dict[str, list[tuple[str, int]]]:
    """Area -> [(region label, row index)] for every numeric region row of a table sheet."""
    out: dict[str, list[tuple[str, int]]] = {}
    area = default_area
    for i, r in df.iterrows():
        lab = str(r.iloc[0]).strip() if pd.notna(r.iloc[0]) else ""
        numeric = pd.to_numeric(r.iloc[1:3], errors="coerce").notna().any()
        if not numeric:
            low = lab.lower()
            if low.startswith("в городской"):
                area = "urban"
            elif low.startswith("в сельской"):
                area = "rural"
            elif low.startswith(("по регионам", "всего")) and default_area is None:
                area = "total"
            continue
        if area is None or not lab:
            continue
        out.setdefault(area, []).append((lab, i))
    return out


def _structure_sources() -> list[tuple[str, object, object, str | None]]:
    """(period, path, sheet, default area) for every consumption-structure sheet."""
    src = []
    rar = KZ_HBS_RAW / "annual_2010_2019"
    for year in range(2010, 2017):
        files = sorted((rar / str(year)).rglob("*.xls"))
        for f in files:
            head = " ".join(str(v) for v in _read(f).head(4).to_numpy().ravel() if pd.notna(v))
            if re.search(r"Структура\s+потребительских\s+расходов", head):
                src.append((str(year), f, 0, _area_of(head.split("Структура")[1])))
    for year in (2017, 2018):
        f = next((rar / str(year)).rglob("1.4_*.xls"))
        for sh, a in (("всего", "total"), ("город", "urban"), ("село", "rural")):
            src.append((str(year), f, sh, a))
    f = next((rar / "2019").rglob("*.xls"))
    for a, sh in (("total", "1.4 всего"), ("urban", "1.4 город"), ("rural", "1.4 село")):
        src.append(("2019", f, sh, a))
    f = KZ_HBS_RAW / "hbs_bulletin_2020.xls"
    for a, sh in (("total", "1.4 всего"), ("urban", "1.4 город"), ("rural", "1.4 село")):
        src.append(("2020", f, sh, a))
    for period, name in (("2021", "hbs_bulletin_2021.xls"), ("2022", "hbs_bulletin_2022.xls"), ("2023", "hbs_bulletin_2023.xlsx"),
                         ("2024", "hbs_bulletin_2024.xlsx"), ("2025", "hbs_bulletin_2025.xlsx"),
                         ("2026Q1", "hbs_bulletin_2026Q1.xlsx"), ("2026Q2", "hbs_bulletin_2026Q2.xlsx")):
        src.append((period, KZ_HBS_RAW / name, "1.4", None))
    return src


def structure_timeseries(regions: bool = False) -> pd.DataFrame:
    """Consumption structure (% of consumer expenditure) by period and area (and region if requested)."""
    rows = []
    for period, path, sheet, default_area in _structure_sources():
        df = _read(path, sheet)
        header = " ".join(str(v) for v in df.head(12).to_numpy().ravel() if pd.notna(v))
        # from 2026 (COICOP 2018) eating out moves out of food goods into paid services
        cols = STRUCT_COLS if "вне дома" in header else [c for c in STRUCT_COLS if c != "eating_out"]
        for area, items in _blocks(df, default_area).items():
            for pos, (lab, i) in enumerate(items):
                is_nat = bool(NATIONAL.search(lab))
                if not regions and not is_nat:
                    continue
                vals = dict(zip(cols, _num(df.loc[i], len(cols))))
                rows.append({"period": period, "area": area, "region": "Kazakhstan" if is_nat else lab, "position": pos,
                             **{c: vals.get(c, np.nan) for c in STRUCT_COLS}, "source_file": path.name, "sheet": str(sheet)})
    out = pd.DataFrame(rows)
    if not regions:
        out = out.drop_duplicates(["period", "area"], keep="first")
    out["year"] = out["period"].str[:4].astype(int)
    out["food_share_check"] = out["food_nonalc"] + out["eating_out"].fillna(0) + out["alcohol"] + out["tobacco"] - out["food_goods"]
    out["total_check"] = out["food_goods"] + out["nonfood_goods"] + out["services"] - out["total"]
    return out.sort_values(["period", "area", "position"]).reset_index(drop=True)


def _detail_table(path, sheet, names: list[str]) -> pd.DataFrame:
    """Wide table split into continuation blocks; returns rows (area, position) x item columns."""
    df = _read(path, sheet)
    blocks = _blocks(df, None)
    # one "По регионам" marker per continuation block; its first data row gives the block's column count
    starts = df.index[df.iloc[:, 0].astype(str).str.strip() == "По регионам"]
    widths = []
    for s in starts:
        first = df.loc[s + 1]
        widths.append(int(pd.to_numeric(first.iloc[1:], errors="coerce").notna().sum()))
    rows = []
    for area, items in blocks.items():
        n_regions = len(items) // len(starts)
        for k, w in enumerate(widths):
            chunk = items[k * n_regions:(k + 1) * n_regions]
            cols = names[sum(widths[:k]):sum(widths[:k]) + w]
            for pos, (lab, i) in enumerate(chunk):
                rows.append({"area": area, "position": pos, **dict(zip(cols, _num(df.loc[i], w)))})
    out = pd.DataFrame(rows).groupby(["area", "position"]).first()
    return out


def detailed_shares(period: str = "2025") -> pd.DataFrame:
    """Item-level shares (% of consumer expenditure) by area x region for one annual bulletin, with COICOP aggregates."""
    path = KZ_HBS_RAW / f"hbs_bulletin_{period}.xlsx"
    base = structure_timeseries(regions=True)
    base = base[base["period"] == period].set_index(["area", "position"])
    nf = _detail_table(path, "1.8 ", ["nonfood_total"] + NONFOOD_ITEMS)
    sv = _detail_table(path, "1.10", ["services_total"] + SERVICE_ITEMS)
    d = base.join(nf).join(sv)
    d["nonfood_item_check"] = d[NONFOOD_ITEMS].sum(axis=1) - d["nonfood_goods"]
    d["services_item_check"] = d[SERVICE_ITEMS].sum(axis=1) - d["services"]
    # The rural blocks of table 1.8 do not add up to the rural non-food share of table 1.4 (up to
    # 3.4 pp in 2025, already in the published table). Items are rescaled to the table 1.4 total so
    # the basket keeps the headline structure; the discrepancy stays in nonfood_item_check.
    scale = d["nonfood_goods"] / d[NONFOOD_ITEMS].sum(axis=1)
    d[NONFOOD_ITEMS] = d[NONFOOD_ITEMS].mul(scale, axis=0)
    for cp in sorted(set(ITEM_COICOP.values())):
        d[cp] = d[[k for k, v in ITEM_COICOP.items() if v == cp]].sum(axis=1)
    d["region_en"] = d["region"].map(REGION_EN).fillna(d["region"])
    return d.reset_index()


def coicop_shares(detail: pd.DataFrame) -> pd.DataFrame:
    """COICOP division shares (fractions summing to 1) per area x region."""
    cps = [f"CP{i:02d}" for i in range(1, 13)]
    s = detail[cps].div(detail[cps].sum(axis=1), axis=0)
    return pd.concat([detail[["area", "region", "region_en"]], s], axis=1)


def kz_coicop_monthly() -> pd.DataFrame:
    """Monthly COICOP-division CPI levels (Dec 2007 = 100) chained from BNS previous-month changes.

    2008-2025 sheets publish growth in percent (1.1 = +1.1%); 2026 publishes previous month = 100
    under COICOP 2018 (13 sections). For 2026 CP12 is 'Личный уход, социальная защита и прочие'
    (insurance and financial services, a new separate section, are left out of CP12).
    """
    path = KZ_BNS_RAW / "kz_cpi_13_sections.xlsx"
    xl = pd.ExcelFile(path)
    rel = {}
    for sh in xl.sheet_names:
        year = int(re.search(r"(\d{4})", sh).group(1))
        df = _read(path, sh)
        stop = df.index[df.iloc[:, 0].astype(str).str.contains("к декабрю", na=False)]
        for block, kind in ((df.loc[:stop[0] - 1], "mom"), (df.loc[stop[0]:stop[0] + 16], "ytd")):
            for _, r in block.iterrows():
                lab = str(r.iloc[0]).strip()
                code = next((c for p, c in SECTION_RU.items() if lab.startswith(p)), None)
                if code is None:
                    continue
                vals = pd.to_numeric(r.iloc[1:13], errors="coerce").to_numpy(dtype=float)
                for m, v in enumerate(vals, start=1):
                    if not np.isnan(v):
                        rel.setdefault((kind, code), {})[(year, m)] = v / 100.0 if v > 50 else 1.0 + v / 100.0
    # a blank previous-month cell (CP02, 2014-02) is recovered from the change since December
    for (kind, code), d in list(rel.items()):
        if kind != "mom":
            continue
        ytd = rel.get(("ytd", code), {})
        for (y, m), v in ytd.items():
            if (y, m) not in d and (m == 1 or (y, m - 1) in ytd):
                d[(y, m)] = v / (ytd[(y, m - 1)] if m > 1 else 1.0)
    rel = {code: d for (kind, code), d in rel.items() if kind == "mom"}
    out = {}
    for code, d in rel.items():
        s = pd.Series({pd.Timestamp(y, m, 1): v for (y, m), v in d.items()}).sort_index()
        out[code] = 100.0 * s.cumprod()
    wide = pd.DataFrame(out).asfreq("MS")
    return wide[[c for c in ["ALL"] + [f"CP{i:02d}" for i in range(1, 13)] if c in wide]]
