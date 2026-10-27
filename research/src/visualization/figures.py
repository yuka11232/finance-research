"""Publication figures (matplotlib, PNG + SVG).

Colour follows the entity, never its rank: each model and each country keeps
one validated categorical slot in every figure (palette validated with the
dataviz validator: adjacent CVD dE >= 9.1, normal-vision dE >= 19.6). Three
slots are below 3:1 contrast on white, so every figure carries a legend,
line-style secondary encoding and an underlying CSV table.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import FIGURES
from src.visualization.layout_check import check_figure

LAYOUT_ISSUES: list[dict] = []  # filled by save(); make_figures.py fails the step if any remain

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_STYLE = {  # colour, linestyle, label
    "seasonal_naive": (MUTED, (0, (4, 2)), "Seasonal naive"),
    "sarima": (SLOTS[0], "-", "SARIMA"),
    "ets": (SLOTS[1], (0, (6, 2, 1, 2)), "ETS"),
    "yu_weighted_fts": (SLOTS[2], "-", "Yu weighted FTS"),
    "yu_equal_fts": (SLOTS[3], (0, (2, 1.5)), "Equal-weight recurrence FTS"),
    "chen_fts": (SLOTS[4], (0, (1, 1.5)), "Chen (1996) FTS"),
    "naive": (INK2, (0, (1, 3)), "Random walk"),
    "ar_aic": (SLOTS[6], (0, (5, 1)), "AR (AIC)"),
}
COUNTRY_STYLE = {"AZE": SLOTS[0], "DZA": SLOTS[1], "KAZ": SLOTS[2], "NOR": SLOTS[3], "RUS": SLOTS[4]}
COUNTRY_NAME = {"AZE": "Azerbaijan", "DZA": "Algeria", "KAZ": "Kazakhstan", "NOR": "Norway", "RUS": "Russia"}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6, "axes.titlelocation": "left",
    })


def save(fig, name: str, note: str | None = None):
    # freeze the constrained layout first: otherwise savefig re-runs it and moves artists after the check
    fig.canvas.draw()
    fig.set_layout_engine("none")
    # figure legends are anchored to the figure edge, which bbox_inches="tight" moves at save time;
    # pin each one to the box it occupies now (figure fraction, centred in that box)
    r = fig.canvas.get_renderer()
    for lg in fig.legends:
        bb = lg.get_window_extent(r).transformed(fig.transFigure.inverted())
        lg.set_bbox_to_anchor(bb, transform=fig.transFigure)
        lg._loc = 10  # "center"
    if note:  # place the note below every other artist so it never collides with labels or legends
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        fig.text(0.01, bb.y0 / fig.get_figheight() - 0.015, note, fontsize=7, color=INK2, ha="left", va="top", wrap=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    # the collision check runs inside the PNG save, on exactly the pixels that are written
    LAYOUT_ISSUES.extend(check_figure(fig, name, lambda: fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")))
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def shade_periods(ax, periods: dict):
    for i, (_, (a, b)) in enumerate(periods.items()):
        if i % 2 == 1:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.45, lw=0, zorder=0)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK, va="center")
    ax.plot([x], [y], "o", ms=3.5, color=color)


def end_labels(ax, items, gap_pt: float = 10.0, dx_pt: float = 6.0):
    """Label series end points [(x, y, text, colour)], pushing labels apart vertically so none overlap."""
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72.0 / fig.dpi
    ys = [ax.transData.transform((mdates_num(x), y))[1] * to_pt for x, y, _, _ in items]
    order = np.argsort(ys)
    placed = list(np.array(ys)[order])
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap_pt)
    shift = (np.mean(np.array(ys)[order]) - np.mean(placed))  # keep the block centred on the data
    placed = [p + shift for p in placed]
    for k, i in enumerate(order):
        x, y, text, color = items[i]
        ax.plot([x], [y], "o", ms=3.5, color=color)
        ax.annotate(text, (x, y), xytext=(dx_pt, placed[k] - ys[i]), textcoords="offset points", fontsize=7.5,
                    color=INK, va="center", arrowprops=dict(arrowstyle="-", color=color, lw=0.6, shrinkA=0, shrinkB=2))


def mdates_num(x):
    import matplotlib.dates as mdates

    return mdates.date2num(x) if isinstance(x, (pd.Timestamp, np.datetime64)) else x
