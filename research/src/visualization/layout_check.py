"""Automatic collision check for rendered figures.

After drawing, every visible text (titles, labels, tick labels, annotations, legend entries,
notes) and every legend box is tested against
  - every other text (overlapping labels), and
  - the plotted data of all axes (lines, markers, bars, error bars) for free-standing texts
    and legends (a label printed over data).
Tick labels, axis labels and titles are only tested against other texts, since data never
reaches them. Bounding boxes are shrunk by PAD pixels so touching is not a collision.
"""
from __future__ import annotations

import numpy as np
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

PAD = 1.0


def _shrink(bb: Bbox, pad: float = PAD) -> Bbox:
    return Bbox.from_extents(bb.x0 + pad, bb.y0 + pad, bb.x1 - pad, bb.y1 - pad)


def _dedupe(drawn):
    seen, out = set(), []
    for t, bb in drawn:
        if id(t) not in seen:
            seen.add(id(t))
            out.append((t, bb))
    return out


def _free_standing(t: Text) -> bool:
    """Texts placed inside the plotting area (annotations, ax.text), not decorations."""
    ax = t.axes
    if ax is None:
        return False
    if t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
        return False
    if t in ax.xaxis.get_ticklabels(which="both") or t in ax.yaxis.get_ticklabels(which="both"):
        return False
    if isinstance(t.get_parent() if hasattr(t, "get_parent") else None, Legend):
        return False
    return any(t is a for a in ax.texts)


def _data_points(fig, renderer):
    """Display-space points (n x 2) and boxes covering plotted data of every axes."""
    pts, boxes = [], []
    for ax in fig.axes:
        clip = ax.get_window_extent(renderer)
        for ln in ax.get_lines():
            if not ln.get_visible() or ln.get_label() == "_layout_ignore":
                continue
            xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), dtype=float))
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy) == 0:
                continue
            if ln.get_linestyle() not in ("None", "", " ") and len(xy) > 1:  # sample along segments
                seg = [np.linspace(a, b, max(2, int(np.hypot(*(b - a)) // 2) + 1)) for a, b in zip(xy[:-1], xy[1:])]
                xy = np.vstack(seg)
            pts.append(xy[_inside(clip, xy)])
        for coll in ax.collections:
            if not coll.get_visible() or coll.get_label() == "_layout_ignore":
                continue
            off = coll.get_offsets()
            if len(off):
                xy = coll.get_offset_transform().transform(np.asarray(off, dtype=float))
                pts.append(xy[_inside(clip, xy)])
            for path in coll.get_paths() if hasattr(coll, "get_segments") else []:
                xy = coll.get_transform().transform(path.vertices)
                if len(xy) > 1:
                    xy = np.vstack([np.linspace(a, b, max(2, int(np.hypot(*(b - a)) // 2) + 1)) for a, b in zip(xy[:-1], xy[1:])])
                pts.append(xy[_inside(clip, xy)])
        for p in ax.patches:
            if isinstance(p, Rectangle) and p.get_visible() and p.get_label() != "_layout_ignore":
                bb = p.get_window_extent(renderer)
                if bb.width > 0.5 and bb.height > 0.5:
                    boxes.append(Bbox.intersection(bb, clip) or bb)
    pts = np.vstack([p for p in pts if len(p)]) if pts else np.empty((0, 2))
    return pts, boxes


def _inside(bb: Bbox, xy: np.ndarray) -> np.ndarray:
    return (xy[:, 0] >= bb.x0) & (xy[:, 0] <= bb.x1) & (xy[:, 1] >= bb.y0) & (xy[:, 1] <= bb.y1)


def check_figure(fig, name: str, draw=None) -> list[dict]:
    """Run `draw` (default: a canvas draw; pass the savefig call to check the saved file) and test the
    figure exactly as that draw rendered it. Only texts that were actually drawn are tested."""
    drawn, result = [], {}
    orig_text_draw = Text.draw
    orig_fig_draw = fig.draw

    def rec(self, renderer):
        orig_text_draw(self, renderer)
        if self.get_visible() and self.get_text().strip():
            bb = self.get_window_extent(renderer)
            if bb.width >= 1 and bb.height >= 1:
                drawn.append((self, _shrink(bb)))

    def fig_draw(renderer):  # savefig may draw more than once; the last draw is the one written
        drawn.clear()
        orig_fig_draw(renderer)
        result["issues"] = _analyse(fig, name, renderer, _dedupe(drawn))

    Text.draw = rec
    fig.draw = fig_draw
    try:
        (draw or fig.canvas.draw)()
    finally:
        Text.draw = orig_text_draw
        del fig.draw
    return result["issues"]


def _analyse(fig, name, r, texts) -> list[dict]:
    issues = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, ba = texts[i]
            b, bb = texts[j]
            if ba.overlaps(bb):
                issues.append({"figure": name, "kind": "text-text", "a": a.get_text()[:60], "b": b.get_text()[:60]})
    pts, boxes = _data_points(fig, r)
    legends = [lg for ax in fig.axes if (lg := ax.get_legend()) is not None] + list(fig.legends)
    # value labels deliberately printed inside a bar must fit inside one bar
    for t, bb in texts:
        if t.get_gid() == "inside-bar" and not any(b.x0 <= bb.x0 and bb.x1 <= b.x1 and b.y0 <= bb.y0 and bb.y1 <= b.y1 for b in boxes):
            issues.append({"figure": name, "kind": "inside-bar-overflow", "a": t.get_text()[:60], "b": "label wider or taller than its bar"})
    probes = [(t.get_text()[:60], bb) for t, bb in texts if _free_standing(t) and t.get_gid() != "inside-bar"]
    probes += [(f"legend[{', '.join(x.get_text() for x in lg.get_texts())[:50]}]", _shrink(lg.get_window_extent(r), 2)) for lg in legends]
    for label, bb in probes:
        n = int(_inside(bb, pts).sum()) if len(pts) else 0
        hit_box = sum(1 for b in boxes if bb.overlaps(_shrink(b, 0.5)))
        if n or hit_box:
            issues.append({"figure": name, "kind": "label-data", "a": label, "b": f"{n} data points, {hit_box} bars"})
    leg_boxes = [(lg, _shrink(lg.get_window_extent(r), 2)) for lg in legends]
    for lg, lb in leg_boxes:
        own = set(id(x) for x in lg.get_texts()) | {id(lg.get_title())}
        for t, tb in texts:
            if id(t) not in own and lb.overlaps(tb) and not _contains_text(lg, t):
                issues.append({"figure": name, "kind": "legend-text", "a": "legend", "b": t.get_text()[:60]})
    return issues


def _contains_text(lg: Legend, t: Text) -> bool:
    return any(t is x for x in lg.findobj(Text))
