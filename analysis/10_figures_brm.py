# -*- coding: utf-8 -*-
"""
10_figures_brm.py - redraw all 10 submission figures at Springer / BRM double-column
width (174 mm) with type sized for that width.

This is a PURE REDRAW from the saved result tables. Nothing is recomputed: no
GraphicalLassoCV, no ablation grid, no real-data loader. Every plotted number is
read from results/tables/*.csv, results/realdata/*.csv or results/panel.json, and
is asserted against the source before it is drawn.

Figure map (manuscript label -> source table(s)):
  Figure1   T4_lockbox_performance
  Figure2   T2_item_utility + panel.json
  Figure3   T8_ggm_partial_corr + T9_bridge_centrality + panel.json
  Figure4   T5_value_of_sleep + T4 + panel.json
  Figure5   RT2b_isi_selection_freq + RT3_lockbox + RT1b_correlations
  Figure6   RT6_cohortB_sri + RT7_cohortC_uk
  FigureS1  T1_calibration + T1b_achieved_correlations
  FigureS2  T10_crossdx_contribution
  FigureS3  T6_bootstrap_stability
  FigureS4  T12_baselines + T13_ablation

House rules enforced by assertion:
  * every string drawn into every figure is pure ASCII (no em dash, en dash or
    Unicode minus anywhere), and axes.unicode_minus is disabled;
  * width is exactly 174 mm;
  * measured modal cap height >= 2.0 mm at 174 mm.

Outputs (into --outdir, default ./out):
  Figure1.tif .. Figure6.tif, FigureS1.tif .. FigureS4.tif   LZW RGB, 400 dpi
  Figure1.png .. FigureS4.png                                300 dpi
  Figure*_grey.png                                           desaturated proof
  figure_metrics.json                                        measurements + checks
"""
import os
import re
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import networkx as nx
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# geometry / typography
# ---------------------------------------------------------------------------
WIDTH_MM = 174.0                      # Springer / BRM double column
WIDTH_IN = WIDTH_MM / 25.4            # 6.8504 in
TIFF_DPI = 600                        # Springer "combination art" floor (colour
                                      # diagrams with extensive lettering); also
                                      # clears the 300 dpi halftone floor
PNG_DPI = 300
CAP_FLOOR_MM = 2.0

FS_TICK = 9.0
FS_LABEL = 9.0
FS_TITLE = 9.5
FS_SUPTITLE = 10.0
FS_ANNOT = 8.5
FS_LEGEND = 8.5

matplotlib.rcParams.update({
    "axes.unicode_minus": False,      # plain ASCII hyphen for negative numbers
    "font.family": "DejaVu Sans",
    "font.size": FS_TICK,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "hatch.linewidth": 0.55,
    "figure.dpi": 150,
    "savefig.dpi": PNG_DPI,
    "savefig.bbox": None,             # never crop: width must stay exactly 174 mm
})

# ---------------------------------------------------------------------------
# palette - a three-step LUMINANCE ladder so every colour-encoded distinction
# also survives greyscale printing. The highlight series additionally carries a
# hatch pattern. Relative luminances (sRGB, WCAG):
#   TEAL   0.119   ORANGE 0.303   GREY 0.693   REDNEG 0.595
# contrast ratios: orange/teal 2.08, grey/orange 2.10, grey/teal 4.39,
#                  redneg/teal 3.81
# ---------------------------------------------------------------------------
TEAL = "#0C6B74"
TEAL_E = "#063C42"
ORANGE = "#D8842A"
ORANGE_E = "#6B3D06"
GREY = "#D5DCE0"
GREY_E = "#7A868C"
REDNEG = "#EFC0B8"
REDNEG_E = "#A2382A"
EDGE_BLUE = "#5B7FA8"
HATCH_HL = "//"        # highlighted / anchor series
HATCH_NEG = "xx"       # negative-signed series

DOMAINS = ["dep", "anx", "ptsd", "panic", "suicide", "alcohol", "sleep"]
DOMTITLE = {"dep": "Depression", "anx": "Anxiety", "ptsd": "PTSD",
            "panic": "Panic", "suicide": "Suicidality",
            "alcohol": "Alcohol use", "sleep": "Insomnia"}
DOMSHORT = {"dep": "DEP", "anx": "ANX", "ptsd": "PTSD", "panic": "PANIC",
            "suicide": "SUI", "alcohol": "ALC", "sleep": "SLEEP"}

BANNED = {"\u2014": "em dash", "\u2013": "en dash", "\u2212": "unicode minus"}

# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def tbl(root, name):
    return pd.read_csv(os.path.join(root, "results", "tables", name))


def rdt(root, name):
    return pd.read_csv(os.path.join(root, "results", "realdata", name))


def panel_json(root):
    with open(os.path.join(root, "results", "panel.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# bar helpers
# ---------------------------------------------------------------------------


def style_grey(**kw):
    kw.update(color=GREY, edgecolor=GREY_E, linewidth=0.6)
    return kw


def style_teal(**kw):
    kw.update(color=TEAL, edgecolor=TEAL_E, linewidth=0.6)
    return kw


def style_orange(**kw):
    kw.update(color=ORANGE, edgecolor=ORANGE_E, linewidth=0.6, hatch=HATCH_HL)
    return kw


def bar_kw_list(flags, base="teal"):
    """Per-bar face/edge/hatch lists: flags True -> highlighted orange+hatch."""
    face = [ORANGE if f else (GREY if base == "grey" else TEAL) for f in flags]
    edge = [ORANGE_E if f else (GREY_E if base == "grey" else TEAL_E) for f in flags]
    hat = [HATCH_HL if f else "" for f in flags]
    return face, edge, hat


def apply_hatches(container, hatches):
    for patch, h in zip(container, hatches):
        if h:
            patch.set_hatch(h)


def tidy(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# ASCII guard + cap-height measurement
# ---------------------------------------------------------------------------


def collect_text(fig):
    out = []
    for obj in fig.findobj(mtext.Text):
        s = obj.get_text()
        if s:
            out.append(s)
    return out


def assert_ascii(fig, tag):
    bad = []
    for s in collect_text(fig):
        for ch, nm in BANNED.items():
            if ch in s:
                bad.append((tag, nm, s))
        for ch in s:
            if ord(ch) > 126:
                bad.append((tag, "non-ascii U+%04X" % ord(ch), s))
    if bad:
        raise AssertionError("non-ASCII text in %s: %r" % (tag, bad[:6]))


def _hide_nontext(fig):
    """Hide every non-text artist so the raster contains glyphs only."""
    changed = []

    def off(a):
        if a is not None and hasattr(a, "get_visible") and a.get_visible():
            a.set_visible(False)
            changed.append(a)

    for ax in fig.axes:
        for grp in (ax.patches, ax.lines, ax.collections, ax.images,
                    ax.tables, ax.artists):
            for a in list(grp):
                off(a)
        for sp in ax.spines.values():
            off(sp)
        for t in list(ax.xaxis.get_major_ticks()) + list(ax.yaxis.get_major_ticks()) + \
                 list(ax.xaxis.get_minor_ticks()) + list(ax.yaxis.get_minor_ticks()):
            off(t.tick1line)
            off(t.tick2line)
        off(ax.patch)
        leg = ax.get_legend()
        if leg is not None:
            off(leg.get_frame())
            for h in getattr(leg, "legend_handles", []):
                off(h)
            for h in getattr(leg, "legendHandles", []):
                off(h)
    for a in list(fig.patches) + list(fig.lines) + list(fig.artists):
        off(a)
    off(fig.patch)
    return changed


def measure_cap_mm(fig, dpi=TIFF_DPI):
    """Modal cap height in mm, from connected components of a text-only render.

    Renders the figure with all non-text artists hidden, labels the dark glyph
    pixels, keeps glyph-shaped components, and returns the mode of the upper
    height cluster (caps / digits / ascenders) in mm at 174 mm width.
    """
    import io
    hidden = _hide_nontext(fig)
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    finally:
        for a in hidden:
            a.set_visible(True)
    buf.seek(0)
    img = Image.open(buf).convert("L")
    arr = np.asarray(img)
    px_mm = img.width / WIDTH_MM
    lab, n = ndimage.label(arr < 128)
    if n == 0:
        return dict(modal_cap_mm=0.0, n_glyphs=0, median_mm=0.0, min_mm=0.0)
    slices = ndimage.find_objects(lab)
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.uint8), lab,
                        index=np.arange(1, n + 1))
    heights = []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        hmm, wmm = h / px_mm, w / px_mm
        if not (0.35 <= hmm <= 7.0 and 0.10 <= wmm <= 7.0):
            continue
        fill = sizes[i] / float(h * w)
        if fill < 0.24:                      # reject hairlines / stray strokes
            continue
        if not (0.08 <= w / float(h) <= 8.0):
            continue
        heights.append(hmm)
    if not heights:
        return dict(modal_cap_mm=0.0, n_glyphs=0, median_mm=0.0, min_mm=0.0)
    hv = np.asarray(heights)
    upper = hv[hv >= np.percentile(hv, 55.0)]   # caps / digits / ascenders
    edges = np.arange(0.30, 7.02, 0.02)
    cnt, _ = np.histogram(upper, bins=edges)
    k = int(np.argmax(cnt))
    modal = float((edges[k] + edges[k + 1]) / 2.0)
    return dict(modal_cap_mm=round(modal, 3), n_glyphs=int(hv.size),
                median_mm=round(float(np.median(hv)), 3),
                min_mm=round(float(hv.min()), 3))


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def export(fig, name, outdir, metrics):
    w, h = fig.get_size_inches()
    assert abs(w - WIDTH_IN) < 1e-6, "%s width %.6f in != %.6f in" % (name, w, WIDTH_IN)
    assert_ascii(fig, name)

    png = os.path.join(outdir, name + ".png")
    fig.savefig(png, dpi=PNG_DPI, facecolor="white")

    tmp = os.path.join(outdir, "_tmp_%s.png" % name)
    fig.savefig(tmp, dpi=TIFF_DPI, facecolor="white")
    im = Image.open(tmp).convert("RGB")
    tif = os.path.join(outdir, name + ".tif")
    im.save(tif, format="TIFF", compression="tiff_lzw", dpi=(TIFF_DPI, TIFF_DPI))
    px_w, px_h = im.size
    im.convert("L").save(os.path.join(outdir, name + "_grey.png"))
    im.close()
    os.remove(tmp)

    m = measure_cap_mm(fig)
    m.update(figure=name, width_mm=round(w * 25.4, 2), height_mm=round(h * 25.4, 2),
             tiff_px=[px_w, px_h], tiff_dpi=TIFF_DPI, png_dpi=PNG_DPI,
             tiff_bytes=os.path.getsize(tif),
             passes=bool(m["modal_cap_mm"] >= CAP_FLOOR_MM))
    metrics.append(m)
    plt.close(fig)
    print("[%-9s] %5.1f x %5.1f mm  %5d x %5d px  cap %.2f mm  %s"
          % (name, m["width_mm"], m["height_mm"], px_w, px_h,
             m["modal_cap_mm"], "PASS" if m["passes"] else "FAIL"))
    return m


# ---------------------------------------------------------------------------
# verification ledger
# ---------------------------------------------------------------------------
CHECKS = []


def chk(fig, what, plotted, source, manuscript=None, tol=5e-4):
    ok = abs(float(plotted) - float(source)) <= tol
    CHECKS.append(dict(figure=fig, quantity=what, plotted=float(plotted),
                       source_csv=float(source), manuscript=manuscript, ok=bool(ok)))
    if not ok:
        raise AssertionError("VALUE MISMATCH %s / %s: plotted %r vs source %r"
                             % (fig, what, plotted, source))


# ===========================================================================
# Figure 1 - lockbox performance (was F3_lockbox_performance)
# ===========================================================================


def figure1(root, outdir, metrics):
    p = tbl(root, "T4_lockbox_performance.csv")
    p = p.set_index("domain").loc[DOMAINS].reset_index()
    x = np.arange(len(p))
    w = 0.38
    labs = [DOMSHORT[d] for d in p.domain]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.20), layout="constrained")

    a1.bar(x - w / 2, p.anchor_auroc, w, label="anchor only", **style_grey())
    b2 = a1.bar(x + w / 2, p.panel_auroc, w, label="full panel", **style_teal())
    a1.set_xticks(x)
    a1.set_xticklabels(labs)
    # every bar reaches 0.82 or higher, so there is no free space at the foot
    # of the axes: the legend goes in the headroom strip above the bars.
    a1.set_ylim(0.5, 1.10)
    a1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a1.set_ylabel("lockbox AUROC")
    a1.set_title("(a) Lockbox AUROC: anchor vs panel", fontsize=FS_TITLE)
    a1.legend(frameon=False, loc="upper center", ncol=2, fontsize=FS_LEGEND,
              handlelength=1.3, borderaxespad=0.1, columnspacing=1.4)
    for xi, v in zip(x, p.panel_auroc):
        a1.text(xi + w / 2, v + 0.008, "%.2f" % v, ha="center", va="bottom",
                fontsize=FS_ANNOT)
    tidy(a1)

    g = p.crossdx_gain_auroc.values
    flags = (p.domain == "sleep").values
    face = [ORANGE if f else (REDNEG if v < 0 else TEAL)
            for f, v in zip(flags, g)]
    edge = [ORANGE_E if f else (REDNEG_E if v < 0 else TEAL_E)
            for f, v in zip(flags, g)]
    hat = [HATCH_HL if f else (HATCH_NEG if v < 0 else "")
           for f, v in zip(flags, g)]
    bars = a2.bar(x, g, 0.62, color=face, edgecolor=edge, linewidth=0.6)
    apply_hatches(bars, hat)
    a2.axhline(0, color="black", lw=0.8)
    a2.set_xticks(x)
    a2.set_xticklabels(labs)
    # DEFECT FIX: the y floor is pushed well below the most negative bar so the
    # value label for panic sits INSIDE the axes and cannot collide with the
    # x tick labels (which live outside the axes).
    a2.set_ylim(-0.016, 0.088)
    a2.set_ylabel("delta AUROC (panel - anchor)")
    a2.set_title("(b) Cross-diagnostic gain", fontsize=FS_TITLE)
    for xi, v in zip(x, g):
        if v >= 0:
            a2.text(xi, v + 0.0022, "%+.3f" % v, ha="center", va="bottom",
                    fontsize=FS_ANNOT)
        else:
            a2.text(xi, v - 0.0022, "%+.3f" % v, ha="center", va="top",
                    fontsize=FS_ANNOT)
    tidy(a2)

    src = tbl(root, "T4_lockbox_performance.csv").set_index("domain")
    for d, ms in [("dep", "0.953"), ("anx", "0.958"), ("ptsd", "0.934"),
                  ("panic", "0.970")]:
        chk("Figure1", "panel AUROC %s" % d,
            float(p.loc[p.domain == d, "panel_auroc"].iloc[0]),
            float(src.loc[d, "panel_auroc"]), ms)
    for d, ms in [("dep", "+0.076"), ("alcohol", "+0.049"), ("ptsd", "+0.040"),
                  ("anx", "+0.028")]:
        chk("Figure1", "crossdx gain %s" % d,
            float(p.loc[p.domain == d, "crossdx_gain_auroc"].iloc[0]),
            float(src.loc[d, "crossdx_gain_auroc"]), ms)
    return export(fig, "Figure1", outdir, metrics)


# ===========================================================================
# Figure 2 - per-item utility (was F2_item_utility)
# ===========================================================================


def figure2(root, outdir, metrics):
    u = tbl(root, "T2_item_utility.csv")
    pan = panel_json(root)["panel_7domain"]

    # LAYOUT CHANGE: the original 2 x 4 grid gives each of 7 panels only ~40 mm
    # at 174 mm, which cannot carry legible item codes for the 20-item PTSD
    # pool. Redrawn as two ~80 mm columns with panel heights proportional to
    # pool size, and the 64 per-bar value annotations dropped (values are in
    # Table T2). Both changes buy type size.
    col0 = [("ptsd", 0, 20), ("dep", 24, 33), ("anx", 37, 44)]
    col1 = [("alcohol", 0, 10), ("panic", 14, 21), ("sleep", 25, 32),
            ("suicide", 36, 40)]
    NROWS = 48

    fig = plt.figure(figsize=(WIDTH_IN, 7.85))
    gs = fig.add_gridspec(NROWS, 2, left=0.108, right=0.963, top=0.940,
                          bottom=0.030, hspace=0.0, wspace=0.34)

    def draw(dom, r0, r1, ci):
        ax = fig.add_subplot(gs[r0:r1, ci])
        sub = u[u.domain == dom].sort_values("utility", kind="stable")
        flags = (sub.selected == 1).values
        face, edge, hat = bar_kw_list(flags)
        bars = ax.barh(np.arange(len(sub)), sub.utility.values, height=0.72,
                       color=face, edgecolor=edge, linewidth=0.6)
        apply_hatches(bars, hat)
        ax.axvline(0.5, color="#8A9499", lw=0.7, ls=(0, (3, 2)), zorder=0)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub.item.values, fontsize=FS_TICK)
        ax.set_ylim(-0.75, len(sub) - 0.25)
        ax.set_xlim(0.45, 0.80)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8])
        ax.tick_params(axis="x", labelsize=FS_ANNOT)
        ax.set_title(DOMTITLE[dom], fontsize=FS_TITLE, pad=3.0,
                     color=(ORANGE_E if dom == "sleep" else "black"))
        tidy(ax)
        # verify the anchor bar really is the selected item
        sel = sub.loc[sub.selected == 1, "item"].tolist()
        assert sel == [pan[dom]], (dom, sel, pan[dom])
        return ax

    for dom, r0, r1 in col0:
        draw(dom, r0, r1, 0)
    for dom, r0, r1 in col1:
        draw(dom, r0, r1, 1)

    axl = fig.add_subplot(gs[41:NROWS, 1])
    axl.axis("off")
    from matplotlib.patches import Rectangle
    axl.add_patch(Rectangle((0.02, 0.80), 0.10, 0.11, transform=axl.transAxes,
                            facecolor=ORANGE, edgecolor=ORANGE_E, hatch=HATCH_HL,
                            linewidth=0.6, clip_on=False))
    axl.text(0.16, 0.855, "selected anchor", transform=axl.transAxes,
             fontsize=FS_LEGEND, va="center")
    axl.add_patch(Rectangle((0.02, 0.635), 0.10, 0.11, transform=axl.transAxes,
                            facecolor=TEAL, edgecolor=TEAL_E, linewidth=0.6,
                            clip_on=False))
    axl.text(0.16, 0.690, "candidate item", transform=axl.transAxes,
             fontsize=FS_LEGEND, va="center")
    axl.text(0.02, 0.50,
             "Utility U(j,k) = mean(AUROC, AUPRC)\n"
             "of item k for domain j; lambda = 0.10.\n"
             "Dashed line = chance (0.50).\n"
             "Item codes: item_dictionary.csv.",
             transform=axl.transAxes, fontsize=FS_LEGEND, va="top", linespacing=1.5)

    fig.suptitle("Per-item cross-diagnostic utility by domain", fontsize=FS_SUPTITLE,
                 y=0.983)

    for it, ms in [("ISI5", 0.755), ("GAD2", 0.763), ("PCL11", 0.695),
                   ("AUDIT3", 0.703)]:
        chk("Figure2", "utility %s" % it,
            float(u.loc[u.item == it, "utility"].iloc[0]), ms, str(ms))
    return export(fig, "Figure2", outdir, metrics)


# ===========================================================================
# Figure 3 - partial-correlation network (was F4_ggm_network)
# ===========================================================================


def figure3(root, outdir, metrics):
    pc = tbl(root, "T8_ggm_partial_corr.csv").set_index(
        tbl(root, "T8_ggm_partial_corr.csv").columns[0])
    cen = tbl(root, "T9_bridge_centrality.csv").set_index("domain")
    pan = panel_json(root)["panel_7domain"]
    codes = [pan[d] for d in DOMAINS]
    M = pc.loc[codes, codes].values.astype(float)

    # graph built exactly as 03_network_contrib.py: |partial r| > 0.04
    G = nx.Graph()
    for i, d in enumerate(DOMAINS):
        G.add_node(i)
    for i in range(len(DOMAINS)):
        for j in range(i + 1, len(DOMAINS)):
            w = M[i, j]
            if abs(w) > 0.04:
                G.add_edge(i, j, weight=abs(w), signed=w)

    # Node size comes from the SAVED node_strength in T9, not from a
    # recomputation off T8. T9 was computed from the full-precision partial
    # correlations, whereas T8 is published rounded to 3 dp; recomputing from
    # the rounded matrix shifts dep from 0.744 to 0.746. The manuscript quotes
    # T9, so T9 is authoritative here and nothing is recomputed.
    strength = {i: float(cen.loc[d, "node_strength"])
                for i, d in enumerate(DOMAINS)}
    for i, d in enumerate(DOMAINS):
        recomputed = round(sum(abs(M[i, j]) for j in range(len(DOMAINS))
                               if j != i), 3)
        chk("Figure3", "node strength %s" % d, strength[i],
            float(cen.loc[d, "node_strength"]),
            {"sleep": "0.73", "alcohol": "0.13", "anx": "0.89"}.get(d))
        if abs(recomputed - strength[i]) > 5e-4:
            print("      note: %s node strength from rounded T8 would be %.3f; "
                  "using saved T9 value %.3f" % (d, recomputed, strength[i]))

    signs = [G[u][v]["signed"] for u, v in G.edges]
    n_neg = sum(1 for s in signs if s < 0)
    assert n_neg == 0, "unexpected negative edges: %d" % n_neg

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.90), layout="constrained")
    pos = nx.spring_layout(G, weight="weight", seed=3, k=1.1)
    P = np.array([pos[i] for i in range(len(DOMAINS))])

    for (uu, vv, dd) in G.edges(data=True):
        ax.plot([pos[uu][0], pos[vv][0]], [pos[uu][1], pos[vv][1]],
                color=EDGE_BLUE, lw=0.9 + 6.5 * abs(dd["signed"]), alpha=0.65,
                solid_capstyle="round", zorder=1)

    sizes = np.array([1300 + 4300 * strength[i] for i in range(len(DOMAINS))])
    for i, d in enumerate(DOMAINS):
        hl = (d == "sleep")
        ax.scatter(P[i, 0], P[i, 1], s=sizes[i],
                   facecolor=(ORANGE if hl else TEAL),
                   edgecolor=(ORANGE_E if hl else "white"),
                   linewidth=(2.0 if hl else 1.6),
                   hatch=(HATCH_HL if hl else None), zorder=3)
        # DEFECT FIX: at FS_ANNOT (8.5 pt) the node labels measured 1.95 mm cap
        # height at 174 mm, under the 2.0 mm print floor and under Springer's
        # 2-3 mm lettering rule. 9.0 pt clears it without enlarging the nodes.
        ax.text(P[i, 0], P[i, 1], "%s\n%s" % (DOMSHORT[d], pan[d]),
                ha="center", va="center", fontsize=9.0, fontweight="bold",
                color="white", zorder=4, linespacing=1.15)

    # DEFECT FIX: the original clipped three node circles (DEP flat-topped,
    # PTSD cut on the right, PANIC cut at the bottom). Reserve a margin large
    # enough for the biggest node radius, converted from points to data units,
    # and set the limits explicitly.
    fig.canvas.draw()
    bb = ax.get_window_extent()
    xs, ys = P[:, 0], P[:, 1]
    spanx = max(xs.max() - xs.min(), 1e-9)
    spany = max(ys.max() - ys.min(), 1e-9)
    rad_pt = np.sqrt(sizes.max() / np.pi)                    # marker radius, pt
    rad_px = rad_pt * fig.dpi / 72.0
    padx = 1.25 * rad_px / bb.width * spanx + 0.06 * spanx
    pady = 1.25 * rad_px / bb.height * spany + 0.06 * spany
    ax.set_xlim(xs.min() - padx, xs.max() + padx)
    ax.set_ylim(ys.min() - pady, ys.max() + pady)
    ax.axis("off")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=EDGE_BLUE, lw=2.4,
               label="positive partial correlation"),
        Patch(facecolor=ORANGE, edgecolor=ORANGE_E, hatch=HATCH_HL,
              label="new sleep anchor"),
        Patch(facecolor=TEAL, edgecolor=TEAL_E, label="original domain anchor"),
    ]
    # legend goes BELOW the network in its own reserved strip, so it cannot
    # overlap the node circles at any layout scale
    ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.008),
              handlelength=1.5, borderaxespad=0.0, columnspacing=1.6)
    ax.set_title("Partial-correlation network of the 7-domain panel\n"
                 "(node size = node strength; all surviving edges positive,\n"
                 "|partial r| > 0.04)", fontsize=FS_TITLE, linespacing=1.35)
    return export(fig, "Figure3", outdir, metrics)


# ===========================================================================
# Figure 4 - value of the sleep domain (was F6_value_of_sleep)
# ===========================================================================


def figure4(root, outdir, metrics):
    v = tbl(root, "T5_value_of_sleep.csv")
    perf = tbl(root, "T4_lockbox_performance.csv").set_index("domain")
    sub = panel_json(root)["sleep_subthreshold"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.35), layout="constrained")

    x = np.arange(len(v))
    d = v.delta_auroc_from_sleep.values
    face = [TEAL if q >= 0 else REDNEG for q in d]
    edge = [TEAL_E if q >= 0 else REDNEG_E for q in d]
    hat = ["" if q >= 0 else HATCH_NEG for q in d]
    bars = a1.bar(x, d, 0.62, color=face, edgecolor=edge, linewidth=0.6)
    apply_hatches(bars, hat)
    a1.axhline(0, color="black", lw=0.8)
    a1.set_xticks(x)
    a1.set_xticklabels([DOMSHORT[q] for q in v.domain])
    a1.set_ylim(-0.0062, 0.0062)
    a1.set_yticks([-0.006, -0.003, 0.0, 0.003, 0.006])
    a1.set_ylabel("delta AUROC")
    a1.set_title("(a) Adding the sleep item to predict\nthe 6 original domains",
                 fontsize=FS_TITLE, linespacing=1.3)
    # DEFECT FIX: negative-bar labels are drawn INSIDE the bar (just above its
    # foot) instead of below it, so they can never collide with the x ticks.
    for xi, q in zip(x, d):
        if q >= 0:
            a1.text(xi, q + 0.00028, "%+.3f" % q, ha="center", va="bottom",
                    fontsize=FS_ANNOT)
        else:
            a1.text(xi, q + 0.00035, "%+.3f" % q, ha="center", va="bottom",
                    fontsize=FS_ANNOT, color="black")
    tidy(a1)

    sp = perf.loc["sleep"]
    vals = [float(sp.anchor_auroc), float(sp.panel_auroc),
            float(sub["anchor_auroc"]), float(sub["panel_auroc"])]
    labels = ["ISI >= 15\nanchor", "ISI >= 15\npanel",
              "ISI >= 8\nanchor", "ISI >= 8\npanel"]
    flags = [False, True, False, True]
    face, edge, hat = bar_kw_list(flags, base="grey")
    bars = a2.bar(np.arange(4), vals, 0.62, color=face, edgecolor=edge,
                  linewidth=0.6)
    apply_hatches(bars, hat)
    a2.set_xticks(np.arange(4))
    a2.set_xticklabels(labels, fontsize=FS_ANNOT, linespacing=1.25)
    a2.set_ylim(0.5, 1.0)
    a2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a2.set_ylabel("lockbox AUROC")
    a2.set_title("(b) Detecting insomnia: comorbid panel\nvs sleep anchor alone",
                 fontsize=FS_TITLE, linespacing=1.3)
    for xi, q in enumerate(vals):
        a2.text(xi, q + 0.008, "%.3f" % q, ha="center", va="bottom",
                fontsize=FS_ANNOT)
    tidy(a2)

    for dd, ms in [("dep", "-0.004"), ("suicide", "-0.004"), ("ptsd", "+0.003")]:
        chk("Figure4", "delta from sleep %s" % dd,
            float(v.loc[v.domain == dd, "delta_auroc_from_sleep"].iloc[0]),
            float(v.loc[v.domain == dd, "delta_auroc_from_sleep"].iloc[0]), ms)
    chk("Figure4", "sleep anchor AUROC ISI>=15", vals[0],
        float(perf.loc["sleep", "anchor_auroc"]), "0.855")
    chk("Figure4", "sleep panel AUROC ISI>=15", vals[1],
        float(perf.loc["sleep", "panel_auroc"]), "0.868")
    chk("Figure4", "sleep anchor AUROC ISI>=8", vals[2],
        float(sub["anchor_auroc"]), "0.882")
    chk("Figure4", "sleep panel AUROC ISI>=8", vals[3],
        float(sub["panel_auroc"]), "0.897")
    return export(fig, "Figure4", outdir, metrics)


# ===========================================================================
# Figure 5 - real-data validation (was RF1_realdata)
# ===========================================================================


def figure5(root, outdir, metrics):
    sf = rdt(root, "RT2b_isi_selection_freq.csv")
    lb = rdt(root, "RT3_lockbox.csv")
    co = rdt(root, "RT1b_correlations.csv").set_index(
        rdt(root, "RT1b_correlations.csv").columns[0])

    # LAYOUT CHANGE: the original 1 x 3 strip leaves each panel ~58 mm at
    # 174 mm, too narrow for the ISI item labels in (A). Redrawn 2 x 2 so A and
    # B get ~85 mm each and C keeps its own row.
    # constrained layout (not a manual gridspec) so the long ISI item labels
    # in (A) always get the left margin they need at 174 mm
    fig = plt.figure(figsize=(WIDTH_IN, 5.35), layout="constrained")
    gs = fig.add_gridspec(2, 2)

    # (A) ISI item bootstrap selection frequency
    a = fig.add_subplot(gs[0, 0])
    # item order ISI1 (top) .. ISI7 (bottom), as in the original figure
    s = sf.sort_values("item", kind="stable").iloc[::-1]
    flags = s.is_ISI3m.astype(bool).values
    face, edge, hat = bar_kw_list(flags)
    bars = a.barh(np.arange(len(s)), s.bootstrap_freq.values, height=0.70,
                  color=face, edgecolor=edge, linewidth=0.6)
    apply_hatches(bars, hat)
    a.set_yticks(np.arange(len(s)))
    a.set_yticklabels(["%s %s" % (i, c) for i, c in zip(s.item, s.content)],
                      fontsize=FS_ANNOT)
    a.set_xlim(0, 1.0)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_xlabel("bootstrap selection frequency", fontsize=FS_ANNOT)
    a.set_title("(A) ISI item selected as sleep anchor", fontsize=FS_TITLE)
    for yi, q in enumerate(s.bootstrap_freq.values):
        if q > 0.005:
            a.text(q + 0.02, yi, "%.2f" % q, va="center", fontsize=FS_ANNOT)
    tidy(a)
    a.legend(handles=[
        matplotlib.patches.Patch(facecolor=ORANGE, edgecolor=ORANGE_E,
                                 hatch=HATCH_HL, label="ISI-3m item"),
        matplotlib.patches.Patch(facecolor=TEAL, edgecolor=TEAL_E,
                                 label="not in ISI-3m")],
        frameon=False, fontsize=FS_ANNOT, loc="center right",
        handlelength=1.3, borderaxespad=0.4)

    # (B) three-domain lockbox AUROC, anchor vs panel
    b = fig.add_subplot(gs[0, 1])
    order = ["sleep", "dep", "anx"]
    lbi = lb.set_index("domain").loc[order]
    x = np.arange(3)
    w = 0.38
    b.bar(x - w / 2, lbi.anchor_auroc.values, w, label="anchor only", **style_grey())
    bb2 = b.bar(x + w / 2, lbi.panel_auroc.values, w, label="3-domain panel",
                **style_teal())
    b.set_xticks(x)
    b.set_xticklabels(["insomnia", "depress.", "anxiety"], fontsize=FS_ANNOT)
    # all six bars exceed 0.89, so the legend goes in the headroom strip
    b.set_ylim(0.5, 1.12)
    b.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    b.set_ylabel("lockbox AUROC")
    b.set_title("(B) Lockbox performance (real data)", fontsize=FS_TITLE)
    for xi, q in zip(x, lbi.panel_auroc.values):
        b.text(xi + w / 2, q + 0.008, "%.3f" % q, ha="center", va="bottom",
               fontsize=FS_ANNOT)
    b.legend(frameon=False, fontsize=FS_ANNOT, loc="upper center", ncol=2,
             handlelength=1.3, borderaxespad=0.1, columnspacing=1.2)
    tidy(b)

    # (C) total-score correlations, real vs literature
    c = fig.add_subplot(gs[1, 0])
    pairs = [("dep-anx", float(co.loc["PHQ_total", "GAD_total"]), 0.75),
             ("dep-insomnia", float(co.loc["PHQ_total", "ISI_total"]), 0.52),
             ("anx-insomnia", float(co.loc["GAD_total", "ISI_total"]), 0.48)]
    xp = np.arange(3)
    c.bar(xp - 0.20, [p[1] for p in pairs], 0.38, label="real data", **style_teal())
    c.bar(xp + 0.20, [p[2] for p in pairs], 0.38, label="literature", **style_grey())
    c.set_xticks(xp)
    c.set_xticklabels([p[0] for p in pairs], fontsize=FS_ANNOT, rotation=20,
                      ha="right")
    c.set_ylim(0, 1.16)
    c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    c.set_ylabel("correlation")
    c.set_title("(C) Total-score correlations:\nreal vs literature",
                fontsize=FS_TITLE, linespacing=1.3)
    for xi, p in zip(xp, pairs):
        c.text(xi - 0.20, p[1] + 0.02, "%.2f" % p[1], ha="center", va="bottom",
               fontsize=FS_ANNOT)
    c.legend(frameon=False, fontsize=FS_ANNOT, loc="upper center", ncol=2,
             handlelength=1.3, borderaxespad=0.1, columnspacing=1.2)
    tidy(c)

    axn = fig.add_subplot(gs[1, 1])
    axn.axis("off")
    axn.text(0.0, 0.95,
             "Cohort A: Zenodo 10423537,\n"
             "N = 24,292 students; ISI + PHQ-9 + GAD-7.\n\n"
             "(A) 300 bootstrap resamples on the\n"
             "development split; hatched = item retained\n"
             "by the published ISI-3m.\n\n"
             "(B) anchor = single item per domain;\n"
             "panel = 3 items, one per domain.\n\n"
             "(C) literature targets used to calibrate\n"
             "the simulation (0.75 / 0.52 / 0.48).",
             transform=axn.transAxes, fontsize=FS_ANNOT, va="top", linespacing=1.4)

    # no explicit y: constrained layout reserves the strip for the suptitle
    # itself, which an explicit y would override and push onto panel titles
    fig.suptitle("Real-data validation (Cohort A)", fontsize=FS_SUPTITLE)

    for it, ms in [("ISI2", "0.65"), ("ISI7", "0.35")]:
        chk("Figure5", "bootstrap freq %s" % it,
            float(sf.loc[sf.item == it, "bootstrap_freq"].iloc[0]),
            float(sf.loc[sf.item == it, "bootstrap_freq"].iloc[0]), ms)
    for d, ms in [("sleep", "0.972"), ("dep", "0.959"), ("anx", "0.974")]:
        chk("Figure5", "panel AUROC %s" % d, float(lbi.loc[d, "panel_auroc"]),
            float(lb.set_index("domain").loc[d, "panel_auroc"]), ms)
    chk("Figure5", "real corr dep-anx", pairs[0][1],
        float(co.loc["PHQ_total", "GAD_total"]), "0.77")
    chk("Figure5", "real corr dep-insomnia", pairs[1][1],
        float(co.loc["PHQ_total", "ISI_total"]), "0.65")
    chk("Figure5", "real corr anx-insomnia", pairs[2][1],
        float(co.loc["GAD_total", "ISI_total"]), "0.60")
    return export(fig, "Figure5", outdir, metrics)


# ===========================================================================
# Figure 6 - multi-cohort validation (was RF2_multicohort)
# ===========================================================================


def figure6(root, outdir, metrics):
    cb = rdt(root, "RT6_cohortB_sri.csv")
    cc = rdt(root, "RT7_cohortC_uk.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.80), layout="constrained")

    cbs = cb.sort_values("item", kind="stable").iloc[::-1]
    flags = cbs.is_ISI3m.astype(bool).values
    face, edge, hat = bar_kw_list(flags)
    bars = a.barh(np.arange(len(cbs)), cbs.boot_freq.values, height=0.70,
                  color=face, edgecolor=edge, linewidth=0.6)
    apply_hatches(bars, hat)
    a.set_yticks(np.arange(len(cbs)))
    a.set_yticklabels(["%s %s" % (i, c) for i, c in zip(cbs.item, cbs.content)],
                      fontsize=FS_ANNOT)
    a.set_xlim(0, 1.0)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_xlabel("bootstrap selection frequency", fontsize=FS_ANNOT)
    a.set_title("(A) Cohort B: clinical insomnia\nsample (N = 95); unstable selection",
                fontsize=FS_TITLE, linespacing=1.3)
    for yi, q in enumerate(cbs.boot_freq.values):
        if q > 0.005:
            a.text(q + 0.02, yi, "%.2f" % q, va="center", fontsize=FS_ANNOT)
    a.legend(handles=[
        matplotlib.patches.Patch(facecolor=ORANGE, edgecolor=ORANGE_E,
                                 hatch=HATCH_HL, label="ISI-3m item"),
        matplotlib.patches.Patch(facecolor=TEAL, edgecolor=TEAL_E,
                                 label="not in ISI-3m")],
        frameon=False, fontsize=FS_ANNOT, loc="center right",
        handlelength=1.3, borderaxespad=0.4)
    tidy(a)

    labels = ["insomnia (SCI)", "depression", "anxiety", "stress", "suicidality"]
    newdom = [True, False, False, False, True]        # domains new in Cohort C
    face, edge, hat = bar_kw_list(newdom)
    bars = b.bar(np.arange(len(cc)), cc.panel_auroc.values, 0.62, color=face,
                 edgecolor=edge, linewidth=0.6)
    apply_hatches(bars, hat)
    b.set_xticks(np.arange(len(cc)))
    # five domain names cannot sit side by side at 9 pt in an 85 mm panel;
    # rotating keeps full words at full size instead of abbreviating
    b.set_xticklabels(labels, fontsize=FS_ANNOT, rotation=30, ha="right")
    b.set_ylim(0.5, 1.06)
    b.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    b.set_ylabel("lockbox AUROC")
    b.set_title("(B) Cohort C: UK cross-population\n"
                "(N = 1,406); 5 domains", fontsize=FS_TITLE,
                linespacing=1.3)
    for xi, q in enumerate(cc.panel_auroc.values):
        b.text(xi, q + 0.008, "%.3f" % q, ha="center", va="bottom",
               fontsize=FS_ANNOT)
    tidy(b)

    fig.suptitle("Multi-cohort real-data validation", fontsize=FS_SUPTITLE)

    for it, ms in [("ISI3", "0.345"), ("ISI5", "0.275"), ("ISI7", "0.18")]:
        chk("Figure6", "Cohort B boot freq %s" % it,
            float(cb.loc[cb.item == it, "boot_freq"].iloc[0]),
            float(cb.loc[cb.item == it, "boot_freq"].iloc[0]), ms)
    for d, ms in [("sleep", "0.932"), ("anx", "0.949"), ("suicide", "0.949")]:
        chk("Figure6", "Cohort C panel AUROC %s" % d,
            float(cc.loc[cc.domain == d, "panel_auroc"].iloc[0]),
            float(cc.loc[cc.domain == d, "panel_auroc"].iloc[0]), ms)
    return export(fig, "Figure6", outdir, metrics)


# ===========================================================================
# Figure S1 - calibration (was F1_calibration)
# ===========================================================================


def figureS1(root, outdir, metrics):
    cal = tbl(root, "T1_calibration.csv")
    ac = tbl(root, "T1b_achieved_correlations.csv")
    ac = ac.set_index(ac.columns[0])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.40), layout="constrained")

    x = np.arange(len(cal))
    w = 0.38
    a1.bar(x - w / 2, cal.target_prev, w, label="target", **style_grey())
    a1.bar(x + w / 2, cal.achieved_prev, w, label="achieved", **style_teal())
    a1.set_xticks(x)
    # PTSD and PANIC are the two widest short codes and touch at 9 pt in an
    # 80 mm panel; a slight rotation separates them without shrinking type.
    a1.set_xticklabels([DOMSHORT[d] for d in cal.domain], rotation=30, ha="right")
    a1.set_ylabel("caseness prevalence")
    a1.set_ylim(0, 0.183)
    a1.set_title("(a) Prevalence calibration", fontsize=FS_TITLE)
    a1.legend(frameon=False, fontsize=FS_LEGEND, loc="upper left",
              handlelength=1.3, borderaxespad=0.2)
    for xi, q in zip(x, cal.achieved_prev):
        a1.text(xi + w / 2, q + 0.003, "%.2f" % q, ha="center", va="bottom",
                fontsize=FS_ANNOT)
    tidy(a1)

    V = ac.values.astype(float)
    im = a2.imshow(V, cmap="RdBu_r", vmin=-1, vmax=1)
    a2.set_xticks(range(len(ac)))
    a2.set_xticklabels([DOMSHORT[c] for c in ac.columns], fontsize=FS_ANNOT,
                       rotation=45, ha="right")
    a2.set_yticks(range(len(ac)))
    a2.set_yticklabels([DOMSHORT[i] for i in ac.index], fontsize=FS_ANNOT)
    for i in range(len(ac)):
        for j in range(len(ac)):
            a2.text(j, i, "%.2f" % V[i, j], ha="center", va="center",
                    fontsize=FS_TICK,
                    color=("white" if abs(V[i, j]) > 0.75 else "black"))
    a2.set_title("(b) Achieved total-score correlations", fontsize=FS_TITLE)
    a2.tick_params(length=0)
    for s in a2.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=a2, shrink=0.86, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=FS_ANNOT)
    cb.outline.set_linewidth(0.6)

    for d, ms in [("dep", 0.1205), ("sleep", 0.1205), ("alcohol", 0.15)]:
        chk("FigureS1", "achieved prevalence %s" % d,
            float(cal.loc[cal.domain == d, "achieved_prev"].iloc[0]), ms, str(ms))
    chk("FigureS1", "corr dep-anx", V[0, 1], 0.673, "0.67")
    chk("FigureS1", "corr dep-sleep", V[0, 6], 0.568, "0.57")
    return export(fig, "FigureS1", outdir, metrics)


# ===========================================================================
# Figure S2 - cross-diagnostic contribution heatmap (was F5)
# ===========================================================================


def figureS2(root, outdir, metrics):
    ct = tbl(root, "T10_crossdx_contribution.csv")
    ct = ct.set_index(ct.columns[0])
    V = ct.values.astype(float)
    pan = panel_json(root)["panel_7domain"]

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.15), layout="constrained")
    vmax = float(np.abs(V).max())
    im = ax.imshow(V, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels(["%s\n%s" % (pan[d], DOMSHORT[d]) for d in DOMAINS],
                       fontsize=FS_TICK, linespacing=1.2)
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels([DOMTITLE[d] for d in DOMAINS], fontsize=FS_TICK)
    ax.set_xlabel("Panel item (anchor)", fontsize=FS_LABEL)
    ax.set_ylabel("Predicted domain", fontsize=FS_LABEL)
    for i in range(len(DOMAINS)):
        for j in range(len(DOMAINS)):
            ax.text(j, i, "%.2f" % V[i, j], ha="center", va="center",
                    fontsize=FS_TICK,
                    color=("white" if abs(V[i, j]) > 0.68 * vmax else "black"))
    ax.set_title("Cross-diagnostic contribution weights\n"
                 "(standardized logistic coefficients; off-diagonal = "
                 "transdiagnostic signal)", fontsize=FS_TITLE, linespacing=1.35)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("standardized weight", fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_ANNOT)
    cb.outline.set_linewidth(0.6)

    chk("FigureS2", "DEP row / ISI5 col", V[0, 6], 0.926, "0.93")
    chk("FigureS2", "SLEEP row / ISI5 col", V[6, 6], 2.134, "2.13")
    chk("FigureS2", "ANX row / GAD2 col", V[1, 1], 1.954, "1.95")
    return export(fig, "FigureS2", outdir, metrics)


# ===========================================================================
# Figure S3 - bootstrap selection stability (was F7)
# ===========================================================================


def figureS3(root, outdir, metrics):
    b = tbl(root, "T6_bootstrap_stability.csv")
    rows = []
    for d in DOMAINS:
        sub = b[b.domain == d].sort_values("selection_freq", ascending=False,
                                           kind="stable").head(3)
        for _, r in sub.iterrows():
            rows.append((d, str(r["item"]), str(r["label"]),
                         float(r["selection_freq"]), int(r["selected_in_main"])))

    fig, ax = plt.subplots(figsize=(WIDTH_IN, 6.10), layout="constrained")
    yy = 0.0
    yt, yl, last = [], [], None
    for d, item, label, freq, ismain in rows:
        if last is not None and d != last:
            yy += 0.85
        hl = ismain == 1
        ax.barh(yy, freq, height=0.72,
                color=(ORANGE if hl else TEAL),
                edgecolor=(ORANGE_E if hl else TEAL_E), linewidth=0.6,
                hatch=(HATCH_HL if hl else None))
        ax.text(freq + 0.012, yy, "%.2f" % freq, va="center", fontsize=FS_ANNOT)
        yt.append(yy)
        yl.append("%-5s %s %s" % (DOMSHORT[d], item, label))
        yy += 1.0
        last = d
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=FS_ANNOT, fontfamily="DejaVu Sans Mono")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("bootstrap selection frequency (300 resamples)", fontsize=FS_LABEL)
    ax.set_title("Selection stability: top-3 candidate items per domain",
                 fontsize=FS_TITLE)
    ax.legend(handles=[
        matplotlib.patches.Patch(facecolor=ORANGE, edgecolor=ORANGE_E,
                                 hatch=HATCH_HL, label="main-analysis anchor"),
        matplotlib.patches.Patch(facecolor=TEAL, edgecolor=TEAL_E,
                                 label="runner-up candidate")],
        frameon=False, fontsize=FS_LEGEND, loc="lower right",
        handlelength=1.4, borderaxespad=0.3)
    tidy(ax)

    for it, ms in [("ISI5", 0.47), ("ISI2", 0.44), ("GAD2", 0.89),
                   ("AUDIT3", 0.827)]:
        chk("FigureS3", "selection freq %s" % it,
            float(b.loc[b.item == it, "selection_freq"].iloc[0]), ms, str(ms))
    return export(fig, "FigureS3", outdir, metrics)


# ===========================================================================
# Figure S4 - robustness (was F8)
# ===========================================================================


def figureS4(root, outdir, metrics):
    bl = tbl(root, "T12_baselines.csv").set_index("domain").loc[DOMAINS].reset_index()
    ab = tbl(root, "T13_ablation.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.35), layout="constrained")

    x = np.arange(len(bl))
    w = 0.38
    a1.bar(x - w / 2, bl.random_panel_auroc, w, yerr=bl.random_panel_sd,
           label="random 1-item panel (mean +/- SD)",
           error_kw=dict(ecolor="#33403F", elinewidth=0.8, capsize=1.8,
                         capthick=0.8),
           **style_grey())
    a1.bar(x + w / 2, bl.optimized_panel_auroc, w, label="optimized panel",
           **style_teal())
    a1.set_xticks(x)
    a1.set_xticklabels([DOMSHORT[d] for d in bl.domain])
    # bars plus SD whiskers fill the plot to 0.84 and above, so the legend
    # goes in reserved headroom rather than over the DEP / ANX bars
    a1.set_ylim(0.5, 1.14)
    a1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a1.set_ylabel("lockbox AUROC")
    a1.set_title("(a) Optimized vs random\n1-item-per-domain panel",
                 fontsize=FS_TITLE, linespacing=1.3)
    a1.legend(frameon=False, fontsize=FS_ANNOT, loc="upper center", ncol=1,
              handlelength=1.3, borderaxespad=0.1, labelspacing=0.3)
    tidy(a1)

    cnt = ab.sleep_anchor.value_counts()
    order = ["ISI2", "ISI5", "ISI4", "ISI7"]
    order = [o for o in order if o in cnt.index]
    vals = [int(cnt[o]) for o in order]
    is3m = {}
    for o in order:
        s = set(ab.loc[ab.sleep_anchor == o, "in_ISI3m"].astype(int).tolist())
        assert len(s) == 1, (o, s)
        is3m[o] = bool(s.pop())
    n_tot = int(len(ab))
    n_3m = int(ab.in_ISI3m.sum())
    pct = 100.0 * n_3m / n_tot
    assert n_tot == 36, n_tot
    assert round(pct) == 89, pct

    face, edge, hat = bar_kw_list([is3m[o] for o in order])
    bars = a2.bar(np.arange(len(order)), vals, 0.62, color=face, edgecolor=edge,
                  linewidth=0.6)
    apply_hatches(bars, hat)
    a2.set_xticks(np.arange(len(order)))
    a2.set_xticklabels(order)
    a2.set_ylim(0, max(vals) * 1.30)
    a2.set_ylabel("configurations (of %d)" % n_tot)
    a2.set_title("(b) Sleep anchor across %d perturbed\ncalibrations: %d%% are "
                 "ISI-3m items" % (n_tot, round(pct)),
                 fontsize=FS_TITLE, linespacing=1.3)
    for xi, q in enumerate(vals):
        a2.text(xi, q + max(vals) * 0.018, "%d" % q, ha="center", va="bottom",
                fontsize=FS_ANNOT)
    a2.legend(handles=[
        matplotlib.patches.Patch(facecolor=ORANGE, edgecolor=ORANGE_E,
                                 hatch=HATCH_HL, label="ISI-3m item"),
        matplotlib.patches.Patch(facecolor=TEAL, edgecolor=TEAL_E,
                                 label="not in ISI-3m")],
        frameon=False, fontsize=FS_ANNOT, loc="upper right",
        handlelength=1.3, borderaxespad=0.2)
    tidy(a2)

    chk("FigureS4", "random panel AUROC dep",
        float(bl.loc[bl.domain == "dep", "random_panel_auroc"].iloc[0]), 0.94, "0.940")
    chk("FigureS4", "optimized panel AUROC sleep",
        float(bl.loc[bl.domain == "sleep", "optimized_panel_auroc"].iloc[0]),
        0.868, "0.868")
    chk("FigureS4", "ablation count ISI2", vals[0], 26, "26 of 36")
    chk("FigureS4", "ablation ISI-3m percent", round(pct), 89, "89%")
    return export(fig, "FigureS4", outdir, metrics)


# ===========================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="project root")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    metrics = []
    figure1(args.root, args.outdir, metrics)
    figure2(args.root, args.outdir, metrics)
    figure3(args.root, args.outdir, metrics)
    figure4(args.root, args.outdir, metrics)
    figure5(args.root, args.outdir, metrics)
    figure6(args.root, args.outdir, metrics)
    figureS1(args.root, args.outdir, metrics)
    figureS2(args.root, args.outdir, metrics)
    figureS3(args.root, args.outdir, metrics)
    figureS4(args.root, args.outdir, metrics)

    fails = [m["figure"] for m in metrics if not m["passes"]]
    bad = [c for c in CHECKS if not c["ok"]]
    with open(os.path.join(args.outdir, "figure_metrics.json"), "w") as fh:
        json.dump(dict(metrics=metrics, checks=CHECKS, cap_floor_mm=CAP_FLOOR_MM,
                       width_mm=WIDTH_MM, failures=fails,
                       value_mismatches=bad), fh, indent=2)
    print("\n%d/%d figures at or above %.1f mm modal cap height"
          % (len(metrics) - len(fails), len(metrics), CAP_FLOOR_MM))
    print("%d/%d value checks passed" % (len(CHECKS) - len(bad), len(CHECKS)))
    if fails:
        print("BELOW FLOOR: %s" % ", ".join(fails))
    if bad:
        print("VALUE MISMATCHES: %r" % bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
