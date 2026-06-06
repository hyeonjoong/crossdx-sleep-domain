# -*- coding: utf-8 -*-
"""
04_figures.py — summary figures from the saved result tables (readable, publication style).
  F1_calibration.png        target vs achieved prevalence + correlation recovery
  F2_item_utility.png       per-domain item utility (short codes; anchor highlighted)
  F3_lockbox_performance.png panel vs anchor AUROC + cross-diagnostic gain
  F6_value_of_sleep.png     incremental value of the sleep domain (both directions)
  F7_bootstrap_stability.png selection frequency (top items per domain)
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TBL = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
RES = os.path.join(ROOT, "results")
TEAL, ORANGE, GRAY, RED, BLUE = "#0E7C86", "#DD8A33", "#9AA7AE", "#C0392B", "#5B9BD5"
DOMTITLE = {d: C.DOMAIN_LABEL[d].replace("  [NEW 7th domain]", "").split(" (")[0] for d in C.DOMAINS}


def _load(t):
    return pd.read_csv(os.path.join(TBL, t))


def fig_calibration():
    cal = _load("T1_calibration.csv")
    ac = pd.read_csv(os.path.join(TBL, "T1b_achieved_correlations.csv"), index_col=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(cal)); w = 0.38
    a1.bar(x - w/2, cal.target_prev, w, label="target", color=GRAY)
    a1.bar(x + w/2, cal.achieved_prev, w, label="achieved", color=TEAL)
    a1.set_xticks(x); a1.set_xticklabels(cal.domain, rotation=25, ha="right")
    a1.set_ylabel("caseness prevalence"); a1.set_title("(a) Prevalence calibration")
    a1.legend(frameon=False)
    for i, r in cal.iterrows():
        a1.text(i + w/2, r.achieved_prev + 0.003, f"{r.achieved_prev:.2f}", ha="center", fontsize=9)
    im = a2.imshow(ac.values, cmap="RdBu_r", vmin=-1, vmax=1)
    a2.set_xticks(range(len(ac))); a2.set_xticklabels(ac.columns, rotation=45, ha="right", fontsize=9)
    a2.set_yticks(range(len(ac))); a2.set_yticklabels(ac.index, fontsize=9)
    for i in range(len(ac)):
        for j in range(len(ac)):
            a2.text(j, i, f"{ac.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    a2.set_title("(b) Achieved total-score correlations")
    fig.colorbar(im, ax=a2, shrink=0.85)
    fig.savefig(os.path.join(FIG, "F1_calibration.png")); plt.close(fig)


def fig_item_utility():
    """Per-domain item utility; SHORT codes on y-axis (full names in item dictionary)."""
    u = _load("T2_item_utility.csv")
    doms = C.DOMAINS
    fig, axes = plt.subplots(2, 4, figsize=(17, 10.5), constrained_layout=True)
    axes = axes.ravel()
    for k, d in enumerate(doms):
        sub = u[u.domain == d].sort_values("utility")
        ax = axes[k]
        colors = [ORANGE if s == 1 else TEAL for s in sub.selected]
        ax.barh(range(len(sub)), sub.utility, color=colors, height=0.7)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub.item, fontsize=9)
        ax.set_xlim(0.45, max(0.82, float(sub.utility.max()) + 0.05))
        for yi, (_, r) in enumerate(sub.iterrows()):
            ax.text(r.utility + 0.004, yi, f"{r.utility:.2f}", va="center", fontsize=8)
        ax.set_title(DOMTITLE[d], fontsize=12, color=(ORANGE if d == "sleep" else "black"))
        ax.axvline(0.5, color=GRAY, lw=0.8, ls="--")
        ax.tick_params(axis="x", labelsize=9)
    axes[-1].axis("off")
    axes[-1].text(0.05, 0.55,
                  "Utility U(j,k) = mean(AUROC, AUPRC)\nof item k for domain j.\n\n"
                  "Orange = selected anchor (λ=0.10).\nItem codes: see item_dictionary.csv\n"
                  "(e.g., ISI5 = daytime interference).",
                  fontsize=11, va="center")
    fig.suptitle("Per-item cross-diagnostic utility by domain", fontsize=14)
    fig.savefig(os.path.join(FIG, "F2_item_utility.png")); plt.close(fig)


def fig_performance():
    p = _load("T4_lockbox_performance.csv")
    x = np.arange(len(p)); w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    a1.bar(x - w/2, p.anchor_auroc, w, label="anchor only", color=GRAY)
    a1.bar(x + w/2, p.panel_auroc, w, label="full panel", color=TEAL)
    a1.set_xticks(x); a1.set_xticklabels(p.domain, rotation=25, ha="right")
    a1.set_ylim(0.5, 1.0); a1.set_ylabel("lockbox AUROC")
    a1.set_title("(a) Lockbox AUROC: anchor vs cross-diagnostic panel")
    a1.legend(frameon=False)
    for i, r in p.iterrows():
        a1.text(i + w/2, r.panel_auroc + 0.006, f"{r.panel_auroc:.2f}", ha="center", fontsize=9)
    colors = [ORANGE if dd == "sleep" else (RED if g < 0 else TEAL)
              for dd, g in zip(p.domain, p.crossdx_gain_auroc)]
    a2.bar(x, p.crossdx_gain_auroc, color=colors)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels(p.domain, rotation=25, ha="right")
    a2.set_ylabel("Δ AUROC (panel − anchor)")
    a2.set_title("(b) Cross-diagnostic gain\n(panic = anchor-dominant; orange = sleep)")
    for i, r in p.iterrows():
        a2.text(i, r.crossdx_gain_auroc + (0.002 if r.crossdx_gain_auroc >= 0 else -0.007),
                f"{r.crossdx_gain_auroc:+.3f}", ha="center", fontsize=9)
    fig.savefig(os.path.join(FIG, "F3_lockbox_performance.png")); plt.close(fig)


def fig_value_of_sleep():
    v = _load("T5_value_of_sleep.csv")
    panel = json.load(open(os.path.join(RES, "panel.json")))
    perf = _load("T4_lockbox_performance.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    x = np.arange(len(v))
    colors = [TEAL if dd >= 0 else RED for dd in v.delta_auroc_from_sleep]
    a1.bar(x, v.delta_auroc_from_sleep, color=colors)
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xticks(x); a1.set_xticklabels(v.domain, rotation=15)
    a1.set_ylabel("Δ AUROC")
    a1.set_ylim(-0.006, 0.006)
    a1.set_title("(a) Adding the sleep item to predict the\n6 original domains (incremental ≈ 0)")
    for i, r in v.iterrows():
        a1.text(i, r.delta_auroc_from_sleep + (0.0004 if r.delta_auroc_from_sleep >= 0 else -0.0008),
                f"{r.delta_auroc_from_sleep:+.3f}", ha="center", fontsize=8.5)
    sp = perf[perf.domain == "sleep"].iloc[0]
    sub = panel["sleep_subthreshold"]
    labels = ["ISI≥15\nanchor", "ISI≥15\npanel", "ISI≥8\nanchor", "ISI≥8\npanel"]
    vals = [sp.anchor_auroc, sp.panel_auroc, sub["anchor_auroc"], sub["panel_auroc"]]
    cols = [GRAY, ORANGE, GRAY, ORANGE]
    a2.bar(range(4), vals, color=cols)
    a2.set_xticks(range(4)); a2.set_xticklabels(labels, fontsize=10)
    a2.set_ylim(0.5, 1.0); a2.set_ylabel("lockbox AUROC")
    a2.set_title("(b) Detecting INSOMNIA: comorbid panel\nimproves over the sleep anchor alone")
    for i, val in enumerate(vals):
        a2.text(i, val + 0.006, f"{val:.2f}", ha="center", fontsize=9.5)
    fig.savefig(os.path.join(FIG, "F6_value_of_sleep.png")); plt.close(fig)


def fig_bootstrap():
    b = _load("T6_bootstrap_stability.csv")
    rows = []
    for d in C.DOMAINS:
        sub = b[b.domain == d].sort_values("selection_freq", ascending=False).head(3)
        for _, r in sub.iterrows():
            rows.append((d, str(r["item"]), str(r["label"]), float(r["selection_freq"]),
                         int(r["selected_in_main"])))
    fig, ax = plt.subplots(figsize=(11.5, 8.5), constrained_layout=True)
    yy = 0; yticks, ylabels = [], []
    last_d = None
    for d, item, label, freq, ismain in rows:
        if last_d is not None and d != last_d:
            yy += 0.7
        color = ORANGE if ismain == 1 else TEAL
        ax.barh(yy, freq, color=color, height=0.7)
        ax.text(freq + 0.01, yy, f"{freq:.2f}", va="center", fontsize=9)
        yticks.append(yy); ylabels.append(f"[{DOMTITLE[d]}]  {item} · {label}")
        yy += 1; last_d = d
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("bootstrap selection frequency (300 resamples)")
    ax.set_title("Selection stability — top-3 candidate items per domain\n(orange = main-analysis anchor)")
    ax.set_xlim(0, 1.02)
    fig.savefig(os.path.join(FIG, "F7_bootstrap_stability.png")); plt.close(fig)


def main():
    fig_calibration(); fig_item_utility(); fig_performance()
    fig_value_of_sleep(); fig_bootstrap()
    print("[04] saved figures F1, F2, F3, F6, F7")


if __name__ == "__main__":
    main()
