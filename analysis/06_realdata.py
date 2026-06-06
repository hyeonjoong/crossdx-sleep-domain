# -*- coding: utf-8 -*-
"""
06_realdata.py — REAL-DATA validation of the sleep-domain extension on a public,
item-level, open dataset.

Dataset: Su et al. (2024) "Temporal dynamics in psychological assessments",
Scientific Data; Zenodo 10423537 (CC-BY 4.0). N=24,292 Chinese university students,
item-level ISI + PHQ-9 + GAD-7 (+ PSS) with a shared respondent key (export_id).
Downloaded to data/realdata/.

This is a 3-DOMAIN subset {insomnia, depression, anxiety} — the clinically central
comorbidity triad and the part of the framework the public data can test. It directly
tests pre-registered H1: is the selected ISI item an ISI-3m item (maintenance,
daytime-interference, or concerns/worry)?

Outputs (results/realdata/):
  RT1_describe.csv         prevalence, ISI mean/SD, cross-correlations, item-total r
  RT2_selected_panel.csv   selected item per domain + utility + bootstrap freq
  RT3_lockbox.csv          panel vs anchor AUROC with bootstrap 95% CI
  RT4_isi_lambda.csv       selected ISI item across lambda (robustness)
  RF1_realdata.png         summary figure
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({"font.size": 11, "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RD = os.path.join(ROOT, "data", "realdata")
OUT = os.path.join(ROOT, "results", "realdata")
os.makedirs(OUT, exist_ok=True)
TEAL, ORANGE, GRAY, RED = "#0E7C86", "#DD8A33", "#9AA7AE", "#C0392B"
SEED = 20260605

# Standard ISI item content (7-item scoring). ISI-3m = {2 maintenance, 5 daytime, 7 worry}.
ISI_CONTENT = {1: "onset", 2: "maintenance", 3: "early-morning", 4: "dissatisfaction",
               5: "daytime interference", 6: "noticeable", 7: "worry/concerns"}
ISI3M = {2, 5, 7}
CUTOFF = {"sleep": 15, "dep": 10, "anx": 10}
LABELFULL = {"sleep": "Insomnia (ISI)", "dep": "Depression (PHQ-9)", "anx": "Anxiety (GAD-7)"}


ZENODO = "https://zenodo.org/records/10423537/files/{}.csv?download=1"


def ensure_downloaded():
    """Auto-download the open CC-BY dataset (Zenodo 10423537) if absent."""
    import urllib.request
    os.makedirs(RD, exist_ok=True)
    for f in ["isi", "phq9", "gad7", "pss", "demographic"]:
        path = os.path.join(RD, f"{f}.csv")
        if not os.path.exists(path):
            print(f"[06] downloading {f}.csv from Zenodo 10423537 (CC-BY 4.0) ...")
            urllib.request.urlretrieve(ZENODO.format(f), path)


def load():
    ensure_downloaded()
    isi = pd.read_csv(os.path.join(RD, "isi.csv"))
    phq = pd.read_csv(os.path.join(RD, "phq9.csv"))
    gad = pd.read_csv(os.path.join(RD, "gad7.csv"))
    # keep item columns only (questionN), rename to ISIn/PHQn/GADn
    def items(df, n, pre):
        cols = [f"question{i}" for i in range(1, n + 1)]
        d = df[["export_id"] + cols].rename(columns={f"question{i}": f"{pre}{i}" for i in range(1, n + 1)})
        d[f"{pre}_total"] = df[cols].sum(axis=1).values
        return d
    isi_i = items(isi, 7, "ISI")
    phq_i = items(phq, 9, "PHQ")
    gad_i = items(gad, 7, "GAD")
    df = isi_i.merge(phq_i, on="export_id").merge(gad_i, on="export_id")
    df = df.dropna().reset_index(drop=True)
    return df


def describe(df):
    rows = []
    for dom, pre, n in [("sleep", "ISI", 7), ("dep", "PHQ", 9), ("anx", "GAD", 7)]:
        tot = df[f"{pre}_total"]
        rows.append({"domain": dom, "instrument": pre, "n_items": n,
                     "total_mean": round(tot.mean(), 2), "total_sd": round(tot.std(), 2),
                     "cutoff": CUTOFF[dom],
                     "prevalence": round((tot >= CUTOFF[dom]).mean(), 4),
                     "n_positive": int((tot >= CUTOFF[dom]).sum())})
    desc = pd.DataFrame(rows)
    # cross-correlations
    corr = df[["ISI_total", "PHQ_total", "GAD_total"]].corr().round(3)
    # ISI item-total correlations
    itc = {}
    for i in range(1, 8):
        itc[f"ISI{i} ({ISI_CONTENT[i]})"] = round(df[f"ISI{i}"].corr(df["ISI_total"]), 3)
    return desc, corr, itc


def labels(df):
    return {d: (df[f"{p}_total"] >= CUTOFF[d]).astype(int).values
            for d, p in [("sleep", "ISI"), ("dep", "PHQ"), ("anx", "GAD")]}


def utility(df, item, idx, y):
    x = df[item].values[idx].astype(float)
    return float(0.5 * (roc_auc_score(y[idx], x) + average_precision_score(y[idx], x)))


def fit_auc(df, dev, lb, feats, y):
    sc = StandardScaler().fit(df.loc[dev, feats].values)
    Xtr, Xte = sc.transform(df.loc[dev, feats].values), sc.transform(df.loc[lb, feats].values)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, y[dev])
    p = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(y[lb], p), average_precision_score(y[lb], p), p, y[lb]


def boot_ci(yt, p, n=1000, seed=1):
    rng = np.random.default_rng(seed); a = []
    for _ in range(n):
        b = rng.choice(len(yt), len(yt), replace=True)
        if 0 < yt[b].sum() < len(b):
            a.append(roc_auc_score(yt[b], p[b]))
    return round(np.percentile(a, 2.5), 3), round(np.percentile(a, 97.5), 3)


POOLS = {"sleep": [f"ISI{i}" for i in range(1, 8)],
         "dep": [f"PHQ{i}" for i in range(1, 10)],
         "anx": [f"GAD{i}" for i in range(1, 8)]}
DOMS = ["sleep", "dep", "anx"]


def optimize(df, idx, ylab, lam, cosfun):
    U = {d: {it: utility(df, it, idx, ylab[d]) for it in POOLS[d]} for d in DOMS}
    # exhaustive (7*9*7=441)
    best, bestval = None, -1e9
    for combo in product(*[POOLS[d] for d in DOMS]):
        sel = dict(zip(DOMS, combo))
        val = sum(U[d][sel[d]] for d in DOMS)
        pen = (cosfun(sel["sleep"], sel["dep"]) + cosfun(sel["sleep"], sel["anx"])
               + cosfun(sel["dep"], sel["anx"]))
        v = val - lam * pen
        if v > bestval:
            bestval, best = v, sel
    return best, U


def main():
    df = load()
    print(f"[06] merged real cohort: N={len(df)} (ISI+PHQ-9+GAD-7, item-level)")
    desc, corr, itc = describe(df)
    print(desc.to_string(index=False))
    print("\n[06] total-score correlations (real data):\n", corr.to_string())
    print("\n[06] ISI item-total correlations:")
    for k, v in itc.items():
        print(f"     {k:28s} {v}")
    desc.to_csv(os.path.join(OUT, "RT1_describe.csv"), index=False)
    corr.to_csv(os.path.join(OUT, "RT1b_correlations.csv"))
    pd.DataFrame([{"item": k, "item_total_r": v} for k, v in itc.items()]).to_csv(
        os.path.join(OUT, "RT1c_isi_item_total.csv"), index=False)

    ylab = labels(df)
    n = len(df)
    rng = np.random.default_rng(SEED + 7)
    perm = rng.permutation(n)
    nlb = int(round(0.15 * n))
    lb = np.sort(perm[:nlb]); dev = np.sort(perm[nlb:])
    print(f"\n[06] development={len(dev)}  lockbox={len(lb)}")

    # cosine redundancy on dev (precompute standardized)
    allitems = POOLS["sleep"] + POOLS["dep"] + POOLS["anx"]
    Z = StandardScaler().fit_transform(df.loc[dev, allitems].values.astype(float))
    Zdf = pd.DataFrame(Z, columns=allitems)
    norms = {c: np.linalg.norm(Zdf[c].values) for c in allitems}

    def cosfun(a, b):
        return float(Zdf[a].values @ Zdf[b].values / (norms[a] * norms[b] + 1e-9))

    LAM = 0.10
    panel, U = optimize(df, dev, ylab, LAM, cosfun)
    isi_sel = int(panel["sleep"].replace("ISI", ""))
    print(f"\n[06] >>> selected panel (lambda={LAM}): {panel}")
    print(f"[06] >>> SELECTED ISI ITEM = ISI{isi_sel} ({ISI_CONTENT[isi_sel]}); "
          f"ISI-3m item? {'YES' if isi_sel in ISI3M else 'NO'}")

    # bootstrap selection stability (300x) on dev
    freq = {d: {it: 0 for it in POOLS[d]} for d in DOMS}
    B = 300
    for b in range(B):
        bidx = rng.choice(dev, len(dev), replace=True)
        # recompute cosine on bootstrap sample
        Zb = StandardScaler().fit_transform(df.loc[bidx, allitems].values.astype(float))
        Zbdf = pd.DataFrame(Zb, columns=allitems)
        nb = {c: np.linalg.norm(Zbdf[c].values) for c in allitems}
        cf = lambda a, c: float(Zbdf[a].values @ Zbdf[c].values / (nb[a] * nb[c] + 1e-9))
        sb, _ = optimize(df, bidx, ylab, LAM, cf)
        for d in DOMS:
            freq[d][sb[d]] += 1

    sel_rows = []
    for d in DOMS:
        code = panel[d]
        sel_rows.append({"domain": d, "instrument_label": LABELFULL[d], "selected_item": code,
                         "content": (ISI_CONTENT[int(code[3:])] if d == "sleep" else ""),
                         "utility": round(U[d][code], 3),
                         "bootstrap_freq": round(freq[d][code] / B, 3),
                         "is_ISI3m": (int(code[3:]) in ISI3M) if d == "sleep" else ""})
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(os.path.join(OUT, "RT2_selected_panel.csv"), index=False)
    # full ISI bootstrap distribution
    isi_freq = pd.DataFrame([{"item": f"ISI{i}", "content": ISI_CONTENT[i],
                              "bootstrap_freq": round(freq["sleep"][f"ISI{i}"] / B, 3),
                              "is_ISI3m": i in ISI3M} for i in range(1, 8)]
                            ).sort_values("bootstrap_freq", ascending=False)
    isi_freq.to_csv(os.path.join(OUT, "RT2b_isi_selection_freq.csv"), index=False)

    # lockbox performance with CI
    pcodes = [panel[d] for d in DOMS]
    perf_rows = []
    for d in DOMS:
        a_roc, a_prc, _, _ = fit_auc(df, dev, lb, [panel[d]], ylab[d])
        p_roc, p_prc, pp, yt = fit_auc(df, dev, lb, pcodes, ylab[d])
        lo, hi = boot_ci(yt, pp)
        perf_rows.append({"domain": d, "label": LABELFULL[d], "anchor_item": panel[d],
                          "anchor_auroc": round(a_roc, 3), "panel_auroc": round(p_roc, 3),
                          "panel_auroc_ci": f"[{lo}, {hi}]", "panel_auprc": round(p_prc, 3),
                          "crossdx_gain": round(p_roc - a_roc, 3)})
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(os.path.join(OUT, "RT3_lockbox.csv"), index=False)
    print("\n[06] lockbox performance (real data):\n", perf.to_string(index=False))

    # ISI item across lambda
    lam_rows = []
    for lam in [0.0, 0.05, 0.10, 0.15, 0.25, 0.50]:
        pl, _ = optimize(df, dev, ylab, lam, cosfun)
        i = int(pl["sleep"].replace("ISI", ""))
        lam_rows.append({"lambda": lam, "isi_item": pl["sleep"], "content": ISI_CONTENT[i],
                         "is_ISI3m": i in ISI3M, "dep_item": pl["dep"], "anx_item": pl["anx"]})
    lamdf = pd.DataFrame(lam_rows)
    lamdf.to_csv(os.path.join(OUT, "RT4_isi_lambda.csv"), index=False)
    print("\n[06] selected ISI item across lambda:\n", lamdf.to_string(index=False))

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), constrained_layout=True)
    # (a) ISI item-total + selection
    a = axes[0]
    items_i = [f"ISI{i}" for i in range(1, 8)]
    itc_vals = [itc[f"ISI{i} ({ISI_CONTENT[i]})"] for i in range(1, 8)]
    bf = [freq["sleep"][f"ISI{i}"] / B for i in range(1, 8)]
    cols = [ORANGE if (i + 1) in ISI3M else TEAL for i in range(7)]
    a.barh(range(7), bf, color=cols)
    a.set_yticks(range(7)); a.set_yticklabels([f"ISI{i+1} {ISI_CONTENT[i+1]}" for i in range(7)], fontsize=9)
    a.invert_yaxis(); a.set_xlabel("bootstrap selection frequency (300)")
    a.set_title("(A) Real data: ISI item selected as sleep anchor\n(orange = ISI-3m items)")
    a.set_xlim(0, 1)
    for i, v in enumerate(bf):
        a.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    # (b) lockbox AUROC
    b = axes[1]
    x = np.arange(len(perf)); w = 0.38
    b.bar(x - w/2, perf.anchor_auroc, w, label="anchor only", color=GRAY)
    b.bar(x + w/2, perf.panel_auroc, w, label="3-domain panel", color=TEAL)
    b.set_xticks(x); b.set_xticklabels(perf.domain); b.set_ylim(0.5, 1.0)
    b.set_ylabel("lockbox AUROC"); b.set_title("(B) Lockbox performance (real data)")
    b.legend(frameon=False)
    for i, r in perf.iterrows():
        b.text(i + w/2, r.panel_auroc + 0.006, f"{r.panel_auroc:.2f}", ha="center", fontsize=9)
    # (c) correlations vs literature
    c = axes[2]
    pairs = [("dep–anx", corr.loc["PHQ_total", "GAD_total"], 0.75),
             ("dep–insomnia", corr.loc["PHQ_total", "ISI_total"], 0.52),
             ("anx–insomnia", corr.loc["GAD_total", "ISI_total"], 0.48)]
    xp = np.arange(len(pairs))
    c.bar(xp - 0.2, [p[1] for p in pairs], 0.4, label="real data", color=TEAL)
    c.bar(xp + 0.2, [p[2] for p in pairs], 0.4, label="literature", color=GRAY)
    c.set_xticks(xp); c.set_xticklabels([p[0] for p in pairs]); c.set_ylim(0, 1)
    c.set_ylabel("correlation"); c.set_title("(C) Total-score correlations\nreal vs literature")
    c.legend(frameon=False)
    for i, p in enumerate(pairs):
        c.text(i - 0.2, p[1] + 0.02, f"{p[1]:.2f}", ha="center", fontsize=8)
    fig.suptitle(f"Real-data validation (Zenodo 10423537, N={len(df):,} students; ISI+PHQ-9+GAD-7)",
                 fontsize=13)
    fig.savefig(os.path.join(OUT, "RF1_realdata.png")); plt.close(fig)
    print("\n[06] saved RT1-RT4 and RF1_realdata.png to results/realdata/")


if __name__ == "__main__":
    main()
