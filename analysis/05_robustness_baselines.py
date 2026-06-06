# -*- coding: utf-8 -*-
"""
05_robustness_baselines.py — reviewer-requested rigor:
  (A) bootstrap 95% CIs for lockbox AUROC (panel & anchor)         -> T11_auroc_ci.csv
  (B) baselines: full-instrument (upper) & random-item panel (low) -> T12_baselines.csv
  (C) calibration ablation: is the sleep anchor stable when the
      generative assumptions (sleep cross-loadings, sleep latent
      correlations) are perturbed?                                  -> T13_ablation.csv
      figure                                                        -> F8_robustness.png

(C) directly answers the central simulation critique ("the result is baked into
the calibration") by showing across which assumption ranges the sleep anchor stays
within the ISI-3m item set {ISI2 maintenance, ISI5 daytime, ISI7 concerns}.
"""
import os, json, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import config as C
import pipeline_core as P
import sim_core

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TBL = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
RES = os.path.join(ROOT, "results")
TEAL, ORANGE, GRAY = "#0E7C86", "#DD8A33", "#9AA7AE"
ISI3M = {"ISI2", "ISI5", "ISI7"}   # ISI-3m items: maintenance, daytime interference, concerns


def _probs(items, meta, dev, lb, feat, domain):
    Xtr = StandardScaler().fit(items.loc[dev, feat].values.astype(float))
    Xt = Xtr.transform(items.loc[dev, feat].values.astype(float))
    Xe = Xtr.transform(items.loc[lb, feat].values.astype(float))
    clf = LogisticRegression(max_iter=1000).fit(Xt, meta.loc[dev, f"label_{domain}"].values)
    return clf.predict_proba(Xe)[:, 1], meta.loc[lb, f"label_{domain}"].values


def bootstrap_ci(y, p, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); aucs = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].sum() == 0 or y[b].sum() == len(b):
            continue
        aucs.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    items, meta = P.load_cohort()
    dev, lb = P.make_split(len(items))
    panel = json.load(open(os.path.join(RES, "panel.json")))["panel_7domain"]
    codes = [panel[d] for d in C.DOMAINS]

    # ---------- (A) bootstrap AUROC CIs ----------
    rows = []
    for d in C.DOMAINS:
        p_pan, y = _probs(items, meta, dev, lb, codes, d)
        p_anc, _ = _probs(items, meta, dev, lb, [panel[d]], d)
        pl, ph = bootstrap_ci(y, p_pan)
        al, ah = bootstrap_ci(y, p_anc)
        rows.append({
            "domain": d, "label": C.DOMAIN_LABEL[d].replace("  [NEW 7th domain]", ""),
            "panel_auroc": round(roc_auc_score(y, p_pan), 3),
            "panel_ci": f"[{pl:.3f}, {ph:.3f}]",
            "anchor_auroc": round(roc_auc_score(y, p_anc), 3),
            "anchor_ci": f"[{al:.3f}, {ah:.3f}]",
        })
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(os.path.join(TBL, "T11_auroc_ci.csv"), index=False)
    print("[05A] bootstrap AUROC 95% CIs:")
    print(ci_df.to_string(index=False))

    # ---------- (B) baselines ----------
    rng = np.random.default_rng(C.SEED + 5)
    base_rows = []
    for d in C.DOMAINS:
        pool_all = [it[0] for dd in C.DOMAINS for it in C.ITEMS[dd]]
        # full instrument of domain d
        inst = P.domain_pool(d)
        p_full, y = _probs(items, meta, dev, lb, inst, d)
        full_auc = roc_auc_score(y, p_full)
        # optimized panel
        p_pan, _ = _probs(items, meta, dev, lb, codes, d)
        pan_auc = roc_auc_score(y, p_pan)
        # random 1-item-per-domain panels
        rand_aucs = []
        for _ in range(200):
            rcodes = [P.domain_pool(dd)[rng.integers(len(P.domain_pool(dd)))] for dd in C.DOMAINS]
            pr, _ = _probs(items, meta, dev, lb, rcodes, d)
            rand_aucs.append(roc_auc_score(y, pr))
        base_rows.append({
            "domain": d, "label": C.DOMAIN_LABEL[d].replace("  [NEW 7th domain]", ""),
            "random_panel_auroc": round(float(np.mean(rand_aucs)), 3),
            "random_panel_sd": round(float(np.std(rand_aucs)), 3),
            "optimized_panel_auroc": round(pan_auc, 3),
            "full_instrument_auroc_TRIVIAL": round(full_auc, 3),
        })
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(os.path.join(TBL, "T12_baselines.csv"), index=False)
    print("\n[05B] baselines (random vs optimized panel; full-instrument = 1.0 trivially,")
    print("      because the caseness label is a deterministic sum of its own items):")
    print(base_df.to_string(index=False))

    # ---------- (C) calibration ablation ----------
    orig_items = copy.deepcopy(C.ITEMS)
    orig_corr = copy.deepcopy(C.LATENT_CORR)
    sleep_idx = C.DOMAINS.index("sleep")
    abl_rows = []
    scales = [0.0, 0.5, 1.0, 1.5]
    jitters = [-0.10, 0.0, 0.10]
    seeds = [0, 1, 2]
    for s in scales:
        for j in jitters:
            for sd in seeds:
                # scale sleep cross-loadings
                new_sleep = []
                for (code, lab, own, nc, cross) in orig_items["sleep"]:
                    new_sleep.append((code, lab, own, nc, [(cd, round(cl * s, 3)) for cd, cl in cross]))
                C.ITEMS["sleep"] = new_sleep
                # jitter sleep latent correlations with the 5 internalizing domains
                M = [row[:] for row in orig_corr]
                for di in range(len(C.DOMAINS)):
                    if di == sleep_idx:
                        continue
                    if C.DOMAINS[di] == "alcohol":
                        continue
                    val = min(0.9, max(0.05, M[sleep_idx][di] + j))
                    M[sleep_idx][di] = val; M[di][sleep_idx] = val
                C.LATENT_CORR = M
                di_items, di_meta, *_ = sim_core.simulate_cohort(seed=C.SEED + 1000 + sd,
                                                                 verbose=False)
                dv, _lb = P.make_split(len(di_items), seed=C.SEED + sd)
                allc = [c for c in di_items.columns if c != "subject_id"]
                Ua = P.univariate_utility(di_items, di_meta, dv)
                cosa = P.cosine_redundancy(di_items, dv, allc)
                sela, _ = P.optimize_panel(Ua, cosa, domains=C.DOMAINS, n_restarts=30, seed=sd)
                anchor = sela["sleep"]
                abl_rows.append({"sleep_crossload_scale": s, "sleep_corr_jitter": j,
                                 "seed": sd, "sleep_anchor": anchor,
                                 "in_ISI3m": int(anchor in ISI3M)})
    C.ITEMS = orig_items; C.LATENT_CORR = orig_corr   # restore
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(os.path.join(TBL, "T13_ablation.csv"), index=False)
    frac = abl["in_ISI3m"].mean()
    print(f"\n[05C] ablation: sleep anchor in ISI-3m set in {frac*100:.0f}% of "
          f"{len(abl)} perturbed configurations")
    print(abl.groupby("sleep_anchor").size().to_string())

    # ---------- figure F8 ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    # (a) baselines: random vs optimized (full-instrument is trivially 1.0 -> omitted)
    x = np.arange(len(base_df)); w = 0.36
    a1.bar(x - w/2, base_df.random_panel_auroc, w, yerr=base_df.random_panel_sd,
           label="random 1-item panel (mean±SD)", color=GRAY, capsize=3)
    a1.bar(x + w/2, base_df.optimized_panel_auroc, w, label="optimized panel", color=TEAL)
    a1.set_xticks(x); a1.set_xticklabels(base_df.domain, rotation=25, ha="right")
    a1.set_ylim(0.5, 1.0); a1.set_ylabel("lockbox AUROC")
    a1.set_title("(a) Optimized vs random 1-item-per-domain panel\n"
                 "(full 64-item battery = 1.0 trivially; omitted)")
    a1.legend(frameon=False, fontsize=9)
    # (b) ablation: anchor identity across grid
    counts = abl.groupby("sleep_anchor").size().sort_values(ascending=False)
    barcols = [ORANGE if a in ISI3M else GRAY for a in counts.index]
    a2.bar(range(len(counts)), counts.values, color=barcols)
    a2.set_xticks(range(len(counts))); a2.set_xticklabels(counts.index, rotation=0)
    a2.set_ylabel(f"# configurations (of {len(abl)})")
    a2.set_title(f"(b) Sleep anchor across {len(abl)} perturbed calibrations\n"
                 f"{frac*100:.0f}% land on an ISI-3m item (orange)")
    fig.savefig(os.path.join(FIG, "F8_robustness.png"), dpi=300); plt.close(fig)

    # save a small summary into panel.json sidecar
    summ = {"ablation_frac_in_ISI3m": round(float(frac), 3),
            "ablation_n": int(len(abl)),
            "ablation_anchor_counts": {k: int(v) for k, v in counts.items()}}
    with open(os.path.join(RES, "robustness_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("[05] saved T11-T13, F8, robustness_summary.json")


if __name__ == "__main__":
    main()
