# -*- coding: utf-8 -*-
"""
02_optimize.py — run the cross-diagnostic optimization (7 domains incl. sleep),
bootstrap selection-stability, lockbox evaluation, and the incremental
value-of-adding-sleep analysis.

Outputs (results/tables/):
  T2_item_utility.csv          per-item utility U(j,k) for each domain
  T3_selected_panel.csv        selected anchor per domain + bootstrap frequency
  T4_lockbox_performance.csv   panel vs anchor-only AUROC/AUPRC on lockbox
  T5_value_of_sleep.csv        incremental AUROC for the 6 original domains from adding sleep
  T6_bootstrap_stability.csv   full selection-frequency table
Outputs (results/):
  panel.json                   selected panels (7-domain, 6-domain) + split sizes
"""
import os, json
import numpy as np
import pandas as pd
import config as C
import pipeline_core as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TBL = os.path.join(ROOT, "results", "tables")
RES = os.path.join(ROOT, "results")
os.makedirs(TBL, exist_ok=True)

ITEM_LABEL = {it[0]: it[1] for d in C.DOMAINS for it in C.ITEMS[d]}
OWN_LOAD = {it[0]: it[2] for d in C.DOMAINS for it in C.ITEMS[d]}
SIX = ["dep", "anx", "ptsd", "panic", "suicide", "alcohol"]


def main():
    items, meta = P.load_cohort()
    n = len(items)
    dev, lb = P.make_split(n)
    print(f"[02] N={n}  development={len(dev)}  lockbox={len(lb)}")

    all_codes = [c for c in items.columns if c != "subject_id"]
    U = P.univariate_utility(items, meta, dev)
    cos_dev = P.cosine_redundancy(items, dev, all_codes)

    # ---- 7-domain optimization (with sleep) ----
    panel7, val7 = P.optimize_panel(U, cos_dev, domains=C.DOMAINS)
    # ---- 6-domain optimization (original Lim et al. set, no sleep) ----
    panel6, val6 = P.optimize_panel(U, cos_dev, domains=SIX)
    print(f"[02] 7-domain objective={val7:.3f} ; 6-domain objective={val6:.3f}")
    print("[02] selected 7-domain panel:")
    for d in C.DOMAINS:
        print(f"     {d:8s} -> {panel7[d]:7s} ({ITEM_LABEL[panel7[d]]})  U={U[d][panel7[d]]:.3f}")

    # ---- bootstrap selection stability (300x) ----
    print(f"[02] bootstrap selection stability ({C.N_BOOTSTRAP} resamples) ...")
    rng = np.random.default_rng(C.SEED + 99)
    freq = {d: {it[0]: 0 for it in C.ITEMS[d]} for d in C.DOMAINS}
    for b in range(C.N_BOOTSTRAP):
        bidx = rng.choice(dev, size=len(dev), replace=True)
        Ub = P.univariate_utility(items, meta, bidx)
        cosb = P.cosine_redundancy(items, bidx, all_codes)
        selb, _ = P.optimize_panel(Ub, cosb, domains=C.DOMAINS, n_restarts=20, seed=b)
        for d in C.DOMAINS:
            freq[d][selb[d]] += 1
    boot_rows = []
    for d in C.DOMAINS:
        for it in C.ITEMS[d]:
            code = it[0]
            boot_rows.append({
                "domain": d, "item": code, "label": ITEM_LABEL[code],
                "selection_freq": round(freq[d][code] / C.N_BOOTSTRAP, 3),
                "selected_in_main": int(panel7[d] == code),
            })
    boot_df = pd.DataFrame(boot_rows).sort_values(
        ["domain", "selection_freq"], ascending=[True, False])
    boot_df.to_csv(os.path.join(TBL, "T6_bootstrap_stability.csv"), index=False)

    # ---- item utility table ----
    u_rows = []
    for d in C.DOMAINS:
        for it in C.ITEMS[d]:
            code = it[0]
            u_rows.append({
                "domain": d, "item": code, "label": ITEM_LABEL[code],
                "own_loading": OWN_LOAD[code], "utility": round(U[d][code], 3),
                "selected": int(panel7[d] == code),
            })
    pd.DataFrame(u_rows).sort_values(["domain", "utility"], ascending=[True, False]) \
        .to_csv(os.path.join(TBL, "T2_item_utility.csv"), index=False)

    # ---- selected panel table (+bootstrap freq) ----
    sel_rows = []
    for d in C.DOMAINS:
        code = panel7[d]
        sel_rows.append({
            "domain": d, "label": C.DOMAIN_LABEL[d], "anchor_item": code,
            "item_label": ITEM_LABEL[code], "utility": round(U[d][code], 3),
            "own_loading": OWN_LOAD[code],
            "bootstrap_selection_freq": round(freq[d][code] / C.N_BOOTSTRAP, 3),
        })
    pd.DataFrame(sel_rows).to_csv(os.path.join(TBL, "T3_selected_panel.csv"), index=False)

    # ---- lockbox evaluation: panel vs anchor-only ----
    perf = P.evaluate_panel(items, meta, dev, lb, panel7, domains=C.DOMAINS)
    perf.to_csv(os.path.join(TBL, "T4_lockbox_performance.csv"), index=False)
    print("[02] lockbox panel performance:")
    print(perf.to_string(index=False))

    # ---- value of adding the sleep domain ----
    # For each of the 6 original domains: predict it with the 6 original anchors,
    # vs the same 6 anchors PLUS the sleep anchor. Incremental AUROC = value of sleep.
    six_codes = [panel6[d] for d in SIX]
    sleep_code = panel7["sleep"]
    vrows = []
    for d in SIX:
        a_roc, a_prc, _, _ = P._fit_eval(items, meta, dev, lb, six_codes, d)
        b_roc, b_prc, _, _ = P._fit_eval(items, meta, dev, lb, six_codes + [sleep_code], d)
        vrows.append({
            "domain": d, "label": C.DOMAIN_LABEL[d],
            "auroc_6anchors": round(a_roc, 3),
            "auroc_6anchors_plus_sleep": round(b_roc, 3),
            "delta_auroc_from_sleep": round(b_roc - a_roc, 3),
            "auprc_6anchors": round(a_prc, 3),
            "auprc_6anchors_plus_sleep": round(b_prc, 3),
            "delta_auprc_from_sleep": round(b_prc - a_prc, 3),
        })
    vdf = pd.DataFrame(vrows)
    vdf.to_csv(os.path.join(TBL, "T5_value_of_sleep.csv"), index=False)
    print("[02] incremental value of adding the sleep anchor (6 original domains):")
    print(vdf[["domain", "auroc_6anchors", "auroc_6anchors_plus_sleep",
               "delta_auroc_from_sleep"]].to_string(index=False))

    # ---- secondary: sleep subthreshold label (ISI>=8) sensitivity ----
    # predict subthreshold insomnia with sleep anchor vs full panel
    panel_codes = [panel7[d] for d in C.DOMAINS]
    meta_sub = meta.copy()
    meta_sub["label_sleep"] = meta_sub["label_sleep_sub"]
    s_anchor = P._fit_eval(items, meta_sub, dev, lb, [sleep_code], "sleep")
    s_panel = P._fit_eval(items, meta_sub, dev, lb, panel_codes, "sleep")

    # ---- lambda sensitivity (robustness of selection) ----
    lam_rows = []
    for lam in [0.0, 0.05, 0.10, 0.15, 0.25, 0.50]:
        sel, _ = P.optimize_panel(U, cos_dev, lam=lam, domains=C.DOMAINS, n_restarts=60, seed=1)
        row = {"lambda": lam}
        for d in C.DOMAINS:
            row[d] = f"{sel[d]} ({ITEM_LABEL[sel[d]]})"
        lam_rows.append(row)
    pd.DataFrame(lam_rows).to_csv(os.path.join(TBL, "T7_lambda_sensitivity.csv"), index=False)

    # ---- save panels ----
    out = {
        "n_subjects": int(n), "n_dev": int(len(dev)), "n_lockbox": int(len(lb)),
        "lambda": C.REDUNDANCY_LAMBDA, "objective_7domain": val7, "objective_6domain": val6,
        "panel_7domain": panel7, "panel_6domain": panel6,
        "sleep_anchor": sleep_code,
        "sleep_subthreshold": {
            "anchor_auroc": round(s_anchor[0], 3), "panel_auroc": round(s_panel[0], 3),
            "anchor_auprc": round(s_anchor[1], 3), "panel_auprc": round(s_panel[1], 3),
        },
    }
    with open(os.path.join(RES, "panel.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[02] saved tables T2-T6 and results/panel.json")


if __name__ == "__main__":
    main()
