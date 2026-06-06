# -*- coding: utf-8 -*-
"""
08_cohortD.py — Cohort D: BELL-001 MoA study (Korean clinical insomnia patients).

REAL, internal clinical data (de-identified): item-level ISI + PHQ-9 + GAD-7,
N=33. Extracted from the study spreadsheet and validated (per-instrument item
sums == provided totals). The de-identified CSV (data/realdata/cohortD_bell/
cohortD_items.csv) is NOT redistributed (git-ignored); regenerate it from the
internal export. This cohort fills the Korean + clinical gap, but is small and —
by inclusion design (PHQ-9<16, GAD-7<16) — has minimal depression/anxiety, so it
tests the SLEEP item selection (pre-registered H1) in a Korean clinical context
rather than the full cross-diagnostic comorbidity structure.

Output: results/realdata/RT8_cohortD.csv (ISI-item selection + verdict)
"""
import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "data", "realdata", "cohortD_bell", "cohortD_items.csv")
OUT = os.path.join(ROOT, "results", "realdata")
SEED = 20260605
ISI_CONTENT = {1: "onset", 2: "maintenance", 3: "early-morning", 4: "dissatisfaction",
               5: "daytime interference", 6: "noticeable", 7: "worry/concerns"}
ISI3M = {2, 5, 7}
POOLS = {"sleep": [f"ISI{i}" for i in range(1, 8)],
         "dep": [f"PHQ{i}" for i in range(1, 10)],
         "anx": [f"GAD{i}" for i in range(1, 8)]}
DOMS = ["sleep", "dep", "anx"]
CUT = {"sleep": 15, "dep": 10, "anx": 10}


def util(col, y, idx):
    x = col.values[idx].astype(float)
    if y[idx].sum() == 0 or y[idx].sum() == len(idx):
        return 0.5
    return float(0.5 * (roc_auc_score(y[idx], x) + average_precision_score(y[idx], x)))


def optimize(df, ylab, idx, lam=0.10):
    U = {d: {it: util(df[it], ylab[d], idx) for it in POOLS[d]} for d in DOMS}
    allc = sum(POOLS.values(), [])
    Z = StandardScaler().fit_transform(df[allc].values[idx].astype(float))
    Z = pd.DataFrame(Z, columns=allc); nrm = {c: np.linalg.norm(Z[c].values) + 1e-9 for c in allc}
    cos = lambda a, b: float(Z[a].values @ Z[b].values / (nrm[a] * nrm[b]))
    best, bv = None, -1e9
    for combo in product(*[POOLS[d] for d in DOMS]):
        s = dict(zip(DOMS, combo))
        v = sum(U[d][s[d]] for d in DOMS) - lam * (cos(s["sleep"], s["dep"]) +
              cos(s["sleep"], s["anx"]) + cos(s["dep"], s["anx"]))
        if v > bv:
            bv, best = v, s
    return best


def main():
    if not os.path.exists(CSV):
        print(f"[08] de-identified Cohort D CSV not found at {CSV} — skipping "
              "(regenerate from the internal MoA export).")
        return
    df = pd.read_csv(CSV)
    n = len(df)
    tot = {d: df[POOLS[d]].sum(axis=1) for d in DOMS}
    ylab = {d: (tot[d] >= CUT[d]).astype(int).values for d in DOMS}
    idx = np.arange(n)
    panel = optimize(df, ylab, idx)
    isi_num = int(panel["sleep"][3:])

    # bootstrap ISI selection stability
    rng = np.random.default_rng(SEED + 4)
    freq = {f"ISI{i}": 0 for i in range(1, 8)}
    B = 300
    for _ in range(B):
        b = rng.choice(idx, n, replace=True)
        try:
            s = optimize(df, ylab, b)
            freq[s["sleep"]] += 1
        except Exception:
            pass
    isi3m_boot = sum(v for k, v in freq.items() if int(k[3:]) in ISI3M) / B

    # ISI item-total correlations
    itc = {f"ISI{i}": round(float(np.corrcoef(df[f"ISI{i}"], tot["sleep"])[0, 1]), 3)
           for i in range(1, 8)}

    rows = [{"item": f"ISI{i}", "content": ISI_CONTENT[i],
             "item_total_r": itc[f"ISI{i}"],
             "boot_freq": round(freq[f"ISI{i}"] / B, 3),
             "is_ISI3m": i in ISI3M} for i in range(1, 8)]
    pd.DataFrame(rows).sort_values("boot_freq", ascending=False).to_csv(
        os.path.join(OUT, "RT8_cohortD.csv"), index=False)

    print(f"[08] Cohort D — BELL-001 MoA (Korean clinical insomnia), N={n}")
    print(f"     ISI>=15: {int(ylab['sleep'].sum())}/{n}  "
          f"PHQ>=10: {int(ylab['dep'].sum())}  GAD>=10: {int(ylab['anx'].sum())}")
    print(f"     selected panel: {panel}")
    print(f"     >>> SLEEP item = {panel['sleep']} ({ISI_CONTENT[isi_num]}); "
          f"ISI-3m? {isi_num in ISI3M}; ISI-3m in {isi3m_boot*100:.0f}% of bootstraps")
    print(f"     (caveat: N={n}, dep/anx truncated by design — exploratory)")
    print("[08] saved RT8_cohortD.csv")
    return {"N": n, "sleep_item": panel["sleep"], "content": ISI_CONTENT[isi_num],
            "is_ISI3m": isi_num in ISI3M, "isi3m_boot": round(isi3m_boot, 3),
            "isi_ge15": int(ylab["sleep"].sum())}


if __name__ == "__main__":
    main()
