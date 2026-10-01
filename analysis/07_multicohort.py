# -*- coding: utf-8 -*-
"""
07_multicohort.py — multi-cohort real-data validation across THREE independent
open datasets, strengthening the single-cohort result of 06_realdata.py.

Cohorts (all open, item-level, auto-downloaded):
  A. Zenodo 10423537  N=24,292  Chinese university students   ISI+PHQ-9+GAD-7(+PSS)
  B. SRI (figshare)   N≈97      US adolescents, enriched for   ISI+BDI-II+STAI+PSS
                                clinical insomnia (REAL ISI, clinical)
  C. UK (figshare)    N=1,408   UK university students         SCI+PHQ-9+GAD-7+PSS+SBQ-R
                                (adds SUICIDALITY domain; insomnia via SCI, not ISI)

Goals: (1) replicate the sleep/insomnia item selection across populations and a
clinical sample; (2) extend domains (perceived stress; suicidality). For the two
cohorts that use the actual ISI (A, B) we test pre-registered H1 (selected sleep
item in ISI-3m set {maintenance, daytime-interference, concerns}). C tests
cross-population framework behaviour and adds the suicide domain.

Outputs (results/realdata/):
  RT5_multicohort_summary.csv   one row per cohort: selected items + H1 verdict
  RT6_cohortB_sri.csv / RT7_cohortC_uk.csv   per-cohort detail
  RF2_multicohort.png           cross-cohort sleep-item selection figure
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from _download import MANIFEST, DownloadError, fetch_verified

# The same rcParams as 04_figures.py. run_all.py runs 04 first in the same
# process, so the committed figures were drawn with these settings; setting them
# here gives the same image when this step is run on its own.
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": False,
})
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RD = os.path.join(ROOT, "data", "realdata")
OUT = os.path.join(ROOT, "results", "realdata")
os.makedirs(OUT, exist_ok=True)
TEAL, ORANGE, GRAY, RED = "#0E7C86", "#DD8A33", "#9AA7AE", "#C0392B"
SEED = 20260605
ISI_CONTENT = {1: "onset", 2: "maintenance", 3: "early-morning", 4: "dissatisfaction",
               5: "daytime interference", 6: "noticeable", 7: "worry/concerns"}
ISI3M = {2, 5, 7}


N_EXPECTED = {"B": 95, "C": 1406}   # complete records analysed in the paper


def dl(url, path):
    """Download unless the file on disk matches the SHA-256 in _download.MANIFEST."""
    rel = os.path.relpath(path, RD).replace(os.sep, "/")
    try:
        fetch_verified(url, path, MANIFEST[rel], tag="[07]")
    except DownloadError as e:
        raise SystemExit("[07] %s" % e)


def check_n(cohort, n):
    if n != N_EXPECTED[cohort]:
        raise SystemExit("[07] Cohort %s has N=%d complete records, expected %d; the "
                         "downloaded file is not the one behind the published results"
                         % (cohort, n, N_EXPECTED[cohort]))


# Cohort A (step 06) enters RT5 as fixed values. They are compared with the
# tables 06 writes, so a stale row is reported instead of passing silently.
COHORT_A_ROW = {"cohort": "A. Chinese students (Zenodo)", "N": 24292, "country": "China",
                "sleep_instrument": "ISI", "sleep_item_selected": "ISI2 maintenance",
                "is_core_ISI3m": "YES (99.4% boot)", "anx_item": "GAD2 worry",
                "extra_domain": "\u2014"}


def cohort_a_row_problems(out_dir=OUT):
    """Differences between COHORT_A_ROW and RT2/RT2b written by 06_realdata.py."""
    try:
        rt2 = pd.read_csv(os.path.join(out_dir, "RT2_selected_panel.csv")).set_index("domain")
        rt2b = pd.read_csv(os.path.join(out_dir, "RT2b_isi_selection_freq.csv"))
    except (OSError, KeyError, pd.errors.ParserError) as e:
        return ["RT2/RT2b from 06_realdata.py not readable (%s)" % e]
    problems = []
    sleep = "%s %s" % (rt2.loc["sleep", "selected_item"], rt2.loc["sleep", "content"])
    if sleep != COHORT_A_ROW["sleep_item_selected"]:
        problems.append("sleep item is %s in RT2" % sleep)
    is3m = str(rt2.loc["sleep", "is_ISI3m"]).strip().lower() == "true"
    boot = 100 * rt2b.loc[rt2b.is_ISI3m.astype(str).str.lower() == "true", "bootstrap_freq"].sum()
    verdict = "%s (%.1f%% boot)" % ("YES" if is3m else "NO", boot)
    if verdict != COHORT_A_ROW["is_core_ISI3m"]:
        problems.append("ISI-3m verdict is %s in RT2/RT2b" % verdict)
    anx = rt2.loc["anx", "selected_item"]
    if anx != COHORT_A_ROW["anx_item"].split()[0]:
        problems.append("anxiety item is %s in RT2" % anx)
    return problems


# ---------- generic engine ----------
def util(col, y, idx):
    x = col.values[idx].astype(float)
    if y[idx].sum() == 0 or y[idx].sum() == len(idx):
        return 0.0
    return float(0.5 * (roc_auc_score(y[idx], x) + average_precision_score(y[idx], x)))


def cosmat(df, cols, idx):
    Z = StandardScaler().fit_transform(df[cols].values[idx].astype(float))
    Z = pd.DataFrame(Z, columns=cols)
    nrm = {c: np.linalg.norm(Z[c].values) + 1e-9 for c in cols}
    return Z, nrm


def optimize(df, pools, ylab, idx, lam, restarts=12):
    """Coordinate ascent with random restarts (fast; exact-equivalent on small pools)."""
    U = {d: {it: util(df[it], ylab[d], idx) for it in pools[d]} for d in pools}
    allc = [c for d in pools for c in pools[d]]
    Z, nrm = cosmat(df, allc, idx)
    cos = lambda a, b: float(Z[a].values @ Z[b].values / (nrm[a] * nrm[b]))
    doms = list(pools)
    rng = np.random.default_rng(SEED + int(idx.sum()) % 9973)
    best, bv = None, -1e9
    for _ in range(restarts):
        sel = {d: pools[d][rng.integers(len(pools[d]))] for d in doms}
        improved = True
        while improved:
            improved = False
            for d in doms:
                others = [sel[o] for o in doms if o != d]
                bk, bl = sel[d], -1e9
                for k in pools[d]:
                    sc = U[d][k] - lam * sum(cos(k, o) for o in others)
                    if sc > bl:
                        bl, bk = sc, k
                if bk != sel[d]:
                    sel[d] = bk; improved = True
        v = sum(U[d][sel[d]] for d in doms) - lam * sum(
            cos(sel[doms[i]], sel[doms[j]]) for i in range(len(doms)) for j in range(i + 1, len(doms)))
        if v > bv:
            bv, best = v, dict(sel)
    return best, U


def boot_isi(df, pools, ylab, idx, lam, sleepdom, nb=200):
    rng = np.random.default_rng(SEED + 3)
    freq = {it: 0 for it in pools[sleepdom]}
    for _ in range(nb):
        b = rng.choice(idx, len(idx), replace=True)
        sel, _ = optimize(df, pools, ylab, b, lam)
        freq[sel[sleepdom]] += 1
    return {k: v / nb for k, v in freq.items()}


def lockbox(df, pools, ylab, dev, lb, panel):
    rows = {}
    pc = [panel[d] for d in pools]
    for d in pools:
        for tag, feats in [("anchor", [panel[d]]), ("panel", pc)]:
            sc = StandardScaler().fit(df[feats].values[dev].astype(float))
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(df[feats].values[dev].astype(float)), ylab[d][dev])
            p = clf.predict_proba(sc.transform(df[feats].values[lb].astype(float)))[:, 1]
            rows[(d, tag)] = round(roc_auc_score(ylab[d][lb], p), 3)
    return rows


def split(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); k = int(round(0.15 * n))
    return np.sort(perm[k:]), np.sort(perm[:k])


# ---------- Cohort B: SRI (real ISI, clinical adolescents) ----------
def cohort_B():
    path = os.path.join(RD, "cohortB_sri", "sri_insomnia_items.csv")
    dl("https://ndownloader.figshare.com/files/36167244", path)
    df = pd.read_csv(path)
    def pick(prefix, n):
        cols = [c for c in df.columns if re.sub(r"\.\d+$", "", c) == prefix][:n]
        return cols
    isi = pick("Insomnia Severity Index (ISI)", 7)
    bdi = pick("Beck Depression Inventory (BDI)", 21)
    stai = pick("State-Trait Anxiety Inventory (STAI-Y2)", 20)
    pss = pick("Perceived Stress Scale (PSS)", 10)
    use = isi + bdi + stai + pss
    d = df[use].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    # rename ISI to ISI1..7
    ren = {c: f"ISI{i+1}" for i, c in enumerate(isi)}
    ren.update({c: f"BDI{i+1}" for i, c in enumerate(bdi)})
    ren.update({c: f"STAI{i+1}" for i, c in enumerate(stai)})
    ren.update({c: f"PSS{i+1}" for i, c in enumerate(pss)})
    d = d.rename(columns=ren)
    check_n("B", len(d))
    pools = {"sleep": [f"ISI{i}" for i in range(1, 8)],
             "dep": [f"BDI{i}" for i in range(1, 22)],
             "anx": [f"STAI{i}" for i in range(1, 21)],
             "stress": [f"PSS{i}" for i in range(1, 11)]}
    tot = {"sleep": d[pools["sleep"]].sum(1), "dep": d[pools["dep"]].sum(1),
           "anx": d[pools["anx"]].sum(1), "stress": d[pools["stress"]].sum(1)}
    ylab = {"sleep": (tot["sleep"] >= 15).astype(int).values,   # ISI>=15
            "dep": (tot["dep"] >= 20).astype(int).values,        # BDI-II>=20 moderate
            "anx": (tot["anx"] >= 45).astype(int).values,        # STAI-Y2>=45
            "stress": (tot["stress"] >= 20).astype(int).values}  # PSS>=20
    idx = np.arange(len(d))
    panel, U = optimize(d, pools, ylab, idx, 0.10)
    bf = boot_isi(d, pools, ylab, idx, 0.10, "sleep", nb=200)
    isi_num = int(panel["sleep"][3:])
    prev = {k: round(float(v.mean()), 3) for k, v in ylab.items()}
    return {"name": "B. SRI US adolescents (clinical insomnia)", "N": len(d),
            "instrument": "ISI", "prev": prev, "panel": panel,
            "sleep_item": panel["sleep"], "sleep_content": ISI_CONTENT[isi_num],
            "is_ISI3m": isi_num in ISI3M, "boot": bf,
            "isi3m_boot": round(sum(v for k, v in bf.items() if int(k[3:]) in ISI3M), 3)}


# ---------- Cohort C: UK Akram (SCI insomnia + suicidality domain) ----------
def cohort_C():
    path = os.path.join(RD, "cohortC_uk", "uk_akram.xlsx")
    dl("https://ndownloader.figshare.com/files/42177492", path)
    df = pd.read_excel(path)
    phq = [f"PHQ_{i}" for i in range(1, 10)]
    gad = [f"GAD7_{i}" for i in range(1, 8)]
    pss = [f"PSS_{i}" for i in range(1, 11)]
    sbq = [f"SBQ{i}" for i in range(1, 5)]
    sci = [f"SCi{i}" for i in range(1, 9)]
    use = phq + gad + pss + sbq + sci
    d = df[use].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    check_n("C", len(d))
    # SCI: higher = better sleep -> reverse so higher = more insomnia (items 0-4)
    for c in sci:
        d[c] = d[c].max() - d[c]
    pools = {"sleep": sci, "dep": phq, "anx": gad, "stress": pss, "suicide": sbq}
    tot = {k: d[v].sum(1) for k, v in pools.items()}
    ylab = {"sleep": (tot["sleep"] >= (d[sci].values.max() * 8 - 16)).astype(int).values,  # SCI<=16 reversed
            "dep": (tot["dep"] >= 10).astype(int).values,
            "anx": (tot["anx"] >= 10).astype(int).values,
            "stress": (tot["stress"] >= 20).astype(int).values,
            "suicide": (tot["suicide"] >= 7).astype(int).values}   # SBQ-R>=7
    dev, lb = split(len(d), SEED + 7)
    panel, U = optimize(d, pools, ylab, dev, 0.10)
    lk = lockbox(d, pools, ylab, dev, lb, panel)
    prev = {k: round(float(v.mean()), 3) for k, v in ylab.items()}
    return {"name": "C. UK Akram students (+suicidality)", "N": len(d),
            "instrument": "SCI (not ISI)", "prev": prev, "panel": panel,
            "sleep_item": panel["sleep"], "lockbox": lk,
            "anx_item": panel["anx"], "suicide_item": panel["suicide"]}


def main():
    print("=" * 60)
    B = cohort_B()
    print(f"\n[07] Cohort {B['name']}  N={B['N']}")
    print(f"     prevalence: {B['prev']}")
    print(f"     selected panel: {B['panel']}")
    print(f"     >>> SLEEP item = {B['sleep_item']} ({B['sleep_content']}); "
          f"ISI-3m? {B['is_ISI3m']}; ISI-3m in {B['isi3m_boot']*100:.0f}% of bootstraps")
    print("=" * 60)
    C = cohort_C()
    print(f"\n[07] Cohort {C['name']}  N={C['N']}")
    print(f"     prevalence: {C['prev']}")
    print(f"     selected panel: {C['panel']}")
    print(f"     anxiety item = {C['anx_item']}; suicidality item = {C['suicide_item']}")
    print(f"     lockbox AUROC: {[(d, C['lockbox'][(d,'panel')]) for d in ['sleep','dep','anx','stress','suicide']]}")

    # summary table (incl. Cohort A from 06)
    for p in cohort_a_row_problems():
        print("[07] WARNING: RT5 row A (fixed values) disagrees with step 06 output: %s" % p)
    summ = pd.DataFrame([
        dict(COHORT_A_ROW),
        {"cohort": B["name"], "N": B["N"], "country": "USA (clinical insomnia)",
         "sleep_instrument": "ISI", "sleep_item_selected": f"{B['sleep_item']} {B['sleep_content']}",
         "is_core_ISI3m": f"{'YES' if B['is_ISI3m'] else 'NO'} ({B['isi3m_boot']*100:.0f}% boot)",
         "anx_item": B["panel"]["anx"], "extra_domain": "stress (PSS)"},
        {"cohort": C["name"], "N": C["N"], "country": "UK",
         "sleep_instrument": "SCI (not ISI)", "sleep_item_selected": C["sleep_item"],
         "is_core_ISI3m": "n/a (SCI)", "anx_item": C["anx_item"],
         "extra_domain": "suicidality (SBQ-R), stress (PSS)"},
    ])
    summ.to_csv(os.path.join(OUT, "RT5_multicohort_summary.csv"), index=False)
    pd.DataFrame([{"item": k, "content": ISI_CONTENT[int(k[3:])], "boot_freq": v,
                   "is_ISI3m": int(k[3:]) in ISI3M} for k, v in B["boot"].items()]
                 ).sort_values("boot_freq", ascending=False).to_csv(
        os.path.join(OUT, "RT6_cohortB_sri.csv"), index=False)
    pd.DataFrame([{"domain": d, "anchor": C["panel"][d], "prevalence": C["prev"][d],
                   "anchor_auroc": C["lockbox"][(d, "anchor")],
                   "panel_auroc": C["lockbox"][(d, "panel")]} for d in C["panel"]]).to_csv(
        os.path.join(OUT, "RT7_cohortC_uk.csv"), index=False)
    print("\n[07] multi-cohort summary:\n", summ.to_string(index=False))

    # ---------- figure ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    items = [f"ISI{i}" for i in range(1, 8)]
    bf = [B["boot"].get(f"ISI{i}", 0) for i in range(1, 8)]
    cols = [ORANGE if (i + 1) in ISI3M else TEAL for i in range(7)]
    a1.barh(range(7), bf, color=cols)
    a1.set_yticks(range(7)); a1.set_yticklabels([f"ISI{i+1} {ISI_CONTENT[i+1]}" for i in range(7)], fontsize=9)
    a1.invert_yaxis(); a1.set_xlim(0, 1); a1.set_xlabel("bootstrap selection frequency")
    a1.set_title(f"(A) Cohort B — CLINICAL insomnia sample (N={B['N']})\n"
                 f"ISI item selected as sleep anchor (orange = ISI-3m)")
    for i, v in enumerate(bf):
        if v > 0.01:
            a1.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    # Cohort C lockbox
    doms = ["sleep", "dep", "anx", "stress", "suicide"]
    av = [C["lockbox"][(d, "panel")] for d in doms]
    a2.bar(range(5), av, color=[ORANGE if d in ("sleep", "suicide") else TEAL for d in doms])
    a2.set_xticks(range(5)); a2.set_xticklabels(["insomnia\n(SCI)", "dep", "anx", "stress", "suicide"], fontsize=9)
    a2.set_ylim(0.5, 1.0); a2.set_ylabel("lockbox AUROC")
    a2.set_title(f"(B) Cohort C — UK cross-population (N={C['N']})\n"
                 f"5-domain panel incl. NEW suicidality domain (orange)")
    for i, v in enumerate(av):
        a2.text(i, v + 0.006, f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle("Multi-cohort real-data validation (3 independent open datasets)", fontsize=13)
    fig.savefig(os.path.join(OUT, "RF2_multicohort.png")); plt.close(fig)
    print("\n[07] saved RT5-RT7 and RF2_multicohort.png")


if __name__ == "__main__":
    main()
