# -*- coding: utf-8 -*-
"""
03_network_contrib.py — Gaussian Graphical Model (partial-correlation network)
and cross-diagnostic contribution analysis (SHAP-like standardized weights),
focusing on the role of the new SLEEP anchor as a transdiagnostic bridge.

Outputs (results/tables/):
  T8_ggm_partial_corr.csv         partial-correlation matrix among 7 anchors
  T9_bridge_centrality.csv        node strength + betweenness (sleep = bridge?)
  T10_crossdx_contribution.csv    domain x item standardized logistic weights
Outputs (results/figures/):
  F4_ggm_network.png              partial-correlation network (sleep highlighted)
  F5_contribution_heatmap.png     cross-diagnostic contribution heatmap
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import config as C
import pipeline_core as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TBL = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

ITEM_LABEL = {it[0]: it[1] for d in C.DOMAINS for it in C.ITEMS[d]}
DOM_SHORT = {"dep": "DEP", "anx": "ANX", "ptsd": "PTSD", "panic": "PANIC",
             "suicide": "SUI", "alcohol": "ALC", "sleep": "SLEEP"}


def main():
    items, meta = P.load_cohort()
    dev, lb = P.make_split(len(items))
    panel = json.load(open(os.path.join(RES, "panel.json")))["panel_7domain"]
    codes = [panel[d] for d in C.DOMAINS]
    node_names = [f"{DOM_SHORT[d]}\n{panel[d]}" for d in C.DOMAINS]

    # ---------- Gaussian Graphical Model (partial correlations) ----------
    X = items.loc[dev, codes].values.astype(float)
    X = StandardScaler().fit_transform(X)
    model = GraphicalLassoCV().fit(X)
    prec = model.precision_
    d = np.sqrt(np.diag(prec))
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    pc_df = pd.DataFrame(pcorr.round(3), index=[panel[x] for x in C.DOMAINS],
                         columns=[panel[x] for x in C.DOMAINS])
    pc_df.to_csv(os.path.join(TBL, "T8_ggm_partial_corr.csv"))

    # ---------- bridge centrality ----------
    G = nx.Graph()
    for i, di in enumerate(C.DOMAINS):
        G.add_node(i, domain=di, label=DOM_SHORT[di], item=panel[di])
    for i in range(len(C.DOMAINS)):
        for j in range(i + 1, len(C.DOMAINS)):
            w = pcorr[i, j]
            if abs(w) > 0.04:  # threshold weak edges
                G.add_edge(i, j, weight=abs(w), signed=w)
    strength = {i: sum(abs(pcorr[i, j]) for j in range(len(C.DOMAINS)) if j != i)
                for i in range(len(C.DOMAINS))}
    btw = nx.betweenness_centrality(G, weight=lambda u, v, dd: 1.0 / (dd["weight"] + 1e-6))
    cen_rows = []
    for i, di in enumerate(C.DOMAINS):
        cen_rows.append({
            "domain": di, "anchor": panel[di], "item_label": ITEM_LABEL[panel[di]],
            "node_strength": round(strength[i], 3),
            "betweenness": round(btw[i], 3),
        })
    cen_df = pd.DataFrame(cen_rows).sort_values("node_strength", ascending=False)
    cen_df.to_csv(os.path.join(TBL, "T9_bridge_centrality.csv"), index=False)
    print("[03] bridge centrality (node strength / betweenness):")
    print(cen_df.to_string(index=False))

    # ---------- network figure ----------
    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.spring_layout(G, weight="weight", seed=3, k=1.1)
    colors = ["#0E7C86" if C.DOMAINS[i] != "sleep" else "#DD8A33" for i in G.nodes]
    sizes = [1500 + 5000 * strength[i] for i in G.nodes]
    for (u, v, dd) in G.edges(data=True):
        w = dd["signed"]
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=("#C0392B" if w < 0 else "#5B9BD5"),
                lw=1 + 8 * abs(w), alpha=0.55, zorder=1)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes,
                           edgecolors="white", linewidths=2, ax=ax)
    labels = {i: f"{DOM_SHORT[C.DOMAINS[i]]}\n{panel[C.DOMAINS[i]]}" for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_color="white",
                            font_weight="bold", ax=ax)
    ax.set_title("Partial-correlation network of the 7-domain panel\n"
                 "(orange = new SLEEP anchor; blue edge = +, red edge = −)",
                 fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "F4_ggm_network.png"), dpi=300)
    plt.close()

    # ---------- cross-diagnostic contribution (standardized logistic weights) ----------
    Xdev = StandardScaler().fit_transform(items.loc[dev, codes].values.astype(float))
    contrib = np.zeros((len(C.DOMAINS), len(C.DOMAINS)))
    for r, d_target in enumerate(C.DOMAINS):
        y = meta.loc[dev, f"label_{d_target}"].values
        clf = LogisticRegression(max_iter=1000).fit(Xdev, y)
        contrib[r, :] = clf.coef_[0]
    contrib_df = pd.DataFrame(contrib.round(3),
                              index=[DOM_SHORT[d] for d in C.DOMAINS],
                              columns=[f"{panel[d]}\n({DOM_SHORT[d]})" for d in C.DOMAINS])
    contrib_df.to_csv(os.path.join(TBL, "T10_crossdx_contribution.csv"))

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    vmax = np.abs(contrib).max()
    im = ax.imshow(contrib, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(C.DOMAINS)))
    ax.set_xticklabels([f"{panel[d]}\n{DOM_SHORT[d]}" for d in C.DOMAINS], fontsize=8)
    ax.set_yticks(range(len(C.DOMAINS)))
    ax.set_yticklabels([C.DOMAIN_LABEL[d].split(" (")[0] for d in C.DOMAINS], fontsize=9)
    ax.set_xlabel("Panel item (anchor)")
    ax.set_ylabel("Predicted domain")
    for r in range(len(C.DOMAINS)):
        for c in range(len(C.DOMAINS)):
            ax.text(c, r, f"{contrib[r, c]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("Cross-diagnostic contribution weights\n"
                 "(standardized logistic coefficients; off-diagonal = transdiagnostic signal)",
                 fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8, label="standardized weight")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "F5_contribution_heatmap.png"), dpi=300)
    plt.close()

    # how much does the SLEEP anchor contribute to OTHER domains? (off-diagonal column)
    sleep_col = list(C.DOMAINS).index("sleep")
    sleep_contrib = {C.DOMAINS[r]: round(float(contrib[r, sleep_col]), 3)
                     for r in range(len(C.DOMAINS))}
    print("\n[03] SLEEP anchor (ISI5) standardized contribution to each domain:")
    for k, v in sleep_contrib.items():
        print(f"     -> {k:8s}: {v:+.3f}")
    print("[03] saved T8-T10 and figures F4, F5")


if __name__ == "__main__":
    main()
