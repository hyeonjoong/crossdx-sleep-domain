# -*- coding: utf-8 -*-
"""
09_deconfounding.py — quantifies the mechanism behind the depression de-confounding:
the selected ISI sleep anchor is more redundant (higher cosine) with the PHQ-9
sleep/fatigue items than with the core depressed-mood item, so once the ISI item
enters the panel the redundancy penalty moves the depression representative off the
somatic (fatigue) item onto the affective core. This is the explicit handling of the
ISI <-> PHQ-9 sleep-content overlap that a six-domain (no-sleep) framework never faced.

Additive script (does not modify the rest of the pipeline).
Output: results/tables/T14_deconfounding.csv
"""
import os, json
import pandas as pd
import pipeline_core as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TBL = os.path.join(ROOT, "results", "tables")

PHQ = {"PHQ3": "PHQ-9 sleep item", "PHQ4": "PHQ-9 fatigue item",
       "PHQ2": "PHQ-9 depressed-mood item", "PHQ1": "PHQ-9 anhedonia item"}


def main():
    items, meta = P.load_cohort()
    dev, _ = P.make_split(len(items))
    pj = json.load(open(os.path.join(ROOT, "results", "panel.json")))
    sleep = pj["panel_7domain"]["sleep"]
    dep6, dep7 = pj["panel_6domain"]["dep"], pj["panel_7domain"]["dep"]
    cos = P.cosine_redundancy(items, dev, [sleep] + list(PHQ))
    rows = [{"pair": f"{sleep} (sleep anchor) — {c}", "phq_item": c, "label": PHQ[c],
             "cosine_redundancy": round(float(cos.loc[sleep, c]), 3)} for c in PHQ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TBL, "T14_deconfounding.csv"), index=False)
    print("[09] depression anchor: 6-domain =", dep6, "-> 7-domain =", dep7,
          "(i.e., fatigue -> depressed mood when the sleep domain is added)")
    print(f"[09] redundancy of the sleep anchor ({sleep}) with PHQ-9 items:")
    print(df.to_string(index=False))
    print("[09] saved T14_deconfounding.csv")


if __name__ == "__main__":
    main()
