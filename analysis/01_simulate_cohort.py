# -*- coding: utf-8 -*-
"""
01_simulate_cohort.py — generate and save the calibrated synthetic cohort.

Outputs (data/):
  simulated_cohort_items.csv   item-level responses (64 items) + subject_id
  simulated_cohort_meta.csv    total scores, binary caseness labels, sex
  item_dictionary.csv          item -> domain/label/loadings map
Outputs (results/tables/):
  T1_calibration.csv           target vs achieved prevalence, totals
  T1b_achieved_correlations.csv total-score correlation matrix

To run on REAL data instead: produce the same two CSVs (items + meta with
label_<domain> columns) from your EMBRAIN / BELL-001 export and skip this step.
"""
import os
import pandas as pd
import sim_core

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
TBL = os.path.join(ROOT, "results", "tables")
os.makedirs(DATA, exist_ok=True)
os.makedirs(TBL, exist_ok=True)


def main():
    df_items, df_meta, item_dict, calib, achieved_corr = sim_core.simulate_cohort()
    df_items.to_csv(os.path.join(DATA, "simulated_cohort_items.csv"), index=False)
    df_meta.to_csv(os.path.join(DATA, "simulated_cohort_meta.csv"), index=False)
    item_dict.to_csv(os.path.join(DATA, "item_dictionary.csv"), index=False)
    calib.to_csv(os.path.join(TBL, "T1_calibration.csv"), index=False)
    achieved_corr.to_csv(os.path.join(TBL, "T1b_achieved_correlations.csv"))
    print("\n[01] saved cohort to data/ and calibration tables to results/tables/")


if __name__ == "__main__":
    main()
