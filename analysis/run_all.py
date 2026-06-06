# -*- coding: utf-8 -*-
"""
run_all.py — reproduce the entire analysis from scratch (fixed seed).
    python analysis/run_all.py
Regenerates: data/, results/tables/*, results/figures/*, results/panel.json
"""
import os, sys, runpy, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEPS = [
    "01_simulate_cohort.py",
    "02_optimize.py",
    "03_network_contrib.py",
    "04_figures.py",
    "05_robustness_baselines.py",
    "06_realdata.py",
    "07_multicohort.py",
    "08_cohortD.py",
]


def main():
    t0 = time.time()
    for s in STEPS:
        print("\n" + "=" * 70 + f"\n[run_all] {s}\n" + "=" * 70)
        runpy.run_path(os.path.join(HERE, s), run_name="__main__")
    print(f"\n[run_all] DONE in {time.time()-t0:.1f}s. See results/ and manuscript/.")


if __name__ == "__main__":
    main()
