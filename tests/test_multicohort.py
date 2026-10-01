# -*- coding: utf-8 -*-
"""07_multicohort.py writes Cohort A's RT5 row from fixed values; it must notice
when the step 06 tables no longer agree with them (work order C12)."""
import importlib.util
import os
import shutil

import pandas as pd

from conftest import ROOT


def _module():
    spec = importlib.util.spec_from_file_location(
        "multicohort", os.path.join(ROOT, "analysis", "07_multicohort.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_row_a_agrees_with_committed_step06_tables():
    m = _module()
    assert m.cohort_a_row_problems(os.path.join(ROOT, "results", "realdata")) == []


def test_row_a_disagreement_is_reported(tmp_path):
    m = _module()
    src = os.path.join(ROOT, "results", "realdata")
    for f in ("RT2_selected_panel.csv", "RT2b_isi_selection_freq.csv"):
        shutil.copy(os.path.join(src, f), tmp_path / f)
    rt2 = pd.read_csv(tmp_path / "RT2_selected_panel.csv")
    rt2.loc[rt2.domain == "anx", "selected_item"] = "GAD1"
    rt2.to_csv(tmp_path / "RT2_selected_panel.csv", index=False)
    problems = m.cohort_a_row_problems(str(tmp_path))
    assert len(problems) == 1 and "GAD1" in problems[0]


def test_row_a_check_without_step06_tables(tmp_path):
    m = _module()
    problems = m.cohort_a_row_problems(str(tmp_path))
    assert problems and "RT2" in problems[0]
