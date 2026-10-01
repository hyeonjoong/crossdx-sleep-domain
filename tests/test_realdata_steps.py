# -*- coding: utf-8 -*-
"""Steps 06 and 07 use the verified download and the N check (work order C6),
and step 03 names the sleep anchor it actually used (C12).

06 and 07 are loaded from a copy of the repository, so their data/realdata/ and
results/realdata/ are those of the copy. No network is used: every way of
opening a URL raises. The open-data files written here are small synthetic
tables, never the real downloads.
"""
import importlib.util
import json
import os
import urllib.error
import urllib.request

import matplotlib
import numpy as np
import pandas as pd
import pytest

import _download
from conftest import ROOT, run_py, sha256


@pytest.fixture
def offline(monkeypatch):
    """Any download attempt fails as if there were no internet."""
    def no_network(*args, **kwargs):
        raise urllib.error.URLError("network disabled in tests")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(urllib.request, "urlretrieve", no_network)
    monkeypatch.setattr(_download, "urlopen", no_network)


def _load_step(root, script, name):
    """Import analysis/<script> of the copy at `root` (its paths point into the copy)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, "analysis", script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_table(path, columns, n, seed, truncate=False):
    """Synthetic integer table; with truncate=True the last line is cut short, like
    an interrupted download."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [",".join(columns)]
    for i in range(n):
        lines.append(",".join([str(i + 1)] + [str(v) for v in rng.integers(0, 4, len(columns) - 1)]))
    text = "\n".join(lines) + "\n"
    if truncate:
        text = text[:-7]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _questions(k):
    return ["export_id"] + ["question%d" % i for i in range(1, k + 1)]


def _part_files(top):
    return [os.path.join(dp, f) for dp, _dn, fn in os.walk(top) for f in fn
            if f.endswith(".part")]


def _committed(rel):
    return sha256(os.path.join(ROOT, rel))


# ---------------------------------------------------------------------------
# 06
# ---------------------------------------------------------------------------
def test_06_does_not_reuse_a_truncated_download(repo_copy, offline):
    rd = os.path.join(repo_copy, "data", "realdata")
    _write_table(os.path.join(rd, "isi.csv"), _questions(7), 60, 1, truncate=True)
    for name, k, seed in (("phq9.csv", 9, 2), ("gad7.csv", 7, 3), ("pss.csv", 10, 4),
                          ("demographic.csv", 3, 5)):
        _write_table(os.path.join(rd, name), _questions(k), 60, seed)
    truncated = sha256(os.path.join(rd, "isi.csv"))
    rt1 = os.path.join("results", "realdata", "RT1_describe.csv")
    with matplotlib.rc_context():
        mod = _load_step(repo_copy, "06_realdata.py", "realdata06_copy")
        with pytest.raises(SystemExit) as e:
            mod.main()
    msg = str(e.value.code)
    assert msg.startswith("[06]") and "isi.csv" in msg
    assert sha256(os.path.join(repo_copy, rt1)) == _committed(rt1)   # nothing written
    assert sha256(os.path.join(rd, "isi.csv")) == truncated            # not replaced
    assert _part_files(rd) == []


def test_06_stops_when_the_cohort_size_is_not_the_published_one(repo_copy, monkeypatch):
    rng = np.random.default_rng(6)
    cols = {}
    for pre, k in (("ISI", 7), ("PHQ", 9), ("GAD", 7)):
        for i in range(1, k + 1):
            cols["%s%d" % (pre, i)] = rng.integers(0, 4, 50)
        cols["%s_total" % pre] = sum(cols["%s%d" % (pre, i)] for i in range(1, k + 1))
    small = pd.DataFrame(dict(export_id=np.arange(50), **cols))
    rt1 = os.path.join("results", "realdata", "RT1_describe.csv")
    with matplotlib.rc_context():
        mod = _load_step(repo_copy, "06_realdata.py", "realdata06_copy_n")
        monkeypatch.setattr(mod, "load", lambda: small)
        with pytest.raises(SystemExit) as e:
            mod.main()
    assert "N=50" in str(e.value.code) and "24292" in str(e.value.code)
    assert sha256(os.path.join(repo_copy, rt1)) == _committed(rt1)


# ---------------------------------------------------------------------------
# 07
# ---------------------------------------------------------------------------
SRI_BLOCKS = (("Insomnia Severity Index (ISI)", 7), ("Beck Depression Inventory (BDI)", 21),
              ("State-Trait Anxiety Inventory (STAI-Y2)", 20),
              ("Perceived Stress Scale (PSS)", 10))


def _sri_columns():
    # the distributed file repeats each scale name once per item
    return ["record"] + [name for name, k in SRI_BLOCKS for _ in range(k)]


def test_07_does_not_reuse_a_truncated_download(repo_copy, offline):
    rd = os.path.join(repo_copy, "data", "realdata")
    path = os.path.join(rd, "cohortB_sri", "sri_insomnia_items.csv")
    _write_table(path, _sri_columns(), 40, 7, truncate=True)
    rt5 = os.path.join("results", "realdata", "RT5_multicohort_summary.csv")
    with matplotlib.rc_context():
        mod = _load_step(repo_copy, "07_multicohort.py", "multicohort07_copy")
        with pytest.raises(SystemExit) as e:
            mod.main()
    msg = str(e.value.code)
    assert msg.startswith("[07]") and "sri_insomnia_items.csv" in msg
    assert sha256(os.path.join(repo_copy, rt5)) == _committed(rt5)
    assert _part_files(rd) == []


def test_07_stops_when_the_cohort_size_is_not_the_published_one(repo_copy, offline,
                                                                 monkeypatch):
    rd = os.path.join(repo_copy, "data", "realdata")
    _write_table(os.path.join(rd, "cohortB_sri", "sri_insomnia_items.csv"),
                 _sri_columns(), 30, 8)
    with matplotlib.rc_context():
        mod = _load_step(repo_copy, "07_multicohort.py", "multicohort07_copy_n")
        monkeypatch.setattr(mod, "dl", lambda url, path: None)   # use the file as it is
        with pytest.raises(SystemExit) as e:
            mod.main()
    msg = str(e.value.code)
    assert "Cohort B" in msg and "N=30" in msg and "95" in msg


# ---------------------------------------------------------------------------
# 03
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_03_names_the_sleep_anchor_of_panel_json(repo_copy):
    pj = os.path.join(repo_copy, "results", "panel.json")
    with open(pj) as fh:
        panel = json.load(fh)
    committed = panel["panel_7domain"]["sleep"]
    other = next(c for c in ("ISI2", "ISI5", "ISI7") if c != committed)
    panel["panel_7domain"]["sleep"] = other
    with open(pj, "w") as fh:
        json.dump(panel, fh, indent=2)
    r = run_py(repo_copy, "03_network_contrib.py")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SLEEP anchor (%s)" % other in r.stdout
    assert "SLEEP anchor (%s)" % committed not in r.stdout
