# -*- coding: utf-8 -*-
"""run_all.py: per-step arguments, failure reporting and --check (work order C1, C2, C5)."""
import os
import shutil
import sys

import pytest

import run_all
from conftest import ROOT, tree_hashes

STEP10 = "10_figures_brm.py"
DATA_STEPS = {"02_optimize.py", "03_network_contrib.py", "05_robustness_baselines.py",
              "09_deconfounding.py"}


@pytest.fixture
def calls(monkeypatch):
    """Replace runpy.run_path so no step really runs; record argv per step."""
    rec = []

    def fake(path, run_name=None):
        rec.append((os.path.basename(path), list(sys.argv)))

    monkeypatch.setattr(run_all.runpy, "run_path", fake)
    monkeypatch.delenv("CROSSDX_DATA_DIR", raising=False)
    return rec


def _path(step):
    return os.path.join(ROOT, "analysis", step)


def test_step_argv(calls):
    saved = list(sys.argv)
    assert run_all.main([]) == 0
    assert [c[0] for c in calls] == run_all.STEPS
    for name, argv in calls:
        if name == STEP10:
            assert argv == [_path(STEP10), "--root", ROOT,
                            "--outdir", os.path.join(ROOT, "out", "brm_figures")]
        else:
            assert argv == [_path(name)]
    assert sys.argv == saved


def test_run_all_flags_do_not_leak_into_steps(calls):
    run_all.main(["--force"])
    argv = dict(calls)
    assert argv["01_simulate_cohort.py"] == [_path("01_simulate_cohort.py"), "--force"]
    assert argv["02_optimize.py"] == [_path("02_optimize.py")]


def test_data_dir_skips_simulation_and_reaches_the_steps(calls, tmp_path, capsys):
    assert run_all.main(["--data-dir", str(tmp_path)]) == 0
    names = [c[0] for c in calls]
    # only the steps that use the cohort or its results: no simulation (01), no
    # open-data or internal cohorts (06-08, they need internet or internal data and
    # do not read the folder), no check of the published numbers (10)
    assert names == ["02_optimize.py", "03_network_contrib.py", "04_figures.py",
                     "05_robustness_baselines.py", "09_deconfounding.py"]
    for name, argv in calls:
        if name in DATA_STEPS:
            assert argv == [_path(name), "--data-dir", str(tmp_path)]
        else:
            assert argv == [_path(name)]
    out = capsys.readouterr().out
    for skipped in ("01_simulate_cohort.py", "06_realdata.py", "07_multicohort.py",
                    "08_cohortD.py", STEP10):
        assert "skipping %s" % skipped in out


def test_data_dir_with_named_open_data_steps_runs_them(calls, tmp_path):
    assert run_all.main(["--data-dir", str(tmp_path), "--steps", "02,06,08"]) == 0
    assert [c[0] for c in calls] == ["02_optimize.py", "06_realdata.py", "08_cohortD.py"]


def test_data_dir_on_committed_cohort_keeps_step10(calls):
    assert run_all.main(["--data-dir", os.path.join(ROOT, "data")]) == 0
    names = [c[0] for c in calls]
    assert names == [s for s in run_all.STEPS if s != "01_simulate_cohort.py"]


def test_env_var_is_honoured(calls, monkeypatch, tmp_path):
    monkeypatch.setenv("CROSSDX_DATA_DIR", str(tmp_path))
    assert run_all.main([]) == 0
    argv = dict(calls)
    assert "01_simulate_cohort.py" not in argv
    assert argv["02_optimize.py"] == [_path("02_optimize.py"), "--data-dir", str(tmp_path)]


def test_steps_option(calls):
    assert run_all.main(["--steps", "02,09"]) == 0
    assert [c[0] for c in calls] == ["02_optimize.py", "09_deconfounding.py"]


def test_unknown_step_is_rejected(calls, capsys):
    assert run_all.main(["--steps", "02,77"]) == 2
    assert calls == []
    assert "77" in capsys.readouterr().out


def test_failure_names_step(monkeypatch, capsys):
    monkeypatch.delenv("CROSSDX_DATA_DIR", raising=False)
    ran = []

    def fake(path, run_name=None):
        ran.append(os.path.basename(path))
        if path.endswith("03_network_contrib.py"):
            raise RuntimeError("boom")

    monkeypatch.setattr(run_all.runpy, "run_path", fake)
    rc = run_all.main([])
    assert rc != 0
    assert "[run_all] FAILED at 03_network_contrib.py" in capsys.readouterr().out
    assert "04_figures.py" not in ran


def test_exit_code_of_a_step_is_kept(monkeypatch, capsys):
    monkeypatch.delenv("CROSSDX_DATA_DIR", raising=False)

    def fake(path, run_name=None):
        if path.endswith("01_simulate_cohort.py"):
            raise SystemExit(3)
        if path.endswith("02_optimize.py"):
            raise SystemExit(0)          # an explicit clean exit is not a failure

    monkeypatch.setattr(run_all.runpy, "run_path", fake)
    assert run_all.main([]) == 3
    assert "FAILED at 01_simulate_cohort.py" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def _fake_runner(edit=None, rewrite=None, rc=0, seen=None):
    def runner(copy_root, child_args):
        if seen is not None:
            seen.append((copy_root, list(child_args)))
        if edit:
            p = os.path.join(copy_root, edit)
            with open(p, "a", encoding="utf-8") as fh:
                fh.write("extra,row\n")
        if rewrite:
            p = os.path.join(copy_root, rewrite)
            with open(p, "rb") as fh:
                data = fh.read()
            os.remove(p)
            with open(p, "wb") as fh:
                fh.write(data)
        return rc
    return runner


def test_check_mode_reports_diff(repo_copy, capsys):
    before, before_copy = tree_hashes(ROOT), tree_hashes(repo_copy)
    seen = []
    rc = run_all.check(root=repo_copy,
                       runner=_fake_runner(edit="results/tables/T14_deconfounding.csv",
                                           seen=seen))
    out = capsys.readouterr().out
    assert rc == 1
    assert "DIFFERS" in out and "T14_deconfounding.csv" in out
    assert tree_hashes(repo_copy) == before_copy  # the checkout is untouched
    assert tree_hashes(ROOT) == before
    copy_root, child_args = seen[0]
    assert copy_root != repo_copy and not os.path.exists(copy_root)   # temp copy removed
    assert "--force" in child_args


def test_check_mode_identical_rewrite_passes(repo_copy, capsys):
    rc = run_all.check(root=repo_copy,
                       runner=_fake_runner(rewrite="results/tables/T14_deconfounding.csv"))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "not rerun" in out             # files no step rewrote are not counted as same
    assert "[check] OK" in out


def test_check_mode_reports_failed_run(repo_copy, capsys):
    rc = run_all.check(root=repo_copy, runner=_fake_runner(rc=2))
    assert rc != 0
    assert "did not finish" in capsys.readouterr().out


def test_check_without_any_regenerated_file_fails(repo_copy, capsys):
    # e.g. --check --steps 08 without the internal Cohort D file: the run succeeds
    # but rewrites nothing, so nothing was verified
    rc = run_all.check(root=repo_copy, runner=_fake_runner())
    out = capsys.readouterr().out
    assert rc == 1
    assert "nothing was compared" in out
    assert "[check] OK" not in out


def test_check_keeps_committed_cohort_with_data_dir(repo_copy, capsys):
    seen = []
    run_all.check(data_dir=os.path.join(repo_copy, "data"), root=repo_copy,
                  runner=_fake_runner(seen=seen))
    copy_root, child_args = seen[0]
    i = child_args.index("--data-dir")
    # the committed data/ of the temporary copy is used, never the checkout
    assert child_args[i + 1] == os.path.join(copy_root, "data")


def _fake_root(tmp_path):
    """A tiny synthetic checkout with files under data/realdata/ (no real data)."""
    root = tmp_path / "root"
    files = ["data/a.csv", "results/tables/T.csv", "data/realdata/SOURCE.md",
             "data/realdata/isi.csv", "data/realdata/notes.txt",
             "data/realdata/cohortB_sri/sri_insomnia_items.csv",
             "data/realdata/cohortB_sri/other.csv",
             "data/realdata/" + run_all.COHORT_D_FILE,
             "data/realdata/cohortD_bell/raw_export.csv"]
    for f in files:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x,y\n1,2\n", encoding="utf-8")
    return root


def _listing(top):
    return sorted(os.path.relpath(os.path.join(dp, f), top).replace(os.sep, "/")
                  for dp, _dn, fn in os.walk(top) for f in fn)


def test_check_copies_only_known_realdata_files_and_removes_read_only_copy(tmp_path, capsys):
    root = _fake_root(tmp_path)
    locked = [root / "data" / "realdata" / "cohortD_bell", root / "data" / "realdata"]
    for d in locked:                       # read-only folders, as in a protected cache
        os.chmod(d, 0o555)
    copied = []

    def runner(copy_root, child_args):
        copied.append((copy_root, _listing(copy_root)))
        p = os.path.join(copy_root, "results", "tables", "T.csv")
        data = open(p, "rb").read()
        os.remove(p)
        with open(p, "wb") as fh:
            fh.write(data)
        return 0

    try:
        rc = run_all.check(root=str(root), runner=runner)
    finally:
        for d in locked:
            os.chmod(d, 0o755)
    out = capsys.readouterr().out
    assert rc == 0, out
    copy_root, listing = copied[0]
    realdata = [f for f in listing if f.startswith("data/realdata/")]
    assert realdata == sorted(["data/realdata/SOURCE.md", "data/realdata/isi.csv",
                               "data/realdata/cohortB_sri/sri_insomnia_items.csv",
                               "data/realdata/" + run_all.COHORT_D_FILE])
    # the temporary folder (and the Cohort D copy in it) is gone
    assert not os.path.exists(os.path.dirname(copy_root))
    assert "could not delete" not in out


def test_check_names_a_temporary_copy_it_could_not_delete(tmp_path, capsys, monkeypatch):
    root = _fake_root(tmp_path)
    seen = []
    monkeypatch.setattr(run_all.shutil, "rmtree", lambda *a, **k: None)
    run_all.check(root=str(root), runner=_fake_runner(seen=seen))
    monkeypatch.undo()
    work = os.path.dirname(seen[0][0])
    try:
        out = capsys.readouterr().out
        assert "could not delete the temporary copy %s" % work in out
    finally:
        shutil.rmtree(work, ignore_errors=True)


@pytest.mark.slow
def test_check_mode_runs_a_real_step(repo_copy, capsys):
    before, before_copy = tree_hashes(ROOT), tree_hashes(repo_copy)
    rc = run_all.check(data_dir=os.path.join(repo_copy, "data"), steps=["09"],
                       root=repo_copy)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "T14_deconfounding.csv" in out
    assert tree_hashes(repo_copy) == before_copy
    assert tree_hashes(ROOT) == before
