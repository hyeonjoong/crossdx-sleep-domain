# -*- coding: utf-8 -*-
"""
run_all.py: run the whole pipeline (fixed seeds).

    python analysis/run_all.py           run every step in this checkout
    python analysis/run_all.py --check   run every step in a temporary copy and
                                         compare its data/ and results/ with this
                                         checkout; nothing here is written

Options:
  --check           exit 0 when files were regenerated and every one of them is
                    identical to this checkout; 1 when a file differs, the run did
                    not finish, or no file under data/ or results/ was regenerated
                    (nothing was compared)
  --keep-temp       with --check: keep the temporary copy and print its path
  --data-dir DIR    read the cohort from DIR instead of data/ (see README,
                    "Running on real data"). Runs the steps that use the cohort
                    or its results (02, 03, 04, 05, 09). Skipped: 01 (it would
                    simulate a new cohort), 06-08 (they analyse the open and
                    internal cohorts, not DIR; still run when named in --steps)
                    and 10 (it checks the published numbers). When DIR is the
                    committed data/, only 01 is skipped. $CROSSDX_DATA_DIR has the
                    same effect.
  --steps 02,09     run only these steps (by number), in pipeline order
  --force           pass --force to step 01 (overwrite the committed cohort)

Writes: data/ (step 01), results/tables, results/figures, results/panel.json,
results/robustness_summary.json, results/realdata (06-08), out/brm_figures (10).
Steps 06 and 07 download about 10 MB of open data on first use.

--check copies the repository, including the open-data files and the internal
Cohort D file (if present) under data/realdata/, to a new temporary folder that
only the current user can read, and deletes it afterwards. If it cannot be
deleted, its path is printed.
"""
import argparse
import os
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from _download import MANIFEST  # noqa: E402  (open-data files copied by --check)

STEPS = [
    "01_simulate_cohort.py",
    "02_optimize.py",
    "03_network_contrib.py",
    "04_figures.py",
    "05_robustness_baselines.py",
    "06_realdata.py",
    "07_multicohort.py",
    "08_cohortD.py",
    "09_deconfounding.py",
    # Redraws the 10 submission figures at journal column width (174 mm) from the
    # result tables written by the steps above. Pure redraw: nothing is recomputed.
    # Key values are checked against the tables and the published numbers; a
    # mismatch is listed in figure_metrics.json and makes the step exit with 1.
    "10_figures_brm.py",
]
SIM_STEP = "01_simulate_cohort.py"
FIG_STEP = "10_figures_brm.py"
# steps that read the cohort and accept --data-dir
DATA_STEPS = {"02_optimize.py", "03_network_contrib.py", "05_robustness_baselines.py",
              "09_deconfounding.py"}
# steps that analyse other cohorts (open data, internal Cohort D), not the one in --data-dir
OTHER_COHORT_STEPS = {"06_realdata.py", "07_multicohort.py", "08_cohortD.py"}
DATA_DIR_ENV = "CROSSDX_DATA_DIR"

# never copied by --check (generated output, caches, private data)
_SKIP_NAMES = {".git", "out", "SUBMISSION", "__pycache__", ".pytest_cache", ".venv", "venv"}
# the only files under data/realdata/ that --check copies: SOURCE.md, the open-data
# files of 06 and 07, and the internal Cohort D file read by 08 (when present)
COHORT_D_FILE = "cohortD_bell/cohortD_items.csv"
_REALDATA_FILES = {"SOURCE.md", COHORT_D_FILE} | set(MANIFEST)
_REALDATA_DIRS = {f.rsplit("/", 1)[0] for f in _REALDATA_FILES if "/" in f}


# ---------------------------------------------------------------------------
# running the steps
# ---------------------------------------------------------------------------
def _same_dir(a, b):
    return os.path.realpath(a) == os.path.realpath(b)


def select_steps(spec):
    """Step names for a spec such as '02,9' (02_optimize.py and 09_deconfounding.py),
    in pipeline order.
    Raises ValueError naming the first unknown entry."""
    if not spec:
        return list(STEPS)
    wanted = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        hit = [s for s in STEPS if s == tok or s.split("_")[0] == tok.zfill(2)]
        if not hit:
            raise ValueError(tok)
        wanted.update(hit)
    return [s for s in STEPS if s in wanted]


def step_argv(step, data_dir=None, force=False):
    """sys.argv for one step. run_all's own options never reach a step."""
    argv = [os.path.join(HERE, step)]
    if step == SIM_STEP and force:
        argv.append("--force")
    if step in DATA_STEPS and data_dir:
        argv += ["--data-dir", data_dir]
    if step == FIG_STEP:
        argv += ["--root", ROOT, "--outdir", os.path.join(ROOT, "out", "brm_figures")]
    return argv


def plan(steps, data_dir=None, named=False):
    """(steps to run, [(skipped step, reason)]). `named` is True when the steps
    were chosen with --steps; 06-08 then run even with another cohort folder."""
    run, skipped = [], []
    own = bool(data_dir) and not _same_dir(data_dir, os.path.join(ROOT, "data"))
    for s in steps:
        if s == SIM_STEP and data_dir:
            skipped.append((s, "the cohort is read from %s" % data_dir))
        elif s == FIG_STEP and own:
            skipped.append((s, "it checks the published numbers of the committed cohort"))
        elif s in OTHER_COHORT_STEPS and own and not named:
            skipped.append((s, "it analyses the open or internal cohorts, not %s (name it "
                               "in --steps to run it anyway)" % data_dir))
        else:
            run.append(s)
    return run, skipped


def run_step(step, argv):
    """Run one step in this process; return its exit status (0 = success)."""
    saved = sys.argv
    sys.argv = list(argv)
    try:
        runpy.run_path(os.path.join(HERE, step), run_name="__main__")
        return 0
    except SystemExit as e:
        if e.code is None or e.code == 0:
            return 0
        if isinstance(e.code, int):
            return e.code
        print(e.code, file=sys.stderr)
        return 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        sys.argv = saved


def run_steps(steps, data_dir=None, force=False, named=False):
    t0 = time.time()
    run, skipped = plan(steps, data_dir, named)
    if data_dir:
        print("[run_all] cohort folder: %s" % data_dir)
    for s, why in skipped:
        print("[run_all] skipping %s: %s" % (s, why))
    for s in run:
        print("\n" + "=" * 70 + f"\n[run_all] {s}\n" + "=" * 70)
        rc = run_step(s, step_argv(s, data_dir, force))
        if rc != 0:
            print("\n[run_all] FAILED at %s (exit status %d)" % (s, rc))
            return rc
    print(f"\n[run_all] DONE in {time.time()-t0:.1f}s. See results/ and out/brm_figures/.")
    return 0


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def _ignore_for_copy(root):
    def ignore(dirpath, names):
        rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
        skip = {n for n in names if n in _SKIP_NAMES or n.endswith((".pyc", ".part"))}
        if rel == "data":
            skip.add("private")
        if rel == "data/realdata" or rel.startswith("data/realdata/"):
            sub = rel[len("data/realdata"):].strip("/")
            for n in names:
                p = "%s/%s" % (sub, n) if sub else n
                if p not in _REALDATA_FILES and p not in _REALDATA_DIRS:
                    skip.add(n)
        return skip
    return ignore


def _remove_tree(path):
    """Delete the temporary copy, also when it holds read-only folders (copytree
    keeps the permissions of the source). Returns True when it is gone."""
    for dirpath, dirnames, _files in os.walk(path):
        for d in dirnames:              # top-down: a folder is fixed before it is listed
            p = os.path.join(dirpath, d)
            if not os.path.islink(p):
                try:
                    os.chmod(p, stat.S_IMODE(os.lstat(p).st_mode) | stat.S_IRWXU)
                except OSError:
                    pass
    shutil.rmtree(path, ignore_errors=True)
    return not os.path.lexists(path)


def output_files(root):
    """Files compared by --check: data/* (top level) and everything under results/."""
    out = []
    data = os.path.join(root, "data")
    if os.path.isdir(data):
        out += [os.path.join("data", f) for f in os.listdir(data)
                if os.path.isfile(os.path.join(data, f)) and not f.startswith(".")]
    for dp, _dn, fn in os.walk(os.path.join(root, "results")):
        out += [os.path.relpath(os.path.join(dp, f), root) for f in fn if not f.startswith(".")]
    return sorted(out)


def _first_line_difference(a, b):
    with open(a, encoding="utf-8", errors="replace") as fa, \
            open(b, encoding="utf-8", errors="replace") as fb:
        la, lb = fa.read().splitlines(), fb.read().splitlines()
    for i in range(max(len(la), len(lb))):
        x = la[i] if i < len(la) else "<end of file>"
        y = lb[i] if i < len(lb) else "<end of file>"
        if x != y:
            k = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            start = max(0, k - 20)
            pre = "..." if start else ""
            return "line %d: %r vs %r" % (i + 1, pre + x[start:start + 60],
                                          pre + y[start:start + 60])
    return "line endings or encoding"


def _same_bytes(a, b):
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def compare(root, copy_root, reference, stamps):
    """[(status, relative path, detail)] for every compared file."""
    rows = []
    new_files = set(output_files(copy_root)) - set(reference)
    for rel in sorted(set(reference) | new_files):
        a, b = os.path.join(root, rel), os.path.join(copy_root, rel)
        if rel in new_files:
            rows.append(("NEW", rel, "written by the run, not in this checkout"))
        elif not os.path.exists(b):
            rows.append(("MISSING", rel, "removed by the run"))
        elif _same_bytes(a, b):
            if os.stat(b).st_mtime_ns == stamps.get(rel):
                rows.append(("not rerun", rel, "no step rewrote it"))
            else:
                rows.append(("same", rel, ""))
        else:
            detail = ""
            if rel.endswith((".csv", ".json", ".txt", ".md")):
                detail = "first difference at " + _first_line_difference(a, b)
            rows.append(("DIFFERS", rel, detail))
    return rows


def _git_changes(root):
    """Uncommitted changes under data/ and results/ (empty if git is unavailable)."""
    if not os.path.isdir(os.path.join(root, ".git")):
        return []
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        r = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "data",
                            "results"], capture_output=True, text=True, timeout=60, env=env)
    except Exception:
        return []
    return r.stdout.splitlines() if r.returncode == 0 else []


def _run_copy(copy_root, child_args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("MPLBACKEND", "Agg")
    env.pop(DATA_DIR_ENV, None)
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.call([sys.executable, os.path.join(copy_root, "analysis", "run_all.py")]
                           + list(child_args), cwd=copy_root, env=env)


def check(data_dir=None, steps=None, keep_temp=False, runner=None, root=ROOT):
    """Run the pipeline in a temporary copy of `root` and compare the outputs.
    `steps` is a list of step numbers or names (None = all). Returns the exit status."""
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="crossdx-check-")
    copy_root = os.path.join(work, "repo")
    try:
        print("[check] copying the repository to %s" % copy_root, flush=True)
        shutil.copytree(root, copy_root, ignore=_ignore_for_copy(root))
        reference = output_files(root)
        stamps = {rel: os.stat(os.path.join(copy_root, rel)).st_mtime_ns
                  for rel in reference if os.path.exists(os.path.join(copy_root, rel))}
        child = ["--force"]
        if data_dir:
            dd = os.path.abspath(data_dir)
            if _same_dir(dd, os.path.join(root, "data")):
                dd = os.path.join(copy_root, "data")
            child += ["--data-dir", dd]
        if steps:
            child += ["--steps", ",".join(str(s) for s in steps)]
        print("[check] running: run_all.py %s" % " ".join(child), flush=True)
        rc = (runner or _run_copy)(copy_root, child)

        rows = compare(root, copy_root, reference, stamps)
        print("\n[check] comparison with %s" % root)
        for status, rel, detail in rows:
            print("  %-10s %s%s" % (status, rel, ("  (%s)" % detail) if detail else ""))
        count = {s: sum(1 for r in rows if r[0] == s)
                 for s in ("same", "DIFFERS", "not rerun", "MISSING", "NEW")}
        print("[check] %d identical, %d differ, %d not rerun (no step rewrote them), "
              "%d missing, %d new" % (count["same"], count["DIFFERS"], count["not rerun"],
                                      count["MISSING"], count["NEW"]))
        changes = _git_changes(root)
        if changes:
            print("[check] note: %d uncommitted change(s) under data/ or results/; the "
                  "comparison is against these working-tree files, not the last commit"
                  % len(changes))
        bad = count["DIFFERS"] + count["MISSING"] + count["NEW"]
        if rc != 0:
            print("[check] the pipeline run did not finish (exit status %d, see FAILED "
                  "above); steps after the failing one were not run" % rc)
        print("[check] finished in %.0f s" % (time.time() - t0))
        if rc != 0 or bad:
            return 1
        if count["same"] == 0:
            print("[check] FAILED: no file under data/ or results/ was regenerated, so "
                  "nothing was compared (the selected steps were skipped, see above, or "
                  "write elsewhere: step 10 writes to out/)")
            return 1
        print("[check] OK: every regenerated file is identical to this checkout")
        return 0
    finally:
        if keep_temp:
            print("[check] temporary copy kept at %s" % copy_root)
        elif not _remove_tree(work):
            print("[check] WARNING: could not delete the temporary copy %s; delete it "
                  "by hand (it holds a copy of this repository, including the files "
                  "under data/realdata/)" % work)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the crossdx-sleep-domain pipeline.",
                                 epilog="See the module docstring or README, Reproduce.")
    ap.add_argument("--check", action="store_true",
                    help="run in a temporary copy and compare outputs with this checkout")
    ap.add_argument("--keep-temp", action="store_true",
                    help="with --check: keep the temporary copy")
    ap.add_argument("--data-dir", default=None,
                    help="folder with the cohort CSVs (default: $%s, else data/)" % DATA_DIR_ENV)
    ap.add_argument("--steps", default=None, help="comma-separated step numbers, e.g. 02,03")
    ap.add_argument("--force", action="store_true",
                    help="let step 01 overwrite the committed cohort")
    args = ap.parse_args(argv)

    data_dir = args.data_dir or os.environ.get(DATA_DIR_ENV) or None
    if data_dir:
        data_dir = os.path.abspath(data_dir)
    try:
        steps = select_steps(args.steps)
    except ValueError as e:
        print("[run_all] unknown step %s; steps are %s"
              % (e, ", ".join(s.split("_")[0] for s in STEPS)))
        return 2
    if args.check:
        spec = [t.strip() for t in args.steps.split(",") if t.strip()] if args.steps else None
        return check(data_dir=data_dir, steps=spec, keep_temp=args.keep_temp)
    return run_steps(steps, data_dir=data_dir, force=args.force, named=bool(args.steps))


if __name__ == "__main__":
    sys.exit(main())
