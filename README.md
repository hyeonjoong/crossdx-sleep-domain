# Sleep as the 7th Domain — extending the cross-diagnostic item-selection framework

This project extends the **cross-diagnostic optimization framework** of Lim et al. (2026)
— which selects one questionnaire item per psychiatric disorder to jointly screen six
disorders — by adding **insomnia (the 7-item Insomnia Severity Index, ISI) as a seventh
domain**. The original authors explicitly flagged sleep-domain integration as future work.

> ## ⚠️ Read this first — data transparency
> The empirical numbers here come from a **literature-calibrated SIMULATED cohort**, not
> the real EMBRAIN data used by the source papers (we do not have access to it). This
> package is therefore an honest **methods extension + reproducible pipeline + proof-of-
> concept + pre-specified analysis protocol**, not a clinical finding. Every number
> in results/ was produced by the code in analysis/ with fixed seeds; see "Reproduce"
> for how to check it on your machine. The pipeline runs unchanged on real item-level
> data (see "Running on real data" below).

## What it does

1. Simulates a Korean general-adult screening cohort (N=2,000; 64 items across PHQ-9,
   GAD-7, PCL-5, AUDIT, PDSS, DSI-SS, ISI) calibrated to published prevalences and
   inter-scale correlations.
2. Re-implements the multi-objective joint optimization
   `max Σ U(j,k) − λ Σ cos(x_p, x_q)`, one item per domain, and **verifies it reproduces
   the source framework** (it independently selects uncontrollable-worry, panic-distress,
   suicidal-ideation, binge-drinking, and the depression/anxiety>panic cross-diagnostic
   gain pattern).
3. Adds ISI as the 7th domain and reports which ISI item is selected, its network role,
   and the incremental value of the sleep domain.
4. Validates with a strict 15% lockbox and 300 bootstrap resamples (item utility is a
   rank statistic on the development split; no cross-validation is used);
   adds a Gaussian graphical model and cross-diagnostic contribution analysis.

## Key results

**Simulated (7 domains):**
- Sleep representative = **ISI daytime-interference**; across λ it alternates only with
  **ISI maintenance** — both **ISI-3m** items. In a 36-condition ablation (incl. sleep
  cross-loadings = 0) the sleep anchor is an ISI-3m item in **89%** of runs.
- Insomnia is an **internalizing bridge** (node strength 0.73; alcohol 0.13).
- Seven-domain lockbox **AUROC 0.87–0.97**. Adding sleep doesn't improve the other domains
  (ΔAUROC ≈ 0) but the comorbid panel improves insomnia detection (lockbox AUROC
  0.855 to 0.868).
- Emergent **de-confounding**: adding sleep shifts the depression anchor *fatigue → mood*.

**Real data (open, N = 24,292; ISI + PHQ-9 + GAD-7 — `06_realdata.py`):**
- Selected sleep anchor = **ISI maintenance**, robust across all λ; an **ISI-3m item in
  99.4%** of 300 bootstraps (maintenance 65% + concerns/worry 35%); onset/early-morning/
  satisfaction/noticeability ≈ 0% — reproducing ISI-3m's keeps and discards **on real data**.
- Anxiety anchor = **uncontrollable worry** (matches the source framework + simulation).
- Three-domain lockbox **AUROC 0.96–0.97**; total-score correlations match the literature.
- Confirms pre-registered **H1** on real, independent data (caveat: young low-prevalence
  Chinese student sample; 3 of 7 domains).

**Multi-cohort + domain extension (`07_multicohort.py`, 3 open datasets):**
- **Cohort C (UK, N=1,408)**: cross-population — anxiety anchor = a **worry** item again;
  **suicidality domain added** successfully (5-domain lockbox AUROC 0.92–0.95, suicide 0.95).
  1,406 of the 1,408 respondents had complete item data and were analysed.
- **Cohort B (US clinical insomnia, N=95)**: underpowered/unstable (ISI-3m item in 55% of
  bootstraps) — reported honestly. Spans China/UK/US and ≈1–47% prevalence.

## Folder map

The accompanying paper has been submitted to Behavior Research Methods.

```
crossdx-sleep-domain/
  README.md                        this file
  CHANGELOG.md                     changes between releases
  analysis/                        runnable pipeline
    config.py                      calibration parameters + item specification
    sim_core.py                    generative model / simulator
    pipeline_core.py               cohort loader with input checks, utility,
                                   redundancy, optimizer, evaluation
    01_simulate_cohort.py          data/*.csv, results/tables/T1, T1b
    02_optimize.py                 results/tables/T2-T7, results/panel.json
    03_network_contrib.py          results/tables/T8-T10, figures F4, F5
    04_figures.py                  figures F1-F3, F6, F7
    05_robustness_baselines.py     CIs, baselines, ablation: tables T11-T13, figure F8,
                                   results/robustness_summary.json
    06_realdata.py                 Cohort A real-data validation (downloads Zenodo
                                   10423537): results/realdata/RT1-RT4, RF1
    07_multicohort.py              Cohorts B and C (downloads from figshare): RT5-RT7, RF2
    08_cohortD.py                  Cohort D, internal BELL-001 data (not distributed;
                                   skipped when absent): RT8
    09_deconfounding.py            de-confounding redundancy analysis: table T14
    10_figures_brm.py              the 10 BRM submission figures (174 mm) redrawn
                                   from the tables: out/brm_figures/
    run_all.py                     runs every step; --check compares a fresh run in a
                                   temporary copy with the committed outputs
    _download.py                   verified downloads of the open datasets (SHA-256)
    build_docx.py                  manuscript .docx builders; they need the manuscript
    build_submission_docs.py       sources (manuscript/), which are not in this repository
    requirements.txt               minimum package versions
    requirements-lock.txt          exact versions used to verify the committed results
  data/
    simulated_cohort_items.csv     simulated cohort, the canonical input of the paper
    simulated_cohort_meta.csv      labels, totals and sex for the simulated cohort
    item_dictionary.csv            item codes, domains and loadings
    realdata/                      downloaded open data (not redistributed) + SOURCE.md
    private/                       your own data (git-ignored; see "Running on real data")
  results/
    tables/                        T1-T14 (.csv)
    figures/                       F1-F8 (.png, 300 dpi)
    realdata/                      RT1-RT8 (.csv), RF1, RF2 (.png)
    panel.json                     selected panels + summary metrics
    robustness_summary.json        ablation summary (step 05)
  protocol/
    analysis_plan.md               pre-specified confirmatory protocol
    research_brief_ISI.md          sourced facts: ISI + Jo et al. 2024
    research_brief_calibration.md  sourced calibration parameters
  tests/                           pytest suite (see "Tests")
```

## Reproduce

Python 3.11 or newer (verified with Python 3.14.2 on macOS).

```bash
cd crossdx-sleep-domain
python -m pip install -r analysis/requirements-lock.txt  # exact versions (Python 3.14); requirements.txt has minimums
python analysis/run_all.py --check                       # verify; changes nothing in this folder
```

`run_all.py --check` copies the repository to a temporary folder, runs every step
there and compares each file under data/ and results/ with this checkout. It
exits with 0 only when every regenerated file is identical. Files that no step
rewrote (RT8 without the internal Cohort D data) are listed as "not rerun" and are
not counted as identical. `--steps 02,03,04,09` checks a subset; `--keep-temp`
keeps the copy for inspection.

Platform note. The committed data/simulated_cohort_*.csv is the canonical input.
Step 01 draws the cohort with numpy's multivariate_normal, whose SVD factor has a
sign convention that depends on the BLAS/LAPACK build, and the step 05 ablation
(T13_ablation.csv, F8_robustness.png, robustness_summary.json) simulates new
cohorts the same way. With the versions in requirements-lock.txt on macOS
(Accelerate), these two do not regenerate the committed files, while steps 02, 03,
04, 06, 07 and 09 on the committed cohort reproduce every committed file they write
byte for byte and step 10 passes all of its checks. Other builds (for example
OpenBLAS on Linux) have not been verified and may differ in the last printed digit.
Step 01 therefore compares its draw with the committed cohort and, when they
differ, writes nothing and stops with exit status 3 (`--force` overwrites).
On such a build `run_all.py --check` lists the cohort and everything computed
from it as different, and `run_all.py --check --data-dir data` (which keeps the
committed cohort) lists only the three ablation files above.

To regenerate in place:

```bash
python analysis/run_all.py                   # stops at step 01 if its draw differs
python analysis/run_all.py --data-dir data   # keeps the committed cohort, runs steps 02-10
```

Where step 01 stops, the second command rewrites T13_ablation.csv, F8_robustness.png
and robustness_summary.json with different values, and step 10 then reports the
ablation mismatch (exit status 1). Do not commit those files; restore them with
`git checkout -- results/`.

Steps 06 and 07 need internet on first use: they download about 10 MB of open data
(Zenodo, figshare) into data/realdata/ and check each file's SHA-256 (see
data/realdata/SOURCE.md). Step 08 needs the internal Cohort D file and is skipped
when it is absent. With the data downloaded, a full run takes about 2 minutes on
a recent laptop (longer when the machine is busy).

BRM submission figures (174 mm column width, TIFF and PNG, with value checks):

```bash
python analysis/10_figures_brm.py --outdir out/brm_figures   # --strict: fail below the 2 mm lettering floor
```

The manuscript builders (build_docx.py, build_submission_docs.py) need the
manuscript sources in manuscript/, which are not part of this repository.

## Tests

```bash
python -m pip install pytest
python -m pytest                  # about 3 minutes, no internet needed
python -m pytest -m "not slow"    # fast checks only, a few seconds
CROSSDX_STRICT=1 python -m pytest # also require byte-identical outputs
```

The slow tests rerun steps 02, 03, 04, 09 and 10 in a temporary copy and compare
them with the committed results (numbers to one unit in the third decimal, text
exactly). The two tests for steps 01 and 05 are expected to fail (xfail) on
builds whose SVD sign convention differs, as described in the platform note.
A GitHub Actions workflow that runs the suite on Ubuntu and macOS is in .github/ci/tests.yml; see .github/ci/README.md to enable it.

## Running on real data (EMBRAIN / BELL-001)

Real item-level data must never be committed to this public repository. Put the
two CSV files in data/private/, which is git-ignored, using the same file names
and layout as the simulated cohort:
- `data/private/simulated_cohort_items.csv`: `subject_id`, then one column per item
  using the codes in `data/item_dictionary.csv` (PHQ1-PHQ9, GAD1-GAD7, PCL1-PCL20,
  AUDIT1-AUDIT10, PDSS1-PDSS7, DSI1-DSI4, ISI1-ISI7), scored from 0 (PHQ, GAD and
  DSI-SS items 0-3; ISI, PCL-5, AUDIT and PDSS items 0-4).
- `data/private/simulated_cohort_meta.csv`: `subject_id` and a 0/1 `label_<domain>`
  per domain (`label_dep`, `label_anx`, `label_ptsd`, `label_panic`, `label_suicide`,
  `label_alcohol`, `label_sleep`) plus `label_sleep_sub` (ISI >= 8). Other columns
  are kept.

Then run

```bash
python analysis/run_all.py --data-dir data/private
```

This skips step 01 (simulation) and step 10 (which checks the published numbers).
Single steps take the same option, for example
`python analysis/02_optimize.py --data-dir data/private`, and the environment
variable CROSSDX_DATA_DIR works for both. The loader matches the label rows to the
item rows by subject_id, reads UTF-8 (with or without BOM) and CP949 files, and
stops with a one-line message (exit status 2) naming the column and rows when a
value is blank, out of range or not 0/1. The results are written to results/ as
usual: do not commit them, and restore the published ones with
`git checkout -- results/`.
The pre-registered confirmatory analysis is in
`protocol/analysis_plan.md`.

## Provenance

Extends: Jo et al., *Sleep & Breathing* 2024;28(4):1819–1830 (ISI-3m); Lim et al.,
*npj Digital Medicine* 2026 (under review, cross-diagnostic 6-item framework).
Built from the internal briefing `ISI_Shortened_Forms_Briefing.pptx`.
