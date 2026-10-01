# Changelog

## Unreleased

Infrastructure, input checks, documentation and tests only. No number, table or
figure that the pipeline produces has changed: the committed data/ and results/
are untouched, and the computations are unchanged (the new input checks return
the committed tables exactly as read).

### How this was checked

On macOS (Accelerate) with the versions in analysis/requirements-lock.txt:
- Steps 02, 03, 04, 06, 07 and 09, each run on its own on the committed cohort
  (06 and 07 on open-data files with the SHA-256 listed in
  data/realdata/SOURCE.md), rewrite every committed file under data/ and
  results/ with identical bytes. RF1 and RF2 are now identical also when 06 and
  07 run on their own (before, only when 04 had run earlier in the same process).
- `python analysis/run_all.py --check --data-dir data` (all steps except 01 in a
  temporary copy): every regenerated file is byte-identical except T13_ablation.csv,
  F8_robustness.png and robustness_summary.json, which come from the step 05
  ablation and already differed on this build with v1.2.0 (SVD sign convention,
  see README, Reproduce).
- `10_figures_brm.py` on the committed tables: all 30 TIFF and PNG files are
  byte-identical to those written by the v1.2.0 script; 57 of 57 value checks
  pass; 10 of 10 figures are at or above the 2.0 mm lettering floor.
- The test suite (`python -m pytest`) passes, including byte-for-byte comparison
  with CROSSDX_STRICT=1.

### Fixed
- run_all.py stopped at step 10 since v1.1.0 ("the following arguments are
  required: --root"). Each step now gets its own command line (run_all's options
  never reach a step), step 10 gets `--root` and `--outdir out/brm_figures`, and a
  failing step is named ("[run_all] FAILED at <step>") with a non-zero exit status.
  `--root` of 10_figures_brm.py now defaults to the repository folder.
- Step 01 no longer replaces the committed cohort silently. It compares its draw
  with data/simulated_cohort_*.csv and, when they differ, writes nothing, says
  what differs (with the numpy version and BLAS/LAPACK library) and exits with
  status 3. `--force` overwrites.
- The cohort loader (pipeline_core.load_cohort) matched item rows and label rows
  by position and accepted any values. It now matches them by subject_id, reads
  UTF-8 (with or without BOM) and CP949 files, and stops with a one-line message
  (English and Korean, exit status 2) naming the column and rows for blank,
  non-numeric or out-of-range scores, labels other than 0/1, missing or renamed
  item columns and subject_id mismatches. Extra non-item columns are ignored with
  a note instead of entering the redundancy matrix.
- Steps 06 and 07 reused any file already on disk, so an interrupted download
  silently shrank Cohort A. Downloads now go through analysis/_download.py: a file
  is reused only when its SHA-256 matches, a download is checked for length and
  SHA-256 before it replaces anything, and the analysed N is checked (A 24,292;
  B 95; C 1,406).
- 10_figures_brm.py measured lettering size from dark pixels only, so the white
  node labels of Figure 3 were not counted and the figure was reported at 1.91 mm
  (FAIL). The measurement now draws all text black for its own render; Figure 3
  measures 2.33 mm. The figures themselves are unchanged. A failed value check is
  now recorded in figure_metrics.json and makes the script exit with status 1
  after all figures are drawn (before, the first mismatch raised and nothing was
  recorded). New `--strict` exits with status 1 when a figure is below the floor.
- build_submission_docs.py created an empty SUBMISSION/ folder on import. Both
  manuscript builders now stop with a clear message when manuscript/ (not part of
  this repository) is missing.
- 05, 06 and 07 now set the same matplotlib settings as 04, so F8, RF1 and RF2 do
  not depend on whether 04 ran first in the same process.
- Console messages of 03 and 09 named fixed items (ISI5; fatigue to depressed
  mood); they now name the items actually selected. 07 warns when its fixed
  Cohort A row of RT5 disagrees with RT2 and RT2b written by 06.

### Added
- `run_all.py --check`: runs the pipeline in a temporary copy and compares every
  file under data/ and results/ with the checkout (exit 0 only if all regenerated
  files are identical; files no step rewrote are listed as "not rerun").
  Also `--data-dir`, `--steps`, `--force` and `--keep-temp`.
- `--data-dir` (and $CROSSDX_DATA_DIR) for steps 02, 03, 05, 09 and run_all.py, so
  real data can live in data/private/ (git-ignored) instead of tracked files.
- analysis/requirements-lock.txt with the versions used for the checks above;
  openpyxl and pillow added to requirements.txt (07 reads an .xlsx file).
- Tests (tests/, pytest.ini) and a GitHub Actions workflow (Ubuntu and macOS).
- CITATION.cff: version 1.2.0 (latest tag), repository-code and url.
  .zenodo.json: related identifiers for the three open datasets and the Cohort A
  data paper (each DOI resolved before adding).

### Documentation
- README: correct folder map and repository name, outputs T1-T14, RT1-RT8,
  RF1-RF2 and F1-F8, the BRM figure command, internet use of 06 and 07, the skip
  of 08, the platform note on the SVD sign, `--check`, tests, and real data in
  data/private/. Removed "5-fold CV" (item utility is a rank statistic on the
  development split; config.N_FOLDS is marked unused), the PLOS ONE target and
  the build_docx step. Insomnia AUROC given as 0.855 to 0.868 and Cohort C as
  1,406 analysed of 1,408.
- data/realdata/SOURCE.md: file checksums and a note that PSS and STAI-Y2 items
  in the distributed files are already reverse-keyed.

When tagging the next release, set `version` and `date-released` in CITATION.cff
to that release.
