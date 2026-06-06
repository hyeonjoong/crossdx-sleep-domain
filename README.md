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
> regenerates from code with a fixed seed. The pipeline runs unchanged on real item-level
> data by swapping the loader — see "Running on real data" below.

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
4. Validates with a strict 15% lockbox, 5-fold CV, and 300 bootstrap resamples;
   adds a Gaussian graphical model and cross-diagnostic contribution analysis.

## Key results

**Simulated (7 domains):**
- Sleep representative = **ISI daytime-interference**; across λ it alternates only with
  **ISI maintenance** — both **ISI-3m** items. In a 36-condition ablation (incl. sleep
  cross-loadings = 0) the sleep anchor is an ISI-3m item in **89%** of runs.
- Insomnia is an **internalizing bridge** (node strength 0.73; alcohol 0.13).
- Seven-domain lockbox **AUROC 0.87–0.97**. Adding sleep doesn't improve the other domains
  (ΔAUROC ≈ 0) but the comorbid panel improves **insomnia detection** (0.85→0.87).
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
- **Cohort B (US clinical insomnia, N=95)**: underpowered/unstable (ISI-3m item in 55% of
  bootstraps) — reported honestly. Spans China/UK/US and ≈1–47% prevalence.

## Folder map

```
BELL_Paper3_Sleep_7th_Domain/
├── README.md                     ← you are here
├── analysis/                     ← runnable pipeline
│   ├── config.py                 calibration parameters + item specification
│   ├── sim_core.py               generative model / simulator
│   ├── pipeline_core.py          utility, redundancy, optimizer, evaluation
│   ├── 01_simulate_cohort.py     → data/*.csv, results/tables/T1*
│   ├── 02_optimize.py            → results/tables/T2–T7, results/panel.json
│   ├── 03_network_contrib.py     → results/tables/T8–T10, figures F4–F5
│   ├── 04_figures.py             → figures F1–F3, F6–F7
│   ├── 05_robustness_baselines.py → CIs, baselines, ablation; tables T11–T13, F8
│   ├── 06_realdata.py            → REAL-DATA validation (auto-downloads Zenodo 10423537)
│   ├── 07_multicohort.py        → 3-cohort validation + suicidality domain (auto-download)
│   ├── run_all.py                ← reproduce everything (≈ 1.5–2 min)
│   ├── build_docx.py             → manuscript/manuscript.docx
│   └── requirements.txt
├── data/
│   ├── simulated_cohort_*.csv    simulated cohort + item dictionary
│   └── realdata/                 open CC-BY dataset (downloaded) + SOURCE.md
├── results/
│   ├── tables/  T1–T13 (.csv)
│   ├── figures/ F1–F8 (.png, 300 dpi)
│   ├── realdata/ RT1–RT4 (.csv) + RF1_realdata.png
│   └── panel.json                selected panels + summary metrics
├── manuscript/                   ← TARGET: PLOS ONE (Research Article)
│   ├── manuscript.md / .docx     full manuscript, PLOS ONE format (figs+tables embedded)
│   ├── cover_letter.md / .docx   PLOS ONE cover letter
│   └── SUBMISSION_CHECKLIST_PLOS_ONE.md   pre-submission checklist + editor-risk note
└── protocol/
    ├── analysis_plan.md          pre-specified confirmatory protocol
    ├── research_brief_ISI.md     sourced facts: ISI + Jo et al. 2024
    └── research_brief_calibration.md  sourced calibration parameters
```

## Reproduce

```bash
cd BELL_Paper3_Sleep_7th_Domain
python -m pip install -r analysis/requirements.txt
python analysis/run_all.py            # regenerates data/, tables, figures
python analysis/build_docx.py         # rebuilds manuscript/manuscript.docx
```

## Running on real data (EMBRAIN / BELL-001)

Produce two CSVs and skip step 01:
- `data/simulated_cohort_items.csv` — columns: `subject_id`, then one column per item
  using the codes in `data/item_dictionary.csv` (PHQ1…PHQ9, GAD1…GAD7, PCL1…PCL20,
  AUDIT1…AUDIT10, PDSS1…PDSS7, DSI1…DSI4, ISI1…ISI7).
- `data/simulated_cohort_meta.csv` — columns: `subject_id`, `sex_male`, and a binary
  `label_<domain>` per domain (`label_dep`, …, `label_sleep`, plus `label_sleep_sub`).

Then run `02 → 03 → 04`. The pre-registered confirmatory analysis is in
`protocol/analysis_plan.md`.

## Provenance

Extends: Jo et al., *Sleep & Breathing* 2024;28(4):1819–1830 (ISI-3m); Lim et al.,
*npj Digital Medicine* 2026 (under review, cross-diagnostic 6-item framework).
Built from the internal briefing `ISI_Shortened_Forms_Briefing.pptx`.
