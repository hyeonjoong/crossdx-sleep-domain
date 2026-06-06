# Real-data source and attribution

These CSVs are a **third-party open dataset** used for the real-data validation
(`analysis/06_realdata.py`). They are **not** redistributed as part of this project;
`06_realdata.py` downloads them from Zenodo on first run.

- **Dataset:** Su Z, Zhao C, et al. *Temporal dynamics in psychological assessments:
  a novel dataset with scales and response times.* Scientific Data (2024).
  doi:10.1038/s41597-024-03888-8
- **Repository:** Zenodo record 10423537 — https://zenodo.org/records/10423537
  (DOI 10.5281/zenodo.10423537)
- **License:** CC-BY 4.0 (reuse permitted with attribution).
- **Contents (item-level):** `isi.csv` (ISI, 7 items), `phq9.csv` (PHQ-9, 9 items),
  `gad7.csv` (GAD-7, 7 items), `pss.csv` (PSS-10), `demographic.csv`. Shared
  respondent key = `export_id`.
- **Sample:** N = 24,292 Chinese university students (health screening, 2021).

If you publish results derived from these files, cite the dataset above.

## Additional cohorts (07_multicohort.py) — auto-downloaded, not redistributed

### Cohort B — SRI adolescent insomnia (clinical) — `cohortB_sri/`
- de Zambotti M, Baker FC, et al. *A dataset reflecting the multidimensionality of
  insomnia symptomatology in adolescence using standardized questionnaires.* figshare.
  doi:10.6084/m9.figshare.20235492 — item-level file (figshare file 36167244).
- License: **MIT** (item-level CSV). N≈95 US adolescents (incl. clinical insomnia).
- Item-level ISI, BDI-II, STAI-Y2, PSS, FIRST, and other sleep scales.

### Cohort C — UK university students — `cohortC_uk/`
- Akram U, et al. *Prevalence of anxiety, depression, mania, insomnia, stress, suicidal
  ideation, psychotic experiences and loneliness in UK university students.* Scientific
  Data (2023). figshare article 24052236 (file 42177492).
- License: **CC-BY 4.0**. N=1,408 UK students.
- Item-level PHQ-9, GAD-7, PSS, SBQ-R (suicidality), SCI (Sleep Condition Indicator).
