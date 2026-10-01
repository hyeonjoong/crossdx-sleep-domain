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

## Scoring notes

- Cohort C: 1,406 of the 1,408 rows have complete PHQ-9, GAD-7, PSS, SBQ-R and SCI
  items and are analysed (complete-case).
- PSS (Cohorts B and C) and STAI-Y2 (Cohort B) items in the distributed files are
  already reverse-keyed: every item-rest correlation is positive, including the
  positively worded items (STAI-Y2 1, 3, 6, 7, 10, 13, 14, 16, 19), which raw
  coding would make negative. The scripts therefore sum the items as distributed.

## File checksums

06_realdata.py and 07_multicohort.py download each file only when it is missing
or its SHA-256 differs from the value below (the files behind the published
results), and stop with a message if a download is incomplete or different
(analysis/_download.py). They also check the analysed N (A 24,292; B 95; C 1,406).

| file | SHA-256 |
|---|---|
| isi.csv | 3e748c0fd7827a271bcd7347a5ca442445596be9c6dbe51743661111ad0d6889 |
| phq9.csv | 0e837b6cf289c4083ebd131e6bc7a52d63eefae3113a8738b57eb8ef51b9aece |
| gad7.csv | e47ec29e61f30cadee85ed66d39092fa47f41fecb47544f30626fc33b021218f |
| pss.csv | ad844acb6c4239be4daf2400bb101dda770b41989523c5c7185c6c468da46397 |
| demographic.csv | 33247e3a6f1a8d9b64da54ab8f482b1bf26decfa3a825398d8c5d4d7564af693 |
| cohortB_sri/sri_insomnia_items.csv | a68e167d8ad8d01b7a3f92b924ba109ae54e8e1c060b049ca586adb6ae9ad415 |
| cohortC_uk/uk_akram.xlsx | e04b2a75dd4c46021c8520ba013fada3db73d03a694fea6afc06769ca0ae4e64 |
