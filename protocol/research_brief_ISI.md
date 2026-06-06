# Research brief — ISI psychometrics + Jo et al. (2024) [sourced]

Compiled during project scoping to ground the simulation calibration and manuscript
claims. Peer-reviewed sources prioritized; uncertainties flagged.

## ISI (Insomnia Severity Index)
- 7 items, 0–4 each, total 0–28, 2-week recall (Bastien 2001; Morin 2011).
- Items: (1) sleep-onset severity, (2) maintenance severity, (3) early-morning awakening,
  (4) sleep dissatisfaction, (5) interference with daytime functioning, (6) noticeability
  to others, (7) worry/distress about sleep. **Caveat:** some papers renumber items 5/6/7;
  content is identical — anchor item numbers to the cited paper.
- Severity bands: 0–7 none; 8–14 subthreshold; 15–21 moderate clinical; 22–28 severe.
  Community screening cutoff ≥10 (Se 86.1%, Sp 87.7%); clinical-discrimination cutoff ≥14
  (Morin 2011). Korean ISI-K optimal ≈ 15.5 (Cho et al.).
- Factor structure debated: 2-factor (nocturnal severity vs daytime impact) most robust;
  3-factor in some samples; noticeability item is the most problematic. ISI is unidimensional
  in original treatment.
- α = 0.90–0.91. Item–total r: satisfaction/interference/worry highest (0.79–0.85),
  nocturnal-symptom items lower (0.50–0.66). **IRT: sleep-onset (item 1) and early-morning
  (item 3) show weak discrimination** — context for why ISI-3m drops onset.
- Total-score benchmarks: general/community ISI ≈ 5–8; clinical insomnia ≈ 17–20
  (sample-dependent; tie any norm to a named cohort).

## Jo et al. (2024) — ISI-3m  [all claims CONFIRMED]
- *Sleep & Breathing* 2024;28(4):1819–1830; doi:10.1007/s11325-024-03037-w; PMID 38684641.
- N = 800, EMBRAIN survey system (Korean panel).
- Method: EFA clusters items → XGBoost picks representative item per cluster.
- ISI-3m = **sleep maintenance + interference with daytime function + concerns/worry about
  sleep**. Sleep-onset and satisfaction NOT selected.
- Performance: R² = 0.910 for ISI total; multi-class (4-level) accuracy 0.965; outperforms
  four prior shortened versions.
- To verify against the paywalled PDF before submission: exact title punctuation; "Korean
  online panel" wording in Methods.

## ISI relationships & comorbidity
- ISI × depression r ≈ 0.48–0.50 (Morin 2011, BDI); ISI × anxiety r ≈ 0.48–0.50
  (STAI/BAI). Higher in young-adult/screening samples (up to ~0.63–0.69 with PHQ-9/GAD-7).
  Defensible range r ≈ 0.45–0.65.
- Comorbidity ~40–50% overall; insomnia present in ~50–70% of mood and ~70–90% of anxiety
  disorders; chronic insomnia ~2–5× odds of incident depression/anxiety (the "40×" figure
  is an outlier — do not cite).

## Primary sources
- Jo et al. 2024 — doi:10.1007/s11325-024-03037-w (PMID 38684641)
- Bastien, Vallières, Morin 2001 — doi:10.1016/S1389-9457(00)00065-4 (PMID 11438246)
- Morin, Belleville, Bélanger, Ivers 2011 — doi:10.1093/sleep/34.5.601 (PMID 21532953)
- ISI factor structure (lemborexant) — doi:10.1186/s41687-024-00744-6 (PMC11217251)
- Comorbid insomnia reviews — PMC11980635; PMC5906087
