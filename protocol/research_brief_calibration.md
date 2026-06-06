# Research brief — calibration parameters [sourced]

Published parameters used to calibrate the simulated EMBRAIN-style Korean general-adult
cohort. Online-panel regime chosen where it differs from nationally representative samples.

## Per-instrument (items, scale, cutoff, prevalence)
| Instrument | Items | Scale | Caseness cutoff | Prevalence (Korean adult / panel) |
|---|---|---|---|---|
| PHQ-9 (depression) | 9 | 0–3 | ≥10 | ~6.5% national (KNHANES); 10–19% online/COVID panel. Used 12%. item3=sleep, item4=fatigue, item5=appetite |
| GAD-7 (anxiety) | 7 | 0–3 | ≥10 | ~5–11% (panel ~10.6%). Used 10% |
| PCL-5 (PTSD) | 20 | 0–4 | ≥31–33 (use 33) | ~3–8% general (Korea-specific thin). Used 6% |
| AUDIT (alcohol) | 10 | 0–4 | ≥8 | high-risk ~15% (men 23.7% ≫ women 4.2%). Used 15% + male effect |
| PDSS (panic) | 7 | 0–4 | ≥8 | ~2–4%. Used 4% |
| DSI-SS (suicidality) | 4 | 0–3 | **≥4 (Korean validation)** | ~15% young samples; lower general. Used 7% |
| ISI (insomnia) | 7 | 0–4 | ≥15 (mod), ≥8 (subthr.) | ≥15: 10–17%; ≥8: 25–40%. Used 12% / ~32% |

**Correction adopted:** DSI-SS Korean validation cutoff is ≥4 (Joiner original 3; some
population work ≥2). AUDIT has a strong male confound (modeled with a sex main effect).

## Cross-instrument structure (HiTOP)
- 2-factor internalizing/externalizing; internalizing splits into Distress
  (depression, GAD, PTSD, suicidality) and Fear (panic, PTSD cross-loads). Alcohol =
  externalizing. **Insomnia loads on internalizing (λ≈0.4–0.6) and acts as a bridge.**

## Pairwise total-score correlations (plug-in, ±0.1)
| Pair | r |
|---|---|
| depression–anxiety | 0.75 |
| depression–insomnia | 0.52 |
| anxiety–insomnia | 0.48 |
| depression–suicidality | 0.58 |
| suicidality–insomnia | 0.28 (weakest sleep link) |
| depression–PTSD | 0.60 |
| panic–anxiety | 0.55 |
| PTSD–insomnia | 0.50 |
| alcohol–internalizing | ~0.15 |

## Bridge symptoms (item-level)
- **Fatigue / low energy** — most consistent insomnia↔depression/anxiety bridge
  (PHQ-9 item 4 mechanically bridges to ISI).
- **Concentration / cognitive impairment** — daytime-dysfunction bridge.
- **Worry / cognitive pre-sleep arousal ("concerns about sleep")** — links insomnia to
  anxiety; high-centrality node.
- Caution: PHQ-9 sleep/fatigue items overlap ISI content → can inflate the depression–
  insomnia correlation (content overlap, not pure comorbidity).

## Selected sources
KNHANES PHQ-9 (PMC7193414); Korean PHQ-9/PHQ-2 validation (Psychiatry Investig 2023);
EMBRAIN online panel COVID survey (PMC8313395); Korean GAD-7 (PMC6431620); PCL-5 review
(PMC10292741); AUDIT-K (PMC3912263); KNHANES drinking (PLoS One 10.1371/journal.pone.0175299);
PDSS (PMID 11591432); Korean DSI-SS (PMC5639125); ISI-K (PMC4101097); HiTOP (PMC9122089);
insomnia bridge networks (PMC9224757; PMC12794031).
