# Pre-specified analysis plan — Sleep as the 7th cross-diagnostic domain

A confirmatory protocol to be run on **real** item-level data, using the identical pipeline
that produced the simulation-based proof-of-concept. Intended for OSF pre-registration.

## 1. Background & rationale
Lim et al. (2026) select one item per disorder to jointly screen six psychiatric domains but
omit sleep; Jo et al. (2024) shortened the ISI alone (ISI-3m). We add insomnia (ISI) as a
seventh domain and test whether the cross-diagnostic framework (a) selects an ISI-3m item,
(b) places insomnia as an internalizing bridge, and (c) yields incremental clinical value.

## 2. Datasets
- **D1 — EMBRAIN panel** (collaboration target with the Kim/Chung group): item-level PHQ-9,
  GAD-7, PCL-5, AUDIT, PDSS, DSI-SS **plus ISI**, Korean general-adult.
- **D2 — BELL-001 clinical cohort** (SERENE confirmatory; DAWN decentralized Phase 2):
  same battery in a clinical insomnia population, with objective markers (EEG high-beta/wPLI,
  HRV/TINN, nearable/Withings sleep parameters).

## 3. Design
Cross-sectional item selection + internal validation. Primary on D1; clinical generalization
+ objective-marker mapping on D2.

## 4. Variables
- Items: all instrument items (codes per `data/item_dictionary.csv`).
- Caseness labels: PHQ-9≥10, GAD-7≥10, PCL-5≥33, PDSS≥8, DSI-SS≥4, AUDIT≥8, ISI≥15
  (primary), ISI≥8 (secondary).

## 5. Primary analyses (identical to `analysis/`)
1. Utility U(j,k)=mean(AUROC,AUPRC) on development set.
2. Joint optimization `max ΣU − λ Σcos`, one item/domain, λ=0.10 (report λ sensitivity).
3. Strict 15% lockbox AUROC/AUPRC; panel vs anchor-only cross-diagnostic gain.
4. 300 bootstrap selection frequencies.
5. GGM partial-correlation network; node strength + betweenness.
6. Cross-diagnostic contribution weights.
7. Value of sleep: forward (Δ on 6 domains) and reverse (insomnia detection: anchor vs panel).

## 6. Pre-registered hypotheses
- **H1.** The sleep representative is an ISI-3m item (daytime-interference, maintenance, or
  concerns/worry). *Pass:* selected anchor ∈ {ISI2, ISI5, ISI7}.
- **H2.** Insomnia is an internalizing bridge: sleep node strength above the 7-node median
  and above the alcohol (externalizing) node.
- **H3.** The comorbid panel improves insomnia detection over the sleep anchor alone
  (lockbox AUROC gain > 0, bootstrap CI excludes 0).
- **H4 (D2 only).** Short-form sleep score correlates with objective markers at least as
  strongly as the ISI total (non-inferiority margin Δr ≥ −0.05).
- **H5 (exploratory).** Adding the sleep domain shifts the depression anchor away from the
  somatic (fatigue/sleep) items toward the affective core (de-confounding).

## 7. Sample size
D1 ≥ 1,500 (matched to source). Lockbox 15%. Rare domains (panic, PTSD) reported with
bootstrap CIs given AUPRC instability.

## 8. Deviations & reporting
Report any deviations from this plan; release code and (where permitted) data; report
selection frequencies and lockbox CIs; declare the simulation-based origin of all prior
parameter choices.

## 9. Ethics & collaboration
Real-data execution requires IRB approval and a data-use agreement; D1 contingent on
collaboration with the source group (collaboration > competition stance). No real data were
used in the present simulation-based manuscript.
