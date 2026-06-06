# -*- coding: utf-8 -*-
"""
config.py — Calibration parameters for the simulated multi-questionnaire cohort.

This file defines a literature-calibrated GENERATIVE MODEL used to produce a
synthetic EMBRAIN-style Korean general-adult screening cohort. All parameters are
grounded in published psychometrics (see references in manuscript). The cohort is
SIMULATED; it is NOT the real EMBRAIN data used by Jo et al. (2024) or Lim et al.
(2026). The same downstream pipeline runs unchanged on real item-level data by
replacing the loader in 01_simulate_cohort.py.

Domain order (7 domains = the 6 of Lim et al. 2026 + Sleep as the 7th):
    dep, anx, ptsd, panic, suicide, alcohol, sleep
"""

SEED = 20260605
N_SUBJECTS = 2000          # Lim et al. used N=1,600 EMBRAIN panel; we use 2,000 for estimate stability
LOCKBOX_FRAC = 0.15        # strict held-out lockbox (Lim et al. used 15%)
N_FOLDS = 5                # 5-fold CV for item-utility estimation
N_BOOTSTRAP = 300          # bootstrap resamples for selection-stability (Lim et al. used 300)
REDUNDANCY_LAMBDA = 0.10   # lambda on the cosine-redundancy penalty (tuned; see T7 sensitivity)

DOMAINS = ["dep", "anx", "ptsd", "panic", "suicide", "alcohol", "sleep"]
DOMAIN_LABEL = {
    "dep": "Depression (PHQ-9)",
    "anx": "Anxiety (GAD-7)",
    "ptsd": "PTSD (PCL-5)",
    "panic": "Panic (PDSS)",
    "suicide": "Suicidality (DSI-SS)",
    "alcohol": "Alcohol use (AUDIT)",
    "sleep": "Insomnia (ISI)  [NEW 7th domain]",
}

# ---------------------------------------------------------------------------
# Latent correlation matrix among the 7 disorder severities (standardized).
# Synthesized from HiTOP CFA loadings + published pairwise correlations
# (depression-anxiety ~.75; depression-insomnia ~.52; anxiety-insomnia ~.48;
#  depression-suicide ~.58; suicide-insomnia ~.28; depression-PTSD ~.60;
#  panic-anxiety ~.55; alcohol-internalizing ~.15; PTSD-insomnia ~.50).
# ---------------------------------------------------------------------------
#                 dep   anx   ptsd  panic suic  alc   sleep
LATENT_CORR = [
    [1.00, 0.75, 0.60, 0.45, 0.58, 0.15, 0.52],   # dep
    [0.75, 1.00, 0.60, 0.55, 0.45, 0.15, 0.48],   # anx
    [0.60, 0.60, 1.00, 0.50, 0.45, 0.18, 0.50],   # ptsd
    [0.45, 0.55, 0.50, 1.00, 0.35, 0.12, 0.40],   # panic
    [0.58, 0.45, 0.45, 0.35, 1.00, 0.20, 0.28],   # suicide
    [0.15, 0.15, 0.18, 0.12, 0.20, 1.00, 0.12],   # alcohol
    [0.52, 0.48, 0.50, 0.40, 0.28, 0.12, 1.00],   # sleep
]

# Alcohol latent gets an additive sex effect (males higher). beta on centered sex (male=+0.5).
SEX_MALE_PROB = 0.5
ALCOHOL_SEX_BETA = 0.70

# ---------------------------------------------------------------------------
# Per-instrument item specification.
#  Each item: (code, short_label, own_loading, n_categories, [ (cross_domain, loading), ... ])
#  - own_loading      : standardized loading on the item's PRIMARY domain latent
#  - cross-loadings   : create realistic transdiagnostic / bridge structure
#  - n_categories     : Likert categories (PHQ/GAD/DSI = 4 -> 0..3 ; ISI/PCL/AUDIT/PDSS = 5 -> 0..4)
#  Residual variance = 1 - sum(loading^2), so total item-eta variance = 1.
#
# Calibration intent for the SLEEP (ISI) items, grounded in literature:
#  - maintenance (ISI2): high own-loading -> strong own-domain anchor (retained in ISI-3m)
#  - worry/concerns (ISI7): bridges to anxiety (cognitive pre-sleep arousal) -> cross-diagnostic
#  - daytime interference (ISI5): bridges to depression (fatigue/daytime) -> cross-diagnostic
#  - onset (ISI1) & early-morning (ISI3): weak IRT discrimination (Morin 2011) -> NOT selected
# ---------------------------------------------------------------------------
ITEMS = {
    # ---- Sleep / Insomnia: ISI (7 items, 0-4) — the NEW 7th domain ----
    "sleep": [
        ("ISI1", "onset difficulty",          0.55, 5, []),
        ("ISI2", "maintenance",               0.78, 5, []),
        ("ISI3", "early-morning awakening",   0.55, 5, []),
        ("ISI4", "sleep dissatisfaction",     0.68, 5, [("dep", 0.15)]),
        ("ISI5", "daytime interference",      0.72, 5, [("dep", 0.30)]),
        ("ISI6", "noticeable to others",      0.58, 5, []),
        ("ISI7", "worry/concerns re sleep",   0.70, 5, [("anx", 0.35)]),
    ],
    # ---- Depression: PHQ-9 (9 items, 0-3) ----
    "dep": [
        ("PHQ1", "anhedonia",                 0.74, 4, []),
        ("PHQ2", "depressed mood",            0.78, 4, []),
        ("PHQ3", "sleep problems",            0.55, 4, [("sleep", 0.45)]),
        ("PHQ4", "fatigue",                   0.68, 4, [("sleep", 0.30), ("anx", 0.15)]),
        ("PHQ5", "appetite change",           0.70, 4, []),
        ("PHQ6", "worthlessness/guilt",       0.72, 4, [("suicide", 0.20)]),
        ("PHQ7", "concentration",             0.66, 4, [("anx", 0.20)]),
        ("PHQ8", "psychomotor",               0.60, 4, []),
        ("PHQ9", "suicidal thoughts",         0.62, 4, [("suicide", 0.55)]),
    ],
    # ---- Anxiety: GAD-7 (7 items, 0-3) ----
    "anx": [
        ("GAD1", "nervous/on edge",           0.78, 4, []),
        ("GAD2", "uncontrollable worry",      0.80, 4, [("dep", 0.20)]),
        ("GAD3", "worry too much",            0.79, 4, []),
        ("GAD4", "trouble relaxing",          0.74, 4, [("panic", 0.20)]),
        ("GAD5", "restlessness",              0.70, 4, [("panic", 0.25)]),
        ("GAD6", "irritable",                 0.66, 4, []),
        ("GAD7", "afraid something awful",    0.70, 4, [("panic", 0.30), ("ptsd", 0.20)]),
    ],
    # ---- PTSD: PCL-5 (20 items, 0-4), DSM-5 clusters B/C/D/E ----
    "ptsd": [
        ("PCL1",  "intrusive memories",       0.72, 5, [("anx", 0.15)]),
        ("PCL2",  "nightmares",               0.70, 5, [("sleep", 0.30)]),
        ("PCL3",  "flashbacks",               0.74, 5, []),
        ("PCL4",  "emotional distress at cue",0.73, 5, [("panic", 0.20)]),
        ("PCL5",  "physiological reactivity", 0.70, 5, [("panic", 0.25)]),
        ("PCL6",  "avoid memories",           0.68, 5, []),
        ("PCL7",  "avoid reminders",          0.67, 5, []),
        ("PCL8",  "amnesia",                  0.58, 5, []),
        ("PCL9",  "negative beliefs",         0.70, 5, [("dep", 0.20)]),
        ("PCL10", "blame",                    0.66, 5, [("dep", 0.20)]),
        ("PCL11", "negative emotions",        0.71, 5, [("dep", 0.25)]),
        ("PCL12", "loss of interest",         0.69, 5, [("dep", 0.30)]),
        ("PCL13", "detachment",               0.67, 5, [("dep", 0.20)]),
        ("PCL14", "emotional numbing",        0.66, 5, [("dep", 0.20)]),
        ("PCL15", "irritability/anger",       0.64, 5, [("alcohol", 0.15)]),
        ("PCL16", "recklessness",             0.60, 5, [("alcohol", 0.20)]),
        ("PCL17", "hypervigilance",           0.68, 5, [("anx", 0.20)]),
        ("PCL18", "startle",                  0.66, 5, [("panic", 0.20)]),
        ("PCL19", "concentration",            0.65, 5, [("dep", 0.20)]),
        ("PCL20", "sleep disturbance",        0.64, 5, [("sleep", 0.40)]),
    ],
    # ---- Panic: PDSS (7 items, 0-4) ----
    "panic": [
        ("PDSS1", "panic frequency",          0.78, 5, []),
        ("PDSS2", "panic distress",           0.80, 5, [("anx", 0.20)]),
        ("PDSS3", "anticipatory anxiety",     0.76, 5, [("anx", 0.30)]),
        ("PDSS4", "agoraphobic avoidance",    0.72, 5, []),
        ("PDSS5", "interoceptive avoidance",  0.70, 5, []),
        ("PDSS6", "work impairment",          0.66, 5, [("dep", 0.20)]),
        ("PDSS7", "social impairment",        0.66, 5, [("dep", 0.20)]),
    ],
    # ---- Suicidality: DSI-SS (4 items, 0-3) ----
    "suicide": [
        ("DSI1", "frequency of ideation",     0.85, 4, [("dep", 0.20)]),
        ("DSI2", "suicidal ideation/plan",    0.86, 4, []),
        ("DSI3", "control over ideation",     0.80, 4, []),
        ("DSI4", "impulses to act",           0.78, 4, []),
    ],
    # ---- Alcohol: AUDIT (10 items, 0-4) ----
    "alcohol": [
        ("AUDIT1",  "frequency",              0.72, 5, []),
        ("AUDIT2",  "typical quantity",       0.74, 5, []),
        ("AUDIT3",  "binge frequency",        0.80, 5, []),
        ("AUDIT4",  "impaired control",       0.76, 5, []),
        ("AUDIT5",  "failed expectations",    0.72, 5, []),
        ("AUDIT6",  "morning drinking",       0.70, 5, []),
        ("AUDIT7",  "guilt after drinking",   0.68, 5, [("dep", 0.20)]),
        ("AUDIT8",  "blackouts",              0.66, 5, []),
        ("AUDIT9",  "injuries",               0.62, 5, []),
        ("AUDIT10", "others concerned",       0.64, 5, []),
    ],
}

# ---------------------------------------------------------------------------
# Binary caseness cutoffs (>= cutoff is a positive label) and target prevalence
# in a Korean general-adult / online-panel regime.
# ---------------------------------------------------------------------------
CUTOFF = {
    "dep": 10,       # PHQ-9 >= 10
    "anx": 10,       # GAD-7 >= 10
    "ptsd": 33,      # PCL-5 >= 33
    "panic": 8,      # PDSS >= 8
    "suicide": 4,    # DSI-SS >= 4 (Korean validation)
    "alcohol": 8,    # AUDIT >= 8 (hazardous)
    "sleep": 15,     # ISI >= 15 (moderate clinical insomnia)  -- primary sleep label
}
SLEEP_SUBTHRESHOLD_CUTOFF = 8   # ISI >= 8 (subthreshold) -- secondary sensitivity analysis

TARGET_PREVALENCE = {
    "dep": 0.12,
    "anx": 0.10,
    "ptsd": 0.06,
    "panic": 0.04,
    "suicide": 0.07,
    "alcohol": 0.15,
    "sleep": 0.12,
}

# Per-instrument graded-response thresholds are defined by a base spacing plus a
# global difficulty offset that 01_simulate_cohort.py tunes to hit the target
# prevalence. Base thresholds for K categories (between-category cut points):
def base_thresholds(n_cat):
    # symmetric-ish spacing; difficulty offset added in the simulator
    if n_cat == 4:   # 0..3
        return [-0.55, 0.45, 1.45]
    if n_cat == 5:   # 0..4
        return [-0.80, 0.10, 0.95, 1.85]
    raise ValueError(n_cat)
