"""Makes a simple fake hospital dataset with the 26 columns named in the case.
Run:  python synthetic_data_generator/generate.py

One row = one hospital discharge.
  data/train.csv    Jan 2021 - Dec 2023  (60,000 rows, has the answer column readmitted_30d)
  data/holdout.csv  Jan - Mar 2024       (8,000 rows, no answer column)

Patients are consistent: a patient_id keeps the same sex, insurance and underlying health
("frailty"), gets one year older each year, is never in two hospital stays at once, and never has a
new stay inside 30 days of a discharge (that would contradict readmitted_30d = 0).
About 30% of patients appear in more than one admission (70% once, 30% two to five times).
The chance of coming back is a simple weighted sum of a few columns. Three problems are planted:
  1. LEAK      post_discharge_call_logged is mostly filled in AFTER a patient is back.
  2. CENSORED  discharges in the last 30 days of 2023 have a blank answer (not known yet).
  3. MISSING   ~4% blanks in the five lab and vital columns, ~3% in num_medications and bmi.
"""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROWS, HOLDOUT_ROWS = 60_000, 8_000
TRAIN_START, TRAIN_DAYS = "2021-01-01", 1095       # Jan 2021 - Dec 2023
SHARE_MULTI = 0.30                                   # share of patients with more than one admission
INTERCEPT = -3.3
SITES = ["LS_North", "LS_South", "LS_East", "LS_West", "LS_Central", "LS_NorthCentral"]   # the six hospital sites
DX_EFFECT = {"Heart failure": 0.5, "COPD": 0.4, "Pneumonia": 0.2, "Sepsis": 0.35, "Kidney disease": 0.3, "Stroke": 0.1,
             "Diabetes": 0.0, "Cardiac arrhythmia": 0.05, "GI bleed": -0.05, "UTI": -0.1, "Hip/knee surgery": -0.4, "Other": -0.2}   # 12 groups
DISCHARGE_EFFECT = {"Home": 0.0, "Home with services": 0.1, "Skilled nursing": 0.25, "Rehab": 0.15}

COLS = ["admission_id", "patient_id", "hospital_site", "admit_date", "discharge_date", "age", "sex",
        "insurance_type", "admission_source", "primary_dx_group", "comorbidity_count", "los_days",
        "n_prior_admissions_12m", "n_ed_visits_6m", "num_medications", "bmi", "lab_creatinine_last",
        "lab_sodium_last", "lab_hemoglobin_last", "systolic_bp_last", "heart_rate_last",
        "discharge_disposition", "weekend_discharge", "followup_visit_scheduled",
        "post_discharge_call_logged", "readmitted_30d"]
assert len(COLS) == 26

# Plan how many admissions each patient has: 70% have one, 30% have two to five.
counts = np.where(rng.random(60_000) < SHARE_MULTI, rng.integers(2, 6, 60_000), 1)
cut = int(np.searchsorted(np.cumsum(counts), TRAIN_ROWS)) + 1
counts = counts[:cut].copy()
counts[-1] -= counts.sum() - TRAIN_ROWS
N_TRAIN_PATIENTS = len(counts)
N_PATIENTS = N_TRAIN_PATIENTS + 6_000                # extra patients appear for the first time in 2024
PAT = pd.DataFrame({
    "sex": rng.choice(["F", "M"], N_PATIENTS),
    "insurance_type": rng.choice(["Medicare", "Medicaid", "Commercial", "Self-pay"], N_PATIENTS, p=[0.45, 0.2, 0.3, 0.05]),
    "frailty": rng.normal(size=N_PATIENTS),     # hidden health: makes sick-looking columns move together
}, index=[f"P{i:05d}" for i in range(N_PATIENTS)])
PAT["age_2021"] = np.clip(np.round(65 + 12 * rng.normal(size=N_PATIENTS) + 3 * PAT.frailty), 18, 95)

HISTORY = {}                     # patient_id -> sorted admit dates of the training stays

def prior_admissions(pids, admits):
    """Number of this patient's earlier admissions in the 365 days before each admit date (true history, not random)."""
    out = np.zeros(len(pids), dtype=int)
    for i, (p, a) in enumerate(zip(pids, admits)):
        h = HISTORY.get(p)
        if h is not None:
            out[i] = np.searchsorted(h, a, side="left") - np.searchsorted(h, a - np.timedelta64(365, "D"), side="left")
    return out

def train_stays():
    """Dates for every training admission. A patient's stays never overlap and are 31+ days apart."""
    pid, disch, los = [], [], []
    for k, n in enumerate(counts):
        l = 1 + rng.poisson(2.8, n)
        while True:
            days = np.sort(rng.integers(0, TRAIN_DAYS, n))
            order = np.argsort(days, kind="stable")
            gap = np.diff(days) - l[order][1:]          # days between a discharge and the next admit
            if n == 1 or (gap > 30).all():
                break
        pid += [k] * n; disch += list(days); los += list(l[order])
    return np.array(pid), np.array(disch), np.array(los)

def make(start, n, days, id_start, patient_idx, discharge_day, los_days):
    d = pd.DataFrame({"admission_id": [f"A{id_start + i:06d}" for i in range(n)]})
    d["hospital_site"] = rng.choice(SITES, n)
    d["discharge_date"] = pd.Timestamp(start) + pd.to_timedelta(discharge_day, unit="D")
    d["los_days"] = los_days
    d["admit_date"] = d.discharge_date - pd.to_timedelta(d.los_days, unit="D")      # always before discharge
    d["patient_id"] = PAT.index.to_numpy()[patient_idx]
    p = PAT.loc[d.patient_id]
    z = p.frailty.to_numpy()
    years = ((d.admit_date - pd.Timestamp("2021-01-01")).dt.days / 365.25).to_numpy()
    d["age"] = np.floor(p.age_2021.to_numpy() + years).astype(int)                  # older each year
    d["sex"] = p.sex.to_numpy()
    d["insurance_type"] = p.insurance_type.to_numpy()
    d["admission_source"] = rng.choice(["Emergency", "Referral", "Transfer"], n, p=[0.6, 0.3, 0.1])
    d["primary_dx_group"] = rng.choice(list(DX_EFFECT), n)
    d["comorbidity_count"] = np.minimum(rng.poisson(np.exp(0.6 + 0.3 * z)), 8)
    d["n_prior_admissions_12m"] = prior_admissions(d.patient_id.to_numpy(), d.admit_date.to_numpy())   # real count from the patient's earlier stays
    d["n_ed_visits_6m"] = np.minimum(rng.poisson(np.exp(-0.8 + 0.4 * z)), 6)
    d["num_medications"] = np.clip(np.round(4 + 1.2 * d.comorbidity_count + rng.normal(0, 2.5, n)), 0, 30)
    d["bmi"] = np.round(np.clip(28 + 4 * rng.normal(size=n) + 0.5 * z, 15, 55), 1)
    d["lab_creatinine_last"] = np.round(np.clip(1.0 + 0.2 * z + 0.25 * rng.normal(size=n), 0.4, 6), 2)
    d["lab_sodium_last"] = np.round(138 - 0.8 * z + rng.normal(0, 3, n), 1)
    d["lab_hemoglobin_last"] = np.round(np.clip(12.5 - 0.5 * z + rng.normal(0, 1.5, n), 6, 18), 1)
    d["systolic_bp_last"] = np.round(np.clip(128 + 15 * rng.normal(size=n), 80, 220))
    d["heart_rate_last"] = np.round(np.clip(78 + 11 * rng.normal(size=n) + 2 * z, 40, 150))
    d["discharge_disposition"] = rng.choice(list(DISCHARGE_EFFECT), n, p=[0.65, 0.15, 0.12, 0.08])
    d["weekend_discharge"] = d.discharge_date.dt.dayofweek.isin([5, 6]).astype(int)
    d["followup_visit_scheduled"] = (rng.random(n) < 0.5).astype(int)
    logit = (INTERCEPT + 0.03 * (d.age - 67) + 0.30 * d.comorbidity_count + 0.45 * d.n_prior_admissions_12m
             + 0.20 * d.n_ed_visits_6m + 0.05 * d.los_days + 0.04 * d.num_medications
             + d.primary_dx_group.map(DX_EFFECT) + d.discharge_disposition.map(DISCHARGE_EFFECT)
             - 0.5 * d.followup_visit_scheduled + 0.4 * (d.lab_sodium_last < 135)
             + 0.3 * (d.lab_creatinine_last > 1.5) + 0.15 * d.weekend_discharge)
    d["readmitted_30d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # 1. leak: a call gets logged much more often for patients who came back
    d["post_discharge_call_logged"] = (rng.random(n) < np.where(d.readmitted_30d == 1, 0.78, 0.18)).astype(int)
    # 3. missing values
    for c, prob in [("lab_creatinine_last", .04), ("lab_sodium_last", .04), ("lab_hemoglobin_last", .04),
                    ("systolic_bp_last", .04), ("heart_rate_last", .04), ("num_medications", .03), ("bmi", .03)]:
        d.loc[rng.random(n) < prob, c] = np.nan
    return d[COLS]

pid, disch, los = train_stays()
_a = pd.Series(pd.Timestamp(TRAIN_START) + pd.to_timedelta(disch - los, unit="D")).groupby(PAT.index.to_numpy()[pid])
HISTORY.update({k: np.sort(v.values) for k, v in _a})
train = make(TRAIN_START, TRAIN_ROWS, TRAIN_DAYS, 0, pid, disch, los)

# Holdout (Jan - Mar 2024): about 30% of the admissions are returning patients from the training years
# (their last discharge was 31+ days earlier), the rest are new patients. Nobody appears twice here.
last = train.assign(k=pid).groupby("k").discharge_date.max()
free = last.index[last < pd.Timestamp("2023-11-25")].to_numpy()
back = rng.choice(free, int(0.30 * HOLDOUT_ROWS), replace=False)
new = rng.choice(np.arange(N_TRAIN_PATIENTS, N_PATIENTS), HOLDOUT_ROWS - len(back), replace=False)
h_pid = rng.permutation(np.concatenate([back, new]))
holdout = make("2024-01-01", HOLDOUT_ROWS, 91, 100000, h_pid, rng.integers(0, 91, HOLDOUT_ROWS), 1 + rng.poisson(2.8, HOLDOUT_ROWS))
# 2. censoring: discharged in the last 30 days of the extract -> outcome not known yet
train["readmitted_30d"] = train["readmitted_30d"].astype("Int64")
train.loc[train.discharge_date > "2023-12-01", "readmitted_30d"] = pd.NA

(ROOT / "data").mkdir(exist_ok=True); (ROOT / "organizer_sim").mkdir(exist_ok=True)
train.to_csv(ROOT / "data/train.csv", index=False)
holdout.drop(columns="readmitted_30d").to_csv(ROOT / "data/holdout.csv", index=False)
holdout[["admission_id", "readmitted_30d"]].to_csv(ROOT / "organizer_sim/holdout_labels_SIMULATED.csv", index=False)
print("columns", len(COLS), "| train", len(train), "| holdout", len(holdout),
      "| base rate", round(train.readmitted_30d.dropna().astype(int).mean(), 3))
