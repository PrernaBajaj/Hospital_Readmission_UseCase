"""Makes a simple fake hospital dataset with the 26 columns named in the case.
Run:  python synthetic_data_generator/generate.py

One row = one hospital discharge, about 600 a week (the case says 60,000 train, 8,000 holdout).
  data/train.csv    Jan 2022 - Dec 2023  (60,000 rows, has the answer column readmitted_30d)
  data/holdout.csv  Jan - Mar 2024       (8,000 rows, no answer column)

Everything is generated in the plainest way: each column is drawn at random, and the chance
of coming back is a simple weighted sum of a few columns. Three problems are planted:
  1. LEAK      post_discharge_call_logged is only filled in AFTER a patient is back.
  2. CENSORED  discharges in the last 30 days of 2023 have a blank answer (not known yet).
  3. MISSING   ~5% blanks in the lab columns, ~3% in num_medications and bmi.
"""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROWS, HOLDOUT_ROWS = 60_000, 8_000
DX_EFFECT = {"Heart failure": 0.5, "COPD": 0.4, "Pneumonia": 0.2, "Diabetes": 0.0, "Other": -0.2}
DISCHARGE_EFFECT = {"Home": 0.0, "Home health": 0.1, "Skilled nursing": 0.25}

COLS = ["admission_id", "patient_id", "hospital_site", "admit_date", "discharge_date", "age", "sex",
        "insurance_type", "admission_source", "primary_dx_group", "comorbidity_count", "los_days",
        "n_prior_admissions_12m", "n_ed_visits_6m", "num_medications", "bmi", "lab_creatinine_last",
        "lab_sodium_last", "lab_hemoglobin_last", "systolic_bp_last", "heart_rate_last",
        "discharge_disposition", "weekend_discharge", "followup_visit_scheduled",
        "post_discharge_call_logged", "readmitted_30d"]
assert len(COLS) == 26

def make(start, n, days, age_shift, intercept, id_start):
    z = rng.normal(size=n)                          # hidden "frailty": makes sick-looking columns move together
    d = pd.DataFrame({
        "admission_id": [f"A{id_start + i:06d}" for i in range(n)],
        "patient_id": [f"P{i:05d}" for i in rng.integers(0, 20000, n)],
        "hospital_site": rng.choice(["LS-North", "LS-South", "LS-East", "LS-West"], n),
    })
    d["discharge_date"] = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n), unit="D")
    d["age"] = np.clip(np.round(67 + age_shift + 12 * rng.normal(size=n) + 3 * z), 18, 98).astype(int)
    d["sex"] = rng.choice(["F", "M"], n)
    d["insurance_type"] = rng.choice(["Medicare", "Medicaid", "Commercial", "Self-pay"], n, p=[0.45, 0.2, 0.3, 0.05])
    d["admission_source"] = rng.choice(["Emergency", "Referral", "Transfer"], n, p=[0.6, 0.3, 0.1])
    d["primary_dx_group"] = rng.choice(list(DX_EFFECT), n)
    d["comorbidity_count"] = np.minimum(rng.poisson(np.exp(0.6 + 0.3 * z)), 8)
    d["los_days"] = 1 + rng.poisson(np.exp(1.0 + 0.15 * z))
    d["admit_date"] = d.discharge_date - pd.to_timedelta(d.los_days, unit="D")
    d["n_prior_admissions_12m"] = np.minimum(rng.poisson(np.exp(-1.2 + 0.5 * z)), 6)
    d["n_ed_visits_6m"] = np.minimum(rng.poisson(np.exp(-0.8 + 0.4 * z)), 6)
    d["num_medications"] = np.clip(np.round(4 + 1.2 * d.comorbidity_count + rng.normal(0, 2.5, n)), 0, 30)
    d["bmi"] = np.round(np.clip(28 + 4 * rng.normal(size=n) + 0.5 * z, 15, 55), 1)
    d["lab_creatinine_last"] = np.round(np.clip(1.0 + 0.2 * z + 0.25 * rng.normal(size=n), 0.4, 6), 2)
    d["lab_sodium_last"] = np.round(138 - 0.8 * z + rng.normal(0, 3, n), 1)
    d["lab_hemoglobin_last"] = np.round(np.clip(12.5 - 0.5 * z + rng.normal(0, 1.5, n), 6, 18), 1)
    d["systolic_bp_last"] = np.round(np.clip(128 + 15 * rng.normal(size=n), 80, 220))
    d["heart_rate_last"] = np.round(np.clip(78 + 11 * rng.normal(size=n) + 2 * z, 40, 150))
    d["discharge_disposition"] = rng.choice(list(DISCHARGE_EFFECT), n, p=[0.75, 0.15, 0.10])
    d["weekend_discharge"] = d.discharge_date.dt.dayofweek.isin([5, 6]).astype(int)
    d["followup_visit_scheduled"] = (rng.random(n) < 0.5).astype(int)
    logit = (intercept + 0.03 * (d.age - 67) + 0.30 * d.comorbidity_count + 0.45 * d.n_prior_admissions_12m
             + 0.20 * d.n_ed_visits_6m + 0.05 * d.los_days + 0.04 * d.num_medications
             + d.primary_dx_group.map(DX_EFFECT) + d.discharge_disposition.map(DISCHARGE_EFFECT)
             - 0.5 * d.followup_visit_scheduled + 0.4 * (d.lab_sodium_last < 135)
             + 0.3 * (d.lab_creatinine_last > 1.5) + 0.15 * d.weekend_discharge)
    d["readmitted_30d"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # 1. leak: a call gets logged much more often for patients who came back
    d["post_discharge_call_logged"] = (rng.random(n) < np.where(d.readmitted_30d == 1, 0.78, 0.18)).astype(int)
    # 3. missing values
    for c, p in [("lab_creatinine_last", .05), ("lab_sodium_last", .05), ("lab_hemoglobin_last", .05),
                 ("num_medications", .03), ("bmi", .03)]:
        d.loc[rng.random(n) < p, c] = np.nan
    return d[COLS]

train = make("2022-01-01", TRAIN_ROWS, 730, 0, -3.3, 0)
holdout = make("2024-01-01", HOLDOUT_ROWS, 91, 2, -3.3, 100000)       # 2024 patients are a little older
# 2. censoring: discharged in the last 30 days of the extract -> outcome not known yet
train["readmitted_30d"] = train["readmitted_30d"].astype("Int64")
train.loc[train.discharge_date > "2023-12-01", "readmitted_30d"] = pd.NA

(ROOT / "data").mkdir(exist_ok=True); (ROOT / "organizer_sim").mkdir(exist_ok=True)
train.to_csv(ROOT / "data/train.csv", index=False)
holdout.drop(columns="readmitted_30d").to_csv(ROOT / "data/holdout.csv", index=False)
holdout[["admission_id", "readmitted_30d"]].to_csv(ROOT / "organizer_sim/holdout_labels_SIMULATED.csv", index=False)
print("columns", len(COLS), "| train", len(train), "| holdout", len(holdout),
      "| base rate", round(train.readmitted_30d.dropna().astype(int).mean(), 3))
