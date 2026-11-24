"""Small web service.  Start it with:  python backend/run.py   then open http://localhost:8000

Endpoints
  GET  /api/health            is it running?
  GET  /api/weeks             list of weeks that have scores
  GET  /api/call-list?week=YYYY-MM-DD&pct=5   the weekly top list (never more than 5%)
  GET  /api/features          the inputs the model uses (chosen by feature importance)
  POST /api/score             score one new discharge (JSON with exactly those input fields)
  POST /api/score-batch       score many discharges at once: {"admissions": [ {admission_id, ...input fields}, ... ]}
"""
import json
from pathlib import Path
from typing import Literal
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ConfigDict, Field, create_model

HERE = Path(__file__).resolve().parent
ART = HERE / "model_artifacts"
def _needs_training():
    """Train on first run, or when the saved model was made with a different scikit-learn."""
    if not (ART / "model.joblib").exists() or not (ART / "info.json").exists() or not (ART / "holdout_scored.csv").exists():
        return True
    import sklearn
    return json.load(open(ART / "info.json")).get("sklearn") != sklearn.__version__

if _needs_training():
    import subprocess, sys
    print("Preparing the model for this computer (takes a few seconds)...")
    subprocess.run([sys.executable, str(HERE.parent / "code/run_all.py")], check=True)

import sys; sys.path.insert(0, str(HERE.parent / "code"))
import run_all as R                                          # noqa: E402

model = joblib.load(ART / "model.joblib")
info = json.load(open(ART / "info.json"))
R.use_features(info["numbers"], info["categories"])           # the inputs the saved model was trained on
FEATURES, reasons = R.FEATURES, R.reasons
scored = pd.read_csv(ART / "holdout_scored.csv")
MAX_PCT = 5

app = FastAPI(title="Lakeshore readmission call list")


# Every candidate input with its allowed range. The form only asks for the inputs the model actually uses.
SPEC = {
    "hospital_site": (Literal["LS_North", "LS_South", "LS_East", "LS_West", "LS_Central", "LS_NorthCentral"], ...),
    "age": (int, Field(ge=18, le=110)),
    "sex": (Literal["F", "M"], ...),
    "insurance_type": (Literal["Medicare", "Medicaid", "Commercial", "Self-pay"], ...),
    "admission_source": (Literal["Emergency", "Referral", "Transfer"], ...),
    "primary_dx_group": (Literal["Heart failure", "COPD", "Pneumonia", "Sepsis", "Kidney disease", "Stroke", "Diabetes", "Cardiac arrhythmia", "GI bleed", "UTI", "Hip/knee surgery", "Other"], ...),
    "comorbidity_count": (int, Field(ge=0, le=20)),
    "los_days": (int, Field(ge=0, le=100)),
    "n_prior_admissions_12m": (int, Field(ge=0, le=30)),
    "n_ed_visits_6m": (int, Field(ge=0, le=30)),
    "num_medications": (float | None, Field(default=None, ge=0, le=60)),
    "bmi": (float | None, Field(default=None, ge=10, le=80)),
    "lab_creatinine_last": (float | None, Field(default=None, ge=0.1, le=15)),
    "lab_sodium_last": (float | None, Field(default=None, ge=100, le=180)),
    "lab_hemoglobin_last": (float | None, Field(default=None, ge=3, le=22)),
    "systolic_bp_last": (float | None, Field(default=None, ge=50, le=260)),
    "heart_rate_last": (float | None, Field(default=None, ge=20, le=220)),
    "discharge_disposition": (Literal["Home", "Home with services", "Skilled nursing", "Rehab"], ...),
    "weekend_discharge": (Literal[0, 1], ...),
    "followup_visit_scheduled": (Literal[0, 1], ...),
}
# One discharge. Values outside sensible ranges are rejected, and so is any input the model does not use.
Discharge = create_model("Discharge", __config__=ConfigDict(extra="forbid"), **{f: SPEC[f] for f in FEATURES})


# One row of a batch: the same inputs as a single discharge, plus an optional admission_id to echo back.
BatchRow = create_model("BatchRow", __config__=ConfigDict(extra="forbid"), admission_id=(str | None, None), **{f: SPEC[f] for f in FEATURES})
Batch = create_model("Batch", admissions=(list[BatchRow], Field(min_length=1, max_length=5000)))


@app.get("/api/health")
def health():
    return {"status": "ok", "model": info["model"], "weeks": int(scored.week_start.nunique())}


@app.get("/api/features")
def features():
    return FEATURES


@app.get("/api/weeks")
def weeks():
    return sorted(scored.week_start.unique().tolist())


@app.get("/api/call-list")
def call_list(week: str, pct: float = Query(5, gt=0)):
    if pct > MAX_PCT:
        raise HTTPException(400, f"The care team can call at most {MAX_PCT}% of discharges a week.")
    w = scored[scored.week_start == week]
    if w.empty:
        raise HTTPException(404, "No discharges for that week.")
    k = max(1, round(len(w) * pct / 100))
    top = w.nsmallest(k, "rank_in_week")
    cols = ["rank_in_week", "admission_id", "patient_id", "hospital_site", "age", "primary_dx_group", "risk_score", "reasons"]
    return {"week": week, "discharges": len(w), "list_size": k, "rows": top[cols].to_dict("records")}


@app.post("/api/score")
def score(d: Discharge):
    x = pd.DataFrame([d.model_dump()])[FEATURES]
    p = float(model.predict_proba(x)[:, 1][0])
    return {"risk_score": round(p, 4), "in_100": round(p * 100), "reasons": reasons(model, x)[0]}


@app.post("/api/score-batch")
def score_batch(b: Batch):
    """Score many discharges in one request. One bad row rejects the whole batch (422) and says which row."""
    rows = [r.model_dump() for r in b.admissions]
    x = pd.DataFrame(rows)[FEATURES]
    p = model.predict_proba(x)[:, 1]
    why = reasons(model, x)
    rank = pd.Series(p).rank(ascending=False, method="first").astype(int).tolist()   # 1 = highest risk in this batch
    out = [{"admission_id": r["admission_id"], "risk_score": round(float(p[i]), 4), "rank_in_batch": rank[i], "reasons": why[i]}
           for i, r in enumerate(rows)]
    return {"count": len(out), "results": out}


@app.middleware("http")
async def no_cache(request, call_next):   # always serve the latest page files
    r = await call_next(request)
    r.headers["Cache-Control"] = "no-store"
    return r

app.mount("/", StaticFiles(directory=HERE.parent / "frontend", html=True), name="frontend")
