# Solution architecture

```
 synthetic_data_generator/generate.py
   (26 columns, 3 planted problems)
                 │
                 ▼
     data/train.csv                data/holdout.csv
     60,000 discharges             8,000 discharges
     Jan 2021 - Dec 2023           Jan - Mar 2024
     (with readmitted_30d)         (no target column)
                 │                         │
                 ▼                         │
 ┌───────────────────────────────────┐     │
 │ code/run_all.py                   │     │
 │  1. audit the data                │     │
 │  2. choose inputs (20 -> 10)      │     │
 │  3. split by time and by patient  │     │
 │  4. compare 3 models, pick        │     │
 │     logistic regression           │     │
 │  5. retrain on all known outcomes │◄────┘  scores every holdout row
 └───────────────────────────────────┘
        │                │                    │
        ▼                ▼                    ▼
 outputs/            predictions.csv    backend/model_artifacts/
validation_results    (8,000 rows,        model.joblib
                       score 0 to 1)      info.json
                                           holdout_scored.csv
                                │
        │                                        ▼
        ▼                          ┌──────────────────────────┐
 decision_log.md                   │ backend/app.py (FastAPI) │
 recommendation.md                 │  - weekly call list      │
                                   │  - score one patient     │
                                   │  - score a batch         │
                                   │  - input checks          │
                                   └─────────────┬────────────┘
                                                 │
                                                 ▼
                                   ┌──────────────────────────┐
                                   │ frontend/                │
                                   │  index.html, style.css,  │
                                   │  app.js                  │
                                   │ (what the care-management│
                                   │  lead sees)              │
                                   └──────────────────────────┘
```

| Piece | What it does |
|---|---|
| `synthetic_data_generator/generate.py` | Generates the synthetic data (26 columns) with 3 planted issues: a leak, unfinished outcomes and blanks. |
| `data/` | `train.csv` (60,000 discharges, Jan 2021 to Dec 2023, with the answer) and `holdout.csv` (8,000 discharges, Jan to Mar 2024, no answer). |
| `code/run_all.py` | The whole analysis in one command: audit, choose inputs, test, compare models, predict. |
| `outputs/` | Evidence for the decision log: audit report, kept and dropped inputs, model comparison, calibration table. |
| `predictions.csv` | One risk score from 0 to 1 for every 2024 discharge. |
| `backend/` | `app.py` serves the weekly list, scores one patient or a batch, and checks inputs; `run.py` starts it; `model_artifacts/` holds the saved model, its info and the scored holdout. |
| `frontend/` | Plain HTML, CSS and JavaScript page the care-management lead uses to pick a week and act on the call list. |
| `README.md`, `decision_log.md`, `recommendation.md`, `requirements.txt`, `walkthrough.ipynb` | How to run it, the reasoning, the answer for the care lead, the packages, and an optional step-by-step notebook. |
