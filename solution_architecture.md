# Solution architecture

```
synthetic_data_generator/generate.py ──> data/train.csv, data/holdout.csv
                                              │
                         code/run_all.py  (audit → features → validate → fit → predict)
                                              │
        ┌─────────────────────────────────────┼───────────────────────┐
   predictions.csv            backend/model_artifacts/            outputs/ (audit, validation,
   (id, risk_score)           model.joblib, holdout_scored.csv     calibration tables)
                                              │
                            backend/app.py  (FastAPI, port 8000)
                                              │
                            frontend/ (HTML + JS: weekly list, score a new discharge)
```

| Piece | What it does | Why this choice |
|---|---|---|
| `synthetic_data_generator/` | Builds fake data (60,000 training and 8,000 holdout rows) with the 26 case columns and 3 planted problems (leak, unfinished outcomes, blanks). | Reproducible, no real patients. |
| `code/run_all.py` | One script, five small steps. Runs in seconds on CPU. | Easy to read and explain. |
| Logistic regression | Weighted sum of inputs turned into a chance. | Explainable, calibrated, as good as boosting here. |
| `backend/app.py` | `/api/call-list` (capped at 5%), `/api/score` (checks every input), `/api/health`. | Small and testable. Rejects impossible values. |
| `frontend/` | Pick a week, see the top list with reasons, download CSV, score one new discharge. Plain HTML/JS, no build step. | Works offline inside the network. |

Constraints met: CPU only, about 5 seconds to run, packages from the internal index only, no pretrained models, no outside data. Monitoring: see `decision_log.md` section 9.
