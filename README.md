# Who gets the call? 30-day readmission ranking (Lakeshore Health Network)

Setup (internal package index only): `pip install -r requirements.txt`

| Do this | Command |
|---|---|
| Make the fake data | `python synthetic_data_generator/generate.py` |
| Regenerate `predictions.csv` (and everything the app needs) | `python code/run_all.py` |
| Start the app (http://localhost:8000) | `python backend/run.py` |

Read in this order: `recommendation.md`, `decision_log.md`, `solution_architecture.md`, then `code/run_all.py` (about 200 lines).
Optional check on simulated answers: `python code/check_simulated_holdout.py`.
Data is fake (60,000 train rows, 8,000 holdout rows); the 26 columns follow the case brief. `organizer_sim/` holds simulated answers and is git-ignored.
