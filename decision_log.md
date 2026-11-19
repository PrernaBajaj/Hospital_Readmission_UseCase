# Decision log

Data is fake (made by `synthetic_data_generator/generate.py`), so numbers show the method, not real hospital results.

## 1. What problem are we solving?
The care team can phone about 5% of discharges a week (about 30 of 600). So the task is **ranking**, not "predict yes/no": who should be in that list of 30? I judge the model by the weekly top 5%:
- **Precision@5%**: of the 30 called, how many came back (the number the lead cares about).
- **Capture@5%**: of all patients who came back that week, how many were on the list.
- Secondary: ROC-AUC and PR-AUC (ranking quality across everyone), and calibration (is "30 in 100" really about 30 in 100?).
Accuracy is useless here: saying "nobody returns" is 84% accurate.

## 2. Data checks and what I did (all from the data, no outside help)
| Finding | Why it matters | Decision |
|---|---|---|
| `post_discharge_call_logged` is 1 for 78% of returners and 18% of others (AUC 0.80 alone) | It is recorded after the patient is back, so it is not known on discharge day. Using it would look brilliant in testing and fail in real use (**leakage**). | Removed. |
| 2,370 discharges in the last 30 days have a blank outcome | The 30 days have not finished. Treating blank as "no" teaches that recent patients never return (**censoring**). | Left out of training. |
| 3-5% blanks in labs, bmi, medications | Blank is not zero. | Filled with the middle value (median). |
| IDs and dates | They identify a row, they do not describe the patient. | Not used as inputs. |
I only use what is known on discharge day: age, sex, site, insurance, admission source, diagnosis group, comorbidities, length of stay, prior admissions, ED visits, medications, bmi, last labs and vitals, discharge destination, weekend flag, follow-up booked.

## 3. How I tested (to look like real use)
Real use = train on the past, predict the future. I trained on everything up to July 2023 (Jan 2022 on), skipped August (outcomes still unfolding), and tested on Sep-Nov 2023. A random split would mix future into training and flatter the result.

## 4. Models compared (test period, weekly top 5%)
| Model | ROC-AUC | PR-AUC | Precision@5% | Capture@5% |
|---|---|---|---|---|
| **Logistic regression** | 0.754 | 0.407 | 56.9% | 18.7% |
| Hist. gradient boosting | 0.753 | 0.407 | 57.4% | 18.9% |
| XGBoost | 0.752 | 0.408 | 56.6% | 18.6% |
Rule I fixed before looking: a boosting model must beat logistic by at least 0.01 PR-AUC to replace it. None did, so **logistic regression ships**. It is also easy to explain: each input has one weight, and the app shows the top reasons. Caveat: this data is a simple weighted sum, so a simple model suits it; on real data boosting must be re-tested.

## 5. Can the scores be trusted?
Yes, reasonably: patients in each risk tenth were predicted and observed close together (top tenth predicted 46%, observed 49%; see `outputs/calibration_table.csv`). So the app can say "46 in 100" honestly, but as a group estimate, not a promise for one person.

## 6. What I deliberately did not do
- Did not tune hard or stack models: gains were tiny and the story gets harder to defend.
- Did not use race or similar fields (none in the data) and did not try to "fix" fairness by guessing; instead I would monitor who ends up on the list (age mix) and ask clinical informatics.
- Did not claim calls prevent readmissions. The list says who is at risk, not who is helped.

## 7. Risks and limits
- Fake data: absolute numbers prove nothing about real patients.
- About 30 calls a week is a small sample: one week's hit rate can swing by about 9 points, so watch four-week averages.
- Once the team calls these patients, their outcomes change, so later outcomes mix risk and the effect of calls. A random hold-back keeps one clean comparison.
- Patients change over time (2024 patients are slightly older), so refit every quarter and watch calibration.

## 8. Questions I would ask clinical informatics
Is the call flag time-stamped? Are planned readmissions excluded from the 30-day outcome? How are blanks in labs created? Did coding or definitions change at any site?

## 9. Monitoring plan
Weekly: four-week hit rate (alert below 45%), calibration, age mix of the list, missing values per column. Quarterly: refit and re-compare models.

## 10. Result on the supplied 2024 holdout
`predictions.csv` has one risk score per discharge (8,000 rows). Outcomes are not supplied, so I cannot score them. As a sanity check I scored against answers from my own generator: precision@5% about 54%, capture about 17% (`code/check_simulated_holdout.py`; not used for any decision).
