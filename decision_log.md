# Decision Log – 30-Day Hospital Readmission Prediction

**Note:** This analysis uses synthetic data generated through `synthetic_data_generator/generate.py`. The results demonstrate the approach and are not actual hospital outcomes.

## 1. Objective
Identify the patients most likely to be readmitted within 30 days of discharge and help the care team at Lakeshore prioritize whom to contact each week, while adhering to a maximum outreach capacity of 5%

## 2. Success Definition & Metrics 
* Precision at top 5% - Model must optimise for Precision at the top 5%. We should focus that the top 5% patients we give the care team are true high-risk patients
* False Negative - Patient is actually re-admitted but the model did not select them for a call.  Hospital misses an opportunity for pro-active intervention and is also clinically concerning.
* False Positive - Model selects a patient for a call, but actually the patient is not re-admitted which consumes the limited operational capacity and might cause unnecessary panic

## 3. Data Audit and Inconsistencies
* `post_discharge_call_logged` showed a strong relationship with readmission but is recorded after the patient returns. This variable is excluded from model training to avoid data leakage.
* 1,763 records had missing outcomes because their 30-day observation period was incomplete. These were excluded from training rather than treated as non-readmissions.
* Roughly 4% of the lab and vital values were missing. Instead of removing them altogether, they were imputed with median
* Numerical features were standardized using StandardScaler to bring them onto a common scale and prevent larger-valued variables from disproportionately influencing the model. 

## 3. Feature Selection
* Started with 20 discharge day inputs. admission_id, patient_id, admit_date, discharge_date, post_discharge_call_logged excluded.
* readmitted_30d is the target variable and hence not included
* Used feature importance on the training data to identify the 10 most relevant features. The remaining 10 were dropped as they had minimal impact on model performance. Since both versions achieved the same ROC-AUC of 0.731, using 10 features keeps the model simpler, easier to explain and equally effective.

## 4. Validation Approach and Model Comparison
* A random 80/20 train-test split approach was rejected due to time and patient leakage risks.
* Training: The model is trained on discharges from January 2021 to July 2023. August is kept as a gap to allow 30-day readmission outcomes for late-July discharges to become available, avoiding incomplete labels. 
* Testing: Model performance is evaluated on September–November 2023 discharges. December is excluded because the 30-day readmission outcomes are not yet fully available.

| Model | ROC-AUC | PR-AUC | Brier | Accuracy | Precision@5% |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.731** | **0.376** | **0.124** | **83.7%** | **54.5%** |
| Histogram Gradient Boosting | 0.725 | 0.363 | 0.125 | 83.4% | 51.6% |
| XGBoost | 0.726 | 0.365 | 0.125 | 83.4% | 51.6% |

Confusion matrix at the weekly top-5%:

| Model | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| **Logistic Regression** | 134 | 112 | 692 | 3,990 |
| Histogram Gradient Boosting | 127 | 119 | 699 | 3,983 |
| XGBoost | 127 | 119 | 699 | 3,983 |

Brier score: Measures how closely the predicted probabilities match actual outcomes, with lower values indicating better performance. Our models score between 0.124 and 0.125, suggesting that their predicted readmission probabilities provide useful insight into which patients are more likely to return.

Logistic Regression was selected because it achieved the lowest number of false positives and false negatives, along with the highest precision within the top 5% of patients prioritized for follow-up.

## 6. Limitations
* Since the dataset is synthetic, model performance may not reflect real-world hospital outcomes and needs validation on actual patient data.
* Limited Coverage and Intervention Bias: A 5% call list captures only 134 of 826 readmitted patients (16%). Additionally, care team calls may influence readmission outcomes, so a randomized or phased rollout is needed to measure their actual impact.

## 7. Final Output

The selected model was retrained on all discharges with known outcomes and used to score all 8,000 admissions in the holdout data. The scores are stored in `predictions.csv` file.

## 8. Monitoring Plan

* Assuming the week starts on Monday, we recommend scheduling the model to run every Sunday at midnight. This ensures the latest patient vitals and clinical data are incorporated, allowing the updated high-risk patient list to be shared with the care team before the start of the following week.

* Evaluate model performance against the original test results and monitor changes in patient demographics, diagnoses, hospital sites, and data quality to identify data drift.
* Capture care-team feedback when predictions are inaccurate (e.g., a patient moved to a skilled nursing facility). Use these clinical insights to improve future model iterations and retraining. 


