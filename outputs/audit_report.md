# Data audit

- Rows: 60,000. Columns: 26. Discharges 2022-01-01 to 2023-12-31.
- Readmission rate (known outcomes): 15.5%.
- **Censored:** 2,370 discharges after 2023-12-01 have no outcome yet. They are left out of training (treating blank as 'no' would teach the model that late discharges never return).
- **Leak:** `post_discharge_call_logged` is 1 for 78% of patients who came back but 18% of those who did not. Alone it scores AUC 0.80. It is written after the fact, so it is excluded.
- **Missing values:** num_medications 3.1%, bmi 3.0%, lab_creatinine_last 5.1%, lab_sodium_last 4.9%, lab_hemoglobin_last 5.0%. Filled with the median (numbers).
