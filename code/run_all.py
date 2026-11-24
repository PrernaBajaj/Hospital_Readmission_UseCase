"""One command does everything:   python code/run_all.py     (about 15 seconds, CPU only)

Steps (each is a small function below):
  1. audit      look at the data, find the problems
  2. select     keep only the inputs that matter (permutation importance on the training months)
  3. features   fill blanks, scale, one-hot (inside every model's pipeline)
  4. validate   train on the past, test on a later period, compare 3 models
  5. fit        train the chosen model (logistic regression) on all known outcomes
  6. predict    score the 2024 holdout -> predictions.csv, and save what the app needs
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
(ROOT / "backend/model_artifacts").mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_30d"
LEAK = "post_discharge_call_logged"                 # only known AFTER the patient is back -> never use
ALL_NUMBERS = ["age", "comorbidity_count", "los_days", "n_prior_admissions_12m", "n_ed_visits_6m",
               "num_medications", "bmi", "lab_creatinine_last", "lab_sodium_last", "lab_hemoglobin_last",
               "systolic_bp_last", "heart_rate_last", "weekend_discharge", "followup_visit_scheduled"]
ALL_CATEGORIES = ["hospital_site", "sex", "insurance_type", "admission_source", "primary_dx_group",
                  "discharge_disposition"]          # the 20 candidates: ids, dates, the leak and the answer are left out
NUMBERS, CATEGORIES = list(ALL_NUMBERS), list(ALL_CATEGORIES)   # replaced by the chosen inputs after step 2
FEATURES = NUMBERS + CATEGORIES
TRAIN_END, TEST_START, TEST_END = "2023-07-31", "2023-09-01", "2023-11-30"   # August is skipped on purpose
SELECT_SPLIT = "2022-12-31"                         # inside the training months: fit before, check after (patients never shared)
MIN_DROP = 0.002                                    # an input must cost the model at least this much ROC-AUC when shuffled
LABEL_LAG = 30                                      # an outcome is only known 30 days after discharge


# ---------------------------------------------------------------- 1. audit
def audit(train):
    known = train.dropna(subset=[TARGET])
    lines = ["# Data audit", ""]
    lines.append(f"- Rows: {len(train):,}. Columns: {train.shape[1]}. Discharges {train.discharge_date.min():%Y-%m-%d} to {train.discharge_date.max():%Y-%m-%d}.")
    per = train.groupby("patient_id").size()
    lines.append(f"- Patients: {len(per):,}; {(per > 1).mean():.0%} have more than one admission. Splits are made by patient_id so nobody is in both train and test.")
    lines.append(f"- Readmission rate (known outcomes): {known[TARGET].mean():.1%}.")
    blank = train[TARGET].isna().sum()
    lines.append(f"- **Censored:** {blank:,} discharges after 2023-12-01 have no outcome yet. They are left out of training (treating blank as 'no' would teach the model that late discharges never return).")
    r = known.groupby(LEAK)[TARGET].mean()
    auc = roc_auc_score(known[TARGET], known[LEAK])
    lines.append(f"- **Leak:** `{LEAK}` is 1 for {known[known[TARGET]==1][LEAK].mean():.0%} of patients who came back but {known[known[TARGET]==0][LEAK].mean():.0%} of those who did not. Alone it scores AUC {auc:.2f}. It is written after the fact, so it is excluded.")
    miss = train[ALL_NUMBERS + ALL_CATEGORIES].isna().mean()
    miss = miss[miss > 0]
    lines.append("- **Missing values:** " + ", ".join(f"{c} {v:.1%}" for c, v in miss.items()) + ". Filled with the median (numbers).")
    print("\n".join(lines))


def patient_split(known, fit_end, test_start, test_end):
    """Past-to-future split with no patient in both parts.
    Test = discharges in [test_start, test_end]. Train = discharges up to fit_end whose patient_id never appears in the test.
    (About 30% of patients have several admissions, so a patient seen in the test months is removed from training.)"""
    test = known[(known.discharge_date >= test_start) & (known.discharge_date <= test_end)]
    fit = known[(known.discharge_date <= fit_end) & ~known.patient_id.isin(test.patient_id)]
    assert not set(fit.patient_id) & set(test.patient_id), "a patient is in both train and test"
    return fit, test


# ---------------------------------------------------------------- 2. choose the inputs that matter
def use_features(numbers, categories):
    """Switch the whole pipeline to this list of inputs (lists are changed in place, so every function sees it)."""
    NUMBERS[:] = list(numbers); CATEGORIES[:] = list(categories); FEATURES[:] = NUMBERS + CATEGORIES


def select_features(train, repeats=20):
    """Permutation importance: fit logistic regression on Jan 2021-Dec 2022, then on Jan-Jul 2023 shuffle one input at a time
    and see how much ROC-AUC drops. Keep an input only if the drop is at least MIN_DROP and clearly above the shuffle noise.
    The test months (Sep-Nov 2023) are never used here."""
    use_features(ALL_NUMBERS, ALL_CATEGORIES)
    known = train.dropna(subset=[TARGET]).copy()
    known[TARGET] = known[TARGET].astype(int)
    window = known[known.discharge_date <= TRAIN_END]
    a, v = patient_split(window, SELECT_SPLIT, str(pd.Timestamp(SELECT_SPLIT) + pd.Timedelta(days=1))[:10], TRAIN_END)
    model = make_model("logistic").fit(a[FEATURES], a[TARGET])
    base = roc_auc_score(v[TARGET], model.predict_proba(v[FEATURES])[:, 1])
    rng = np.random.default_rng(0)
    rows = []
    for col in ALL_NUMBERS + ALL_CATEGORIES:
        drops = []
        for _ in range(repeats):
            x = v[FEATURES].copy()
            x[col] = rng.permutation(x[col].values)
            drops.append(base - roc_auc_score(v[TARGET], model.predict_proba(x)[:, 1]))
        rows.append(dict(feature=col, auc_drop=np.mean(drops), spread=np.std(drops)))
    table = pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
    table["keep"] = (table.auc_drop >= MIN_DROP) & (table.auc_drop > 2 * table.spread)
    keep = set(table[table.keep].feature)
    use_features([c for c in ALL_NUMBERS if c in keep], [c for c in ALL_CATEGORIES if c in keep])
    print(f"\nInputs kept ({len(FEATURES)} of {len(table)}):", ", ".join(FEATURES))
    print("Inputs dropped:", ", ".join(table[~table.keep].feature))
    return table


# ---------------------------------------------------------------- 3. features + models
def preprocessor():
    num = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, NUMBERS), ("cat", cat, CATEGORIES)])


def make_model(name):
    if name == "logistic":
        clf = LogisticRegression(C=1.0, max_iter=1000)
    elif name == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
    elif name == "xgboost":
        clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                            tree_method="hist", n_jobs=2, random_state=0, eval_metric="logloss")
    return Pipeline([("prep", preprocessor()), ("clf", clf)])


# ---------------------------------------------------------------- 4. validation
def weekly_flags(df, score, k=0.05):
    """1 for the top 5% of each week by score (the call list), 0 for everyone else."""
    d = df.assign(score=np.asarray(score), week=df.discharge_date.dt.to_period("W"))
    flag = pd.Series(0, index=df.index)
    for _, w in d.groupby("week"):
        m = max(1, int(round(k * len(w))))
        flag.loc[w.nlargest(m, "score").index] = 1
    return flag.values


def top5_results(df, score, k=0.05):
    """Confusion matrix of the weekly call list, plus accuracy and precision (share of the list that came back)."""
    tn, fp, fn, tp = confusion_matrix(df[TARGET].values, weekly_flags(df, score, k)).ravel()
    return dict(accuracy=(tp + tn) / (tp + tn + fp + fn), precision_top5=tp / (tp + fp),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def validate(train):
    known = train.dropna(subset=[TARGET]).copy()
    known[TARGET] = known[TARGET].astype(int)
    # Train on everything up to July 2023. Skip August (outcomes of late-July discharges are still unfolding). Test on Sep-Nov.
    fit, test = patient_split(known, TRAIN_END, TEST_START, TEST_END)
    print(f"\nTrain {len(fit):,} discharges / {fit.patient_id.nunique():,} patients | test {len(test):,} / {test.patient_id.nunique():,} | patients in both: 0")
    rows = []
    for name in ["logistic", "hist_gradient_boosting", "xgboost"]:
        m = make_model(name).fit(fit[FEATURES], fit[TARGET])
        p = m.predict_proba(test[FEATURES])[:, 1]
        rows.append(dict(model=name, roc_auc=roc_auc_score(test[TARGET], p),
                         pr_auc=average_precision_score(test[TARGET], p), brier=brier_score_loss(test[TARGET], p), **top5_results(test, p)))
        if name == "logistic":
            test = test.assign(p=p)
    res = pd.DataFrame(rows)
    res[["roc_auc", "pr_auc", "brier", "accuracy", "precision_top5"]] = res[["roc_auc", "pr_auc", "brier", "accuracy", "precision_top5"]].round(3)
    res.to_csv(OUT / "validation_results.csv", index=False)
    print("\nValidation (train Jan 2021-Jul 2023 (patients in the test removed), test Sep-Nov 2023, weekly top 5%)\n", res.to_string(index=False))
    # Calibration: do "30 in 100" scores really come back about 30 in 100?
    test["band"] = pd.qcut(test.p, 10, duplicates="drop")
    cal = test.groupby("band", observed=True).agg(predicted=("p", "mean"), observed=(TARGET, "mean"), n=(TARGET, "size")).round(3)
    print("\nCalibration by risk decile\n", cal.to_string(index=False))
    return res


# ---------------------------------------------------------------- 5 + 6. fit and predict
def plain(name, value, high=True):
    """One reason in everyday words, e.g. 'Heart failure diagnosis' or '3 hospital admissions in the past year'."""
    n = lambda v: f"{v:g}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
    plural = lambda v, one, many: f"{n(v)} {one if float(v) == 1 else many}"
    if name in NUMBERS:
        if pd.isna(value):
            return None
        phrase = {
            "age": f"Age {n(value)}",
            "comorbidity_count": plural(value, "chronic condition", "chronic conditions"),
            "los_days": f"{n(value)}-day hospital stay",
            "n_prior_admissions_12m": plural(value, "hospital admission in the past year", "hospital admissions in the past year"),
            "n_ed_visits_6m": plural(value, "emergency visit in the past 6 months", "emergency visits in the past 6 months"),
            "num_medications": plural(value, "medication at discharge", "medications at discharge"),
            "lab_sodium_last": f"{'High' if high else 'Low'} sodium ({n(value)})",
            "lab_creatinine_last": f"{'High' if high else 'Low'} creatinine ({n(value)})",
            "lab_hemoglobin_last": f"{'High' if high else 'Low'} hemoglobin ({n(value)})",
            "systolic_bp_last": f"{'High' if high else 'Low'} blood pressure ({n(value)})",
            "heart_rate_last": f"{'High' if high else 'Low'} heart rate ({n(value)})",
            "bmi": f"{'High' if high else 'Low'} BMI ({n(value)})",
            "followup_visit_scheduled": "Follow-up visit booked" if float(value) == 1 else "No follow-up visit booked",
            "weekend_discharge": "Discharged on a weekend" if float(value) == 1 else "Discharged on a weekday",
        }
        return phrase.get(name, f"{name.replace('_', ' ')}: {n(value)}")
    v = str(value)
    if name == "followup_visit_scheduled":
        return "Follow-up visit booked" if v in ("1", "1.0") else "No follow-up visit booked"
    if name == "primary_dx_group":
        return f"{v} diagnosis"
    if name == "discharge_disposition":
        return {"Home": "Discharged home", "Home with services": "Discharged home with services",
                "Skilled nursing": "Discharged to skilled nursing", "Rehab": "Discharged to rehab"}.get(v, v)
    if name == "weekend_discharge":
        return "Discharged on a weekend" if v in ("1", "1.0") else "Discharged on a weekday"
    return f"{name.replace('_', ' ').capitalize()}: {v}"


def reasons(model, X, top=3):
    """Top reasons a patient's score is high, in plain words: coefficient x (how far the value is from typical)."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Z = prep.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    contrib = Z * clf.coef_[0]
    cats = [(c, v) for c, cs in zip(CATEGORIES, prep.named_transformers_["cat"].categories_) for v in cs]
    names = [(n, None) for n in NUMBERS] + cats
    out = []
    for i in range(len(X)):
        txt = []
        for j in np.argsort(-contrib[i]):
            if contrib[i, j] <= 0 or len(txt) == top:
                break
            col, cat = names[j]
            phrase = plain(col, X.iloc[i][col] if cat is None else cat, high=Z[i, j] >= 0)
            if phrase and phrase not in txt:
                txt.append(phrase)
        out.append("; ".join(txt))
    return out


def main():
    train = pd.read_csv(ROOT / "data/train.csv", parse_dates=["admit_date", "discharge_date"])
    holdout = pd.read_csv(ROOT / "data/holdout.csv", parse_dates=["admit_date", "discharge_date"])
    audit(train)
    importance = select_features(train)
    res = validate(train)

    # Choose: logistic regression unless a booster is clearly better (>= 0.01 PR-AUC).
    best = res.set_index("model").pr_auc
    chosen = "logistic"
    for m in ["hist_gradient_boosting", "xgboost"]:
        if best[m] >= best["logistic"] + 0.01:
            chosen = m
    print(f"\nChosen model: {chosen}")
    assert chosen == "logistic", "the explanations in the app assume logistic regression"

    known = train.dropna(subset=[TARGET]).copy()
    known[TARGET] = known[TARGET].astype(int)
    model = make_model(chosen).fit(known[FEATURES], known[TARGET])

    holdout["risk_score"] = model.predict_proba(holdout[FEATURES])[:, 1].round(4)
    holdout[["admission_id", "risk_score"]].to_csv(ROOT / "predictions.csv", index=False)

    holdout["week_start"] = holdout.discharge_date.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    holdout["rank_in_week"] = holdout.groupby("week_start").risk_score.rank(ascending=False, method="first").astype(int)
    holdout["reasons"] = reasons(model, holdout[FEATURES])
    holdout.to_csv(ROOT / "backend/model_artifacts/holdout_scored.csv", index=False)
    joblib.dump(model, ROOT / "backend/model_artifacts/model.joblib")
    import sklearn
    json.dump({"model": chosen, "sklearn": sklearn.__version__, "features": list(FEATURES), "numbers": list(NUMBERS),
               "categories": list(CATEGORIES), "dropped": importance[~importance.keep].feature.tolist(),
               "train_rows": int(len(known)), "validation": res.set_index("model").to_dict("index")},
              open(ROOT / "backend/model_artifacts/info.json", "w"), indent=2)
    print(f"\nWrote predictions.csv ({len(holdout):,} rows) and the files the app needs.")


if __name__ == "__main__":
    main()
