"""One command does everything:   python code/run_all.py     (about 1 minute, CPU only)

Steps (each is a small function below):
  1. audit      look at the data, find the problems
  2. features   choose columns, fill blanks, scale, one-hot
  3. validate   train on the past, test on a later period, compare 3 models
  4. fit        train the chosen model (logistic regression) on all known outcomes
  5. predict    score the 2024 holdout -> predictions.csv, and save what the app needs
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
(ROOT / "backend/model_artifacts").mkdir(parents=True, exist_ok=True)

TARGET = "readmitted_30d"
LEAK = "post_discharge_call_logged"                 # only known AFTER the patient is back -> never use
NUMBERS = ["age", "comorbidity_count", "los_days", "n_prior_admissions_12m", "n_ed_visits_6m",
           "num_medications", "bmi", "lab_creatinine_last", "lab_sodium_last", "lab_hemoglobin_last",
           "systolic_bp_last", "heart_rate_last", "weekend_discharge", "followup_visit_scheduled"]
CATEGORIES = ["hospital_site", "sex", "insurance_type", "admission_source", "primary_dx_group",
              "discharge_disposition"]
FEATURES = NUMBERS + CATEGORIES                     # ids, dates, the leak and the answer are left out
LABEL_LAG = 30                                      # an outcome is only known 30 days after discharge


# ---------------------------------------------------------------- 1. audit
def audit(train):
    known = train.dropna(subset=[TARGET])
    lines = ["# Data audit", ""]
    lines.append(f"- Rows: {len(train):,}. Columns: {train.shape[1]}. Discharges {train.discharge_date.min():%Y-%m-%d} to {train.discharge_date.max():%Y-%m-%d}.")
    lines.append(f"- Readmission rate (known outcomes): {known[TARGET].mean():.1%}.")
    blank = train[TARGET].isna().sum()
    lines.append(f"- **Censored:** {blank:,} discharges after 2023-12-01 have no outcome yet. They are left out of training (treating blank as 'no' would teach the model that late discharges never return).")
    r = known.groupby(LEAK)[TARGET].mean()
    auc = roc_auc_score(known[TARGET], known[LEAK])
    lines.append(f"- **Leak:** `{LEAK}` is 1 for {known[known[TARGET]==1][LEAK].mean():.0%} of patients who came back but {known[known[TARGET]==0][LEAK].mean():.0%} of those who did not. Alone it scores AUC {auc:.2f}. It is written after the fact, so it is excluded.")
    miss = train[FEATURES].isna().mean()
    miss = miss[miss > 0]
    lines.append("- **Missing values:** " + ", ".join(f"{c} {v:.1%}" for c, v in miss.items()) + ". Filled with the median (numbers).")
    (OUT / "audit_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- 2. features + models
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


# ---------------------------------------------------------------- 3. validation
def weekly_top5(df, score, k=0.05):
    """Each week keep the top 5% by score. Return precision (share of the list that came back)
    and capture (share of all readmissions that were on the list)."""
    d = df.assign(score=score, week=df.discharge_date.dt.to_period("W"))
    hit = n_list = tot = 0
    for _, w in d.groupby("week"):
        m = max(1, int(round(k * len(w))))
        top = w.nlargest(m, "score")
        hit += top[TARGET].sum(); n_list += m; tot += w[TARGET].sum()
    return hit / n_list, hit / tot


def validate(train):
    known = train.dropna(subset=[TARGET]).copy()
    known[TARGET] = known[TARGET].astype(int)
    # Train on everything up to July 2023. Skip August (outcomes of late-July discharges are still unfolding). Test on Sep-Nov.
    fit = known[known.discharge_date <= "2023-07-31"]
    test = known[(known.discharge_date >= "2023-09-01") & (known.discharge_date <= "2023-11-30")]
    rows = []
    for name in ["logistic", "hist_gradient_boosting", "xgboost"]:
        m = make_model(name).fit(fit[FEATURES], fit[TARGET])
        p = m.predict_proba(test[FEATURES])[:, 1]
        prec, cap = weekly_top5(test, p)
        rows.append(dict(model=name, roc_auc=roc_auc_score(test[TARGET], p),
                         pr_auc=average_precision_score(test[TARGET], p),
                         precision_top5=prec, capture_top5=cap,
                         brier=float(np.mean((p - test[TARGET]) ** 2))))
        if name == "logistic":
            test = test.assign(p=p)
    res = pd.DataFrame(rows).round(3)
    res.to_csv(OUT / "validation_results.csv", index=False)
    print("\nValidation (train Jan 2022-Jul 2023, test Sep-Nov 2023, weekly top 5%)\n", res.to_string(index=False))
    # Calibration: do "30 in 100" scores really come back about 30 in 100?
    test["band"] = pd.qcut(test.p, 10, duplicates="drop")
    cal = test.groupby("band", observed=True).agg(predicted=("p", "mean"), observed=(TARGET, "mean"), n=(TARGET, "size")).round(3)
    cal.to_csv(OUT / "calibration_table.csv")
    print("\nCalibration by risk decile\n", cal.to_string(index=False))
    return res


# ---------------------------------------------------------------- 4 + 5. fit and predict
def reasons(model, X, top=3):
    """Top reasons a patient's score is high: coefficient x (how far the value is from typical)."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Z = prep.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    contrib = Z * clf.coef_[0]
    names = list(NUMBERS) + [f"{c}={v}" for c, cats in zip(CATEGORIES, prep.named_transformers_["cat"].categories_) for v in cats]
    out = []
    for i in range(len(X)):
        idx = np.argsort(-contrib[i])[:top]
        txt = []
        for j in idx:
            if contrib[i, j] <= 0:
                continue
            n = names[j]
            txt.append(f"{n.replace('_', ' ')}: {X.iloc[i][n]}" if n in NUMBERS else n.replace("_", " ").replace("=", ": "))
        out.append(" | ".join(txt))
    return out


def main():
    train = pd.read_csv(ROOT / "data/train.csv", parse_dates=["admit_date", "discharge_date"])
    holdout = pd.read_csv(ROOT / "data/holdout.csv", parse_dates=["admit_date", "discharge_date"])
    audit(train)
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
    json.dump({"model": chosen, "sklearn": sklearn.__version__, "features": FEATURES, "numbers": NUMBERS, "categories": CATEGORIES,
               "train_rows": int(len(known)), "validation": res.set_index("model").to_dict("index")},
              open(ROOT / "backend/model_artifacts/info.json", "w"), indent=2)
    print(f"\nWrote predictions.csv ({len(holdout):,} rows) and the files the app needs.")


if __name__ == "__main__":
    main()
