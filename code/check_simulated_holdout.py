"""Optional, after the fact: score predictions.csv against the SIMULATED holdout answers.
These answers come from our own generator, so this is a sanity check only. Never used to choose anything."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[1]
p = pd.read_csv(ROOT / "predictions.csv")
h = pd.read_csv(ROOT / "data/holdout.csv", parse_dates=["discharge_date"])[["admission_id", "discharge_date"]]
y = pd.read_csv(ROOT / "organizer_sim/holdout_labels_SIMULATED.csv")
d = p.merge(h).merge(y)
d["week"] = d.discharge_date.dt.to_period("W")
hit = n = tot = 0
for _, w in d.groupby("week"):
    m = max(1, round(0.05 * len(w))); top = w.nlargest(m, "risk_score")
    hit += top.readmitted_30d.sum(); n += m; tot += w.readmitted_30d.sum()
print(f"rows {len(d):,} | base rate {d.readmitted_30d.mean():.1%} | ROC-AUC {roc_auc_score(d.readmitted_30d, d.risk_score):.3f} | "
      f"PR-AUC {average_precision_score(d.readmitted_30d, d.risk_score):.3f} | weekly top-5% precision {hit/n:.1%} | capture {hit/tot:.1%}")
