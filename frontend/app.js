const $ = id => document.getElementById(id);
let current = null;

// ---- weekly list
async function loadList() {
  const r = await fetch(`/api/call-list?week=${$("week").value}&pct=${$("pct").value}`);
  const d = await r.json();
  if (!r.ok) { $("summary").innerHTML = `<span class="err">${d.detail}</span>`; $("rows").innerHTML = ""; return; }
  current = d;
  $("summary").textContent = `${d.list_size} patients to call out of ${d.discharges} discharges.`;
  $("rows").innerHTML = d.rows.map(x => `<tr><td>${x.rank_in_week}</td><td>${x.admission_id}</td><td>${x.hospital_site}</td>
    <td>${x.age}</td><td>${x.primary_dx_group}</td><td>${Math.round(x.risk_score * 100)} in 100</td><td>${x.reasons}</td></tr>`).join("");
}
$("csv").onclick = () => {
  if (!current) return;
  const lines = ["rank,admission_id,site,age,diagnosis,risk_score,reasons",
    ...current.rows.map(x => [x.rank_in_week, x.admission_id, x.hospital_site, x.age, x.primary_dx_group, x.risk_score, `"${x.reasons}"`].join(","))];
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], {type: "text/csv"}));
  a.download = `call_list_${$("week").value}.csv`; a.click();
};
$("pct").onchange = $("week").onchange = loadList;

// ---- score a new discharge
const FIELDS = {
  hospital_site: ["LS-North", "LS-South", "LS-East", "LS-West"], age: 70, sex: ["F", "M"],
  insurance_type: ["Medicare", "Medicaid", "Commercial", "Self-pay"], admission_source: ["Emergency", "Referral", "Transfer"],
  primary_dx_group: ["Heart failure", "COPD", "Pneumonia", "Diabetes", "Other"], comorbidity_count: 3, los_days: 5,
  n_prior_admissions_12m: 1, n_ed_visits_6m: 1, num_medications: 8, bmi: 28, lab_creatinine_last: 1.1,
  lab_sodium_last: 138, lab_hemoglobin_last: 12.5, systolic_bp_last: 128, heart_rate_last: 78,
  discharge_disposition: ["Home", "Home health", "Skilled nursing"], weekend_discharge: [0, 1], followup_visit_scheduled: [1, 0]
};
$("form").innerHTML = Object.entries(FIELDS).map(([k, v]) => `<label>${k.replaceAll("_", " ")}` +
  (Array.isArray(v) ? `<select name="${k}">${v.map(o => `<option>${o}</option>`).join("")}</select>`
                    : `<input name="${k}" type="number" step="any" value="${v}">`) + `</label>`).join("") + `<button>Score</button>`;
$("form").onsubmit = async e => {
  e.preventDefault();
  const body = {};
  for (const [k, v] of new FormData(e.target)) body[k] = v === "" ? null : (isNaN(v) || ["hospital_site","sex"].includes(k) ? v : Number(v));
  const r = await fetch("/api/score", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const d = await r.json();
  $("result").innerHTML = r.ok ? `${d.in_100} in 100 <small>(main reasons: ${d.reasons || "none above typical"})</small>`
    : `<span class="err">${JSON.stringify(d.detail)}</span>`;
};

// ---- start
(async () => {
  const weeks = await (await fetch("/api/weeks")).json();
  $("week").innerHTML = weeks.map(w => `<option>${w}</option>`).join("");
  loadList();
})();
