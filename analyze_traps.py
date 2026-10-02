"""
Where does the trap act? Analysis of funnel_synthetic.csv (synthetic data).
1. Stage-by-stage pass rates, trapped vs not: does the trap hurt anywhere before issuance?
2. Issuance by IT trap x business trap, with a logistic model including their interaction.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

d = pd.read_csv("funnel_synthetic.csv")
stages = ["storefront", "quote_start", "quote_complete", "app_start", "app_complete", "issued"]
names = ["Store → Quote start", "Quote start → Quote done", "Quote done → App start",
         "App start → App done", "App done → Issued"]
rng = np.random.default_rng(1)


def rate_ci(s):
    k, n = s.sum(), s.size
    lo, hi = np.percentile(rng.beta(1 + k, 1 + n - k, 20_000), [5, 95])
    return k / n, lo, hi


# 1. Pass rates by stage
rows = []
for (a, b), name in zip(zip(stages, stages[1:]), names):
    r = d[d[a] == 1]
    p = chi2_contingency(pd.crosstab(r.it_trap, r[b]))[1]
    for t in (0, 1):
        rate, lo, hi = rate_ci(r.loc[r.it_trap == t, b])
        rows.append(dict(step=name, it_trap=t, rate=rate, lo=lo, hi=hi, n=(r.it_trap == t).sum(), p=p))
stage_tbl = pd.DataFrame(rows)
print(stage_tbl.pivot(index="step", columns="it_trap", values="rate").reindex(names).round(3)
      .assign(p_value=stage_tbl.groupby("step").p.first().reindex(names).map("{:.2g}".format)))

# 2. Issuance among those who reached App Complete
ac = d[d.app_complete == 1].copy()
ac["not_issued"] = 1 - ac.issued
grid = ac.groupby(["it_trap", "business_trap"]).issued.agg(["mean", "count"])
print("\nIssuance among App Complete:\n", grid.round(3))
fit = smf.logit("not_issued ~ it_trap * business_trap", ac).fit(disp=0)
ci = np.exp(fit.conf_int())
print("\nOdds ratios for NOT being issued (95% CI):")
for term in fit.params.index[1:]:
    print(f"  {term:<24} {np.exp(fit.params[term]):.2f}  ({ci.loc[term, 0]:.2f} to {ci.loc[term, 1]:.2f})  p={fit.pvalues[term]:.2g}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
x = np.arange(len(names))
for t, color, label, off in [(0, "#5F5E5A", "No IT trap", -0.12), (1, "#A32D2D", "IT trap", 0.12)]:
    s = stage_tbl[stage_tbl.it_trap == t].set_index("step").reindex(names)
    ax1.errorbar(x + off, s.rate, yerr=[s.rate - s.lo, s.hi - s.rate], fmt="o", color=color, capsize=4, label=label)
for i, name in enumerate(names):
    p = stage_tbl[stage_tbl.step == name].p.iloc[0]
    ax1.text(i, 0.6, f"p = {p:.2g}", ha="center", fontsize=9,
             color="#A32D2D" if p < 0.001 else "#5F5E5A", fontweight="bold" if p < 0.001 else "normal")
ax1.set_xticks(x, names, rotation=15, fontsize=9)
ax1.set_ylim(0.55, 0.95)
ax1.set_ylabel("Share passing this step (90% interval)")
ax1.set_title("The IT trap changes nothing until the last step")
ax1.legend(frameon=False, loc="upper left")
ax1.spines[["top", "right"]].set_visible(False)

labels = ["Neither", "Business\ntrap only", "IT trap\nonly", "Both"]
keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
vals = [grid.loc[k, "mean"] for k in keys]
ns = [grid.loc[k, "count"] for k in keys]
bars = ax2.bar(labels, vals, color=["#5F5E5A", "#378ADD", "#A32D2D", "#712B13"], width=0.6)
for bar, v, n in zip(bars, vals, ns):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.1%}\n(n={n:,})", ha="center", fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("Share issued, among App Complete")
ax2.set_title("Issuance by trap type")
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("trap_analysis.png", dpi=150)
plt.show()
