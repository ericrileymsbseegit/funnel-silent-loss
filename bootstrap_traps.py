"""
How stable are the trap conclusions? Bootstrap funnel_synthetic.csv 1,000 times
(resample members with replacement) and re-run the tests each time.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

d = pd.read_csv("funnel_synthetic.csv")
stages = ["storefront", "quote_start", "quote_complete", "app_start", "app_complete", "issued"]
B = 1000
rng = np.random.default_rng(2026)

early_p, late_p, gap, inter_or, both_vs_it = [], [], [], [], []
for _ in range(B):
    s = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
    ps = []
    for a, b in zip(stages, stages[1:]):
        r = s[s[a] == 1]
        ps.append(chi2_contingency(pd.crosstab(r.it_trap, r[b]))[1])
    early_p.append(ps[:4])
    late_p.append(ps[4])
    ac = s[s.app_complete == 1].assign(not_issued=lambda x: 1 - x.issued)
    rates = ac.groupby("it_trap").issued.mean()
    gap.append(rates[0] - rates[1])
    g = ac.groupby(["it_trap", "business_trap"]).issued.mean()
    both_vs_it.append(g[(1, 0)] - g[(1, 1)])
    fit = smf.logit("not_issued ~ it_trap * business_trap", ac).fit(disp=0)
    inter_or.append(np.exp(fit.params["it_trap:business_trap"]))

early_p = np.array(early_p)
late_p = np.array(late_p)
gap = np.array(gap)
inter_or = np.array(inter_or)
both_vs_it = np.array(both_vs_it)

pattern = (early_p.min(axis=1) > 0.01) & (late_p < 0.001)
print(f"Bootstrap resamples: {B:,}")
print(f"Last step (App done -> Issued) significant at p < 0.001:   {np.mean(late_p < 0.001):.1%}")
print(f"No earlier step significant at p < 0.01:                   {np.mean(early_p.min(axis=1) > 0.01):.1%}")
print(f"  each earlier step separately (p < 0.05 rate):            "
      + ", ".join(f"{np.mean(early_p[:, i] < 0.05):.0%}" for i in range(4)))
print(f"Full pattern (effect only at the last step):               {pattern.mean():.1%}")
print(f"Issuance gap, clean minus trapped: {gap.mean():.1%}  (90% interval {np.percentile(gap, 5):.1%} to {np.percentile(gap, 95):.1%})")
print(f"Interaction odds ratio:            {np.median(inter_or):.2f}  (90% interval {np.percentile(inter_or, 5):.2f} to {np.percentile(inter_or, 95):.2f});"
      f"  below 1 in {np.mean(inter_or < 1):.1%} of resamples")
print(f"Both traps vs IT only, extra drop: {both_vs_it.mean():.1%}  (90% interval {np.percentile(both_vs_it, 5):.1%} to {np.percentile(both_vs_it, 95):.1%})")

# Effect size at each step: pass rate (no trap) minus pass rate (IT trap), with 90% bootstrap interval
print("\nPass-rate difference at each step (no trap minus IT trap), 90% bootstrap interval:")
names = ["Store -> Quote start", "Quote start -> Quote done", "Quote done -> App start",
         "App start -> App done", "App done -> Issued"]
diffs = np.zeros((B, 5))
for i in range(B):
    s = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
    for j, (a, b) in enumerate(zip(stages, stages[1:])):
        r = s[s[a] == 1]
        m = r.groupby("it_trap")[b].mean()
        diffs[i, j] = m[0] - m[1]
for j, n in enumerate(names):
    lo, hi = np.percentile(diffs[:, j], [5, 95])
    print(f"  {n:<28} {diffs[:, j].mean():+6.1%}   ({lo:+.1%} to {hi:+.1%})")
