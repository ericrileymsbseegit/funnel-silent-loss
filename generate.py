"""
Synthetic funnel generator.

Produces member-level journey records through a six-gate acquisition funnel with
1.Storefront
2.Quote Start
3.Quote Complete
4.App Start
5.App Complete
6.Aquire Product

Two independent exposure flags, IT trap and business trap, are drawn at each gate a member reaches, 
so a member's exposure varies across their journey.

The terminal-gate effect for each exposure is one of two parameters, IT Trap and Business Trap, which is what makes
the diagnostic in diagnose.py falsifiable: plant a known effect, check that it is recovered;
plant none, check that the procedure stays quiet.

Business-rule exposure is a member-level attribute, constant across the journey. IT trap is a per-gate event. 
Through the first five gates, IT-trapped members advance at organic rates. At the terminal gate they acquire 
at roughly 15 points below their organic counterparts. Stratifying on business-rule exposure leaves the IT effect
essentially unchanged, so the deficit is not business rules in disguise.

 """
import numpy as np
import pandas as pd

STAGES = ["storefront", "quote_start", "quote_complete", "app_start", "app_complete"]

# Advance rates through the five pre-terminal gates. Applied to every member
# regardless of exposure -- the exposures act only at the terminal gate.
GATE_RATES = [0.980, 0.872, 0.853, 0.809, 0.830]

BASELINE_ISSUE = 0.890  # P(issued | app complete) for unexposed members


def generate(n=10_000,
             p_it=0.215,
             p_biz=0.150,
             it_effect=-0.170,
             biz_effect=-0.110,
             interaction=0.060,
             gate_rates=GATE_RATES,
             baseline=BASELINE_ISSUE,
             seed=0):
    """Generate n member journeys.

    p_it, p_biz     marginal probability of each exposure flag (drawn independently)
    it_effect       additive shift in P(issued | app complete) for IT-flagged members
    biz_effect      same, for business-rule-flagged members
    interaction     departure from additivity when both flags are present
    """
    rng = np.random.default_rng(seed)

    it = rng.random(n) < p_it
    biz = rng.random(n) < p_biz

    reached = np.ones(n, dtype=bool)
    cols = {}
    for stage, rate in zip(STAGES, gate_rates):
        reached = reached & (rng.random(n) < rate)
        cols[stage] = reached.astype(int)

    p_issue = np.full(n, baseline)
    p_issue += it * it_effect
    p_issue += biz * biz_effect
    p_issue += (it & biz) * interaction
    p_issue = np.clip(p_issue, 0.0, 1.0)

    issued = reached & (rng.random(n) < p_issue)

    return pd.DataFrame({
        "member_id": np.arange(1, n + 1),
        **cols,
        "it_trap": it.astype(int),
        "business_trap": biz.astype(int),
        "issued": issued.astype(int),
    })


if __name__ == "__main__":
    df = generate(seed=20260904)
    df.to_csv("funnel_synthetic.csv", index=False)
    print(f"wrote funnel_synthetic.csv  n={len(df)}")
    ac = df["app_complete"] == 1
    print(f"reached app_complete: {ac.sum()}")
    for a in (0, 1):
        for b in (0, 1):
            s = df[ac & (df.it_trap == a) & (df.business_trap == b)]
            print(f"  it={a} biz={b}: n={len(s):5d}  issue={s.issued.mean():.4f}")
