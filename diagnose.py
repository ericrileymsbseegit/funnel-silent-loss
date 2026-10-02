"""Posterior predictive diagnostic for a declared funnel assumption.

The assumption under test: "the exposure causes some drop-off along the journey,
but a member who completes an application converts like anyone else."

Encode it as a generative model. Fit every pre-terminal gate rate from the
exposed cohort's own data, so those gates cannot fail -- they are fit to the
thing being tested. Borrow exactly one parameter, the terminal conversion rate,
from unexposed members. That is the assumption, stated as a number.

The terminal count is then a quantity the model was never fit to, which is what
makes it a test rather than a mirror.
"""
import numpy as np
import pandas as pd

STAGES = ["storefront", "quote_start", "quote_complete", "app_start", "app_complete"]


def _chain(df, mask):
    sub = df[mask]
    return [len(sub)] + [int((sub[s] == 1).sum()) for s in STAGES]


def ppc(df, exposure, baseline_mask=None, n_draws=40_000, seed=0):
    """Run the posterior predictive check for one exposure column.

    exposure        name of a 0/1 column
    baseline_mask   boolean Series selecting the comparison cohort. Defaults to
                    members carrying no exposure flag at all, which keeps the
                    baseline uncontaminated by the other exposure.
    """
    rng = np.random.default_rng(seed)
    exposed = df[exposure] == 1
    ac = df["app_complete"] == 1

    if baseline_mask is None:
        flags = [c for c in ("it_trap", "business_trap") if c in df.columns]
        baseline_mask = ~df[flags].any(axis=1)

    counts = _chain(df, exposed)
    gates = [(counts[i + 1], counts[i]) for i in range(5)]

    base = df[ac & baseline_mask]
    k6, n6 = int(base.issued.sum()), len(base)

    observed = int(df[ac & exposed].issued.sum())

    # Jeffreys prior on every gate; uncertainty propagates down the chain.
    x = np.full(n_draws, counts[0])
    for k, n in gates:
        x = rng.binomial(x, rng.beta(k + 0.5, n - k + 0.5, n_draws))
    predicted = rng.binomial(x, rng.beta(k6 + 0.5, n6 - k6 + 0.5, n_draws))

    mu, sd = predicted.mean(), predicted.std(ddof=1)
    return {
        "exposure": exposure,
        "cohort": counts[0],
        "at_terminal": counts[-1],
        "observed": observed,
        "predicted": mu,
        "sd": sd,
        "z": (observed - mu) / sd,
        "tail_count": int((predicted <= observed).sum()),
        "n_draws": n_draws,
        "borrowed_rate": k6 / n6,
        "gate_rates": [k / n for k, n in gates],
    }


def localize(df, exposure, baseline_mask=None):
    """Where does the effect live? Test each gate against the baseline cohort,
    then the terminal gate. A terminal-only effect shows |z| small everywhere
    except the last row."""
    exposed = df[exposure] == 1
    if baseline_mask is None:
        flags = [c for c in ("it_trap", "business_trap") if c in df.columns]
        baseline_mask = ~df[flags].any(axis=1)

    ce, cb = _chain(df, exposed), _chain(df, baseline_mask)
    rows = []
    for i, stage in enumerate(STAGES):
        rows.append(_two_prop(f"->{stage}", ce[i + 1], ce[i], cb[i + 1], cb[i]))

    ac = df["app_complete"] == 1
    e, b = df[ac & exposed], df[ac & baseline_mask]
    rows.append(_two_prop("->issued", int(e.issued.sum()), len(e),
                          int(b.issued.sum()), len(b)))
    return pd.DataFrame(rows)


def _two_prop(label, k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return {"gate": label, "exposed": p1, "baseline": p2,
            "diff_pp": 100 * (p1 - p2), "z": (p1 - p2) / se if se else np.nan}


def precision_entry(df, exposure):
    """Off-diagonal precision-matrix entry for (exposure, issued), conditioned on
    reaching app complete. A pure chain through the funnel requires this to be
    zero; a nonzero entry is the fingerprint of a direct path."""
    d = df[df["app_complete"] == 1]
    m = np.c_[d[exposure].to_numpy(float), d["issued"].to_numpy(float)]
    Q = np.linalg.inv(np.cov(m.T))
    r = np.corrcoef(m.T)[0, 1]
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(len(d) - 3)
    return {"n": len(d), "partial_corr": r, "fisher_z": z, "Q_offdiag": Q[0, 1]}


def report(res):
    print(f"[{res['exposure']}] cohort={res['cohort']} at_terminal={res['at_terminal']}")
    print(f"  borrowed terminal rate : {res['borrowed_rate']:.4f}")
    print(f"  predicted              : {res['predicted']:.1f} +/- {res['sd']:.1f}")
    print(f"  observed               : {res['observed']}")
    print(f"  Z                      : {res['z']:+.2f}")
    print(f"  draws at or below obs  : {res['tail_count']} / {res['n_draws']}")


if __name__ == "__main__":
    df = pd.read_csv("funnel_synthetic.csv")
    for col in ("it_trap", "business_trap"):
        report(ppc(df, col, seed=1))
        print(localize(df, col).to_string(index=False,
              formatters={"exposed": "{:.4f}".format, "baseline": "{:.4f}".format,
                          "diff_pp": "{:+.2f}".format, "z": "{:+.2f}".format}))
        print("  precision:", {k: round(v, 4) for k, v in precision_entry(df, col).items()})
        print()
