"""Figure: the predictive distribution against what actually happened.

Draws the posterior predictive distribution of terminal conversions under the
declared assumption, with the observed count marked. The gap between the two is
the finding.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose import ppc, STAGES

NAVY = "#1B2A4A"
RED = "#C8322D"
GREY = "#8A94A6"


def _draws(df, exposure, n_draws=40_000, seed=1):
    """Re-run the chain, returning the draws themselves rather than a summary."""
    rng = np.random.default_rng(seed)
    exposed = df[exposure] == 1
    ac = df["app_complete"] == 1
    flags = [c for c in ("it_trap", "business_trap") if c in df.columns]
    base_mask = ~df[flags].any(axis=1)

    sub = df[exposed]
    counts = [len(sub)] + [int((sub[s] == 1).sum()) for s in STAGES]
    base = df[ac & base_mask]
    k6, n6 = int(base.issued.sum()), len(base)

    x = np.full(n_draws, counts[0])
    for i in range(5):
        k, n = counts[i + 1], counts[i]
        x = rng.binomial(x, rng.beta(k + 0.5, n - k + 0.5, n_draws))
    return rng.binomial(x, rng.beta(k6 + 0.5, n6 - k6 + 0.5, n_draws))


def main(csv="funnel_synthetic.csv", out="predictive_gap.png"):
    df = pd.read_csv(csv)
    draws = _draws(df, "it_trap")
    res = ppc(df, "it_trap", seed=1)
    observed = res["observed"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(draws, bins=60, color=NAVY, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(observed, color=RED, linewidth=2.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)   # headroom for the callouts

    ax.annotate(f"what actually\nhappened: {observed}",
                xy=(observed, ax.get_ylim()[1] * 0.45),
                xytext=(observed - (draws.mean() - observed) * 0.75,
                        ax.get_ylim()[1] * 0.62),
                color=RED, fontsize=11, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.6))
    ax.annotate(f"what the assumption\npredicts: {res['predicted']:.0f} \u00b1 {res['sd']:.0f}",
                xy=(draws.mean(), ax.get_ylim()[1] * 0.88),
                color=NAVY, fontsize=11, fontweight="bold", ha="center")

    ax.set_title("I ran what IT believed 40,000 times.\nIt never produced what happened.",
                 fontsize=15, fontweight="bold", color=NAVY, loc="left", pad=16)
    ax.set_xlabel("policies issued to members who hit a system error", color=GREY)
    ax.set_ylabel("draws out of 40,000", color=GREY)
    ax.text(0.0, -0.19,
            f"Z = {res['z']:+.1f}   ·   {res['tail_count']} of 40,000 draws reached the observed count"
            f"   ·   lowest draw: {draws.min()}",
            transform=ax.transAxes, fontsize=9.5, color=GREY)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=GREY)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}  (observed={observed}, predicted={res['predicted']:.0f}, Z={res['z']:+.2f})")


if __name__ == "__main__":
    main()
