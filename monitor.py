"""Daily monitoring: is the population drifting, or is the process breaking?

diagnose.py answers a question once. A model in production has to be watched
every day, and two different things can go wrong:

  Population drift  -- the people arriving change (a new channel, a price move).
                       The process may be fine; the baseline is just stale.
                       Caught by PSI and KS on an input.
  Process break     -- the people are the same, but a step stops working.
                       Drift checks cannot see this: nothing about the inputs
                       moved. Caught by the posterior predictive check.

This script simulates 60 nightly batches from generate.py, plants one of each
failure on a known day, and runs all three checks every night. Each tool should
fire on its own failure and stay quiet on the other.

  Days  1-20  baseline window (drift checks compare against this)
  Day  30     population drift: quoted premium shifts up ~22%, process unchanged
  Day  45     process break: system-error members start issuing 17 points lower

This is a demonstration of the monitoring design on synthetic data, not a
record of a deployed system.

    python monitor.py            # writes monitor_log.csv and monitor.png
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose import ppc
from generate import generate

N_DAYS = 60
BATCH = 4_000                # journeys per night
BASELINE_DAYS = 20
DRIFT_DAY = 30
BREAK_DAY = 45

PREMIUM_MU, PREMIUM_SIGMA = 7.0, 0.35     # log-scale; median about $1,100
DRIFT_SHIFT = 0.20                        # +0.20 on the log scale, about +22%

# Thresholds. PSI bands are the industry convention, not a statistical test.
# KS uses the D statistic, not its p-value: with thousands of records a day the
# p-value flags trivial shifts, the same reason PSI exists. PPC uses |Z|; with a
# 3-sigma alert and 40 monitored nights, expect roughly one false alert in ten
# runs of this script -- a real deployment would tune that.
PSI_WATCH, PSI_ALERT = 0.10, 0.25
KS_WATCH, KS_ALERT = 0.05, 0.10
Z_WATCH, Z_ALERT = 2.0, 3.0

NAVY, RED, GREY, AMBER = "#1B2A4A", "#C8322D", "#8A94A6", "#B7791F"


# --------------------------------------------------------------------------
# Simulate a night
# --------------------------------------------------------------------------
def nightly_batch(day, seed_base=20260929):
    """One night of journeys. The funnel comes from generate.py unchanged;
    a continuous premium column is added so KS has something continuous to test."""
    broken = day >= BREAK_DAY
    df = generate(n=BATCH,
                  it_effect=-0.17 if broken else 0.0,
                  biz_effect=0.0, interaction=0.0,
                  seed=seed_base + day)
    rng = np.random.default_rng(seed_base + 10_000 + day)
    mu = PREMIUM_MU + (DRIFT_SHIFT if day >= DRIFT_DAY else 0.0)
    df["premium"] = np.round(rng.lognormal(mu, PREMIUM_SIGMA, len(df)), 2)
    df["day"] = day
    return df


# --------------------------------------------------------------------------
# The three checks
# --------------------------------------------------------------------------
def psi(baseline, current, edges, eps=1e-4):
    """Population Stability Index on fixed bins.
    e_i = baseline share in bin i, a_i = current share in bin i.
    PSI = sum (a_i - e_i) * ln(a_i / e_i)."""
    e = np.histogram(baseline, edges)[0] / len(baseline)
    a = np.histogram(current, edges)[0] / len(current)
    e, a = np.maximum(e, eps), np.maximum(a, eps)
    return float(np.sum((a - e) * np.log(a / e)))


def ks(baseline, current):
    """Two-sample Kolmogorov-Smirnov: D = max |F_baseline(x) - F_current(x)|,
    with the asymptotic p-value (reported, not used for alerting)."""
    x, y = np.sort(baseline), np.sort(current)
    grid = np.concatenate([x, y])
    d = float(np.max(np.abs(np.searchsorted(x, grid, "right") / len(x)
                            - np.searchsorted(y, grid, "right") / len(y))))
    ne = len(x) * len(y) / (len(x) + len(y))
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


def status(value, watch, alert):
    v = abs(value)
    return "ALERT" if v >= alert else "WATCH" if v >= watch else "PASS"


def check_night(batch, baseline_premium, edges, day):
    """Everything one nightly run would do. In production this is the unit
    a scheduler calls; main() below just calls it 60 times."""
    p = psi(baseline_premium, batch["premium"].to_numpy(), edges)
    d, ks_p = ks(baseline_premium, batch["premium"].to_numpy())
    z = ppc(batch, "it_trap", n_draws=4_000, seed=day)["z"]
    row = {"day": day, "n": len(batch),
           "trap_rate": batch["it_trap"].mean(),
           "psi_premium": p, "ks_d": d, "ks_p": ks_p, "ppc_z": z,
           "psi_status": status(p, PSI_WATCH, PSI_ALERT),
           "ks_status": status(d, KS_WATCH, KS_ALERT),
           "ppc_status": status(z, Z_WATCH, Z_ALERT)}
    drift = "ALERT" in (row["psi_status"], row["ks_status"])
    broke = row["ppc_status"] == "ALERT"
    row["action"] = ("page owner: process break + drift" if drift and broke else
                     "page owner: process break" if broke else
                     "review baseline: population drift" if drift else "none")
    return row


# --------------------------------------------------------------------------
# Run the simulation
# --------------------------------------------------------------------------
def run():
    batches = [nightly_batch(day) for day in range(1, N_DAYS + 1)]
    base = pd.concat(batches[:BASELINE_DAYS])
    base_prem = base["premium"].to_numpy()
    edges = np.quantile(base_prem, np.linspace(0, 1, 11))     # baseline deciles
    edges[0], edges[-1] = -np.inf, np.inf                     # catch out-of-range values
    rows = [check_night(b, base_prem, edges, day)
            for day, b in enumerate(batches, start=1) if day > BASELINE_DAYS]
    return pd.DataFrame(rows)


def plot(log, out="monitor.png"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True)
    panels = [
        ("psi_premium", "PSI on premium", PSI_WATCH, PSI_ALERT, False),
        ("ks_d", "KS distance on premium", KS_WATCH, KS_ALERT, False),
        ("ppc_z", "PPC Z at the last gate", Z_WATCH, Z_ALERT, True),
    ]
    for ax, (col, title, watch, alert, signed) in zip(axes, panels):
        y = log[col]
        lo = -max(alert * 1.6, abs(y.min()) * 1.15) if signed else 0
        hi = max(alert * 1.6, y.max() * 1.15) if not signed else alert * 1.6
        ax.axhspan(watch, alert, color=AMBER, alpha=0.10, lw=0)
        ax.axhspan(alert, hi, color=RED, alpha=0.08, lw=0)
        if signed:
            ax.axhspan(-alert, -watch, color=AMBER, alpha=0.10, lw=0)
            ax.axhspan(lo, -alert, color=RED, alpha=0.08, lw=0)
            ax.axhline(0, color=GREY, lw=0.6)
        ax.plot(log["day"], y, color=NAVY, lw=2, marker="o", ms=4)
        flagged = log[[status(v, watch, alert) == "ALERT" for v in y]]
        ax.plot(flagged["day"], flagged[col], "o", color=RED, ms=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ax.set_ylim(lo, hi)
        ax.set_title(title, loc="left", fontsize=11, color=NAVY, fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GREY)
        ax.tick_params(colors=GREY)
        for day in (DRIFT_DAY, BREAK_DAY):
            ax.axvline(day - 0.5, color=GREY, lw=1, ls="--")

    axes[0].text(DRIFT_DAY - 0.3, axes[0].get_ylim()[1] * 0.92,
                 "day 30: applicants change", color=GREY, fontsize=9, va="top")
    axes[0].text(BREAK_DAY - 0.3, axes[0].get_ylim()[1] * 0.92,
                 "day 45: last gate breaks", color=GREY, fontsize=9, va="top")
    axes[-1].set_xlabel("day (days 1-20 are the baseline window)", color=GREY)
    fig.suptitle("Drift checks catch a changed population. The PPC catches a broken process.",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=NAVY)
    fig.text(0.01, 0.005, "Amber band = watch   ·   red band = alert   ·   red dots = alert nights",
             fontsize=9, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize(log):
    def alerts(col, lo, hi):
        w = log[(log.day >= lo) & (log.day < hi)]
        return int((w[col] == "ALERT").sum()), len(w)
    windows = [("days 21-29 (nothing changed)", 21, DRIFT_DAY),
               ("days 30-44 (population drift)", DRIFT_DAY, BREAK_DAY),
               ("days 45-60 (drift + break)", BREAK_DAY, N_DAYS + 1)]
    print(f"{'window':32s} {'PSI':>8s} {'KS':>8s} {'PPC':>8s}")
    for label, lo, hi in windows:
        cells = [alerts(c, lo, hi) for c in ("psi_status", "ks_status", "ppc_status")]
        print(f"{label:32s} " + " ".join(f"{a:>3d}/{n:<4d}" for a, n in cells))


if __name__ == "__main__":
    log = run()
    log.to_csv("monitor_log.csv", index=False)
    plot(log)
    summarize(log)
    print("\nwrote monitor_log.csv and monitor.png")
