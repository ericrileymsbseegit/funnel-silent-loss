"""Validation: does the diagnostic behave?

Two properties a diagnostic must have before its output means anything.

  Recovery -- plant a terminal-gate effect of known size, confirm the procedure
              finds it, at the right gate, with a magnitude that tracks truth.
  Null     -- plant no effect, confirm the procedure stays quiet at roughly the
              nominal rate. A test that fires on null data is worthless on real data.
"""
import numpy as np
import pandas as pd
from generate import generate
from diagnose import ppc, localize

RECOVERY_EFFECTS = [0.00, -0.02, -0.05, -0.10, -0.17, -0.25]
N_NULL_REPS = 200
ALPHA_Z = 1.96


def recovery(n=10_000, reps=20):
    rows = []
    for eff in RECOVERY_EFFECTS:
        z_vals, est = [], []
        for r in range(reps):
            df = generate(n=n, it_effect=eff, biz_effect=0.0, interaction=0.0,
                          seed=1000 + r)
            res = ppc(df, "it_trap", n_draws=8_000, seed=r)
            loc = localize(df, "it_trap")
            z_vals.append(res["z"])
            est.append(loc.iloc[-1]["diff_pp"])
        rows.append({"planted_pp": 100 * eff,
                     "recovered_pp": np.mean(est),
                     "mean_Z": np.mean(z_vals),
                     "detected": np.mean([abs(z) > 5 for z in z_vals])})
    return pd.DataFrame(rows)


def null_calibration(n=10_000, reps=N_NULL_REPS):
    """No effect anywhere. Count how often each gate falsely trips |z| > 1.96."""
    trips = np.zeros(6)
    for r in range(reps):
        df = generate(n=n, it_effect=0.0, biz_effect=0.0, interaction=0.0,
                      seed=5000 + r)
        loc = localize(df, "it_trap")
        trips += (loc["z"].abs() > ALPHA_Z).to_numpy()
    return pd.DataFrame({"gate": localize(generate(n=n, seed=1), "it_trap")["gate"],
                         "false_trip_rate": trips / reps})


if __name__ == "__main__":
    print("RECOVERY -- planted terminal effect vs what the procedure reports\n")
    print(recovery().to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    print("\n\nNULL CALIBRATION -- no effect planted; nominal false-trip rate is 0.05\n")
    print(null_calibration().to_string(index=False, float_format=lambda x: f"{x:.3f}"))
