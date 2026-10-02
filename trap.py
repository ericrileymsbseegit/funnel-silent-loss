import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
S = 40_000

# Counts backed out of the slides:
#   4,907 reached App Complete, 4,116 issued
#   trapped: 1,074 reached App Complete, 768 issued (71.5%)
#   clean:   4,907 - 1,074 = 3,833 reached, 4,116 - 768 = 3,348 issued (87.3%)
k_C, n_C = 3348, 3833   # clean path: issued, reached App Complete
k_T, n_T = 768, 1074    # trapped path: replace with an EARLIER month's counts if you have them
N_T = 1074              # trapped applications this month
actual = 768

p_null = rng.beta(1 + k_C, 1 + n_C - k_C, S)
p_trap = rng.beta(1 + k_T, 1 + n_T - k_T, S)
y_null = rng.binomial(N_T, p_null)
y_trap = rng.binomial(N_T, p_trap)

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.arange(min(y_trap.min(), actual) - 10, y_null.max() + 10, 5)
ax.hist(y_null, bins=bins, color="#9AA8B5", label="If the trap made no difference")
ax.hist(y_trap, bins=bins, color="#7A1F2B", alpha=0.85, label="If the trap matters")
ax.axvline(actual, color="#E5484D", lw=3)
ax.text(actual - 4, ax.get_ylim()[1] * 0.92, f"actual {actual}", color="#E5484D", fontweight="bold", ha="right")
ax.set_title("40,000 simulated months under each explanation")
ax.set_xlabel("policies issued to trapped members")
ax.set_ylabel("simulated months")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper center")
fig.savefig("trap_two_models.png", dpi=200, bbox_inches="tight")

print("no-difference mean:", round(y_null.mean()), " range:", y_null.min(), "-", y_null.max())
print("share of no-difference draws <= actual:", (y_null <= actual).mean())
print("trap-model 90% interval:", np.percentile(y_trap, [5, 95]))
