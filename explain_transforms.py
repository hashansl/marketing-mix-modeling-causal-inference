"""Teaching figures for adstock and saturation, using the project's transforms."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from src.transforms import geometric_adstock, hill_saturation

FIG = "outputs/figures"
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- ADSTOCK ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))

# (A) impulse response: spend $1 in week 0 only, watch the effect linger
weeks = 16
imp = np.zeros(weeks); imp[0] = 1.0
for theta, name, c in [(0.65, "TV  (theta=0.65, long memory)", "#185FA5"),
                       (0.35, "social (theta=0.35)", "#2E8B74"),
                       (0.15, "search (theta=0.15, short memory)", "#BA7517")]:
    resp = geometric_adstock(imp, theta=theta, L=12, normalize=False)
    ax[0].plot(range(weeks), resp, marker="o", ms=4, label=name, color=c)
ax[0].set_title("ADSTOCK: one week of spend, effect spread over time")
ax[0].set_xlabel("weeks after the spend"); ax[0].set_ylabel("fraction of effect still active")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# (B) on a real bursty spend series: raw vs adstocked
rng = np.random.default_rng(1)
spend = np.clip(rng.normal(20, 6, 60), 1, None)
spend[10] = 60; spend[35] = 55  # two big bursts
ax[1].bar(range(60), spend, color="#cccccc", label="raw weekly spend")
ax[1].plot(range(60), geometric_adstock(spend, 0.65, 12, normalize=True),
           color="#185FA5", lw=2, label="adstocked (theta=0.65)")
ax[1].set_title("ADSTOCK smooths bursts into sustained pressure")
ax[1].set_xlabel("week"); ax[1].set_ylabel("spend / effective pressure")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/explain_adstock.png", dpi=130, bbox_inches="tight")
print("saved explain_adstock.png")

# ------------------------------------------------------------- SATURATION ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
x = np.linspace(0, 80, 400)

# (A) response curves for a few channels (their real alpha, kappa)
chans = [("tv", 2.0, 28.0, "#185FA5"), ("search", 1.6, 18.0, "#BA7517"),
         ("email", 1.7, 5.0, "#2E8B74")]
for name, a, k, c in chans:
    ax[0].plot(x, hill_saturation(x, a, k), color=c, lw=2, label=f"{name} (kappa={k})")
    ax[0].axvline(k, color=c, ls=":", alpha=0.5)
ax[0].axhline(0.5, color="gray", ls="--", alpha=0.5)
ax[0].set_title("SATURATION: diminishing returns (Hill curve)")
ax[0].set_xlabel("(adstocked) spend"); ax[0].set_ylabel("response (0 to 1)")
ax[0].text(29, 0.05, "half-saturation\npoints (kappa)", fontsize=8, color="gray")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# (B) marginal return: the extra response from one more dollar
for name, a, k, c in chans:
    s = hill_saturation(x, a, k)
    marg = np.gradient(s, x)
    ax[1].plot(x, marg, color=c, lw=2, label=name)
ax[1].set_title("MARGINAL return falls as spend rises")
ax[1].set_xlabel("(adstocked) spend"); ax[1].set_ylabel("extra response per extra $")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/explain_saturation.png", dpi=130, bbox_inches="tight")
print("saved explain_saturation.png")
