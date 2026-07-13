"""Teaching figures for adstock and saturation, using the project's transforms."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from src.transforms import geometric_adstock, delayed_adstock, hill_saturation

FIG = "outputs/figures"
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- ADSTOCK ----
fig, ax = plt.subplots(figsize=(6, 4))

# impulse response: spend $1 in week 0 only, watch the effect linger
weeks = 16
imp = np.zeros(weeks); imp[0] = 1.0
for theta, name, c in [(0.65, "TV  (theta=0.65, long memory)", "#185FA5"),
                       (0.35, "social (theta=0.35)", "#2E8B74"),
                       (0.15, "search (theta=0.15, short memory)", "#BA7517")]:
    resp = geometric_adstock(imp, theta=theta, L=12, normalize=False)
    ax.plot(range(weeks), resp, marker="o", ms=4, label=name, color=c)
# ax.set_title("ADSTOCK: one week of spend, effect spread over time")
ax.set_xlabel("Weeks after the spend", fontsize=15); ax.set_ylabel("Fraction of effect still active", fontsize=15)

# increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/adstock_impulse_response.png", dpi=130, bbox_inches="tight")
print("saved adstock_impulse_response.png")

# on a real bursty spend series: raw vs adstocked
fig, ax = plt.subplots(figsize=(6, 4))
rng = np.random.default_rng(1)
spend = np.clip(rng.normal(20, 6, 60), 1, None)
spend[10] = 60; spend[35] = 55  # two big bursts
ax.bar(range(60), spend, color="#cccccc", label="raw weekly spend")
ax.plot(range(60), geometric_adstock(spend, 0.65, 12, normalize=True),
        color="#185FA5", lw=2, label="adstocked (theta=0.65)")
# ax.set_title("ADSTOCK smooths bursts into sustained pressure", fontsize=18)
ax.set_xlabel("Week", fontsize=15); ax.set_ylabel("Spend / Effective Pressure", fontsize=15)

# increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/adstock_smoothing.png", dpi=130, bbox_inches="tight")
print("saved adstock_smoothing.png")

# geometric vs delayed adstock
fig, ax = plt.subplots(figsize=(6, 4))
lags = np.arange(16)
geo_adstock = geometric_adstock(imp, theta=0.8, L=15, normalize=False)
delayed_adstock_5 = delayed_adstock(imp, alpha=0.8, theta=5, L=15, normalize=False)
ax.plot(lags, geo_adstock, marker="o", ms=4, label="geometric (alpha=0.8)", color="black")
ax.plot(lags, delayed_adstock_5, marker="s", ms=4, label="delayed (alpha=0.8, theta=5)", color="red")
# ax.set_title("Geometric vs Delayed Adstock", fontsize=18)
ax.set_xlabel("Lag", fontsize=15); ax.set_ylabel("Adstock", fontsize=15)

# increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/adstock_comparison.png", dpi=130, bbox_inches="tight")
print("saved adstock_comparison.png")

# ------------------------------------------------------------- SATURATION ----
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 80, 400)

# response curves for a few channels (their real alpha, kappa)
chans = [("tv", 2.0, 28.0, "#185FA5"), ("search", 1.6, 18.0, "#BA7517"),
         ("email", 1.7, 5.0, "#2E8B74")]
for name, a, k, c in chans:
    ax.plot(x, hill_saturation(x, a, k), color=c, lw=2, label=f"{name} (kappa={k})")
    ax.axvline(k, color=c, ls=":", alpha=0.5)
ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
# ax.set_title("SATURATION: diminishing returns (Hill curve)", fontsize=18)
ax.set_xlabel("(Adstocked) Spend", fontsize=15); ax.set_ylabel("Response (0 to 1)", fontsize=15)
ax.text(29, 0.05, "half-saturation\npoints (kappa)", fontsize=8, color="gray")

# increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/saturation_curves.png", dpi=130, bbox_inches="tight")
print("saved saturation_curves.png")

fig, ax = plt.subplots(figsize=(6, 4))
# marginal return: the extra response from one more dollar
for name, a, k, c in chans:
    s = hill_saturation(x, a, k)
    marg = np.gradient(s, x)
    ax.plot(x, marg, color=c, lw=2, label=name)
# ax.set_title("MARGINAL return falls as spend rises", fontsize=18)
ax.set_xlabel("(Adstocked) Spend", fontsize=15); ax.set_ylabel("Extra response per extra $", fontsize=15)

# increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/marginal_return.png", dpi=130, bbox_inches="tight")
print("saved marginal_return.png")