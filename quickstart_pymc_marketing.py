"""
PyMC-Marketing quickstart.

Purpose: end-to-end reference showing how to fit a Bayesian MMM on our
clean synthetic data, mapping the library's API to the math we implemented
ourselves. This script is written to be RUN OFFLINE - a real fit takes
10-30 minutes on a laptop.

For a very fast smoke test (proves the pipeline works but ROIs are unreliable):
  Change SMOKE_TEST=True below. That mode uses 150 draws / 1 chain / 3 channels.

For a proper fit (Week 2 settings, produces meaningful ROI recovery):
  Leave SMOKE_TEST=False. Uses 1000 draws / 4 chains / all 5 channels.

Library <-> our notation cheat-sheet:
  adstock_alpha       = theta       (geometric adstock retention rate)
  saturation_slope    = alpha       (Hill saturation shape)
  saturation_kappa    = kappa       (Hill half-saturation point)
  saturation_beta     = beta        (channel coefficient)
  gamma_control                     (coefficients on our linear controls)
  gamma_fourier                     (Fourier seasonality coefficients)
  y_sigma             = sigma       (observation noise std)

Run:  python quickstart_pymc_marketing.py
"""

import json
import os
import warnings

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SMOKE_TEST = False   # flip to True for the fast, low-quality shakeout run

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================ load data ====
df = pd.read_csv(os.path.join(HERE, "data", "synthetic_clean.csv"),
                 parse_dates=["date"])

with open(os.path.join(HERE, "data", "true_params.json")) as f:
    truth = json.load(f)
true_roi = truth["variants"]["clean"]["true_roi"]

spend_cols = [c for c in df.columns if c.endswith("_spend")]
channels = [c.replace("_spend", "") for c in spend_cols]

if SMOKE_TEST:
    df = df.iloc[:104].copy()
    spend_cols = ["tv_spend", "search_spend", "social_spend"]
    channels = ["tv", "search", "social"]

df["t"] = np.arange(len(df))
print(f"data: {df.shape}   channels: {channels}")


# ================================================== build the MMM object ===
mmm = MMM(
    date_column="date",
    channel_columns=spend_cols,
    control_columns=["t"],                       # trend as a linear control
    adstock=GeometricAdstock(l_max=12),          # matches our L=12
    saturation=HillSaturation(),                 # matches our Hill
    yearly_seasonality=3,                        # matches our N_FOURIER=3
)

X = df[["date", "t"] + spend_cols]
y = df["revenue"]


# ================================================================ fit ====
if SMOKE_TEST:
    draws, tune, chains = 150, 150, 1
    target_accept = 0.98
else:
    draws, tune, chains = 1000, 1000, 4
    target_accept = 0.98

print(f"sampling: draws={draws}  tune={tune}  chains={chains} "
      f"target_accept={target_accept}")
print("(this can take 10-30 minutes for the full-quality run)")

idata = mmm.fit(
    X=X, y=y,
    draws=draws, tune=tune, chains=chains, cores=min(chains, 4),
    target_accept=target_accept, random_seed=42, progressbar=True,
)


# ============================================================ diagnostics ===
summary = az.summary(
    idata,
    var_names=["intercept", "adstock_alpha",
               "saturation_slope", "saturation_kappa", "saturation_beta",
               "gamma_control", "gamma_fourier", "y_sigma"],
    round_to=3,
)
print("\n--- posterior summary ---")
print(summary[["mean", "sd", "r_hat", "ess_bulk"]])

max_rhat = float(summary["r_hat"].max())
n_diverging = int(idata.sample_stats["diverging"].sum())
print(f"\nmax R-hat: {max_rhat:.3f}   (want < 1.05)")
print(f"divergences: {n_diverging}   (want 0)")


# ================================================ recovered ROI vs truth ==
contrib = mmm.compute_channel_contribution_original_scale()
total_contrib = contrib.sum(dim="date")
total_spend = df[spend_cols].sum().values

print("\n--- ROI: fitted vs true (clean synthetic) ---")
print(f"{'channel':<10s} {'true':>7s}   {'mean':>7s}   {'95% CI':>18s}")
means, los, his, truths = [], [], [], []
for i, ch in enumerate(channels):
    samples = total_contrib.isel(channel=i).values.ravel() / total_spend[i]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    print(f"{ch:<10s} {true_roi[ch]:>7.2f}   {samples.mean():>7.2f}   "
          f"[{lo:>6.2f}, {hi:>6.2f}]")
    means.append(samples.mean()); los.append(lo); his.append(hi)
    truths.append(true_roi[ch])


# ============================================================ figure =======
fig, ax = plt.subplots(figsize=(9, 4.8))
x_pos = np.arange(len(channels))
ax.errorbar(x_pos, means,
            yerr=[np.array(means) - np.array(los),
                  np.array(his) - np.array(means)],
            fmt="o", color="#185FA5", capsize=4, lw=1.5, ms=7,
            label="fitted (95% CI)")
ax.scatter(x_pos, truths, marker="D", s=80, color="#BA7517", zorder=5,
           label="true ROI")
ax.set_xticks(x_pos); ax.set_xticklabels(channels)
ax.set_ylabel("ROI")
ax.set_title(f"PyMC-Marketing quickstart: fitted vs true ROI "
             f"({'smoke test' if SMOKE_TEST else 'proper fit'}, clean synthetic)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
out = os.path.join(FIG_DIR, "pymc_marketing_quickstart.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved {out}")
