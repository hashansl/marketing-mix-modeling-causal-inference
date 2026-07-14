"""
Build the parameter-recovery figures

Reads the slim files saved by quickstart_pymc_marketing.py:
  outputs/model2_posterior.nc   parameter posteriors
  outputs/model2_roi.nc         per-channel ROI samples

Produces two figures:
  outputs/figures/recovery_parameters.png
      5 channels x 4 params (theta, alpha, kappa, beta) grid of posteriors,
      each with a vertical line at the TRUE value. If the true line sits
      inside the bulk of the posterior, that parameter is recovered.

  outputs/figures/recovery_roi.png
      Per-channel ROI: posterior distribution with true ROI marked.

Run:  python make_recovery_figure.py
"""
import json
import os

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================ load truth ===
with open(os.path.join(HERE, "data", "true_params.json")) as f:
    truth = json.load(f)
clean = truth["variants"]["clean"]
true_roi = clean["true_roi"]
channel_params = clean["channel_params"]

# The channel order the model used (matches spend_cols order in the fit).
df = pd.read_csv(os.path.join(HERE, "data", "synthetic_clean.csv"), nrows=1)
spend_cols = [c for c in df.columns if c.endswith("_spend")]
channels = [c.replace("_spend", "") for c in spend_cols]

# true values per parameter, in the RAW scale we simulated on
true_vals = {
    "adstock_alpha":    {ch: channel_params[ch]["decay"] for ch in channels},  # theta
    "saturation_slope": {ch: channel_params[ch]["alpha"] for ch in channels},  # alpha
    # kappa/beta live on the raw scale in truth but on the [0,1] scale in the
    # model, so we don't draw a true line for those two - we show the posterior
    # only and rely on ROI (which IS on a comparable scale) for the headline.
}

# ============================================================ load fit =====
import xarray as xr

post = az.from_netcdf(os.path.join(HERE, "outputs", "model2_posterior.nc"))

# roi was saved as an xarray DataArray -> load it with xarray, not arviz.
roi_path = os.path.join(HERE, "outputs", "model2_roi.nc")
try:
    roi_da = xr.open_dataarray(roi_path)
except Exception:
    # fallback: it was saved as a Dataset; grab the single data variable
    ds = xr.open_dataset(roi_path)
    roi_da = ds[list(ds.data_vars)[0]]


# =============================================== figure 1: adstock + slope ==
# We can compare adstock_alpha (theta) and saturation_slope (alpha) directly
# to their true values because those priors were on the same scale.
params_to_plot = ["adstock_alpha", "saturation_slope"]
param_titles = {"adstock_alpha": "Adstock retention  (theta)",
                "saturation_slope": "Saturation shape  (alpha)"}

fig, axes = plt.subplots(len(params_to_plot), len(channels),
                         figsize=(3.0 * len(channels), 2.6 * len(params_to_plot)),
                         squeeze=False)

for r, param in enumerate(params_to_plot):
    samples_all = post.posterior[param]  # dims: chain, draw, channel
    for c, ch in enumerate(channels):
        ax = axes[r][c]
        s = samples_all.isel(channel=c).values.ravel()
        ax.hist(s, bins=40, color="#185FA5", alpha=0.75, density=True)
        # true value line (only for params where truth is on same scale)
        if param in true_vals:
            tv = true_vals[param][ch]
            ax.axvline(tv, color="#BA7517", lw=2.2, label="true")
            # mark whether the true value is inside the 95% CI
            lo, hi = np.percentile(s, [2.5, 97.5])
            inside = lo <= tv <= hi
            # ax.text(0.5, 0.92, "covered" if inside else "MISSED",
            #         transform=ax.transAxes, ha="center", fontsize=15,
            #         color="#1a7a3a" if inside else "#b00")
        if r == 0:
            ax.set_title(ch, fontsize=11)
        if c == 0:
            ax.set_ylabel(param_titles[param], fontsize=12)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

# fig.suptitle("Parameter recovery: posterior (blue) vs true value (orange)",
            #  fontsize=12, y=1.01)
fig.tight_layout()
out1 = os.path.join(FIG_DIR, "recovery_parameters.png")
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"saved {out1}")


# ===================================================== figure 2: ROI recovery
fig, axes = plt.subplots(1, len(channels),
                         figsize=(3.0 * len(channels), 3.2), squeeze=False)
axes = axes[0]
for c, ch in enumerate(channels):
    ax = axes[c]
    s = roi_da.isel(channel=c).values.ravel()
    # clip extreme tail for display only (keeps the plot readable)
    disp_hi = np.percentile(s, 99)
    ax.hist(s[s <= disp_hi], bins=40, color="#2E8B74", alpha=0.75, density=True)
    tv = true_roi[ch]
    ax.axvline(tv, color="#BA7517", lw=2.2, label="true ROI")
    med = np.median(s)
    ax.axvline(med, color="#185FA5", lw=1.6, ls="--", label="posterior median")
    lo, hi = np.percentile(s, [2.5, 97.5])
    inside = lo <= tv <= hi
    ax.set_title(f"{ch}\ntrue={tv:.2f}  med={med:.2f}", fontsize=15)
    # ax.text(0.5, 0.9, "covered" if inside else "MISSED",
    #         transform=ax.transAxes, ha="center", fontsize=8,
    #         color="#1a7a3a" if inside else "#b00")
    ax.set_yticks([]); ax.tick_params(labelsize=7)
    if c == 4:
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=12, loc="upper right")

# fig.suptitle("ROI recovery: posterior (green), true (orange), median (blue dashed)",
            #  fontsize=12, y=1.03)
fig.tight_layout()
out2 = os.path.join(FIG_DIR, "recovery_roi.png")
fig.savefig(out2, dpi=130, bbox_inches="tight")
print(f"saved {out2}")

# ===================================================== quick text summary ===
print("\n--- recovery summary ---")
print(f"{'channel':<10s} {'true ROI':>9s} {'post med':>9s} {'covered?':>9s}")
for c, ch in enumerate(channels):
    s = roi_da.isel(channel=c).values.ravel()
    lo, hi = np.percentile(s, [2.5, 97.5])
    med = np.median(s)
    inside = "yes" if lo <= true_roi[ch] <= hi else "NO"
    print(f"{ch:<10s} {true_roi[ch]:>9.2f} {med:>9.2f} {inside:>9s}")