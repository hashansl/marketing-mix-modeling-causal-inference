"""
Model comparison figure - the three-model ladder in one picture.

Shows, per channel, the ROI estimate from each model against the true value:
  Model 0  naive OLS (raw spend)                     - point estimate
  Model 1  oracle OLS with true transforms           - point estimate
  Model 2  Bayesian MMM (PyMC-Marketing)             - median + 95% CI

Reads:
  outputs/ols_baselines.json    Models 0 and 1 (from models_ols_baselines.py)
  outputs/model2_roi.nc         Model 2 ROI samples (from quickstart v2)
  data/true_params.json         ground truth

Produces:
  outputs/figures/model_comparison.png    per-channel grouped comparison
  outputs/figures/model_comparison_rmse.png   overall ROI RMSE by model

Run:  python make_model_comparison.py
"""
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================ load truth ===
with open(os.path.join(HERE, "data", "true_params.json")) as f:
    truth = json.load(f)
true_roi = truth["variants"]["clean"]["true_roi"]

df = pd.read_csv(os.path.join(HERE, "data", "synthetic_clean.csv"), nrows=1)
spend_cols = [c for c in df.columns if c.endswith("_spend")]
channels = [c.replace("_spend", "") for c in spend_cols]

# ============================================== load Models 0 & 1 (OLS) ====
with open(os.path.join(HERE, "outputs", "ols_baselines.json")) as f:
    ols = json.load(f)
roi_m0 = ols["model_0_naive_ols"]["roi"]
roi_m1 = ols["model_1_oracle_ols"]["roi"]

# ============================================== load Model 2 (Bayesian) ====
roi_path = os.path.join(HERE, "outputs", "model2_roi.nc")
try:
    roi_da = xr.open_dataarray(roi_path)
except Exception:
    ds = xr.open_dataset(roi_path)
    roi_da = ds[list(ds.data_vars)[0]]

m2_med, m2_lo, m2_hi = {}, {}, {}
for c, ch in enumerate(channels):
    s = roi_da.isel(channel=c).values.ravel()
    m2_med[ch] = float(np.median(s))
    m2_lo[ch], m2_hi[ch] = np.percentile(s, [2.5, 97.5])

# ===================================================== figure 1: per channel
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(channels))
w = 0.22

# true value as a wide grey bar behind everything
ax.bar(x, [true_roi[ch] for ch in channels], width=0.8, color="#e8e8e8",
       label="true ROI", zorder=1)

# model point estimates as grouped markers
ax.scatter(x - w, [roi_m0[ch] for ch in channels], s=70, color="#C15B5B",
           marker="s", zorder=3, label="Model 0: naive OLS")
ax.scatter(x, [roi_m1[ch] for ch in channels], s=70, color="#BA7517",
           marker="^", zorder=3, label="Model 1: oracle OLS + transforms")

# Model 2 with error bars
ax.errorbar(x + w, [m2_med[ch] for ch in channels],
            yerr=[[m2_med[ch] - m2_lo[ch] for ch in channels],
                  [m2_hi[ch] - m2_med[ch] for ch in channels]],
            fmt="o", color="#185FA5", capsize=4, ms=7, lw=1.5, zorder=3,
            label="Model 2: Bayesian (median + 95% CI)")

ax.set_xticks(x); ax.set_xticklabels(channels, fontsize=11)
ax.set_ylabel("ROI", fontsize=11)
# ax.set_title("Model comparison: ROI estimate vs truth, by channel", fontsize=12)
# cap y so the Bayesian tails don't blow out the scale; note clipping

# Increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=12)

ymax = max(6, max(true_roi.values()) * 1.6)
ax.set_ylim(0, ymax)
ax.legend(fontsize=12, loc="upper right")
ax.grid(alpha=0.3, axis="y")
# annotate any Model-2 upper CI that got clipped
for c, ch in enumerate(channels):
    if m2_hi[ch] > ymax:
        ax.annotate(f"CI to {m2_hi[ch]:.0f}", (x[c] + w, ymax * 0.97),
                    fontsize=12, ha="center", color="#185FA5", rotation=90,
                    va="top")
fig.tight_layout()
out1 = os.path.join(FIG_DIR, "model_comparison.png")
fig.savefig(out1, dpi=130, bbox_inches="tight")
print(f"saved {out1}")

# ===================================================== figure 2: RMSE bars ==
def rmse(roi_dict):
    return float(np.sqrt(np.mean([(roi_dict[ch] - true_roi[ch]) ** 2
                                   for ch in channels])))

rmse_vals = {
    "Model 0\nnaive OLS": rmse(roi_m0),
    "Model 1\noracle OLS": rmse(roi_m1),
    "Model 2\nBayesian (median)": rmse(m2_med),
}
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(list(rmse_vals.keys()), list(rmse_vals.values()),
              color=["#C15B5B", "#BA7517", "#185FA5"])
for b, v in zip(bars, rmse_vals.values()):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
            ha="center", fontsize=10)
ax.set_ylabel("ROI RMSE vs truth (lower = better)", fontsize=11)
# ax.set_title("Overall ROI recovery error by model", fontsize=12)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
out2 = os.path.join(FIG_DIR, "model_comparison_rmse.png")
fig.savefig(out2, dpi=130, bbox_inches="tight")
print(f"saved {out2}")

# ===================================================== text summary =========
print("\n--- ROI by model vs truth ---")
print(f"{'channel':<10s} {'true':>7s} {'M0':>7s} {'M1':>7s} {'M2 med':>8s}")
for ch in channels:
    print(f"{ch:<10s} {true_roi[ch]:>7.2f} {roi_m0[ch]:>7.2f} "
          f"{roi_m1[ch]:>7.2f} {m2_med[ch]:>8.2f}")
print(f"\nRMSE: M0={rmse(roi_m0):.3f}  M1={rmse(roi_m1):.3f}  M2={rmse(m2_med):.3f}")
