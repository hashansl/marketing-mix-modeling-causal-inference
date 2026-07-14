"""
Calibration comparison figure - reads three slim files from earlier steps
and overlays search's ROI posterior under three conditions.

Reads:
  outputs/confounded_roi_naive.nc       (from confounding_experiment.py)
  outputs/confounded_roi_adjusted.nc    (from confounding_experiment.py)
  outputs/confounded_roi_calibrated.nc  (from lift_test_calibration.py)

Produces:
  outputs/figures/calibration_search.png     three overlaid ROI posteriors
                                              for the calibrated channel
  outputs/figures/calibration_summary.png    all channels: how calibration
                                              affects each

No sampling.  Run:  python make_calibration_figure.py
"""
import json, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "outputs", "figures")

with open(os.path.join(HERE, "data", "true_params.json")) as f:
    truth = json.load(f)
true_roi = truth["variants"]["confounded"]["true_roi"]

df = pd.read_csv(os.path.join(HERE, "data", "synthetic_confounded.csv"), nrows=1)
spend_cols = [c for c in df.columns if c.endswith("_spend")]
channels = [c.replace("_spend", "") for c in spend_cols]

def load(name):
    p = os.path.join(HERE, "outputs", name)
    try:
        return xr.open_dataarray(p)
    except Exception:
        ds = xr.open_dataset(p); return ds[list(ds.data_vars)[0]]

naive = load("confounded_roi_naive.nc")
adj = load("confounded_roi_adjusted.nc")
calib = load("confounded_roi_calibrated.nc")

CALIB_CH = "search"
LIFT_TEST_ROI = 3.9
LIFT_TEST_SE = 0.4

# ============================== figure 1: three overlaid posteriors =========
i = channels.index(CALIB_CH)
sn = naive.isel(channel=i).values.ravel()
sa = adj.isel(channel=i).values.ravel()
sc = calib.isel(channel=i).values.ravel()

fig, ax = plt.subplots(figsize=(9, 5.5))
# clip x-axis at 99th percentile of naive (which has the heaviest tail) so
# the informative posteriors don't get squashed
xmax = np.percentile(sn, 99)
bins = np.linspace(0, xmax, 60)
ax.hist(sn[sn <= xmax], bins=bins, alpha=0.45, density=True, color="#C15B5B",
        label=f"naive (backdoor OPEN): median {np.median(sn):.1f}")
ax.hist(sa[sa <= xmax], bins=bins, alpha=0.45, density=True, color="#185FA5",
        label=f"adjusted (backdoor CLOSED): median {np.median(sa):.1f}")
ax.hist(sc[sc <= xmax], bins=bins, alpha=0.55, density=True, color="#2E8B74",
        label=f"CALIBRATED (Model 3): median {np.median(sc):.1f}")
ax.axvline(true_roi[CALIB_CH], color="#333", lw=2.2, ls="--",
           label=f"true ROI = {true_roi[CALIB_CH]:.2f}")
ax.axvline(LIFT_TEST_ROI, color="#BA7517", lw=1.6, ls=":",
           label=f"lift-test estimate = {LIFT_TEST_ROI} ± {LIFT_TEST_SE}")
ax.set_xlabel(f"ROI ({CALIB_CH})")
ax.set_ylabel("Posterior density")
# ax.set_title(f"Lift-test calibration on {CALIB_CH}:\n"
            #  f"experimental prior tightens the posterior around truth "
            #  f"despite the backdoor being open")
ax.legend(fontsize=12); ax.grid(alpha=0.3)
fig.tight_layout()
out1 = os.path.join(FIG, "calibration_search.png")
fig.savefig(out1, dpi=130, bbox_inches="tight"); print(f"saved {out1}")

# ============================== figure 2: all channels summary ==============
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(channels)); w = 0.22

def med_ci(da):
    m = [float(np.median(da.isel(channel=i).values.ravel())) for i in range(len(channels))]
    lo = [float(np.percentile(da.isel(channel=i).values.ravel(), 2.5)) for i in range(len(channels))]
    hi = [float(np.percentile(da.isel(channel=i).values.ravel(), 97.5)) for i in range(len(channels))]
    return np.array(m), np.array(lo), np.array(hi)

truths = np.array([true_roi[c] for c in channels])
mn, ln, hn = med_ci(naive)
ma, la, ha = med_ci(adj)
mc, lc, hc = med_ci(calib)

ax.bar(x - 1.5*w, truths, w, color="#9aa0a6", label="true ROI")
ax.errorbar(x - 0.5*w, mn, yerr=[mn-ln, hn-mn], fmt="s", color="#C15B5B",
            capsize=3, ms=8, lw=1.4, label="naive")
ax.errorbar(x + 0.5*w, ma, yerr=[ma-la, ha-ma], fmt="^", color="#185FA5",
            capsize=3, ms=8, lw=1.4, label="adjusted")
ax.errorbar(x + 1.5*w, mc, yerr=[mc-lc, hc-mc], fmt="o", color="#2E8B74",
            capsize=3, ms=8, lw=1.4, label="CALIBRATED")
# highlight the calibrated channel
ax.axvspan(x[channels.index(CALIB_CH)] - 0.5, x[channels.index(CALIB_CH)] + 0.5,
           color="#fff4b3", alpha=0.4, zorder=0, label=f"calibrated: {CALIB_CH}")

ax.set_xticks(x); ax.set_xticklabels(channels)
ax.set_ylabel("ROI (median ± 95% CI)")
# ax.set_title(f"Calibrating {CALIB_CH} tightens its posterior;\n"
            #  f"other channels: see whether they moved as a side effect")


# Increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ymax = max(6, truths.max() * 2.4)
ax.set_ylim(0, ymax)
ax.legend(fontsize=12); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
out2 = os.path.join(FIG, "calibration_summary.png")
fig.savefig(out2, dpi=130, bbox_inches="tight"); print(f"saved {out2}")

# ============================== printed summary =============================
print(f"\n=== CALIBRATION SUMMARY (calibrated channel: {CALIB_CH}) ===")
print(f"lift-test injected: {LIFT_TEST_ROI} ± {LIFT_TEST_SE}   (true: {true_roi[CALIB_CH]:.2f})")
print(f"\n{'channel':<9s} {'true':>7s} {'naive':>7s} {'adj':>7s} {'CALIB':>7s}  {'CI_calib':>16s}")
for c, ch in enumerate(channels):
    print(f"{ch:<9s} {true_roi[ch]:>7.2f} {mn[c]:>7.2f} {ma[c]:>7.2f} "
          f"{mc[c]:>7.2f}  [{lc[c]:>5.2f}, {hc[c]:>5.2f}]")
