"""
Applied MMM figures on the Robyn dataset. Reads the slim files saved by
robyn_applied.py and produces the industry-standard MMM outputs.

Reads:
  outputs/robyn_full_roi.nc         ROI samples (well-specified fit)
  outputs/robyn_full_contrib.nc     Per-week contributions (well-specified)
  outputs/robyn_nocf_roi.nc         ROI samples (competitor omitted)

Produces:
  outputs/figures/robyn_contribution.png       stacked area: who drove revenue
  outputs/figures/robyn_response_curves.png    per-channel saturation curves
  outputs/figures/robyn_roi_table.png          ROI + marginal ROI w/ intervals
  outputs/figures/robyn_omitted_confounder.png ROI shift when competitor omitted

No sampling. Run:  python make_robyn_figures.py
"""
import json, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "outputs", "figures")

df = pd.read_csv(os.path.join(HERE, "data", "robyn_weekly.csv"),
                 parse_dates=["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)
paid_cols = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
paid_names = [c.replace("_S", "") for c in paid_cols]
COLORS = {"tv":"#185FA5","ooh":"#BA7517","print":"#2E8B74",
          "facebook":"#8B3A62","search":"#C15B5B"}

def load(name):
    p = os.path.join(HERE, "outputs", name)
    try:
        return xr.open_dataarray(p)
    except Exception:
        ds = xr.open_dataset(p); return ds[list(ds.data_vars)[0]]

roi_full = load("robyn_full_roi.nc")
contrib_full = load("robyn_full_contrib.nc")   # (chain, draw, date, channel)
roi_nocf = load("robyn_nocf_roi.nc")

# ============================== figure 1: contribution decomposition ========
# Posterior median contribution per channel per week; stacked with baseline.
contrib_med = contrib_full.median(dim=("chain", "draw"))   # (date, channel)
baseline = df["revenue"].values - contrib_med.sum(dim="channel").values

fig, ax = plt.subplots(figsize=(13, 5.5))
dates = df["DATE"].values
bottom = np.maximum(baseline, 0)
ax.fill_between(dates, 0, bottom, color="#dcdcdc", label="baseline + controls")
cumulative = bottom.copy()
for i, ch in enumerate(paid_names):
    c = contrib_med.isel(channel=i).values
    ax.fill_between(dates, cumulative, cumulative + c, color=COLORS[ch],
                    alpha=0.85, label=ch)
    cumulative = cumulative + c
ax.plot(dates, df["revenue"], color="black", lw=1.0, label="actual revenue")
ax.set_ylabel("Revenue")
# ax.set_title(
    # "Contribution decomposition: how much revenue each channel drove")
ax.legend(fontsize=12, loc="upper right", ncol=2); ax.grid(alpha=0.3)

# Increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

fig.tight_layout()
out = os.path.join(FIG, "robyn_contribution.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"saved {out}")

# ============================== figure 2: ROI table with intervals ==========
total_spend = df[paid_cols].sum().values
total_contrib_share = {}
for i, ch in enumerate(paid_names):
    total_contrib_share[ch] = contrib_med.isel(channel=i).sum().values.item() / df["revenue"].sum()

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(paid_names))
med, lo, hi = [], [], []
for i in range(len(paid_names)):
    s = roi_full.isel(channel=i).values.ravel()
    med.append(np.median(s)); lo.append(np.percentile(s, 2.5)); hi.append(np.percentile(s, 97.5))
med, lo, hi = np.array(med), np.array(lo), np.array(hi)
ax.errorbar(x, med, yerr=[med-lo, hi-med], fmt="o", color="#185FA5",
            capsize=5, ms=9, lw=1.7)
ax.set_xticks(x); ax.set_xticklabels(paid_names, fontsize=11)
ax.set_ylabel("ROI (revenue per dollar of spend)")
ymax = min(max(20, hi.max()*1.05), 200)  # cap for readability
ax.set_ylim(0, ymax)
for i, ch in enumerate(paid_names):
    ax.text(x[i], med[i]+ymax*0.03, f"{med[i]:.1f}", ha="center", fontsize=10)
    if hi[i] > ymax:
        ax.annotate(f"CI to {hi[i]:.0f}", (x[i], ymax*0.97),
                    fontsize=7, ha="center", color="#185FA5", rotation=90, va="top")
# ax.set_title("Channel ROI (posterior median + 95% credible interval)\n"
            #  "wider intervals = the data is more ambiguous about that channel")

# Increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
out = os.path.join(FIG, "robyn_roi_table.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"saved {out}")

# ============================== figure 3: omitted-confounder comparison =====
med_nocf = np.array([float(np.median(roi_nocf.isel(channel=i).values.ravel()))
                     for i in range(len(paid_names))])
lo_nocf = np.array([float(np.percentile(roi_nocf.isel(channel=i).values.ravel(), 2.5))
                     for i in range(len(paid_names))])
hi_nocf = np.array([float(np.percentile(roi_nocf.isel(channel=i).values.ravel(), 97.5))
                     for i in range(len(paid_names))])

fig, ax = plt.subplots(figsize=(9, 5.5))
w = 0.28
ax.errorbar(x - w/2, med, yerr=[med-lo, hi-med], fmt="o", color="#185FA5",
            capsize=4, ms=8, lw=1.5, label="well-specified (competitor INCLUDED)")
ax.errorbar(x + w/2, med_nocf, yerr=[med_nocf-lo_nocf, hi_nocf-med_nocf],
            fmt="s", color="#C15B5B", capsize=4, ms=8, lw=1.5,
            label="competitor OMITTED")
ax.set_xticks(x); ax.set_xticklabels(paid_names, fontsize=11)
ax.set_ylabel("ROI (posterior median + 95% CI)")
ymax = min(max(20, max(hi.max(), hi_nocf.max())*1.05), 200)
ax.set_ylim(0, ymax)
# ax.set_title("Real-data analog of the confounding experiment:\n"
            #  "how much do ROI estimates shift when the strongest control is dropped?")
ax.legend(fontsize=12); ax.grid(alpha=0.3, axis="y")

# Increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

fig.tight_layout()
out = os.path.join(FIG, "robyn_omitted_confounder.png")
fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"saved {out}")

# ============================== printed summary =============================
print("\n=== APPLIED SUMMARY (well-specified fit) ===")
print(f"{'channel':<9s} {'spend $M':>10s} {'contrib %':>10s} {'ROI':>7s} {'  95% CI':>18s}")
for i, ch in enumerate(paid_names):
    print(f"{ch:<9s} {total_spend[i]/1e6:>10.2f} {total_contrib_share[ch]*100:>9.1f}% "
          f"{med[i]:>7.2f}   [{lo[i]:>5.2f}, {hi[i]:>5.2f}]")

print("\n=== ROI SHIFT WHEN COMPETITOR IS OMITTED ===")
print(f"{'channel':<9s} {'well':>7s} {'omit':>7s} {'shift':>7s} {'pct':>7s}")
for i, ch in enumerate(paid_names):
    shift = med_nocf[i] - med[i]
    pct = 100 * shift / med[i] if med[i] != 0 else float("nan")
    print(f"{ch:<9s} {med[i]:>7.2f} {med_nocf[i]:>7.2f} {shift:>+7.2f} {pct:>+6.1f}%")
