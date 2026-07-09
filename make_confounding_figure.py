"""
Confounding bias figure - reads the slim outputs from confounding_experiment.py
and builds the bias table + bar chart.

  outputs/confounded_roi_naive.nc      (backdoor open)
  outputs/confounded_roi_adjusted.nc   (backdoor closed)

Produces:
  outputs/figures/confounding_bias.png       three bars/channel: true, naive, adjusted
  outputs/figures/confounding_bias_vs_beta.png  bias magnitude vs demand_beta

No sampling. Run:  python make_confounding_figure.py
"""
import json, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(HERE, "data", "true_params.json")) as f:
    truth = json.load(f)
true_roi = truth["variants"]["confounded"]["true_roi"]
params = truth["variants"]["confounded"]["channel_params"]

df = pd.read_csv(os.path.join(HERE, "data", "synthetic_confounded.csv"), nrows=1)
spend_cols = [c for c in df.columns if c.endswith("_spend")]
channels = [c.replace("_spend", "") for c in spend_cols]

def load(path):
    p = os.path.join(HERE, "outputs", path)
    try:
        return xr.open_dataarray(p)
    except Exception:
        ds = xr.open_dataset(p); return ds[list(ds.data_vars)[0]]

roi_naive = load("confounded_roi_naive.nc")
roi_adj = load("confounded_roi_adjusted.nc")

# gather medians + CIs
rows = []
for i, ch in enumerate(channels):
    sn = roi_naive.isel(channel=i).values.ravel()
    sa = roi_adj.isel(channel=i).values.ravel()
    rows.append(dict(
        channel=ch, true=true_roi[ch],
        naive_med=np.median(sn), naive_lo=np.percentile(sn, 2.5), naive_hi=np.percentile(sn, 97.5),
        adj_med=np.median(sa), adj_lo=np.percentile(sa, 2.5), adj_hi=np.percentile(sa, 97.5),
        demand_beta=params[ch]["demand_beta"],
    ))
T = pd.DataFrame(rows)

# ============================================== figure 1: grouped bars ======
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(channels)); w = 0.26
ax.bar(x - w, T["true"], width=w, color="#9aa0a6", label="true ROI")
ax.bar(x, T["naive_med"], width=w, color="#C15B5B",
       label="naive (backdoor OPEN)")
ax.bar(x + w, T["adj_med"], width=w, color="#185FA5",
       label="adjusted (backdoor CLOSED)")
# CI whiskers on the two fitted bars
ax.errorbar(x, T["naive_med"], yerr=[T["naive_med"]-T["naive_lo"], T["naive_hi"]-T["naive_med"]],
            fmt="none", ecolor="#7a2a2a", capsize=3, lw=1)
ax.errorbar(x + w, T["adj_med"], yerr=[T["adj_med"]-T["adj_lo"], T["adj_hi"]-T["adj_med"]],
            fmt="none", ecolor="#0d3a66", capsize=3, lw=1)
ax.set_xticks(x); ax.set_xticklabels(channels, fontsize=11)
ax.set_ylabel("ROI"); ax.set_title(
    "Confounding bias: naive (open backdoor) overshoots true ROI; adjustment recovers it")
ymax = max(6, T["true"].max() * 2.2)
ax.set_ylim(0, ymax)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
for c, ch in enumerate(channels):
    if T["naive_hi"].iloc[c] > ymax:
        ax.annotate(f"to {T['naive_hi'].iloc[c]:.0f}", (x[c], ymax*0.96),
                    fontsize=7, ha="center", color="#7a2a2a", rotation=90, va="top")
fig.tight_layout()
out1 = os.path.join(FIG, "confounding_bias.png")
fig.savefig(out1, dpi=130, bbox_inches="tight"); print(f"saved {out1}")

# ============================================== figure 2: bias vs demand_beta
fig, ax = plt.subplots(figsize=(6.5, 5))
naive_bias = T["naive_med"] - T["true"]
ax.scatter(T["demand_beta"], naive_bias, s=90, color="#C15B5B", zorder=3)
for _, r in T.iterrows():
    ax.annotate(r["channel"], (r["demand_beta"], r["naive_med"]-r["true"]),
                fontsize=9, xytext=(5, 5), textcoords="offset points")
ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xlabel("demand_beta  (strength of spend-demand confounding link)")
ax.set_ylabel("naive ROI bias  (naive median - true)")
ax.set_title("Confounding bias grows with the strength of the backdoor path")
ax.grid(alpha=0.3)
fig.tight_layout()
out2 = os.path.join(FIG, "confounding_bias_vs_beta.png")
fig.savefig(out2, dpi=130, bbox_inches="tight"); print(f"saved {out2}")

# ============================================== bias table ==================
print("\n" + "=" * 72)
print("CONFOUNDING BIAS TABLE")
print("=" * 72)
print(f"{'channel':<9s} {'true':>7s} {'naive':>8s} {'nbias':>7s} {'adjust':>8s} {'abias':>7s} {'demand_b':>9s}")
for _, r in T.iterrows():
    print(f"{r['channel']:<9s} {r['true']:>7.2f} {r['naive_med']:>8.2f} "
          f"{r['naive_med']-r['true']:>+7.2f} {r['adj_med']:>8.2f} "
          f"{r['adj_med']-r['true']:>+7.2f} {r['demand_beta']:>9.2f}")
