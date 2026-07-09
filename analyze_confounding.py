"""
Refined analysis of the confounding experiment, accounting for the
collinearity-induced bias-smearing across correlated channels.

Produces three exhibits:

  1. outputs/figures/confounding_identified_only.png
     Bar chart restricted to TV, search, social (the identified channels).
     The clean per-channel story.

  2. outputs/figures/confounding_aggregate.png
     Aggregate metrics: total naive vs total adjusted media contribution
     as a share of revenue, against truth. Robust to per-channel noise
     because summing averages it out.

  3. outputs/figures/confounding_channel_correlation.png
     The confounded dataset's channel-vs-channel spend correlation matrix.
     Visualizes WHY the per-channel bias story doesn't decompose cleanly:
     the channels are correlated with each other (through their shared
     response to demand), which makes the regression's attribution
     ambiguous. This is the "collinearity smears confounding bias" story.

No sampling. Run:  python analyze_confounding.py
"""
import json, os
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "outputs", "figures")

with open(os.path.join(HERE, "data", "true_params.json")) as f:
    truth = json.load(f)
true_roi = truth["variants"]["confounded"]["true_roi"]

df = pd.read_csv(os.path.join(HERE, "data", "synthetic_confounded.csv"),
                 parse_dates=["date"])
spend_cols = [c for c in df.columns if c.endswith("_spend")]
channels = [c.replace("_spend", "") for c in spend_cols]

def load(name):
    p = os.path.join(HERE, "outputs", name)
    try:
        return xr.open_dataarray(p)
    except Exception:
        ds = xr.open_dataset(p); return ds[list(ds.data_vars)[0]]

roi_naive = load("confounded_roi_naive.nc")
roi_adj = load("confounded_roi_adjusted.nc")

# ============================== figure 1: identified channels only ==========
IDENTIFIED = ["tv", "search", "social"]
idx = [channels.index(c) for c in IDENTIFIED]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(IDENTIFIED)); w = 0.26
truths = [true_roi[c] for c in IDENTIFIED]
n_med = [float(np.median(roi_naive.isel(channel=i).values.ravel())) for i in idx]
a_med = [float(np.median(roi_adj.isel(channel=i).values.ravel())) for i in idx]
n_lo, n_hi, a_lo, a_hi = [], [], [], []
for i in idx:
    sn = roi_naive.isel(channel=i).values.ravel()
    sa = roi_adj.isel(channel=i).values.ravel()
    n_lo.append(np.percentile(sn, 2.5)); n_hi.append(np.percentile(sn, 97.5))
    a_lo.append(np.percentile(sa, 2.5)); a_hi.append(np.percentile(sa, 97.5))

ax.bar(x - w, truths, w, color="#9aa0a6", label="true ROI")
ax.bar(x, n_med, w, color="#C15B5B", label="naive (backdoor OPEN)")
ax.bar(x + w, a_med, w, color="#185FA5", label="adjusted (backdoor CLOSED)")
ax.errorbar(x, n_med, yerr=[np.array(n_med)-np.array(n_lo), np.array(n_hi)-np.array(n_med)],
            fmt="none", ecolor="#7a2a2a", capsize=3)
ax.errorbar(x + w, a_med, yerr=[np.array(a_med)-np.array(a_lo), np.array(a_hi)-np.array(a_med)],
            fmt="none", ecolor="#0d3a66", capsize=3)
ax.set_xticks(x); ax.set_xticklabels(IDENTIFIED)
ax.set_ylabel("ROI")
ax.set_title("Confounding bias — identified channels only\n"
             "(display/email omitted: unidentifiable regardless of adjustment)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
out1 = os.path.join(FIG, "confounding_identified_only.png")
fig.savefig(out1, dpi=130, bbox_inches="tight"); print(f"saved {out1}")

# ============================== figure 2: aggregate exhibit ==================
# For each posterior draw, sum media contribution across ALL channels.
# Compare naive vs adjusted total contribution against the true total.
# This is robust to per-channel noise because it averages out.

# True total media contribution (in the confounded world) - we can compute
# it from the DGP: for each channel, true contribution = true_roi * total_spend.
total_spend_per_ch = df[spend_cols].sum().values
true_total_contrib = sum(true_roi[c] * total_spend_per_ch[i]
                          for i, c in enumerate(channels))

# From posterior: total contribution per draw
def total_contrib_samples(roi_da):
    # roi = contrib/spend, so contrib = roi * spend
    contrib_per_ch = roi_da * total_spend_per_ch   # broadcasting: (chain,draw,channel) * (channel,)
    return contrib_per_ch.sum(dim="channel").values.ravel()

naive_total = total_contrib_samples(roi_naive)
adj_total = total_contrib_samples(roi_adj)

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(naive_total / 1e3, bins=40, alpha=0.55, color="#C15B5B",
        label="naive (backdoor OPEN)", density=True)
ax.hist(adj_total / 1e3, bins=40, alpha=0.55, color="#185FA5",
        label="adjusted (backdoor CLOSED)", density=True)
ax.axvline(true_total_contrib / 1e3, color="#333", lw=2.2, ls="--", label="true total")
ax.set_xlabel("Total media contribution ($M, over 3 years)")
ax.set_ylabel("posterior density")
ax.set_title("Aggregate media contribution: naive overshoots, adjustment recovers")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
out2 = os.path.join(FIG, "confounding_aggregate.png")
fig.savefig(out2, dpi=130, bbox_inches="tight"); print(f"saved {out2}")

# quantitative summary
print("\n=== AGGREGATE MEDIA CONTRIBUTION (over 3 years) ===")
print(f"true total (from DGP):    ${true_total_contrib:>10,.0f}")
print(f"naive posterior median:   ${np.median(naive_total):>10,.0f}  "
      f"({(np.median(naive_total)/true_total_contrib - 1)*100:+.1f}% vs truth)")
print(f"adjusted posterior median:${np.median(adj_total):>10,.0f}  "
      f"({(np.median(adj_total)/true_total_contrib - 1)*100:+.1f}% vs truth)")

# ============================== figure 3: channel collinearity ===============
corr = df[spend_cols].corr()
labels = [c.replace("_spend", "") for c in spend_cols]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if val < 0.6 else "black", fontsize=8)
ax.set_title("Channel spend correlations (confounded data)\n"
             "Correlated channels share the confounder's effect")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
out3 = os.path.join(FIG, "confounding_channel_correlation.png")
fig.savefig(out3, dpi=130, bbox_inches="tight"); print(f"saved {out3}")

# ============================== short table ==================================
print("\n=== IDENTIFIED CHANNELS: bias summary ===")
print(f"{'channel':<9s} {'true':>7s} {'naive':>8s} {'nbias':>7s} {'adjust':>8s} {'abias':>7s}")
for c, i in zip(IDENTIFIED, idx):
    sn = roi_naive.isel(channel=i).values.ravel()
    sa = roi_adj.isel(channel=i).values.ravel()
    mn, ma = np.median(sn), np.median(sa)
    print(f"{c:<9s} {true_roi[c]:>7.2f} {mn:>8.2f} {mn-true_roi[c]:>+7.2f} "
          f"{ma:>8.2f} {ma-true_roi[c]:>+7.2f}")
