"""
Quick visual sanity check of the synthetic data.
Run:  python data/eda_synthetic.py
Saves a figure to outputs/figures/synthetic_overview.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

clean = pd.read_csv(os.path.join(HERE, "synthetic_clean.csv"), parse_dates=["date"])
conf = pd.read_csv(os.path.join(HERE, "synthetic_confounded.csv"), parse_dates=["date"])
spend_cols = [c for c in clean.columns if c.endswith("_spend")]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# 1. Revenue, both variants
ax = axes[0, 0]
ax.plot(clean.date, clean.revenue, lw=1.2, label="clean")
ax.plot(conf.date, conf.revenue, lw=1.2, alpha=0.8, label="confounded")
ax.set_title("Weekly revenue ($000)")
ax.legend(fontsize=8)

# 2. Spend series (clean)
ax = axes[0, 1]
for c in spend_cols:
    ax.plot(clean.date, clean[c], lw=1.0, label=c.replace("_spend", ""))
ax.set_title("Channel spend, clean ($000/week)")
ax.legend(fontsize=8, ncol=2)

# 3. Confounding: search spend vs hidden demand
ax = axes[1, 0]
ax.scatter(conf["demand"], conf["search_spend"], s=12, alpha=0.6)
r = np.corrcoef(conf["demand"], conf["search_spend"])[0, 1]
ax.set_xlabel("hidden demand (z)")
ax.set_ylabel("search spend")
ax.set_title(f"Confounding: spend rises with demand (r={r:+.2f})")

# 4. Channel spend correlation heatmap (confounded)
ax = axes[1, 1]
corr = conf[spend_cols].corr()
im = ax.imshow(corr, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(spend_cols)))
ax.set_yticks(range(len(spend_cols)))
labels = [c.replace("_spend", "") for c in spend_cols]
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.set_title("Channel spend correlation (confounded)")
for i in range(len(spend_cols)):
    for j in range(len(spend_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                color="white" if corr.iloc[i, j] < 0.6 else "black", fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046)

fig.tight_layout()
out = os.path.join(FIG_DIR, "synthetic_overview.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
