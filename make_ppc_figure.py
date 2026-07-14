"""
Posterior predictive check (PPC) for Model 2.

Recovery asked: did we get the PARAMETERS right?
PPC asks:       can the model REPRODUCE the observed revenue?

These are different questions. A model can cover the true parameters yet
predict poorly (or vice-versa). The PPC overlays the model's predicted
revenue - with credible bands - on the actual revenue series, and reports
what fraction of weeks fall inside the 90% band (should be ~90% if the
model's uncertainty is calibrated).

This script rebuilds the model and draws posterior-predictive samples. The
slim saved files don't contain predictions (we dropped them to keep the file
small), so we re-fit. To stay fast, it uses fewer draws than the main run -
the PPC doesn't need the full sample to show fit quality.

Run:  python make_ppc_figure.py    (~3-5 min; it re-samples a short chain)
"""
import json, os, warnings
import arviz as az, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_marketing.prior import Prior

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================ load data ====
df = pd.read_csv(os.path.join(HERE, "data", "synthetic_clean.csv"), parse_dates=["date"])
spend_cols = [c for c in df.columns if c.endswith("_spend")]
df["t"] = np.arange(len(df))

model_config = {
    "adstock_alpha":    Prior("Beta", alpha=1, beta=3, dims="channel"),
    "saturation_slope": Prior("Gamma", alpha=3, beta=1, dims="channel"),
    "saturation_kappa": Prior("Beta", alpha=2, beta=2, dims="channel"),
    "saturation_beta":  Prior("HalfNormal", sigma=1.0, dims="channel"),
    "gamma_control":    Prior("Normal", mu=0, sigma=2, dims="control"),
    "gamma_fourier":    Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
    "intercept":        Prior("Normal", mu=0, sigma=2, dims=()),
    "likelihood":       Prior("Normal", sigma=Prior("HalfNormal", sigma=2, dims=()), dims="date"),
}

mmm = MMM(
    date_column="date", channel_columns=spend_cols, control_columns=["t"],
    adstock=GeometricAdstock(l_max=12), saturation=HillSaturation(),
    yearly_seasonality=3, model_config=model_config,
)
X = df[["date", "t"] + spend_cols]
y = df["revenue"]

# short re-fit (PPC doesn't need the full sample)
print("re-fitting a short chain for the PPC (~3-5 min)...")
idata = mmm.fit(X=X, y=y, draws=500, tune=500, chains=2, cores=2,
                target_accept=0.95, max_treedepth=12,
                random_seed=42, progressbar=False)

# posterior predictive
print("sampling posterior predictive...")
mmm.sample_posterior_predictive(X, extend_idata=True, combined=True)

# ============================================== extract predicted revenue ==
# original-scale predictions
pp = mmm.idata.posterior_predictive["y"]   # dims include sample, date
# collapse sample dims -> quantiles over draws per week
pp_vals = pp.values.reshape(-1, pp.shape[-1])   # (draws, weeks)
lo = np.percentile(pp_vals, 5, axis=0)
hi = np.percentile(pp_vals, 95, axis=0)
med = np.percentile(pp_vals, 50, axis=0)

actual = df["revenue"].values
inside = np.mean((actual >= lo) & (actual <= hi)) * 100

# ============================================== figure =====================
fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(df["date"], lo, hi, color="#185FA5", alpha=0.25,
                label="90% predictive band")
ax.plot(df["date"], med, color="#185FA5", lw=1.5, label="predicted median")
ax.plot(df["date"], actual, color="#BA7517", lw=1.3, label="actual revenue")
# ax.set_title(f"Posterior predictive check: {inside:.0f}% of weeks inside the 90% band "
            #  f"(target ~90%)", fontsize=12)
ax.set_ylabel("Revenue ($000)", fontsize=12); ax.set_xlabel("Date", fontsize=12)


# Increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ax.legend(fontsize=12); ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG_DIR, "posterior_predictive_check.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")

# ============================================== scatter: predicted vs actual
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(actual, med, s=25, color="#185FA5", alpha=0.6)
lims = [min(actual.min(), med.min()), max(actual.max(), med.max())]
ax.plot(lims, lims, color="#BA7517", lw=1.5, ls="--", label="perfect fit")
ax.set_xlabel("Actual revenue", fontsize=12); ax.set_ylabel("Predicted revenue (median)", fontsize=12)
r2 = 1 - np.sum((actual - med)**2) / np.sum((actual - actual.mean())**2)
# ax.set_title(f"Predicted vs actual (R^2 = {r2:.3f})", fontsize=12)

# Increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ax.legend(fontsize=12); ax.grid(alpha=0.3)
fig.tight_layout()
out2 = os.path.join(FIG_DIR, "posterior_predictive_scatter.png")
fig.savefig(out2, dpi=130, bbox_inches="tight")
print(f"saved {out2}")

print(f"\ncoverage: {inside:.1f}% of weeks inside 90% band")
print(f"predictive R^2: {r2:.3f}")
