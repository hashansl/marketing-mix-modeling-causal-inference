"""
PyMC-Marketing quickstart, v2 (fixed priors + sampler settings).

WHAT'S DIFFERENT FROM v1
------------------------
1. The library uses MaxAbsScaler internally: spend and revenue live in [0, 1].
   Priors are set to be sensible ON THAT SCALE, not the raw scale.

2. saturation_kappa gets a Beta(2, 2) prior instead of HalfNormal(1.5).
   Reason: kappa (half-saturation) on scaled data must be in [0, 1] AND real
   MMMs almost always operate on the curved part of the response (spend not
   deeply saturated, not barely started). Beta(2,2) is bounded, centered on
   0.5, with moderate spread - the right shape given what we know.
   HalfNormal(1.5) was letting the sampler wander into implausible regions.

3. saturation_slope (the Hill shape / alpha parameter) gets Gamma(3, 1):
   mode near 2, tail out to ~5. Real MMM slopes are usually 1-4.

4. saturation_beta stays HalfNormal but tighter (sigma=1 instead of 1.5).
   The channel effect on scaled revenue is bounded above by 1, so we don't
   need mass out at 5+.

5. Sampler: target_accept=0.98 (up from 0.95) and max_treedepth=12 (up from
   10). These help NUTS navigate the Hill-saturation ridge.

6. Report POSTERIOR MEDIAN, not mean. When the posterior has a long right
   tail (as it can with ratio quantities like ROI), the median is a much
   more honest point estimate.

EXPECTED BEHAVIOR
-----------------
Compared to v1, v2 should show:
  - Substantially fewer divergences (target: 0)
  - R-hat < 1.05 across all parameters (target: < 1.01)
  - Narrower 95% ROI intervals - the wildly-wide intervals of v1 came from
    posterior spread into implausible regions that these priors close off
  - ROI point estimates in the same ballpark as truth, with intervals that
    contain the true value

RUNTIME
-------
Roughly 5-10 minutes on a laptop with 4 CPU cores. Higher target_accept
means each draw needs more evaluations. Set SMOKE_TEST=True for a quick
sanity check that runs in ~2 minutes.

Run:  python quickstart_pymc_marketing_v2.py
"""
import json, os, warnings
import arviz as az, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_marketing.prior import Prior

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SMOKE_TEST = False   # True -> 250 draws x 2 chains for a fast smoke test

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
df["t"] = np.arange(len(df))
print(f"data: {df.shape}   channels: {channels}")

# ==================================================== priors on scaled data =
# Everything below is in the [0, 1] internal scale used by the library.
model_config = {
    # Adstock retention - Beta(1,3) is the library default, sensible.
    "adstock_alpha":    Prior("Beta", alpha=1, beta=3, dims="channel"),

    # Hill shape (alpha in our math): Gamma(3, 1) puts mode near 2.
    "saturation_slope": Prior("Gamma", alpha=3, beta=1, dims="channel"),

    # Half-saturation kappa: MUST be in [0, 1] on scaled data. Beta(2, 2)
    # is symmetric around 0.5 with moderate spread. Encodes "we're most
    # likely operating on the curved part of the response".
    "saturation_kappa": Prior("Beta", alpha=2, beta=2, dims="channel"),

    # Channel coefficient on scaled response. Bounded above at 1 in principle.
    "saturation_beta":  Prior("HalfNormal", sigma=1.0, dims="channel"),

    "gamma_control":    Prior("Normal", mu=0, sigma=2, dims="control"),
    "gamma_fourier":    Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
    "intercept":        Prior("Normal", mu=0, sigma=2, dims=()),
    "likelihood":       Prior("Normal",
                              sigma=Prior("HalfNormal", sigma=2, dims=()),
                              dims="date"),
}

mmm = MMM(
    date_column="date",
    channel_columns=spend_cols,
    control_columns=["t"],
    adstock=GeometricAdstock(l_max=12),
    saturation=HillSaturation(),
    yearly_seasonality=3,
    model_config=model_config,
)

X = df[["date", "t"] + spend_cols]
y = df["revenue"]

# ================================================================ fit ====
if SMOKE_TEST:
    draws, tune, chains = 250, 250, 2
else:
    draws, tune, chains = 1000, 1000, 4
target_accept = 0.98
max_treedepth = 12

print(f"sampling: draws={draws}, tune={tune}, chains={chains}")
print(f"          target_accept={target_accept}, max_treedepth={max_treedepth}")

idata = mmm.fit(
    X=X, y=y,
    draws=draws, tune=tune, chains=chains, cores=min(chains, 4),
    target_accept=target_accept, max_treedepth=max_treedepth,
    random_seed=42, progressbar=True,
)

# ============================================================ diagnostics ===
summary = az.summary(
    idata,
    var_names=["intercept", "adstock_alpha", "saturation_slope",
               "saturation_kappa", "saturation_beta", "y_sigma"],
    round_to=3,
)
print("\n--- posterior summary ---")
print(summary[["mean", "sd", "r_hat", "ess_bulk"]])
max_rhat = float(summary["r_hat"].max())
n_diverging = int(idata.sample_stats["diverging"].sum())
print(f"\nmax R-hat:   {max_rhat:.3f}   (want < 1.05, ideally < 1.01)")
print(f"divergences: {n_diverging}   (want 0)")

# ================================================ recovered ROI vs truth ==
contrib = mmm.compute_channel_contribution_original_scale()
total_contrib = contrib.sum(dim="date")
total_spend = df[spend_cols].sum().values

print("\n--- ROI: fitted vs true (clean synthetic) ---")
print(f"{'channel':<10s} {'true':>7s}   {'median':>7s}   {'95% CI':>18s}")
med, los, his, truths = [], [], [], []
for i, ch in enumerate(channels):
    samples = total_contrib.isel(channel=i).values.ravel() / total_spend[i]
    m = float(np.median(samples))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    covered = "[COVERED]" if lo <= true_roi[ch] <= hi else "[MISSED]"
    print(f"{ch:<10s} {true_roi[ch]:>7.2f}   {m:>7.2f}   "
          f"[{lo:>6.2f}, {hi:>6.2f}]  {covered}")
    med.append(m); los.append(lo); his.append(hi); truths.append(true_roi[ch])

# ================================================ recovery of adstock params
print("\n--- Adstock retention (theta) recovery ---")
truth_theta = {"tv": 0.65, "search": 0.15, "social": 0.35, "display": 0.45, "email": 0.20}
posterior = idata.posterior["adstock_alpha"]
print(f"{'channel':<10s} {'true':>7s}   {'median':>7s}   {'95% CI':>18s}")
for i, ch in enumerate(channels):
    samples = posterior.isel(channel=i).values.ravel()
    m = float(np.median(samples))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    covered = "[COVERED]" if lo <= truth_theta[ch] <= hi else "[MISSED]"
    print(f"{ch:<10s} {truth_theta[ch]:>7.2f}   {m:>7.2f}   "
          f"[{lo:>6.2f}, {hi:>6.2f}]  {covered}")

# ============================================================ figure =======
fig, ax = plt.subplots(figsize=(9, 4.8))
x_pos = np.arange(len(channels))
ax.errorbar(x_pos, med,
            yerr=[np.array(med) - np.array(los), np.array(his) - np.array(med)],
            fmt="o", color="#185FA5", capsize=4, lw=1.5, ms=7,
            label="fitted (95% CI)")
ax.scatter(x_pos, truths, marker="D", s=80, color="#BA7517", zorder=5,
           label="true ROI")
ax.set_xticks(x_pos); ax.set_xticklabels(channels)
ax.set_ylabel("ROI")
ax.set_title(f"PyMC-Marketing v2 (tightened priors): fitted vs true ROI")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
out = os.path.join(FIG_DIR, "pymc_marketing_quickstart_v2.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved {out}")
