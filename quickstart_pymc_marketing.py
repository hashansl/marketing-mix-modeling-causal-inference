"""
PyMC-Marketing quickstart -- Model 2 (Bayesian MMM)

See Methods §2.3.3 of the paper.

Fits a Bayesian Marketing Mix Model on the clean synthetic dataset. Unlike
Model 1 (oracle OLS), this model estimates the adstock and saturation
parameters jointly with the channel coefficients, and reports full posterior
uncertainty for every quantity.

MODEL SPECIFICATION
-------------------
For channel c and week t, the model is:

    Likelihood:
        y_t ~ Normal(mu_t, sigma^2)

    Mean structure:
        mu_t = beta_0                                     (intercept)
             + tau * t                                    (linear trend)
             + sum_{k=1..K} [gamma_k sin(2*pi*k*t / P) + delta_k cos(2*pi*k*t / P)]
                                                          (Fourier seasonality, K=3, P=52.13)
             + sum_{c=1..C} beta_c * s_{c,t}              (media contributions)

    Adstock (geometric, causal weighted convolution):
        a_{c,t} = sum_{l=0..L} theta_c^l * x_{c,t-l}  /  sum_{l=0..L} theta_c^l

    Saturation (Hill):
        s_{c,t} = a_{c,t}^alpha_c  /  (kappa_c^alpha_c + a_{c,t}^alpha_c)

    Priors (specified on the internal [0,1] MaxAbsScaler scale):
        theta_c ~ Beta(1, 3)                              (adstock retention)
        alpha_c ~ Gamma(3, 1)                             (Hill shape)
        kappa_c ~ Beta(2, 2)                              (half-saturation)
        beta_c  ~ HalfNormal(1)                           (channel coefficient)
        beta_0  ~ Normal(0, 2)                            (intercept)
        tau     ~ Normal(0, 2)                            (control coefficients)
        gamma_k, delta_k ~ Laplace(0, 1)                  (Fourier coefficients)
        sigma   ~ HalfNormal(2)                           (observation noise)


Fitting is done via NUTS (No-U-Turn Sampler) with 1000 tuning + 1000 draws
per chain, 4 chains, target_accept=0.98, max_treedepth=13. Elevated
target_accept is required because Hill parameters (alpha, kappa) trade off
along a curved posterior ridge, a well-documented identification difficulty
in media mix models.

ROI is reported as the posterior MEDIAN with 2.5%-97.5% credible interval.
Median (not mean) is used because ratio quantities like ROI have right-skewed
posteriors; the mean would be pulled up by tail samples and misrepresent the
typical value.


OUTPUTS SAVED (both small, safe to commit)
------------------------------------------
  outputs/model2_posterior.nc   parameter posteriors only (a few MB)
  outputs/model2_roi.nc         per-channel ROI samples (tiny)
  outputs/figures/divergences_tv_adstock_saturation.png
  outputs/figures/fitted_vs_true_roi.png

RUNTIME
-------
Roughly 5-12 minutes on a laptop with 4 CPU cores. Set SMOKE_TEST=True for a
quick sanity check that runs in ~2 minutes.

Run:  python quickstart_pymc_marketing.py
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
    draws, tune, chains = 1000, 2000, 4
target_accept = 0.99
max_treedepth = 13

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

# --- PAIR PLOT CODE ---
if n_diverging > 0:
    print("\nDivergences detected! Plotting adstock vs saturation for TV...")
    az.plot_pair(
        idata,
        var_names=["adstock_alpha", "saturation_kappa"],
        coords={"channel": ["tv_spend"]}, 
        kind="scatter",
        divergences=True,
        scatter_kwargs={"alpha": 0.1}
    )
    plt.title("TV: Adstock vs Saturation (Divergences Highlighted)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "divergences_tv_adstock_saturation.png"), dpi=130)
# --------------------------

# ============================================ save slim posterior (few MB) ==
# NOTE: idata.to_netcdf() would save ~100MB because it includes the
# posterior-predictive and per-observation arrays. We keep only the
# posterior group (the parameter samples), which is all the figures need.
az.InferenceData(posterior=idata.posterior).to_netcdf(
    os.path.join(HERE, "outputs", "model2_posterior.nc"))

# ================================================ recovered ROI vs truth ==
contrib = mmm.compute_channel_contribution_original_scale()
total_contrib = contrib.sum(dim="date")
total_spend = df[spend_cols].sum().values

# save per-channel ROI samples (draws x channel - tiny) for the figures
roi_da = total_contrib / total_spend
roi_da.to_netcdf(os.path.join(HERE, "outputs", "model2_roi.nc"))
print("saved outputs/model2_posterior.nc and outputs/model2_roi.nc")

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
ax.set_xticklabels(channels, fontsize=14)           # <-- Kept fontsize here
ax.set_ylabel("ROI", fontsize=18)
# ax.set_title("PyMC-Marketing v2 (tightened priors): fitted vs true ROI")

# Increase the font size of the actual numbers on the x and y axes
ax.tick_params(axis="both", which="major", labelsize=14)

ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
out = os.path.join(FIG_DIR, "fitted_vs_true_roi.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved {out}")