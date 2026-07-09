"""
LIFT-TEST CALIBRATION - Model 3, the experimental-anchor experiment.

Setup: fit the Bayesian MMM on the CONFOUNDED data WITHOUT the demand
control (backdoor open, worst-case observational analysis), but replace
search's coefficient prior with a tight, informative prior derived from a
simulated geo lift test.

Question: can experimental information injected via one channel's prior
rescue observational identification when the backdoor is open?

Simulated lift test result:
  ROI estimate = 3.9 (unbiased for true 3.83 - real experiments have noise)
  Standard error = 0.4 (typical mid-power geo test)

Prior conversion:
  Model 2 works on MaxAbsScaler-scaled data. On that scale,
      beta_scaled = ROI * (channel_max_spend / target_max_revenue) *
                    (total_saturated_response / total_spend)
  We compute the saturation factor from the true parameters (an oracle
  choice for the demonstration; in practice you'd use a plug-in estimate).

Runs one fit (~10-12 min); saves slim output.

Run:  python lift_test_calibration.py
"""
import json, os, warnings
import arviz as az, numpy as np, pandas as pd, xarray as xr
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_marketing.prior import Prior

# reuse our own transforms
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from transforms import geometric_adstock, hill_saturation

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SMOKE_TEST = False
CALIB_CHANNEL = "search"          # calibrate search - highest confounding gap
LIFT_TEST_ROI = 3.9               # simulated experimental estimate
LIFT_TEST_SE = 0.4                # standard error - moderately informative

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ============================================================ load data ====
df = pd.read_csv(os.path.join(HERE, "data", "synthetic_confounded.csv"),
                 parse_dates=["date"])
with open(os.path.join(HERE, "data", "true_params.json")) as f:
    truth = json.load(f)
true_roi = truth["variants"]["confounded"]["true_roi"]
channel_params = truth["variants"]["confounded"]["channel_params"]

spend_cols = [c for c in df.columns if c.endswith("_spend")]
channels = [c.replace("_spend", "") for c in spend_cols]
df["t"] = np.arange(len(df))

# ==================================================== compute prior on beta =
# We want a Normal prior on the scaled saturation_beta for CALIB_CHANNEL
# that corresponds to "ROI = LIFT_TEST_ROI +/- LIFT_TEST_SE" on the raw scale.
#
# On the internal scale:
#   contribution_scaled_t = beta_scaled * saturated_scaled_t
#   contribution_raw_t = contribution_scaled_t * y_max
# So sum_contribution_raw = beta_scaled * y_max * sum(saturated_scaled)
# And ROI = sum_contribution_raw / sum_spend_raw
#        = beta_scaled * y_max * sum(saturated_scaled) / sum_spend_raw
# Solve for beta_scaled given ROI:
#   beta_scaled = ROI * sum_spend_raw / (y_max * sum(saturated_scaled))
#
# We need sum(saturated_scaled). It depends on the (adstock, saturation)
# parameters of this channel. We use the TRUE parameters here as an oracle
# plug-in - this is a demonstration of what calibration BUYS when done
# correctly. (An honest limitation to note in the paper.)
i_calib = channels.index(CALIB_CHANNEL)
p = channel_params[CALIB_CHANNEL]
spend_ch = df[f"{CALIB_CHANNEL}_spend"].values
x_max_ch = spend_ch.max()
spend_scaled = spend_ch / x_max_ch                       # MaxAbsScaler
adstocked = geometric_adstock(spend_scaled, theta=p["decay"],
                              L=12, normalize=True)
# saturation is on scaled data; kappa in raw units -> convert to scaled
kappa_scaled = p["kappa"] / x_max_ch
saturated_scaled = hill_saturation(adstocked, alpha=p["alpha"],
                                    kappa=kappa_scaled)
sum_sat = saturated_scaled.sum()

y_max = df["revenue"].max()
sum_spend = spend_ch.sum()

roi_to_beta = sum_spend / (y_max * sum_sat)
mu_beta = LIFT_TEST_ROI * roi_to_beta
sd_beta = LIFT_TEST_SE * roi_to_beta

print(f"Lift test injected on channel: {CALIB_CHANNEL}")
print(f"  experimental ROI: {LIFT_TEST_ROI} +/- {LIFT_TEST_SE} (true: {true_roi[CALIB_CHANNEL]:.2f})")
print(f"  -> Normal({mu_beta:.4f}, {sd_beta:.4f}) prior on saturation_beta[{CALIB_CHANNEL}]")

# ============================== per-channel priors: informative on calib ====
# saturation_beta needs a channel-indexed prior. Easiest: keep the default
# HalfNormal for everyone, then override with a per-channel Normal by using
# a NumPy vector of prior means/sds. But PyMC-Marketing's Prior class doesn't
# accept a vector directly. Workaround: build a truncated normal centered
# where we want, per-channel, using a slightly different approach:
# construct the prior as a plain Normal with per-channel dims via a
# named vector when possible.
#
# Simpler and robust: keep HalfNormal(1.0) for non-calibrated channels
# and use TruncatedNormal(mu_beta, sd_beta, lower=0) for the calibrated
# channel. To let PyMC-Marketing accept per-channel structure, we set the
# whole prior as TruncatedNormal with mu and sigma as ARRAYS indexed by
# channel: mu = [0,...,mu_beta at calib_idx,...,0] with fallback prior for
# the others via a large sigma.

n_ch = len(channels)
mu_vec = np.zeros(n_ch)
sd_vec = np.ones(n_ch) * 1.0        # matches HalfNormal(1.0) width for others
mu_vec[i_calib] = mu_beta
sd_vec[i_calib] = sd_beta

# The library's Prior class accepts numeric args - we pass arrays.
sat_beta_prior = Prior("TruncatedNormal",
                       mu=mu_vec.tolist(),
                       sigma=sd_vec.tolist(),
                       lower=0.0,
                       dims="channel")

model_config = {
    "adstock_alpha":    Prior("Beta", alpha=1, beta=3, dims="channel"),
    "saturation_slope": Prior("Gamma", alpha=3, beta=1, dims="channel"),
    "saturation_kappa": Prior("Beta", alpha=2, beta=2, dims="channel"),
    "saturation_beta":  sat_beta_prior,       # <- the calibrated prior
    "gamma_control":    Prior("Normal", mu=0, sigma=2, dims="control"),
    "gamma_fourier":    Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
    "intercept":        Prior("Normal", mu=0, sigma=2, dims=()),
    "likelihood":       Prior("Normal",
                              sigma=Prior("HalfNormal", sigma=2, dims=()),
                              dims="date"),
}

# ============================================================ fit ==========
mmm = MMM(
    date_column="date",
    channel_columns=spend_cols,
    control_columns=["t"],        # NO demand control - backdoor open
    adstock=GeometricAdstock(l_max=12),
    saturation=HillSaturation(),
    yearly_seasonality=3,
    model_config=model_config,
)
X = df[["date", "t"] + spend_cols]
y = df["revenue"]

if SMOKE_TEST:
    draws, tune, chains = 250, 250, 2
else:
    draws, tune, chains = 1000, 1000, 4

print(f"\nsampling: draws={draws}, chains={chains}, target_accept=0.98")
idata = mmm.fit(X=X, y=y,
                draws=draws, tune=tune, chains=chains, cores=min(chains, 4),
                target_accept=0.98, max_treedepth=12,
                random_seed=42, progressbar=False)

n_div = int(idata.sample_stats["diverging"].sum())
print(f"divergences: {n_div}")

contrib = mmm.compute_channel_contribution_original_scale()
total_spend_v = df[spend_cols].sum().values
roi_da = contrib.sum(dim="date") / total_spend_v
roi_da.to_netcdf(os.path.join(OUT, "confounded_roi_calibrated.nc"))

# ============================================================ summary ======
print("\n=== CALIBRATED FIT: ROI vs true (confounded world) ===")
print(f"{'channel':<9s} {'true':>7s} {'calib med':>10s} {'  95% CI':>18s}")
for i, ch in enumerate(channels):
    s = roi_da.isel(channel=i).values.ravel()
    lo, hi = np.percentile(s, [2.5, 97.5])
    print(f"{ch:<9s} {true_roi[ch]:>7.2f} {np.median(s):>10.2f} [{lo:>6.2f}, {hi:>6.2f}]")

print(f"\nsaved outputs/confounded_roi_calibrated.nc")
