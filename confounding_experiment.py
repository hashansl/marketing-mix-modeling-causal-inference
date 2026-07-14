"""
THE CONFOUNDING EXPERIMENT - the causal centerpiece of the project.

Fits the SAME Bayesian MMM on the CONFOUNDED dataset twice:

  Fit A (naive)    : controls = [trend]            -> demand backdoor OPEN
  Fit B (adjusted) : controls = [trend, demand]    -> demand backdoor CLOSED

Everything else is identical. So any difference in recovered ROI is caused
purely by adjusting for the demand confounder. This isolates the effect of
backdoor adjustment - the core idea of observational causal inference.

Ground truth: we score against the CONFOUNDED world's true ROI (not the
clean world's), because concentrating spend in high-demand weeks shifts each
channel's realized ROI. Scoring against the right baseline matters.

Prediction (state before looking at results): channels with the strongest
spend-demand link (search demand_beta=0.60, tv=0.45) should show the MOST
upward bias in Fit A; weak-link channels (email=0.10) the least.

Reads:
  data/synthetic_confounded.csv        historical marketing spend and revenue data influenced by the demand confounder
  data/true_params.json                ground truth parameter configuration (used to fetch the confounded world's true ROI baseline)

Produces:
  outputs/confounded_roi_naive.nc      naive posterior ROI samples (backdoor path left open)
  outputs/confounded_roi_adjusted.nc   adjusted posterior ROI samples (backdoor path closed via controls)

Saves slim outputs so the figure regenerates without re-sampling
Runtime: two ~10-12 min fits. Set SMOKE_TEST=True for a fast shakeout.

Run:  python confounding_experiment.py
"""
import json, os, warnings
import arviz as az, numpy as np, pandas as pd, xarray as xr
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_marketing.prior import Prior

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SMOKE_TEST = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ============================================================ load data ====
df = pd.read_csv(os.path.join(HERE, "data", "synthetic_confounded.csv"),
                 parse_dates=["date"])
with open(os.path.join(HERE, "data", "true_params.json")) as f:
    truth = json.load(f)
# IMPORTANT: score against the CONFOUNDED world's true ROI
true_roi = truth["variants"]["confounded"]["true_roi"]

spend_cols = [c for c in df.columns if c.endswith("_spend")]
channels = [c.replace("_spend", "") for c in spend_cols]
df["t"] = np.arange(len(df))
print(f"data: {df.shape}  channels: {channels}")
print(f"confounded true ROI: {true_roi}")

# ==================================================== shared priors =========
def make_config():
    return {
        "adstock_alpha":    Prior("Beta", alpha=1, beta=3, dims="channel"),
        "saturation_slope": Prior("Gamma", alpha=3, beta=1, dims="channel"),
        "saturation_kappa": Prior("Beta", alpha=2, beta=2, dims="channel"),
        "saturation_beta":  Prior("HalfNormal", sigma=1.0, dims="channel"),
        "gamma_control":    Prior("Normal", mu=0, sigma=2, dims="control"),
        "gamma_fourier":    Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
        "intercept":        Prior("Normal", mu=0, sigma=2, dims=()),
        "likelihood":       Prior("Normal",
                                  sigma=Prior("HalfNormal", sigma=2, dims=()),
                                  dims="date"),
    }

if SMOKE_TEST:
    draws, tune, chains = 250, 250, 2
else:
    draws, tune, chains = 1000, 2000, 4


def fit_and_roi(control_columns, tag):
    """Fit the MMM with a given control set, return per-channel ROI samples."""
    print(f"\n=== Fit {tag}: controls = {control_columns} ===")
    mmm = MMM(
        date_column="date",
        channel_columns=spend_cols,
        control_columns=control_columns,
        adstock=GeometricAdstock(l_max=12),
        saturation=HillSaturation(),
        yearly_seasonality=3,
        model_config=make_config(),
    )
    X = df[["date"] + control_columns + spend_cols]
    y = df["revenue"]
    idata = mmm.fit(X=X, y=y, draws=draws, tune=tune, chains=chains,
                    cores=min(chains, 4), target_accept=0.99, max_treedepth=13,
                    random_seed=42, progressbar=False)
    n_div = int(idata.sample_stats["diverging"].sum())
    print(f"  divergences: {n_div}")

    contrib = mmm.compute_channel_contribution_original_scale()
    total_spend = df[spend_cols].sum().values
    roi = contrib.sum(dim="date") / total_spend   # draws x channel
    return roi


# =============================================================== run both ==
roi_naive = fit_and_roi(["t"], "A (naive, backdoor OPEN)")
roi_naive.to_netcdf(os.path.join(OUT, "confounded_roi_naive.nc"))

roi_adjusted = fit_and_roi(["t", "demand"], "B (adjusted, backdoor CLOSED)")
roi_adjusted.to_netcdf(os.path.join(OUT, "confounded_roi_adjusted.nc"))


# =============================================================== bias table =
print("\n" + "=" * 72)
print("CONFOUNDING BIAS TABLE  (ROI)")
print("=" * 72)
print(f"{'channel':<9s} {'true':>7s} {'naive':>8s} {'nbias':>7s} "
      f"{'adjust':>8s} {'abias':>7s} {'demand_b':>9s}")
for i, ch in enumerate(channels):
    sn = roi_naive.isel(channel=i).values.ravel()
    sa = roi_adjusted.isel(channel=i).values.ravel()
    mn, ma = np.median(sn), np.median(sa)
    db = truth["variants"]["confounded"]["channel_params"][ch]["demand_beta"]
    print(f"{ch:<9s} {true_roi[ch]:>7.2f} {mn:>8.2f} {mn-true_roi[ch]:>+7.2f} "
          f"{ma:>8.2f} {ma-true_roi[ch]:>+7.2f} {db:>9.2f}")

print("\nReading: 'nbias' = naive bias (backdoor open). 'abias' = adjusted bias")
print("(backdoor closed). Expect nbias > 0 and shrinking with demand_beta;")
print("abias near 0. The gap (nbias - abias) is the confounding removed.")
print("\nsaved outputs/confounded_roi_naive.nc, outputs/confounded_roi_adjusted.nc")
