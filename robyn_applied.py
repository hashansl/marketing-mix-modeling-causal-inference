# """
# APPLIED FIT: PyMC-Marketing Bayesian MMM on the Robyn dt_simulated_weekly
# dataset. This is where the whole apparatus meets real-looking data.

# Two fits:

#   A - well-specified: paid channels + adstock/saturation + linear controls
#       (competitor_sales_B, newsletter, trend) + Fourier seasonality.
#       Produces the applied deliverables.

#   B - counterfactual: same as A but competitor_sales_B REMOVED from controls.
#       Not a full replay of the synthetic confounding experiment (no ground
#       truth here), but shows how much ROI moves when the strongest available
#       confounder is omitted - the real-data analog of the "backdoor open"
#       case.


# Reads:
#   data/robyn_weekly.csv          Robyn's open-source benchmark dataset containing weekly historical spend, revenue, and competitive landscape variables

# Produces:
#   outputs/robyn_full_posterior.nc    posterior samples for the well-specified model
#   outputs/robyn_full_roi.nc          posterior ROI samples for the well-specified model
#   outputs/robyn_full_contrib.nc      channel contribution time-series for the well-specified model
#   outputs/robyn_nocf_posterior.nc    posterior samples for the counterfactual model (omitted confounder)
#   outputs/robyn_nocf_roi.nc          posterior ROI samples for the counterfactual model (omitted confounder)
#   outputs/robyn_nocf_contrib.nc      channel contribution time-series for the counterfactual model (omitted confounder)

# Saves slim outputs so figures regenerate without re-sampling.
# Runtime: ~15-20 min per fit (208 weeks x 5 channels x tight sampler).
# Set SMOKE_TEST=True for a fast shakeout.

# Run:  python robyn_applied.py
# """
# import json, os, warnings
# import arviz as az, numpy as np, pandas as pd
# from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
# from pymc_marketing.prior import Prior

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

# SMOKE_TEST = False

# HERE = os.path.dirname(os.path.abspath(__file__))
# OUT = os.path.join(HERE, "outputs")
# os.makedirs(OUT, exist_ok=True)

# # ============================================================ load data ====
# df = pd.read_csv(os.path.join(HERE, "data", "robyn_weekly.csv"),
#                  parse_dates=["DATE"])
# df = df.sort_values("DATE").reset_index(drop=True)
# df = df.rename(columns={"DATE": "date"})
# df["t"] = np.arange(len(df))

# # paid media channels (Robyn's _S = spend suffix)
# paid_cols = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
# paid_names = [c.replace("_S", "") for c in paid_cols]

# # controls: competitor sales is the big one (r=+0.92 with revenue).
# # newsletter is organic marketing (r=+0.41); dropped events (only 2 non-na
# # weeks in 208 - too few to identify an effect).
# controls_full = ["competitor_sales_B", "newsletter", "t"]
# controls_no_confounder = ["newsletter", "t"]   # competitor OMITTED

# print(f"data: {df.shape}   paid channels: {paid_names}")
# print(f"controls (full):          {controls_full}")
# print(f"controls (no confounder): {controls_no_confounder}")

# # ==================================================== priors (same as v2) ===
# def make_config():
#     return {
#         "adstock_alpha":    Prior("Beta", alpha=1, beta=3, dims="channel"),
#         "saturation_slope": Prior("Gamma", alpha=3, beta=1, dims="channel"),
#         "saturation_kappa": Prior("Beta", alpha=2, beta=2, dims="channel"),
#         "saturation_beta":  Prior("HalfNormal", sigma=1.0, dims="channel"),
#         "gamma_control":    Prior("Normal", mu=0, sigma=2, dims="control"),
#         "gamma_fourier":    Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
#         "intercept":        Prior("Normal", mu=0, sigma=2, dims=()),
#         "likelihood":       Prior("Normal",
#                                   sigma=Prior("HalfNormal", sigma=2, dims=()),
#                                   dims="date"),
#     }

# if SMOKE_TEST:
#     draws, tune, chains = 250, 250, 2
# else:
#     draws, tune, chains = 1000, 2000, 4


# def fit_and_save(controls, tag, save_prefix):
#     print(f"\n=== Fit {tag}: controls = {controls} ===")
#     mmm = MMM(
#         date_column="date",
#         channel_columns=paid_cols,
#         control_columns=controls,
#         adstock=GeometricAdstock(l_max=12),
#         saturation=HillSaturation(),
#         yearly_seasonality=3,
#         model_config=make_config(),
#     )
#     X = df[["date"] + controls + paid_cols]
#     y = df["revenue"]
#     idata = mmm.fit(X=X, y=y, draws=draws, tune=tune, chains=chains,
#                     cores=min(chains, 4), target_accept=0.98, max_treedepth=13,
#                     random_seed=42, progressbar=False)
#     n_div = int(idata.sample_stats["diverging"].sum())
#     print(f"  divergences: {n_div}")

#     # slim saves
#     az.InferenceData(posterior=idata.posterior).to_netcdf(
#         os.path.join(OUT, f"{save_prefix}_posterior.nc"))

#     contrib = mmm.compute_channel_contribution_original_scale()
#     total_spend = df[paid_cols].sum().values
#     roi = contrib.sum(dim="date") / total_spend
#     roi.to_netcdf(os.path.join(OUT, f"{save_prefix}_roi.nc"))

#     # save full contribution time-series (needed for stacked area figure)
#     contrib.to_netcdf(os.path.join(OUT, f"{save_prefix}_contrib.nc"))

#     print(f"  saved outputs/{save_prefix}_*.nc")
#     return roi


# # =============================================================== run both ==
# roi_full = fit_and_save(controls_full, "A (well-specified)", "robyn_full")
# roi_nocf = fit_and_save(controls_no_confounder,
#                         "B (competitor OMITTED)", "robyn_nocf")

# # =============================================================== compare ===
# print("\n" + "=" * 72)
# print("ROBYN: ROI shift when competitor_sales_B is OMITTED as control")
# print("=" * 72)
# print(f"{'channel':<9s} {'well-spec':>10s} {'omit-cf':>10s} {'shift':>7s} {'pct':>7s}")
# for i, ch in enumerate(paid_names):
#     a = float(np.median(roi_full.isel(channel=i).values.ravel()))
#     b = float(np.median(roi_nocf.isel(channel=i).values.ravel()))
#     shift = b - a
#     pct = 100 * shift / a if a != 0 else float("nan")
#     print(f"{ch:<9s} {a:>10.2f} {b:>10.2f} {shift:>+7.2f} {pct:>+6.1f}%")

# print("\nReading: 'shift' is the ROI change when the confounder is omitted.")
# print("On synthetic data, omitting the confounder inflates ROI. If Robyn")
# print("shows the same pattern, that's consistent evidence (though not proof,")
# print("since we lack ground truth here).")

# RUN 2: Takes longer! 2+ hours!

"""
APPLIED FIT: PyMC-Marketing Bayesian MMM on the Robyn dt_simulated_weekly
dataset. This is where the whole apparatus meets real-looking data.

Two fits:

  A - well-specified: paid channels + adstock/saturation + linear controls
      (competitor_sales_B, newsletter, trend) + Fourier seasonality.
      Produces the applied deliverables.

  B - counterfactual: same as A but competitor_sales_B REMOVED from controls.
      Not a full replay of the synthetic confounding experiment (no ground
      truth here), but shows how much ROI moves when the strongest available
      confounder is omitted - the real-data analog of the "backdoor open"
      case.

Reads:
  data/robyn_weekly.csv          Robyn's open-source benchmark dataset containing weekly historical spend, revenue, and competitive landscape variables

Produces:
  outputs/robyn_full_posterior.nc    posterior samples for the well-specified model
  outputs/robyn_full_roi.nc          posterior ROI samples for the well-specified model
  outputs/robyn_full_contrib.nc      channel contribution time-series for the well-specified model
  outputs/robyn_nocf_posterior.nc    posterior samples for the counterfactual model (omitted confounder)
  outputs/robyn_nocf_roi.nc          posterior ROI samples for the counterfactual model (omitted confounder)
  outputs/robyn_nocf_contrib.nc      channel contribution time-series for the counterfactual model (omitted confounder)

Saves slim outputs so figures regenerate without re-sampling.
Runtime: ~15-20 min per fit (208 weeks x 5 channels x tight sampler).
Set SMOKE_TEST=True for a fast shakeout.

Run:  python robyn_applied.py
"""
import json, os, warnings
import arviz as az, numpy as np, pandas as pd
from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation
from pymc_marketing.prior import Prior

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SMOKE_TEST = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ============================================================ load data ====
df = pd.read_csv(os.path.join(HERE, "data", "robyn_weekly.csv"),
                 parse_dates=["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)
df = df.rename(columns={"DATE": "date"})
df["t"] = np.arange(len(df))

# paid media channels (Robyn's _S = spend suffix)
paid_cols = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
paid_names = [c.replace("_S", "") for c in paid_cols]

# controls: competitor sales is the big one (r=+0.92 with revenue).
# newsletter is organic marketing (r=+0.41); dropped events (only 2 non-na
# weeks in 208 - too few to identify an effect).
controls_full = ["competitor_sales_B", "newsletter", "t"]
controls_no_confounder = ["newsletter", "t"]   # competitor OMITTED

print(f"data: {df.shape}   paid channels: {paid_names}")
print(f"controls (full):          {controls_full}")
print(f"controls (no confounder): {controls_no_confounder}")

# ==================================================== priors (same as v2) ===
def make_config():
    return {
        "adstock_alpha":    Prior("Beta", alpha=1, beta=3, dims="channel"),
        "saturation_slope": Prior("Gamma", alpha=3, beta=1, dims="channel"),
        "saturation_kappa": Prior("Beta", alpha=2, beta=2, dims="channel"),
        "saturation_beta":  Prior("HalfNormal", sigma=1.0, dims="channel"),
        
        # --- FIX 1: TIGHTENED PRIORS FOR REGULARIZATION ---
        # Dropped gamma_control sigma from 2 to 0.5 to prevent wild coefficient
        # swings caused by the extreme collinearity of competitor_sales_B
        "gamma_control":    Prior("Normal", mu=0, sigma=0.5, dims="control"),
        "gamma_fourier":    Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
        "intercept":        Prior("Normal", mu=0, sigma=1, dims=()), # Dropped from 2 to 1
        "likelihood":       Prior("Normal",
                                  sigma=Prior("HalfNormal", sigma=2, dims=()),
                                  dims="date"),
    }

if SMOKE_TEST:
    draws, tune, chains = 250, 250, 2
else:
    draws, tune, chains = 1000, 2000, 4


def fit_and_save(controls, tag, save_prefix):
    print(f"\n=== Fit {tag}: controls = {controls} ===")
    mmm = MMM(
        date_column="date",
        channel_columns=paid_cols,
        control_columns=controls,
        adstock=GeometricAdstock(l_max=12),
        saturation=HillSaturation(),
        yearly_seasonality=3,
        model_config=make_config(),
    )
    X = df[["date"] + controls + paid_cols]
    y = df["revenue"]
    
    # --- FIX 2: MORE CAUTIOUS SAMPLER ---
    # Increased target_accept from 0.98 to 0.995 (forces smaller steps)
    # Increased max_treedepth from 13 to 15 (gives runway for those smaller steps)
    idata = mmm.fit(X=X, y=y, draws=draws, tune=tune, chains=chains,
                    cores=min(chains, 4), target_accept=0.995, max_treedepth=15,
                    random_seed=42, progressbar=False)
                    
    n_div = int(idata.sample_stats["diverging"].sum())
    print(f"  divergences: {n_div}")

    # slim saves
    az.InferenceData(posterior=idata.posterior).to_netcdf(
        os.path.join(OUT, f"{save_prefix}_posterior.nc"))

    contrib = mmm.compute_channel_contribution_original_scale()
    total_spend = df[paid_cols].sum().values
    roi = contrib.sum(dim="date") / total_spend
    roi.to_netcdf(os.path.join(OUT, f"{save_prefix}_roi.nc"))

    # save full contribution time-series (needed for stacked area figure)
    contrib.to_netcdf(os.path.join(OUT, f"{save_prefix}_contrib.nc"))

    print(f"  saved outputs/{save_prefix}_*.nc")
    return roi


# =============================================================== run both ==
roi_full = fit_and_save(controls_full, "A (well-specified)", "robyn_full")
roi_nocf = fit_and_save(controls_no_confounder,
                        "B (competitor OMITTED)", "robyn_nocf")

# =============================================================== compare ===
print("\n" + "=" * 72)
print("ROBYN: ROI shift when competitor_sales_B is OMITTED as control")
print("=" * 72)
print(f"{'channel':<9s} {'well-spec':>10s} {'omit-cf':>10s} {'shift':>7s} {'pct':>7s}")
for i, ch in enumerate(paid_names):
    a = float(np.median(roi_full.isel(channel=i).values.ravel()))
    b = float(np.median(roi_nocf.isel(channel=i).values.ravel()))
    shift = b - a
    pct = 100 * shift / a if a != 0 else float("nan")
    print(f"{ch:<9s} {a:>10.2f} {b:>10.2f} {shift:>+7.2f} {pct:>+6.1f}%")

print("\nReading: 'shift' is the ROI change when the confounder is omitted.")
print("On synthetic data, omitting the confounder inflates ROI. If Robyn")
print("shows the same pattern, that's consistent evidence (though not proof,")
print("since we lack ground truth here).")