"""
Model 0 and Model 1 - the frequentist ladder that sets up Model 2.

Model 0 (naive OLS)
-------------------
    revenue ~ raw_spend_c + trend + Fourier
No adstock, no saturation. Answers: does the model recover ROI if we ignore
carryover and diminishing returns?  (Answer: no. That's the point.)

Model 1 (oracle OLS with transforms)
------------------------------------
    revenue ~ Hill(adstock(spend_c; theta_c*); alpha_c*, kappa_c*)
             + trend + Fourier
Same OLS, but spend is first put through geometric adstock and Hill
saturation using the TRUE parameters (theta*, alpha*, kappa* from
true_params.json).  Answers: does OLS work once we have the right shapes?
(Answer: mostly yes. This isolates what the transforms buy vs what the
Bayesian machinery buys.)

Both models are scored against the true ROI recorded in true_params.json.

Runs in seconds. No sampling.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# reuse our own transforms - the same code that generated the data
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from transforms import geometric_adstock, hill_saturation

ADSTOCK_L = 12
N_FOURIER = 3
PERIOD = 52.13


# ============================================================ load data ====
df = pd.read_csv(os.path.join(HERE, "data", "synthetic_clean.csv"),
                 parse_dates=["date"])
with open(os.path.join(HERE, "data", "true_params.json")) as f:
    truth = json.load(f)
true_roi = truth["variants"]["clean"]["true_roi"]
channel_params = truth["variants"]["clean"]["channel_params"]

spend_cols = [c for c in df.columns if c.endswith("_spend")]
channels = [c.replace("_spend", "") for c in spend_cols]
T = len(df)
t = np.arange(T)


# ================================== shared baseline features (trend+Fourier)
def baseline_features(t, period=PERIOD, n_fourier=N_FOURIER):
    """Trend + Fourier seasonality columns. Same basis the DGP used, so any
    OLS remaining bias comes from the media side, not the baseline."""
    cols = {"trend": t.astype(float)}
    for k in range(1, n_fourier + 1):
        cols[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols)

base = baseline_features(t)


# =========================================== ROI computation shared helper
def ols_roi(spend_matrix, transformed_matrix, y, X_extra):
    """
    Fit OLS: y = intercept + coeffs * transformed_matrix + gammas * X_extra
    Then for each channel c, ROI_c = sum(coef_c * transformed_c) / sum(spend_c).

    Returns per-channel ROI, fitted intercept, and predicted revenue.
    """
    X = np.hstack([transformed_matrix, X_extra.values])
    reg = LinearRegression().fit(X, y)
    y_hat = reg.predict(X)

    n_ch = transformed_matrix.shape[1]
    beta = reg.coef_[:n_ch]                               # media coefficients
    contribution = transformed_matrix * beta              # (T, n_ch)
    total_contrib = contribution.sum(axis=0)              # per channel
    total_spend = spend_matrix.sum(axis=0)
    roi = total_contrib / total_spend
    return roi, reg.intercept_, y_hat, beta


# =============================================================== Model 0 ==
# Naive OLS on RAW spend.
spend_matrix = df[spend_cols].values
roi_m0, intercept_m0, yhat_m0, beta_m0 = ols_roi(
    spend_matrix=spend_matrix,
    transformed_matrix=spend_matrix,   # no transform
    y=df["revenue"].values,
    X_extra=base,
)

# =============================================================== Model 1 ==
# Oracle OLS: apply adstock + Hill saturation with the TRUE parameters.
transformed = np.zeros_like(spend_matrix, dtype=float)
for i, ch in enumerate(channels):
    p = channel_params[ch]
    adstocked = geometric_adstock(spend_matrix[:, i], theta=p["decay"],
                                  L=ADSTOCK_L, normalize=True)
    saturated = hill_saturation(adstocked, alpha=p["alpha"], kappa=p["kappa"])
    transformed[:, i] = saturated

roi_m1, intercept_m1, yhat_m1, beta_m1 = ols_roi(
    spend_matrix=spend_matrix,
    transformed_matrix=transformed,
    y=df["revenue"].values,
    X_extra=base,
)


# =================================================================== R^2 ==
def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

y = df["revenue"].values
r2_m0 = r2(y, yhat_m0)
r2_m1 = r2(y, yhat_m1)


# =================================================================== print
print("=" * 65)
print("MODEL 0 - naive OLS (no adstock, no saturation)")
print("=" * 65)
print(f"R^2: {r2_m0:.4f}")
print(f"\n{'channel':<10s} {'true':>8s} {'M0 ROI':>10s} {'error':>10s}")
for i, ch in enumerate(channels):
    err = roi_m0[i] - true_roi[ch]
    print(f"{ch:<10s} {true_roi[ch]:>8.2f} {roi_m0[i]:>10.2f} {err:>+10.2f}")

print("\n" + "=" * 65)
print("MODEL 1 - oracle OLS with true adstock + Hill saturation")
print("=" * 65)
print(f"R^2: {r2_m1:.4f}")
print(f"\n{'channel':<10s} {'true':>8s} {'M1 ROI':>10s} {'error':>10s}")
for i, ch in enumerate(channels):
    err = roi_m1[i] - true_roi[ch]
    print(f"{ch:<10s} {true_roi[ch]:>8.2f} {roi_m1[i]:>10.2f} {err:>+10.2f}")

# summary table
rmse_m0 = np.sqrt(np.mean([(roi_m0[i] - true_roi[ch]) ** 2
                            for i, ch in enumerate(channels)]))
rmse_m1 = np.sqrt(np.mean([(roi_m1[i] - true_roi[ch]) ** 2
                            for i, ch in enumerate(channels)]))
print("\n" + "=" * 65)
print("SUMMARY - ROI RMSE vs truth (lower = better)")
print("=" * 65)
print(f"Model 0 (naive OLS)                 :  RMSE = {rmse_m0:.3f}   R^2 = {r2_m0:.4f}")
print(f"Model 1 (oracle OLS with transforms):  RMSE = {rmse_m1:.3f}   R^2 = {r2_m1:.4f}")

# save outputs for the comparison table
out = {
    "model_0_naive_ols": {
        "roi": {ch: float(roi_m0[i]) for i, ch in enumerate(channels)},
        "rmse": float(rmse_m0), "r2": float(r2_m0),
    },
    "model_1_oracle_ols": {
        "roi": {ch: float(roi_m1[i]) for i, ch in enumerate(channels)},
        "rmse": float(rmse_m1), "r2": float(r2_m1),
    },
    "true_roi": true_roi,
}
with open(os.path.join(HERE, "outputs", "ols_baselines.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nsaved outputs/ols_baselines.json")
