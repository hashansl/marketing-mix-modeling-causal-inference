"""
Synthetic MMM data generator.

Produces two datasets from the SAME channel parameters:

  synthetic_clean.csv       - media spend is set independently of demand.
                              Conditional ignorability holds. The model
                              should recover true ROI here.

  synthetic_confounded.csv  - a hidden 'demand' driver raises BOTH spend and
                              revenue (the backdoor path from the DAG). Fit
                              WITHOUT controlling for demand and channel ROI
                              is biased UP; fit WITH it and the bias closes.

Because we set every parameter, we can compute the TRUE ROI of each channel
from its own realized contribution. That ground truth (saved to
true_params.json) is what the recovery and confounding experiments score
against - the whole reason for simulating rather than only using real data.

Run:  python data/generate_synthetic.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from transforms import geometric_adstock, hill_saturation  # noqa: E402

SEED = 42
N_WEEKS = 156          # 3 years of weekly data
ADSTOCK_L = 12         # carryover window (weeks)
START_DATE = "2021-01-04"

# ----------------------------------------------------------------------------
# Channel parameters - THESE ARE THE GROUND TRUTH.
# All money is in THOUSANDS OF DOLLARS.
#
# decay (theta): carryover. TV long, search short - matches real-world priors.
# alpha, kappa : Hill saturation shape and half-saturation point (on the
#                adstocked-spend scale, set near each channel's spend level so
#                we operate on the curved part of the response).
# target_roi   : the TRUE return on investment we want this channel to have.
#                We derive beta from it (see calibrate_beta) rather than guessing
#                a coefficient - cleaner to state and to validate against.
# spend_mean/sd: weekly spend level and volatility.
# demand_beta  : how strongly the hidden demand driver pushes this channel's
#                spend in the CONFOUNDED world (0 effect in the clean world).
# ----------------------------------------------------------------------------
CHANNELS = {
    "tv":      dict(decay=0.65, alpha=2.0, kappa=28.0, target_roi=3.0, spend_mean=30.0, spend_sd=7.0, demand_beta=0.45),
    "search":  dict(decay=0.15, alpha=1.6, kappa=18.0, target_roi=4.5, spend_mean=20.0, spend_sd=5.0, demand_beta=0.60),
    "social":  dict(decay=0.35, alpha=1.8, kappa=16.0, target_roi=2.5, spend_mean=18.0, spend_sd=4.5, demand_beta=0.30),
    "display": dict(decay=0.45, alpha=1.5, kappa=11.0, target_roi=1.8, spend_mean=12.0, spend_sd=3.0, demand_beta=0.20),
    "email":   dict(decay=0.20, alpha=1.7, kappa=5.0,  target_roi=3.5, spend_mean=6.0,  spend_sd=1.5, demand_beta=0.10),
}

# Baseline (non-media) structure - thousands of dollars.
INTERCEPT = 300.0             # baseline weekly revenue with zero media
TREND_PER_WEEK = 0.6          # gentle upward drift
SEASON_AMPLITUDE = 40.0       # yearly seasonality strength
N_FOURIER = 3                 # number of Fourier pairs for seasonality
DEMAND_ON_SALES = 45.0        # how much the hidden demand driver moves revenue
NOISE_SD = 18.0              # observation noise on revenue (~3-4% of revenue)


def fourier_seasonality(t, period=52.13, n_terms=N_FOURIER, rng=None):
    """Smooth yearly seasonality as a sum of sine/cosine pairs."""
    # Think of N_FOURIER as how much detail is allowed inside one year's shape 
    season = np.zeros_like(t, dtype=float)
    coeffs = []
    for k in range(1, n_terms + 1):
        a = rng.normal(0, 1)
        b = rng.normal(0, 1)
        coeffs.append((a, b))
        season += a * np.sin(2 * np.pi * k * t / period) + b * np.cos(2 * np.pi * k * t / period)
    # normalize to unit std then scale, so amplitude is controllable
    season = season / season.std()
    return season, coeffs


# confounding injected - Generates one channels weekly spend
def make_spend(name, p, confounded, demand_z, rng):
    """Generate a single channel's weekly spend series."""
    # not confounded -  Spend is a free variable — statistically independent of demand, seasonality, trend, and revenue's baseline.
    # creates set of random numbers(from a normal distribution) based on mean and sd given below - total size N_weeks
    base = rng.normal(p["spend_mean"], p["spend_sd"], size=N_WEEKS)
    if confounded:
        # demand pushes spend up multiplicatively: marketers spend more when
        # they expect high demand. This is the confounding mechanism.
        # In the confounded world (confounded=True), the base spend gets multiplied by exp(demand_beta * demand_z). Now spend and demand are correlated by construction. 
        # demand beta in the CHANNEL data - different channels would have different betas
        base = base * np.exp(p["demand_beta"] * demand_z)

    return np.clip(base, 0.5, None)  # keep strictly positive


def calibrate_betas(rng_seed):
    """
    Derive each channel's beta from its target ROI, using the CLEAN world's
    spend. Since realized ROI = sum(beta * saturated) / sum(spend), we set

        beta = target_roi * sum(spend) / sum(saturated)

    so the clean-world ROI lands on target by construction. These structural
    betas are then reused unchanged in the confounded world.
    """
    # the design goal is: let me specify each channel's true ROI, and derive whatever β makes that happen. That inversion is exactly what calibrate_betas does.
    
    
    rng = np.random.default_rng(rng_seed)
    betas = {}
    for name, p in CHANNELS.items():
        spend = make_spend(name, p, confounded=False, demand_z=None, rng=rng)
        adstocked = geometric_adstock(spend, theta=p["decay"], L=ADSTOCK_L, normalize=True)
        saturated = hill_saturation(adstocked, alpha=p["alpha"], kappa=p["kappa"])
        betas[name] = float(p["target_roi"] * spend.sum() / saturated.sum())
    return betas


def build_variant(variant, betas, demand_z, season, trend, rng):
    """
    Build one dataset variant using fixed structural betas.

    variant : 'clean' or 'confounded'
    betas   : calibrated coefficients (shared across variants)
    demand_z: standardized hidden demand series (used only when confounded)
    """
    confounded = variant == "confounded"
    T = N_WEEKS
    columns = {}
    contributions = {}
    true_roi = {}

    for name, p in CHANNELS.items():
        spend = make_spend(name, p, confounded, demand_z, rng)

        # --- media response: adstock -> saturate -> scale by beta ---
        adstocked = geometric_adstock(spend, theta=p["decay"], L=ADSTOCK_L, normalize=True)
        saturated = hill_saturation(adstocked, alpha=p["alpha"], kappa=p["kappa"])
        contrib = betas[name] * saturated

        columns[f"{name}_spend"] = spend
        contributions[name] = contrib
        true_roi[name] = float(contrib.sum() / spend.sum())

    media_total = np.sum([contributions[c] for c in CHANNELS], axis=0)

    # --- assemble revenue ---
    mu = INTERCEPT + trend + season + media_total
    if confounded:
        mu = mu + DEMAND_ON_SALES * demand_z   # demand also lifts sales directly
    revenue = mu + rng.normal(0, NOISE_SD, size=T)

    # --- dataframe ---
    dates = pd.date_range(START_DATE, periods=T, freq="W-MON")
    df = pd.DataFrame({"week": np.arange(T), "date": dates, "revenue": revenue})
    for col, vals in columns.items():
        df[col] = vals
    if confounded:
        # expose the true confounder so the experiment can fit 'with control'.
        # (In reality you'd only have a noisy proxy - a nice extension.)
        df["demand"] = demand_z

    meta = {
        "variant": variant,
        "true_roi": true_roi,
        "derived_betas": betas,
        "channel_params": {k: CHANNELS[k] for k in CHANNELS},
        "baseline": {
            "intercept": INTERCEPT,
            "trend_per_week": TREND_PER_WEEK,
            "season_amplitude": SEASON_AMPLITUDE,
            "demand_on_sales": DEMAND_ON_SALES if confounded else 0.0,
            "noise_sd": NOISE_SD,
        },
    }
    return df, meta


def main():
    rng = np.random.default_rng(SEED)
    t = np.arange(N_WEEKS)
    trend = TREND_PER_WEEK * t
    season_raw, season_coeffs = fourier_seasonality(t, rng=rng)
    season = SEASON_AMPLITUDE * season_raw
    # I want the seasonal signal to have a standard deviation of $40k-(40) around baseline.

    # hidden demand driver: its own smooth seasonal + trend signal + noise,
    # It builds a hidden demand driver — a synthetic time series representing "how much people wanted to buy this week, apart from advertising." This variable will do two jobs later
    # 1. Push spend up in high-demand weeks (in make_spend via demand_beta) — one arm of the backdoor.
    # 2. Push revenue up directly (in build_variant via DEMAND_ON_SALES) — the other arm.
    # standardized. This is what confounds spend and sales in the confounded set.
    demand_raw = (
        np.sin(2 * np.pi * t / 52.13)                   # the annual demand cycle
        + 0.4 * np.cos(2 * np.pi * 2 * t / 52.13)       # smaller secondary cycle
        + 0.01 * t                                      # a slow upward drift
        + rng.normal(0, 0.3, size=N_WEEKS)              # Standard-normal noise scaled down to std=0.3, one draw per week. This adds week-to-week randomness on top of the structural signal
    )
    demand_z = (demand_raw - demand_raw.mean()) / demand_raw.std()

    # derive structural betas from target ROIs (calibrated on clean-world spend)
    betas = calibrate_betas(SEED)

    out_dir = os.path.join(os.path.dirname(__file__))
    all_meta = {"seed": SEED, "n_weeks": N_WEEKS, "adstock_L": ADSTOCK_L, "variants": {}}

    for variant in ("clean", "confounded"):
        df, meta = build_variant(variant, betas, demand_z, season, trend, rng)
        path = os.path.join(out_dir, f"synthetic_{variant}.csv")
        df.to_csv(path, index=False)
        all_meta["variants"][variant] = meta
        print(f"\n=== {variant.upper()} ===")
        print(f"  saved: {path}  ({df.shape[0]} weeks x {df.shape[1]} cols)")
        print(f"  revenue: mean={df.revenue.mean():,.0f}  min={df.revenue.min():,.0f}  max={df.revenue.max():,.0f}")
        print("  TRUE ROI per channel:")
        for ch, roi in meta["true_roi"].items():
            print(f"    {ch:8s}  {roi:6.3f}")

    # correlation check: in the confounded set, spend should correlate with demand
    conf = pd.read_csv(os.path.join(out_dir, "synthetic_confounded.csv"))
    print("\n=== CONFOUNDING CHECK (corr of spend with hidden demand) ===")
    for ch in CHANNELS:
        r = np.corrcoef(conf[f"{ch}_spend"], conf["demand"])[0, 1]
        print(f"    {ch:8s}  corr(spend, demand) = {r:+.3f}")

    with open(os.path.join(out_dir, "true_params.json"), "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\n  saved ground truth: {os.path.join(out_dir, 'true_params.json')}")


if __name__ == "__main__":
    main()
