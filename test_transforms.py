"""
Tests for the media transforms. Run with:  python src/test_transforms.py

These are the checks that let you trust the transforms before building
anything on top of them. If these pass, Model 1 and the recovery tests
are standing on solid ground.
"""

import numpy as np
from transforms import geometric_adstock, hill_saturation, logistic_saturation


def test_adstock_impulse_decays_geometrically():
    """An impulse (1 at t=0, 0 after) must produce weights theta**l / sum."""
    theta, L = 0.5, 6
    x = np.zeros(20)
    x[0] = 1.0
    out = geometric_adstock(x, theta=theta, L=L, normalize=True)

    expected_weights = theta ** np.arange(L + 1)
    expected_weights = expected_weights / expected_weights.sum()
    # out[0..L] should equal the normalized weights
    assert np.allclose(out[: L + 1], expected_weights), "impulse response wrong"
    # ratio of consecutive nonzero taps equals theta
    assert np.isclose(out[1] / out[0], theta), "decay rate is not theta"
    print("  ok  adstock impulse decays at rate theta")


def test_adstock_normalized_preserves_level():
    """Normalized adstock of a constant series returns (approx) the constant."""
    x = np.full(50, 10.0)
    out = geometric_adstock(x, theta=0.7, L=12, normalize=True)
    # after the warm-up window the output should sit at the input level
    assert np.allclose(out[20:], 10.0, atol=1e-6), "normalized level not preserved"
    print("  ok  normalized adstock preserves the level of a constant series")


def test_adstock_is_causal():
    """Output at t must not depend on future spend."""
    x = np.zeros(20)
    x[10] = 1.0
    out = geometric_adstock(x, theta=0.5, L=6, normalize=True)
    assert np.all(out[:10] == 0.0), "adstock leaked information backward in time"
    print("  ok  adstock is causal (no leakage from the future)")


def test_adstock_theta_bounds():
    for bad in (-0.1, 1.0, 1.5):
        try:
            geometric_adstock([1, 2, 3], theta=bad)
        except ValueError:
            continue
        raise AssertionError(f"theta={bad} should have raised")
    print("  ok  adstock rejects theta outside [0, 1)")


def test_hill_bounded_and_monotonic():
    x = np.linspace(0, 1000, 500)
    s = hill_saturation(x, alpha=2.0, kappa=200.0)
    assert np.all(s >= 0) and np.all(s < 1.0), "hill not bounded in [0,1)"
    assert np.all(np.diff(s) >= -1e-12), "hill not monotonically increasing"
    print("  ok  hill saturation is bounded in [0,1) and monotonic")


def test_hill_half_saturation_point():
    """s(kappa) must equal 0.5 - that's what makes kappa interpretable."""
    s = hill_saturation(np.array([200.0]), alpha=2.0, kappa=200.0)
    assert np.isclose(s[0], 0.5), "s(kappa) != 0.5"
    print("  ok  hill reaches 0.5 exactly at the half-saturation point kappa")


def test_logistic_bounded_and_monotonic():
    x = np.linspace(0, 50, 500)
    s = logistic_saturation(x, lam=0.1)
    assert np.all(s >= 0) and np.all(s < 1.0), "logistic not bounded in [0,1)"
    assert np.all(np.diff(s) >= -1e-12), "logistic not monotonic"
    print("  ok  logistic saturation is bounded in [0,1) and monotonic")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} transform tests...")
    for t in tests:
        t()
    print("All transform tests passed.")
