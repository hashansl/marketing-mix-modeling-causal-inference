## Marketing Mix Modeling as Causal Inference

Independent study: estimating the causal effect (ROI) of marketing channels
from observational, aggregated time-series data, and stress-testing what
happens when the identification assumptions fail.

### Research question

> Can a Bayesian Marketing Mix Model recover the true causal effect (ROI) of
> each marketing channel from observational, aggregated time-series data — and
> how badly do those estimates break when the core identification assumptions
> fail?

Sub-questions:
1. Recovery — when the truth is known (synthetic), do the posteriors cover it?
2. Confounding — how much does omitting a demand confounder bias ROI, and does
   adjusting for it fix the bias?
3. Calibration — can an experimental lift estimate, used as a prior, correct an
   MMM that observational data alone gets wrong?

### Why this is causal inference, not regression

- The estimand is a counterfactual: ROI = incremental revenue per dollar, the
  difference between the observed world and one with different spend.
- Identification is by assumption, not design: conditional ignorability
  (controls block the demand backdoor path), correct functional form (adstock
  and saturation encode the temporal and dose-response shape of the effect),
  and no interference beyond the adstock window.
- The experiments below show the model recovering truth when assumptions hold,
  breaking under an omitted confounder, and being rescued by experimental
  calibration.

### Repo layout

```
src/
  transforms.py        adstock + saturation (NumPy reference) + tests
  test_transforms.py   unit tests (impulse decay, boundedness, monotonicity)
data/
  generate_synthetic.py  builds clean + confounded datasets, saves true ROI
  eda_synthetic.py       sanity-check figure
  synthetic_clean.csv      (generated)
  synthetic_confounded.csv (generated)
  true_params.json         (generated) ground-truth ROI + parameters
outputs/figures/         generated figures
notebooks/               one per experiment (recovery, confounding, etc.)
```

### Datasets

- Synthetic (generated here) — clean and confounded variants from the same
  channel parameters; only the confounded set ties spend to a hidden demand
  driver. True ROI is known, so recovery and bias can be measured.
- Robyn `dt_simulated_weekly` (to add) — ~208 weeks, industry-standard, used
  for the applied outputs (contributions, response curves, budget allocation).

### Reproduce

```
python src/test_transforms.py      # validate the transforms
python data/generate_synthetic.py  # build datasets + ground truth
python data/eda_synthetic.py       # sanity-check figure
```

Seed is fixed (42) so everything is reproducible.
