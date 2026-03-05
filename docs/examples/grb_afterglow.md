# GRB Afterglow Detection with Rubin LSST

This example simulates GRB afterglow detection with the Rubin LSST 10-year baseline survey using the `BlastwaveModel` for full relativistic blast wave + synchrotron radiation.

## Overview

The simulation pipeline:

1. Loads Rubin LSST survey observations (ugrizy, ~18,000 deg² over 10 years)
2. Samples GRB parameters from a pre-computed catalog of 42,803 parameter sets
3. For each GRB, solves the relativistic blast wave PDE and computes multi-band synchrotron afterglow emission
4. Applies detection criteria to determine which afterglows are recoverable

## Setup

```python
from survey_sim import (
    SurveyStore,
    GrbPopulation,
    BlastwaveModel,
    DetectionCriteria,
    SimulationPipeline,
)
```

## Load the survey

```python
survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db")
```

This loads ~2 million observations from the OpSim database: 6 bands (ugrizy), 10-year duration.

## Define the GRB population

```python
grb_pop = GrbPopulation(
    "GRB_afterglows_argus.csv",
    rate=1.0,      # Gpc^-3 yr^-1
    z_max=6.0,
)
```

The CSV catalog contains pre-drawn parameters spanning the GRB afterglow parameter space:

| Parameter | Typical range | Description |
|-----------|--------------|-------------|
| \(E_\mathrm{iso}\) | \(10^{47}\)--\(10^{54}\) erg | Isotropic equivalent energy |
| \(\Gamma_0\) | 1--500 | Initial bulk Lorentz factor |
| \(\theta_v\) | 0--\(\pi/2\) rad | Viewing angle |
| \(\theta_c\) | \(10^{-2}\)--\(10^{-0.5}\) rad | Jet half-opening angle |
| \(n_0\) | \(10^{-4}\)--\(10^{1}\) cm\(^{-3}\) | ISM density |
| \(\epsilon_e\) | \(10^{-3}\)--\(10^{0}\) | Electron energy fraction |
| \(\epsilon_B\) | \(10^{-5}\)--\(10^{0}\) | Magnetic energy fraction |
| \(p\) | 2.0--3.0 | Electron spectral index |

## Configure the blastwave model

For Rubin's 6-band system, provide the effective frequency of each band:

```python
model = BlastwaveModel(
    radiation_model="sync_ssa_smooth",
    band_frequencies={
        "u": 8.5e14,   # ~352 nm
        "g": 6.3e14,   # ~477 nm
        "r": 4.8e14,   # ~622 nm
        "i": 3.9e14,   # ~763 nm
        "z": 3.3e14,   # ~905 nm
        "y": 3.0e14,   # ~1000 nm
    },
)
```

The model computes separate afterglow flux densities at each frequency, naturally producing multi-band lightcurves and color evolution from the underlying synchrotron spectrum.

### Radiation model options

| Model | Description | Speed |
|-------|-------------|-------|
| `sync` | Optically thin synchrotron | Fastest |
| `sync_smooth` | Smooth power-law transitions | Fast |
| `sync_ssa` | Synchrotron self-absorption | Moderate |
| `sync_ssa_smooth` | SSA + smooth transitions | Moderate |
| `sync_dnp` | Deep Newtonian phase | Moderate |
| `sync_ssc` | Synchrotron self-Compton | Slowest |

## Run the simulation

```python
det = DetectionCriteria(min_detections=2, snr_threshold=5.0)

pipe = SimulationPipeline(
    survey,
    [grb_pop],
    {"Afterglow": model},
    det,
    n_transients=100,
    seed=42,
)

result = pipe.run()
print(result)
```

## What happens inside BlastwaveModel

For each GRB transient instance, the model:

1. **Builds a top-hat jet** on a 128-cell arcsinh-spaced angular grid clustered near \(\theta_c\). The grid resolves the jet core and extends to \(\pi\) for counter-jet emission.

2. **Solves relativistic hydrodynamics** via PDE spreading (finite-volume Godunov scheme with CFL-limited RK2). The forward shock sweeps up ISM material while lateral spreading redistributes energy across cells.

3. **Solves the reverse shock** with coupled forward-reverse dynamics, including magnetization effects.

4. **Computes afterglow radiation** at each observation time:
    - Configures synchrotron microphysics (\(\epsilon_e, \epsilon_B, p\)) for both forward and reverse shocks
    - Evaluates the Equal Arrival Time Surface (EATS) integral to compute specific luminosity \(L_\nu\)
    - Converts to flux density and AB magnitude with host + MW extinction

### Multi-band advantage with Rubin

Because the blastwave model computes flux at arbitrary frequencies, Rubin's 6-band coverage provides:

- **Spectral slope** --- the afterglow synchrotron spectrum \(F_\nu \propto \nu^{-(p-1)/2}\) above the cooling frequency produces a characteristic red-to-blue color evolution
- **Break frequencies** --- the cooling break \(\nu_c\) and self-absorption frequency \(\nu_a\) may pass through the optical bands during the afterglow evolution
- **Color selection** --- multi-band detections enable color-based rejection of contaminants (SNe, asteroids)

## Performance considerations

Each GRB requires ~1--5 seconds of computation (hydro solve + EATS integration). The model is fully parallelized via Rayon:

```bash
# Use all cores (default)
python run_sim.py

# Limit to 8 threads
RAYON_NUM_THREADS=8 python run_sim.py
```

For large samples:

| \(N_\mathrm{transients}\) | Cores | Approx. time |
|--------------------------|-------|-------------|
| 100 | 8 | ~1 min |
| 1,000 | 32 | ~5 min |
| 10,000 | 64 | ~30 min |

## Combined kilonova + afterglow simulation

Run both transient types against Rubin simultaneously:

```python
from survey_sim import (
    SurveyStore, KilonovaPopulation, GrbPopulation,
    MetzgerKNModel, BlastwaveModel,
    DetectionCriteria, SimulationPipeline,
)

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db")

populations = [
    KilonovaPopulation(rate=1000.0, z_max=0.3, peak_abs_mag=-16.0),
    GrbPopulation("GRB_afterglows_argus.csv", rate=1.0, z_max=6.0),
]

models = {
    "Kilonova": MetzgerKNModel(),
    "Afterglow": BlastwaveModel(
        radiation_model="sync_ssa_smooth",
        band_frequencies={
            "u": 8.5e14, "g": 6.3e14, "r": 4.8e14,
            "i": 3.9e14, "z": 3.3e14, "y": 3.0e14,
        },
    ),
}

det = DetectionCriteria(min_detections=2, snr_threshold=5.0)

pipe = SimulationPipeline(
    survey, populations, models, det,
    n_transients=1000, seed=42,
)
result = pipe.run()

for rs in result.rate_summaries:
    print(rs)
```
