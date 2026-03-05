# survey-sim

High-performance survey simulation framework for transient astronomy.

A Rust library with Python bindings for simulating the detection of astrophysical transients by wide-field time-domain surveys. Generates synthetic transient populations, evaluates multi-band lightcurves, and applies realistic detection criteria to compute detection efficiencies and volumetric rates.

## Key Features

- **Multi-survey support** --- Rubin LSST (OpSim), ZTF, Argus Array, and custom instruments via YAML
- **Physics-based lightcurves** --- parametric models (Metzger KN, Bazin, Villar) and full relativistic blast wave afterglow simulation via [blastwave](https://github.com/nuclear-multimessenger-astronomy/blastwave)
- **Python surrogate models** --- wrap any Python model with a `.predict()` method (e.g., fiestaEM)
- **3-phase pipeline** --- spatial matching, lightcurve evaluation, detection criteria, all with Rayon parallelism
- **Efficiency grids** --- N-dimensional binning of detection efficiency vs. redshift, sky position, and intrinsic parameters
- **Rate recovery** --- extrapolate from Monte Carlo efficiency to all-sky volumetric detection rates

## Supported Transients

| Type | Population | Lightcurve Model |
|------|-----------|-----------------|
| Kilonovae | `KilonovaPopulation`, `Bu2026KilonovaPopulation` | `MetzgerKNModel`, fiestaEM surrogate |
| Type Ia SNe | `SupernovaIaPopulation` | Bazin parametric |
| Type II SNe | `SupernovaIIPopulation` | Villar parametric |
| TDEs | `TdePopulation` | TDE parametric |
| GRB Afterglows | `GrbPopulation` | `BlastwaveModel` (hydro + synchrotron) |

## Quick Example

```python
from survey_sim import (
    SurveyStore, KilonovaPopulation, MetzgerKNModel,
    DetectionCriteria, SimulationPipeline,
)

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db")
kn_pop = KilonovaPopulation(rate=1000.0, z_max=0.3)
model = MetzgerKNModel()
det = DetectionCriteria(min_detections=2, snr_threshold=5.0)

pipe = SimulationPipeline(survey, [kn_pop], {"Kilonova": model}, det, n_transients=10000)
result = pipe.run()
print(result)
# SimulationResult(n_simulated=10000, n_detected=265, efficiency=0.0265)
```

## Getting Started

See the [Installation](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md) guides.
