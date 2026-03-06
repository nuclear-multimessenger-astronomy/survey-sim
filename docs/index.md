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
| Kilonovae | `KilonovaPopulation`, `FixedMetzgerKilonovaPopulation`, `Bu2026KilonovaPopulation`, `FixedBu2026KilonovaPopulation` | `MetzgerKNModel` (blackbody), fiestaEM surrogate |
| Type Ia SNe | `SupernovaIaPopulation` | SALT3 (fiesta/JAX) |
| Type II SNe | `SupernovaIIPopulation` | Villar parametric |
| TDEs | `TdePopulation` | TDE parametric |
| GRB Afterglows | `GrbPopulation`, `OnAxisGrbPopulation`, `OffAxisGrbPopulation` | `BlastwaveModel` (hydro + synchrotron), fiesta surrogate |

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

## Example Results

| Transient | Survey | Expected detections |
|-----------|--------|---------------------|
| [Kilonovae](examples/kilonova.md) | ZTF (3yr) | R_upper ~ 700–2200 Gpc⁻³ yr⁻¹ |
| [Kilonovae](examples/kilonova.md) | Rubin (10yr) | ~10 (BNS+NSBH) |
| [GRB Afterglows](examples/grb_afterglow.md) | ZTF (1yr) | ~10 on-axis, ~1 orphan |
| [GRB Afterglows](examples/grb_afterglow.md) | Rubin (10yr) | ~49 (44 on-axis + 5 orphan) |
| [Type Ia SNe](examples/snia.md) | ZTF DR2 (2.8yr) | 3,550 (observed: 3,628) |
| [Type Ia SNe](examples/snia.md) | Rubin (10yr) | ~1.4M (142K/yr) |

## Getting Started

See the [Installation](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md) guides.
