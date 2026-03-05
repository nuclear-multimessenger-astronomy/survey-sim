# Quick Start

## Core workflow

1. **Load a survey** --- read observation metadata (sky positions, times, bands, depths) into a spatially-indexed store
2. **Define populations** --- choose transient types, volumetric rates, and redshift ranges
3. **Choose lightcurve models** --- map each transient type to a model that computes apparent magnitudes
4. **Set detection criteria** --- minimum detections, SNR thresholds, cadence requirements
5. **Run the pipeline** --- generate transients, match to observations, evaluate lightcurves, apply detection cuts

## Loading a survey

```python
from survey_sim import SurveyStore

# Rubin LSST from OpSim database
survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db")

# Argus Array from Parquet files
survey = SurveyStore.from_argus([
    "argussim_hpx_6131.parquet",
    "argussim_hpx_6132.parquet",
])
```

## Kilonova detection efficiency

```python
from survey_sim import (
    SurveyStore, KilonovaPopulation, MetzgerKNModel,
    DetectionCriteria, SimulationPipeline,
)

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db")

# Population: 1000 Gpc^-3 yr^-1, z < 0.3
kn_pop = KilonovaPopulation(rate=1000.0, z_max=0.3, peak_abs_mag=-16.0)

# Parametric lightcurve model (Rust-native, parallel)
model = MetzgerKNModel()

# At least 2 detections at 5-sigma
det = DetectionCriteria(min_detections=2, snr_threshold=5.0)

pipe = SimulationPipeline(
    survey,
    [kn_pop],
    {"Kilonova": model},
    det,
    n_transients=10000,
    seed=42,
)
result = pipe.run()
print(result)
```

## GRB afterglow detection

The `BlastwaveModel` runs a full relativistic blast wave simulation for each GRB, computing synchrotron radiation from forward and reverse shocks:

```python
from survey_sim import (
    SurveyStore, GrbPopulation, BlastwaveModel,
    DetectionCriteria, SimulationPipeline,
)

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db")

# Pre-drawn GRB parameters from CSV catalog
grb_pop = GrbPopulation(
    "GRB_afterglows_argus.csv",
    rate=1.0,      # Gpc^-3 yr^-1
    z_max=6.0,
)

# Blastwave model: synchrotron + self-absorption, Rubin ugrizy bands
model = BlastwaveModel(
    radiation_model="sync_ssa_smooth",
    band_frequencies={
        "u": 8.5e14, "g": 6.3e14, "r": 4.8e14,
        "i": 3.9e14, "z": 3.3e14, "y": 3.0e14,
    },
)

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

!!! note "Blastwave performance"
    Each GRB requires solving PDE hydrodynamics on a 128-cell angular grid followed by EATS integration at each observation time. This is compute-intensive (~seconds per GRB) but fully parallelized via Rayon. Set `n_transients` accordingly.

## Using a Python surrogate model

Any Python object with a `.predict(params) -> (times, {band: mags})` interface can be used as a lightcurve model:

```python
from fiesta.inference.lightcurve_model import SurrogateModel

fiesta_model = SurrogateModel("Bu2026ts")

pipe = SimulationPipeline(
    survey,
    [Bu2026KilonovaPopulation(rate=1000.0, z_max=0.3)],
    {"Kilonova": fiesta_model},
    det,
    n_transients=1000,
)
result = pipe.run()
```

!!! warning "GIL constraint"
    Python callback models run sequentially (one transient at a time) because they require the Python GIL. For large-scale simulations, prefer Rust-native models like `MetzgerKNModel` or `BlastwaveModel`.

## Multi-population simulation

Run multiple transient types simultaneously:

```python
from survey_sim import (
    SurveyStore, KilonovaPopulation, SupernovaIaPopulation,
    TdePopulation, MetzgerKNModel, DetectionCriteria,
    SimulationPipeline,
)

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db")

populations = [
    KilonovaPopulation(rate=1000.0, z_max=0.3, peak_abs_mag=-16.0),
    SupernovaIaPopulation(rate=30000.0, z_max=1.0, peak_abs_mag=-19.3),
    TdePopulation(rate=100.0, z_max=0.5, peak_abs_mag=-20.0),
]

# Each population's TransientType maps to a model key
models = {
    "Kilonova": MetzgerKNModel(),
    "SNIa": MetzgerKNModel(),  # placeholder
    "TDE": MetzgerKNModel(),   # placeholder
}

det = DetectionCriteria(min_detections=2, snr_threshold=5.0)

pipe = SimulationPipeline(
    survey, populations, models, det,
    n_transients=10000, seed=42,
)
result = pipe.run()

for rs in result.rate_summaries:
    print(rs)
```

## Detection criteria

```python
from survey_sim import DetectionCriteria

# Standard criteria
det = DetectionCriteria(
    min_detections=2,       # minimum number of detections
    snr_threshold=5.0,      # primary SNR threshold
)
```
