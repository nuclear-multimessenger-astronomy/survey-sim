# Kilonova Detection Efficiency

This example computes kilonova detection efficiency with the Rubin LSST 10-year baseline survey.

## Setup

```python
from survey_sim import (
    SurveyStore,
    KilonovaPopulation,
    MetzgerKNModel,
    DetectionCriteria,
    SimulationPipeline,
)
```

## Load the survey

```python
survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db")
```

This loads ~2 million observations from the OpSim database: 6 bands (ugrizy), 10-year duration.

## Define the population

```python
kn_pop = KilonovaPopulation(
    rate=1000.0,        # Gpc^-3 yr^-1
    z_max=0.3,          # maximum redshift
    peak_abs_mag=-16.0, # central peak absolute magnitude
)
```

Each transient is drawn with:

- Redshift from \(dN/dz \propto dV/dz\)
- Ejecta mass log-uniform in \([10^{-3}, 0.1]\) M\(_\odot\)
- Ejecta velocity Gaussian \(\mathcal{N}(0.2c, 0.05c)\)
- Opacity log-uniform in \([0.5, 30]\) cm\(^2\)/g
- Random sky position and explosion time

## Configure the model and detection

```python
model = MetzgerKNModel(peak_abs_mag=-16.0)

det = DetectionCriteria(
    min_detections=2,
    snr_threshold=5.0,
)
```

## Run the simulation

```python
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
# SimulationResult(n_simulated=10000, n_detected=265, efficiency=0.0265)
```

## Interpret the results

```python
for rs in result.rate_summaries:
    print(f"Type: {rs.transient_type}")
    print(f"  Volumetric rate: {rs.volumetric_rate:.0f} Gpc^-3/yr")
    print(f"  Detection efficiency: {rs.overall_efficiency:.4f}")
    print(f"  Detections/year: {rs.detections_per_year:.1f}")
```

## Scaling to larger samples

For production runs, increase `n_transients` for better statistics:

```python
pipe = SimulationPipeline(
    survey, [kn_pop], {"Kilonova": model}, det,
    n_transients=100000,  # 100K for sub-percent efficiency precision
    seed=42,
)
```

The parametric Metzger model is fast (~microseconds per evaluation), so 100K transients completes in minutes on a modern workstation.

## Using the Bu2026 surrogate

For more physical kilonova lightcurves, use the Bu2026 two-component model with fiestaEM:

```python
from survey_sim import Bu2026KilonovaPopulation
from fiesta.inference.lightcurve_model import SurrogateModel

bu_pop = Bu2026KilonovaPopulation(rate=1000.0, z_max=0.3)
fiesta_model = SurrogateModel("Bu2026ts")

pipe = SimulationPipeline(
    survey, [bu_pop], {"Kilonova": fiesta_model}, det,
    n_transients=1000,  # smaller sample (Python model is slower)
)
result = pipe.run()
```

!!! warning "Performance note"
    The fiestaEM surrogate requires the Python GIL, so lightcurve evaluation runs sequentially. Use smaller sample sizes (~1000) or run on a single node.
