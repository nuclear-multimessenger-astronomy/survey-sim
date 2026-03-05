# Python API

All classes are imported from the `survey_sim` package:

```python
from survey_sim import (
    SurveyStore,
    KilonovaPopulation, Bu2026KilonovaPopulation,
    FixedBu2026KilonovaPopulation,
    SupernovaIaPopulation, SupernovaIIPopulation,
    TdePopulation, GrbPopulation,
    MetzgerKNModel, BlastwaveModel,
    DetectionCriteria, DetectionResult,
    SimulationPipeline, SimulationResult, RateSummary,
)
```

---

## Survey

### `SurveyStore`

Spatially-indexed observation store. Constructed via class methods:

```python
# Rubin LSST from OpSim database
survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db")

# Argus Array from Parquet files
survey = SurveyStore.from_argus([
    "argussim_hpx_6131.parquet",
    "argussim_hpx_6132.parquet",
])

# ZTF from HDF5
survey = SurveyStore.from_ztf("ztf_fields.hdf5")
```

---

## Populations

All population classes share a common interface. They are passed as a list to `SimulationPipeline`.

### `KilonovaPopulation`

```python
KilonovaPopulation(rate=1000.0, z_max=0.3, peak_abs_mag=-16.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | 1000.0 | Volumetric rate (Gpc\(^{-3}\) yr\(^{-1}\)) |
| `z_max` | float | 0.3 | Maximum redshift |
| `peak_abs_mag` | float | -16.0 | Peak absolute magnitude |

### `Bu2026KilonovaPopulation`

```python
Bu2026KilonovaPopulation(rate=1000.0, z_max=0.3)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | 1000.0 | Volumetric rate |
| `z_max` | float | 0.3 | Maximum redshift |

### `FixedBu2026KilonovaPopulation`

```python
FixedBu2026KilonovaPopulation(
    log10_mej_dyn, v_ej_dyn, ye_dyn,
    log10_mej_wind, v_ej_wind, ye_wind,
    inclination_em,
    rate=1000.0, z_max=0.3,
)
```

All ejecta parameters are required positional arguments.

### `SupernovaIaPopulation`

```python
SupernovaIaPopulation(rate=30000.0, z_max=1.0, peak_abs_mag=-19.3)
```

### `SupernovaIIPopulation`

```python
SupernovaIIPopulation(rate=70000.0, z_max=0.5, peak_abs_mag=-17.0)
```

### `TdePopulation`

```python
TdePopulation(rate=100.0, z_max=0.5, peak_abs_mag=-20.0)
```

### `GrbPopulation`

```python
GrbPopulation(csv_path, rate=1.0, z_max=6.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | str | *required* | Path to GRB parameter catalog CSV |
| `rate` | float | 1.0 | Volumetric rate (Gpc\(^{-3}\) yr\(^{-1}\)) |
| `z_max` | float | 6.0 | Maximum redshift |

The CSV must contain columns: `z`, `d_L`, `Eiso`, `Gamma_0`, `thv`, `logn0`, `logepse`, `logepsB`, `logthc`, `p`, `av`, `p_rvs`, `logepse_rvs`, `logepsB_rvs`, `peak_mag`.

---

## Lightcurve Models

Models are passed as a dictionary mapping `TransientType` name to model object: `{"Kilonova": model, "Afterglow": model}`.

### `MetzgerKNModel`

```python
MetzgerKNModel(peak_abs_mag=-16.0)
```

Parametric kilonova model using `lightcurve-fitting`. Rust-native, fully parallel.

### `BlastwaveModel`

```python
BlastwaveModel(radiation_model="sync_ssa_smooth", band_frequencies=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radiation_model` | str | `"sync_ssa_smooth"` | Synchrotron radiation model |
| `band_frequencies` | dict or None | `{"g": 6.3e14}` | Band name → frequency (Hz) |

Available radiation models: `sync`, `sync_smooth`, `sync_ssa`, `sync_ssa_smooth`, `sync_dnp`, `sync_ssc`.

Rust-native, fully parallel. Runs relativistic hydrodynamics + EATS afterglow computation per transient.

### Python callback models

Any Python object with a `.predict(params)` method:

```python
pipe = SimulationPipeline(
    survey, [pop], {"Kilonova": my_python_model}, det
)
```

The `.predict()` method receives a dict with model parameters plus `_obs_times_mjd`, `_obs_bands`, `_t_exp`, `redshift`, and `luminosity_distance`. Must return `(times, {band_name: magnitudes})`.

---

## Detection

### `DetectionCriteria`

```python
DetectionCriteria(
    min_detections=2,
    snr_threshold=5.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_detections` | int | 2 | Minimum number of detections |
| `snr_threshold` | float | 5.0 | Primary SNR threshold (sigma) |

---

## Pipeline

### `SimulationPipeline`

```python
SimulationPipeline(
    survey,         # SurveyStore
    populations,    # list of population objects
    models,         # dict: TransientType name → model
    detection,      # DetectionCriteria
    n_transients=100000,
    seed=42,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `survey` | SurveyStore | *required* | Survey observation store |
| `populations` | list | *required* | List of population generators |
| `models` | dict | *required* | Map of type name to lightcurve model |
| `detection` | DetectionCriteria | *required* | Detection criteria |
| `n_transients` | int | 100000 | Number of transients per population |
| `seed` | int | 42 | RNG seed for reproducibility |

#### `run() -> SimulationResult`

Execute the 3-phase pipeline. Returns a `SimulationResult`.

### `SimulationResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_simulated` | int | Total transients simulated |
| `n_detected` | int | Total transients detected |
| `rate_summaries` | list[RateSummary] | Per-population rate summaries |

### `RateSummary`

| Attribute | Type | Description |
|-----------|------|-------------|
| `transient_type` | str | Population type name |
| `volumetric_rate` | float | Input rate (Gpc\(^{-3}\) yr\(^{-1}\)) |
| `overall_efficiency` | float | Detection efficiency \(N_\mathrm{det}/N_\mathrm{sim}\) |
| `detections_per_year` | float | Extrapolated all-sky rate |
| `detections_total` | float | Total detections over survey duration |
