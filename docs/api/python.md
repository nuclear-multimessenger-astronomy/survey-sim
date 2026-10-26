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

### `SupernovaIIbPopulation`

```python
SupernovaIIbPopulation(rate=1200.0, z_max=0.15, peak_abs_mag=-16.5)
```

### `SupernovaIaxPopulation`

```python
SupernovaIaxPopulation(rate=3000.0, z_max=0.20, peak_abs_mag=-15.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | 3000.0 | Volumetric rate (Gpc\(^{-3}\) yr\(^{-1}\)) |
| `z_max` | float | 0.20 | Maximum redshift |
| `peak_abs_mag` | float | -15.0 | Peak absolute magnitude |

### `CsmInteractionPopulation`

```python
CsmInteractionPopulation(rate=250.0, z_max=0.25, peak_abs_mag=-17.5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | 250.0 | Volumetric rate (Gpc\(^{-3}\) yr\(^{-1}\)) |
| `z_max` | float | 0.25 | Maximum redshift |
| `peak_abs_mag` | float | -17.5 | Peak absolute magnitude |

### `ShockedCocoonPopulation`

```python
ShockedCocoonPopulation(rate=30.0, z_max=0.30, peak_abs_mag=-16.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | 30.0 | Volumetric rate (Gpc\(^{-3}\) yr\(^{-1}\)) |
| `z_max` | float | 0.30 | Maximum redshift |
| `peak_abs_mag` | float | -16.0 | Peak absolute magnitude |

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

### `FiestaKNModel`

```python
from survey_sim import FiestaKNModel

FiestaKNModel(survey="ztf")   # defaults to ZTF filters ("ztfg", "ztfr", "ztfi")
# or for Rubin LSST:
FiestaKNModel(survey="lsst")  # LSST filters ("lsstu", "lsstg", ...)
```

Two-component kilonova neural network surrogate model via fiestaEM/JAX. Pass `survey="ztf"` or `survey="lsst"` to automatically configure instrument filter passbands.

### `FiestaShockCoolingModel`

```python
from survey_sim import FiestaShockCoolingModel

FiestaShockCoolingModel(survey="ztf")  # automatically uses ZTF filters
```

Analytical shock-cooling light-curve model wrapping fiestaEM's `ShockCoolingModel` (Piro 2021) for Type IIb supernovae. Pass `survey="ztf"` or `survey="lsst"` to configure passbands.

### `FiestaCocoonModel`

```python
from survey_sim import FiestaCocoonModel

FiestaCocoonModel(survey="ztf")
```

Analytical model adapter for relativistic shocked cocoons (Nakar & Piran 2017; Gottlieb et al. 2018). Evaluated via fiestaEM/JAX.

### `FiestaCsmModel`

```python
from survey_sim import FiestaCsmModel

FiestaCsmModel(survey="ztf")
```

Analytical model adapter for dense circumstellar interaction transients (Chatzopoulos et al. 2012; Moriya et al. 2013). Evaluated via fiestaEM/JAX.

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

#### `RateSummary.analyze_imposters()`

```python
rate_summary.analyze_imposters(
    imposter_counts=None,      # list of (name, count, optional_sys_err)
    imposter_summaries=None,   # list of other RateSummary instances
    default_sys_err=0.20,      # fractional systematic error on imposter rates
    followup_budget=None,      # maximum follow-up target capacity
) -> KilonovaImposterAnalysis
```

#### `SimulationResult.analyze_imposters()`

```python
# Multi-transient simulation containing Kilonova + other populations:
result = pipeline.run()

# Analyze contamination directly from the multi-population simulation result:
analysis = result.analyze_imposters(
    signal_type="Kilonova",    # target transient population name
    default_sys_err=0.20,      # fractional systematic uncertainty on imposter rates
    followup_budget=25.0,      # optional maximum target capacity for spectroscopic follow-up
)
```

---

## Imposter Analysis

### `KilonovaImposterAnalysis`

Analyzes sample purity, contamination, false discovery rate (FDR), effective (pure) detection efficiency, follow-up budget dilution, and decontaminated rate recovery under astrophysical imposters (e.g. Type IIb SN shock cooling, FBOTs, flare stars).

Construct from counts, from rate summaries, or directly from an end-to-end `SimulationResult`:

```python
# 1. From a built-in survey preset ("ztf" or "lsst"):
analysis = KilonovaImposterAnalysis.for_survey(
    survey="ztf",
    volumetric_rate=45.0,
    signal_name="BNS Kilonova",
    default_sys_err=0.20,
    followup_budget=20.0,
)

# 2. From an end-to-end multi-population SimulationResult:
analysis = KilonovaImposterAnalysis.from_simulation_result(
    result,
    signal_type="Kilonova",
    default_sys_err=0.20,
    followup_budget=25.0,
)

# 3. From a list of RateSummary objects:
analysis = KilonovaImposterAnalysis.from_rate_summaries(
    result.rate_summaries,
    signal_type="Kilonova",
    default_sys_err=0.20,
)

# 4. From manual counts / forecasts:
analysis = KilonovaImposterAnalysis.from_counts(
    signal_name="BNS Kilonova",
    volumetric_rate=50.0,
    expected_detections=10.0,
    effective_vt=0.20,
    overall_efficiency=0.40,
    imposters=[
        ("Type IIb Shock Cooling", 20.0, 0.20),
        ("FBOT", 10.0, 0.30),
    ],
    followup_budget=25.0,
)

# Serialization compatibility:
from survey_sim.serialization import save_result, load_result

# Save analysis to JSON:
save_result(analysis, "imposter_analysis.json")
restored = load_result("imposter_analysis.json")
```

| Property | Type | Description |
|----------|------|-------------|
| `purity` | float | Sample precision \(N_\mathrm{KN} / (N_\mathrm{KN} + N_\mathrm{imp})\) |
| `false_discovery_rate` | float | FDR \(1 - P\) |
| `imposter_to_signal_ratio` | float | Contamination ratio \(N_\mathrm{imp} / N_\mathrm{KN}\) |
| `pure_efficiency` | float | Effective efficiency \(\epsilon_\mathrm{KN} 	imes P\) |
| `raw_efficiency` | float | Completeness without impurity penalty |
| `f1_score` | float | Harmonic mean of efficiency and purity |
| `significance_fom` | float | Poisson significance \(S / \sqrt{S + B}\) |
| `naive_recovered_rate` | float | Rate assuming zero imposters (biased high) |
| `rate_bias_absolute` | float | Rate overestimation bias \(\Delta R\) |
| `rate_bias_fractional` | float | Fractional bias \(\Delta R / R\) |
| `decontaminated_recovered_rate` | float | Background-subtracted recovered rate |
| `decontaminated_1sig_interval` | tuple[float, float] | 1-sigma interval with background variance |
| `decontaminated_2sig_interval` | tuple[float, float] | 2-sigma interval with background variance |
| `variance_inflation_factor` | float | Variance inflation from imposter noise |
| `uncertainty_inflation_factor` | float | Fractional uncertainty widening \(\sqrt{\mathrm{VIF}}\) |
| `followup_metrics` | dict or None | Follow-up capacity and confirmed event metrics |
