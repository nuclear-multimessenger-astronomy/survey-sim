# Rust Crate API

The `survey-sim` crate is a Rust workspace with two members:

- **`survey-sim`** --- core library (`src/`)
- **`survey-sim-python`** --- PyO3 extension module (`python/`)

## Core Modules

### `types`

Foundational types used throughout the crate.

```rust
pub struct SkyCoord { pub ra: f64, pub dec: f64 }
pub struct Band(pub String);
pub enum TransientType { Kilonova, SupernovaIa, SupernovaII, SupernovaIbc, Tde, Afterglow, Custom }
pub struct Cosmology { pub h: f64, pub omega_m: f64, pub omega_lambda: f64 }
pub struct TransientInstance {
    pub coord: SkyCoord,
    pub z: f64,
    pub d_l: f64,
    pub t_exp: f64,
    pub peak_abs_mag: f64,
    pub transient_type: TransientType,
    pub model_params: HashMap<String, f64>,
    pub mw_extinction_av: f64,
    pub host_extinction_av: f64,
}
```

### `survey`

Survey data loading and spatial-temporal querying.

```rust
pub trait SurveyLoader { fn load(&self) -> Vec<SurveyObservation>; }
pub struct SurveyStore { /* HEALPix-indexed observations */ }
impl SurveyStore {
    pub fn new(observations: Vec<SurveyObservation>, ...) -> Self;
    pub fn query(&self, coord: &SkyCoord, mjd_min: f64, mjd_max: f64) -> Vec<usize>;
    pub fn get(&self, index: usize) -> &SurveyObservation;
}
```

### `population`

Transient population generation.

```rust
pub trait PopulationGenerator: Send + Sync {
    fn generate(&self, n: usize, rng: &mut dyn RngCore) -> Vec<TransientInstance>;
    fn volumetric_rate(&self) -> f64;
    fn transient_type(&self) -> TransientType;
}
```

Implementations: `KilonovaPopulation`, `Bu2026KilonovaPopulation`, `FixedBu2026KilonovaPopulation`, `SupernovaIaPopulation`, `SupernovaIIPopulation`, `SupernovaIbcPopulation`, `TdePopulation`, `AfterglowPopulation`, `GrbPopulation`.

### `lightcurve`

Lightcurve model evaluation.

```rust
pub trait LightcurveModel: Send + Sync {
    fn evaluate(&self, instance: &TransientInstance, times_mjd: &[f64], bands: &[Band])
        -> Result<LightcurveEvaluation>;
    fn requires_gil(&self) -> bool { false }
}

pub struct LightcurveEvaluation {
    pub apparent_mags: HashMap<String, Vec<f64>>,
    pub times_mjd: Vec<f64>,
}
```

Implementations: `ParametricModel`, `BlastwaveModel`.

### `detection`

Detection criteria and evaluation.

```rust
pub struct DetectionCriteria { pub min_detections: usize, pub snr_threshold: f64, /* ... */ }
pub struct DetectionResult { pub detected: bool, pub n_detections: usize, /* ... */ }
pub fn evaluate_detection(eval: &LightcurveEvaluation, obs: &[&SurveyObservation], criteria: &DetectionCriteria) -> DetectionResult;
```

### `pipeline`

End-to-end simulation pipeline.

```rust
pub struct SimulationPipeline { /* survey, populations, models, criteria */ }
pub struct SimulationResult { pub n_simulated: usize, pub n_detected: usize, /* ... */ }
impl SimulationPipeline {
    pub fn run(&self) -> SimulationResult;
}
```

### `efficiency`

N-dimensional efficiency grids and rate recovery.

```rust
pub struct EfficiencyGrid { /* N-dimensional binning */ }
impl EfficiencyGrid {
    pub fn record(&mut self, values: &[f64], detected: bool);
    pub fn efficiency_at(&self, values: &[f64]) -> f64;
    pub fn marginalize_over(&self, axis: usize) -> Vec<(f64, f64)>;
}
```

### `instrument`

Telescope and detector configuration.

```rust
pub struct InstrumentConfig { pub telescope: TelescopeConfig, pub detector: DetectorConfig, /* ... */ }
impl InstrumentConfig {
    pub fn rubin() -> Self;
    pub fn ztf() -> Self;
    pub fn argus() -> Self;
    pub fn from_yaml(path: &str) -> Self;
}
```

### `spatial`

HEALPix spatial indexing.

```rust
pub struct SpatialIndex { /* pixel → observation index mapping */ }
impl SpatialIndex {
    pub fn new(coords: &[(f64, f64)], nside: u32) -> Self;
    pub fn query_cone(&self, ra: f64, dec: f64, radius_deg: f64) -> Vec<usize>;
}
```

### `cadence`

Survey cadence analysis.

```rust
pub struct ReturnTimeAnalysis { pub per_band: HashMap<String, BandCadenceStats>, /* ... */ }
pub struct BandCadenceStats { pub mean: f64, pub median: f64, pub std: f64, /* ... */ }
```

### `lightcurve::cosmology`

Flat LCDM cosmology utilities.

```rust
impl Cosmology {
    pub fn luminosity_distance(&self, z: f64) -> f64;  // Mpc
    pub fn distance_modulus(&self, z: f64) -> f64;
    pub fn comoving_volume(&self, z: f64) -> f64;       // Gpc^3
    pub fn dv_dz(&self, z: f64) -> f64;                 // Gpc^3/sr
}
pub fn extinction_in_band(a_v: f64, band: &str) -> f64;
pub fn k_correction_bolometric(z: f64) -> f64;
```
