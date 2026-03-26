# survey-sim

A high-performance survey simulation framework for transient astronomy, written in Rust with Python bindings. Simulates the detection of astrophysical transients (kilonovae, supernovae, TDEs, GRB afterglows) by wide-field time-domain surveys, computing detection efficiencies and volumetric rates. Detailed documentation and examples are available [here](https://nuclear-multimessenger-astronomy.github.io/survey-sim).

## Features

### Survey Ingestion
- **Multi-format loaders** --- OpSim SQLite (Rubin LSST), Parquet (Argus Array), HDF5, and CSV
- **HEALPix spatial indexing** --- fast cone-search observation matching via `cdshealpix`
- **Built-in instruments** --- Rubin LSST (6 bands, 9.6 deg²), ZTF (3 bands, 47 deg²), Argus Array (1 band, 8000 deg²)
- **YAML-configurable** --- define custom instruments with per-band depth, extinction coefficients, and observing constraints

### Transient Populations
- **Kilonovae** --- Metzger (2017) parametric model and Bu2026 two-component surrogate (via [fiestaEM](https://github.com/nuclear-multimessenger-astronomy/fiesta))
- **Supernovae** --- Type Ia (SALT2-like), Type II (plateau), Type Ibc
- **Tidal disruption events** --- luminosity-function sampling
- **GRB afterglows** --- full relativistic blast wave simulation via [blastwave](https://github.com/nuclear-multimessenger-astronomy/blastwave), with forward + reverse shock synchrotron radiation from pre-drawn CSV catalogs
- **Extensible** --- implement `PopulationGenerator` trait for custom populations

### Lightcurve Models
- **Parametric** (Rust-native) --- Metzger KN, Bazin, Villar, TDE, Afterglow templates via [lightcurve-fitting](https://github.com/boom-astro/lightcurve-fitting)
- **Blastwave** (Rust-native) --- relativistic hydrodynamics + synchrotron radiation with self-absorption (SSA), PDE lateral spreading, and reverse shock; fully parallel via Rayon
- **Python callback** --- wrap any Python model with a `.predict()` method (e.g., fiestaEM surrogate); runs sequentially due to GIL

### Detection Pipeline
- **3-phase architecture**: spatial matching → lightcurve evaluation → detection criteria
- **Rayon parallelism** --- phases 1 and 3 always parallel; phase 2 parallel for Rust models, sequential for Python/GIL models
- **Configurable criteria** --- minimum detections, SNR thresholds, rise/fade rates, band requirements, timespan limits
- **Pre-built presets** --- ZTFReST fast-transient criteria

### Efficiency & Rates
- **N-dimensional efficiency grids** --- bin detection efficiency vs. redshift, sky position, and intrinsic parameters
- **Volumetric rate recovery** --- extrapolate from efficiency to all-sky detection rates (Gpc⁻³ yr⁻¹)
- **Cadence analysis** --- inter-visit return time statistics per band and all-filter

## Project Structure

```
survey-sim/
├── src/                    # Rust core library
│   ├── cadence/            # Return-time cadence analysis
│   ├── detection/          # Detection criteria and evaluation
│   ├── efficiency/         # N-dimensional efficiency grids, rate recovery
│   ├── lightcurve/         # Lightcurve model trait and implementations
│   │   ├── parametric.rs   #   Parametric models (Metzger, Bazin, Villar)
│   │   ├── blastwave_model.rs  #   GRB afterglow via blastwave crate
│   │   ├── cosmology.rs    #   Flat LCDM cosmology, extinction
│   │   └── python_model.rs #   Python callback bridge
│   ├── population/         # Population generators
│   │   ├── generator.rs    #   KN, SNe, TDE, Afterglow populations
│   │   ├── grb.rs          #   GRB catalog-based population (CSV)
│   │   └── distributions.rs #  Sampling utilities (redshift, sky, time)
│   ├── spatial/            # HEALPix spatial index
│   ├── survey/             # Survey loaders (OpSim, Parquet, HDF5)
│   ├── instrument.rs       # Telescope/detector configuration
│   ├── pipeline.rs         # 3-phase simulation pipeline
│   ├── config.rs           # YAML configuration
│   └── types.rs            # Core types (SkyCoord, Band, TransientInstance)
├── python/                 # PyO3 Python extension
│   ├── src/                #   Rust bindings
│   └── survey_sim/         #   Python package
├── instruments/            # YAML instrument definitions
└── tests/                  # Integration tests
```

## Installation

### Requirements

- Rust toolchain (stable, 1.70+)
- Python >= 3.9
- [maturin](https://github.com/PyO3/maturin)
- HDF5 library (for OpSim/HDF5 survey loading)

### Build

```bash
# Rust library only
cargo build --release

# Python bindings
cd python
maturin develop --release
```

!!! note "HPC / module systems"
    On HPC clusters you may need to load compiler, HDF5, and Python modules first:
    ```bash
    module load gcc/13.2.0 openmpi/4.1.6 hdf5/1.14.3 python/3.11.5
    source /path/to/your/venv/bin/activate
    cd python && maturin develop --release
    ```

### Run tests

```bash
# Rust unit tests
cargo test -p survey-sim --lib

# Integration tests (requires OpSim database)
cargo test -p survey-sim --test test_survey -- --ignored
```

## Quick Start

### Python API

```python
from survey_sim import (
    SurveyStore,
    KilonovaPopulation,
    MetzgerKNModel,
    DetectionCriteria,
    SimulationPipeline,
)

# Load survey observations
survey = SurveyStore.from_rubin("/path/to/baseline_v5.1.1_10yrs.db")

# Define population and lightcurve model
kn_pop = KilonovaPopulation(rate=1000.0, z_max=0.3, peak_abs_mag=-16.0)
kn_model = MetzgerKNModel()

# Detection criteria
det = DetectionCriteria(min_detections=2, snr_threshold=5.0)

# Run pipeline
pipe = SimulationPipeline(
    survey,
    [kn_pop],
    {"Kilonova": kn_model},
    det,
    n_transients=10000,
    seed=42,
)
result = pipe.run()
print(result)
# SimulationResult(n_simulated=10000, n_detected=265, efficiency=0.0265)
```

### GRB Afterglow Simulation

```python
from survey_sim import (
    SurveyStore,
    GrbPopulation,
    BlastwaveModel,
    DetectionCriteria,
    SimulationPipeline,
)

# Load Rubin LSST survey
survey = SurveyStore.from_rubin("/path/to/baseline_v5.1.1_10yrs.db")

# GRB population from pre-drawn catalog
grb_pop = GrbPopulation(
    "/path/to/GRB_afterglows_argus.csv",
    rate=1.0,      # Gpc^-3 yr^-1
    z_max=6.0,
)

# Blastwave afterglow model (Rust-native, fully parallel)
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

### Python Surrogate Model (fiestaEM)

```python
from survey_sim import (
    SurveyStore,
    Bu2026KilonovaPopulation,
    SimulationPipeline,
    DetectionCriteria,
)
from fiesta.inference.lightcurve_model import SurrogateModel

survey = SurveyStore.from_rubin("/path/to/opsim.db")
kn_pop = Bu2026KilonovaPopulation(rate=1000.0, z_max=0.3)
fiesta_model = SurrogateModel("Bu2026ts")  # Python .predict() interface

det = DetectionCriteria(min_detections=2, snr_threshold=5.0)
pipe = SimulationPipeline(
    survey,
    [kn_pop],
    {"Kilonova": fiesta_model},
    det,
    n_transients=1000,
)
result = pipe.run()
```

## Available Populations

| Class | Type | Parameters | Source |
|-------|------|------------|--------|
| `KilonovaPopulation` | Kilonova | `mej`, `vej`, `kappa` | Random sampling |
| `Bu2026KilonovaPopulation` | Kilonova | 7 Bu2026 params | Uniform in training bounds |
| `FixedBu2026KilonovaPopulation` | Kilonova | Fixed Bu2026 params | User-specified |
| `SupernovaIaPopulation` | SNIa | SALT2 `x1`, `c` | Gaussian sampling |
| `SupernovaIIPopulation` | SNII | Plateau duration/slope | Gaussian sampling |
| `TdePopulation` | TDE | `m_bh`, `m_star` | Log-uniform / Gaussian |
| `GrbPopulation` | Afterglow | Full blastwave params | CSV catalog sampling |

## Available Lightcurve Models

| Class | Type | Engine | Parallel |
|-------|------|--------|----------|
| `MetzgerKNModel` | Parametric | lightcurve-fitting | Yes (Rayon) |
| `BlastwaveModel` | Physics-based | blastwave hydro+afterglow | Yes (Rayon) |
| Python `.predict()` | Callback | Any Python model | No (GIL) |

## Controlling Parallelism

survey-sim uses [Rayon](https://github.com/rayon-rs/rayon) for parallel lightcurve evaluation and detection. By default it uses all available cores:

```bash
export RAYON_NUM_THREADS=4  # limit to 4 threads
```

## License

See [LICENSE](LICENSE) for details.
