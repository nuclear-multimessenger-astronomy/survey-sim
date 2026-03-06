# ZTF SN Ia DR2 Validation

This example reproduces the ZTF Bright Transient Survey (BTS) SN Ia DR2 sample
from [Rigault et al. (2024, arXiv:2409.04650)](https://arxiv.org/abs/2409.04650),
which found **3628 spectroscopically classified SNe Ia** observed by ZTF between
March 2018 and December 2020.

We use the SALT3 spectral template model (via [fiesta](https://github.com/nuclear-multimessenger-astronomy/fiesta)/JAX)
with population parameters from [skysurvey](https://github.com/skysurvey/skysurvey)
and the ZTF boom-pipeline survey observations.

## Reference parameters

From Rigault et al. (2024) and skysurvey:

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Volumetric rate | \(2.35 \times 10^4\) Gpc\(^{-3}\) yr\(^{-1}\) | Perley (2020) |
| Stretch \(x_1\) | Bimodal Gaussian | Nicolas (2021) |
| Color \(c\) | Intrinsic + dust | Ginolin (2024) |
| Absolute magnitude \(M_B\) | \(-19.3\) | Tripp relation |
| Standardization | \(\alpha = -0.14\), \(\beta = 3.15\) | Tripp (1998) |

## Setup

```python
import sys
sys.path.insert(0, "/path/to/survey-sim/python")
sys.path.insert(0, "/path/to/fiestaEM/src")
import survey_sim.gpu_setup  # auto-configures JAX for GPU

import math
import glob

from survey_sim import (
    SurveyStore,
    SupernovaIaPopulation,
    DetectionCriteria,
    SimulationPipeline,
)
from survey_sim.salt3_model import FiestaSALT3Model
```

## Load the ZTF survey

The ZTF boom-pipeline HDF5 files contain per-observation metadata (MJD, band,
five-sigma depth, sky position) indexed with HEALPix:

```python
boom_files = sorted(glob.glob("/path/to/ztf_boom/*.h5"))

# DR2 covers 2018-03 to 2020-12
dr2_files = [f for f in boom_files if any(
    f"ztf_{ym}" in f for ym in
    [f"{y:04d}{m:02d}" for y in range(2018, 2021) for m in range(1, 13)]
)]

survey = SurveyStore.from_ztf_boom(dr2_files, nside=64)
print(f"Observations: {survey.n_observations}")  # ~460K
print(f"Duration: {survey.duration_years:.2f} years")  # ~2.84 yr
print(f"Bands: {survey.bands}")  # ['g', 'r', 'i']
```

## Define the SN Ia population

```python
pop = SupernovaIaPopulation(
    rate=23500.0,       # Gpc^-3 yr^-1 (Perley 2020)
    z_max=0.12,         # ZTF DR2 redshift cut
    peak_abs_mag=-19.3, # Tripp M_B
)
```

The population draws each SN Ia with:

- Redshift from \(dN/dz \propto dV/dz\) up to \(z = 0.12\)
- Stretch \(x_1\): bimodal Gaussian (Nicolas 2021 parameters)
- Color \(c\): intrinsic + dust model (Ginolin 2024)
- Random sky position and explosion time within the survey window

## SALT3 model

The `FiestaSALT3Model` wraps fiesta's JAX-native SALT3 implementation. It computes
`log10(x0)` from the Tripp standardization relation:

\[
m_B = M_B + \mu(z) - \alpha x_1 + \beta c
\]

\[
\log_{10}(x_0) = -\frac{m_B - 10.682}{2.5}
\]

```python
model = FiestaSALT3Model(filters=["g", "r", "i"])
model.warm_up(z_max=0.12, dz=0.01, batch_size=1024)
```

The `warm_up()` call pre-compiles the JAX JIT for each redshift bin (rounded to
0.01) using both scalar and `vmap` paths, then runs a dummy batch to fully warm
the GPU execution pipeline. This takes ~100s but eliminates first-call latency.

### Batch evaluation

For efficiency, the model uses three levels of optimization:

1. **Redshift binning**: Group transients by \(\Delta z = 0.01\) bins, sharing the
   same SALT3 model instance per bin
2. **JAX vmap**: Vectorized evaluation within each bin using fixed-size padded arrays
   (avoids JIT recompilation on shape changes)
3. **Columnar array passing**: Rust passes flat numpy arrays across the PyO3 boundary
   instead of per-transient Python dicts, and receives flat results back

This processes ~66K transients in ~20s (10s JAX GPU + 3s numpy interpolation + 5s
Rust array handling).

## Detection criteria

The DR2 selection is approximated by combining photometric quality cuts with
a magnitude-dependent spectroscopic completeness function:

### Photometric quality cuts

From Rigault et al. (2024):

| Criterion | Value | Purpose |
|-----------|-------|---------|
| SNR threshold | \(\geq 5\sigma\) | Reliable photometry |
| Min detections | \(\geq 7\) total | Well-sampled light curve |
| Min bands | \(\geq 2\) (g + r) | Color information |
| Min per band | \(\geq 3\) | Per-band SALT2 fitting |
| Pre-peak detections | \(\geq 1\) | Rising phase coverage |
| Post-peak detections | \(\geq 3\) | Decline phase coverage |
| Phase range | \(\geq 30\) days | Sufficient temporal baseline |
| Time baseline | \(\geq 24\) hours | Multi-night confirmation |
| Galactic latitude | \(\lvert b \rvert > 15°\) | Exclude galactic plane |

### BTS spectroscopic completeness

The ZTF DR2 sample is dominated by the Bright Transient Survey (BTS; Perley et al. 2020,
Fremling et al. 2020), which has magnitude-dependent classification completeness:

| Peak mag | BTS completeness |
|----------|-----------------|
| 18.0     | 97%             |
| 18.5     | 93%             |
| 19.0     | 75%             |

We model this as a logistic function applied probabilistically to each detected transient:

\[
P(\text{classified}) = \frac{1}{1 + e^{k \cdot (m_\text{peak} - m_0)}}
\]

with \(k = 2.378\) and \(m_0 = 19.9\), which gives:

| Peak mag | Model completeness |
|----------|--------------------|
| 18.0     | 99%                |
| 18.5     | 97%                |
| 19.0     | 89%                |
| 19.5     | 72%                |
| 20.0     | 44%                |

!!! note "BTS + non-BTS contributions"
    The \(m_0 = 19.9\) midpoint is slightly fainter than a pure BTS fit (\(m_0 \approx 19.5\))
    because 21% of DR2 targets come from programs other than BTS, which contribute
    additional fainter classifications. The combined model reproduces the DR2 count
    within ~2%.

```python
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=5.0,
    min_detections=7,
    min_detections_primary=7,
    min_bands=2,
    min_per_band=3,
    max_timespan_days=100.0,
    min_time_separation_hours=24.0,
    require_fast_transient=False,
    min_rise_rate=0.0,
    min_fade_rate=0.0,
    min_pre_peak_detections=1,
    min_post_peak_detections=3,
    min_phase_range_days=30.0,
    min_galactic_lat=15.0,
    spectroscopic_completeness_k=2.378,
    spectroscopic_completeness_m0=19.90,
)
```

## Run the pipeline

```python
N = 100000
pipeline = SimulationPipeline(
    survey=survey,
    populations=[pop],
    models={"SNIa": model},
    detection=det,
    n_transients=N,
    seed=42,
)
result = pipeline.run()
```

## Results

```
Phase 1 spatial match: 1.2s, 66546 matched of 100000 generated
Phase 2 lightcurve eval: 20.1s, 66546 evaluations
Phase 3 detection: 1.1s

Simulated: 100000
Detected:  20465
Efficiency: 0.2046 (20.46%)
```

### Expected counts

```python
duration = survey.duration_years  # 2.84 yr
z_max = 0.12
d_c_mpc = 509.0  # comoving distance at z=0.12
V_c = (4.0/3.0) * math.pi * (d_c_mpc / 1000.0)**3  # 0.55 Gpc^3

f_sky = 0.47  # ZTF sky coverage
R = 23500.0   # Gpc^-3 yr^-1
eff = result.n_detected / result.n_simulated

N_total = R * V_c * f_sky * duration  # ~17,326
N_detected = N_total * eff            # ~3,546
```

| Quantity | Value |
|----------|-------|
| Comoving volume (\(z < 0.12\)) | 0.55 Gpc\(^3\) |
| ZTF sky fraction | 47% |
| Survey duration | 2.84 yr |
| Total SNe Ia in volume | ~17,300 |
| Detection efficiency | 20.5% |
| **Expected detections** | **~3,550** |
| **ZTF DR2 actual** | **3,628** |
| **Agreement** | **2.3%** |

## Effect of individual cuts

The table below shows how each criterion contributes to the overall efficiency
(cumulative, starting from photometric detection only):

| Criteria added | Efficiency | Expected | Notes |
|----------------|-----------|----------|-------|
| Photometric only (SNR \(\geq 5\), \(\geq 2\) det) | ~51% | ~8,800 | Raw detection |
| + Quality cuts (7 det, 2 bands, phase, etc.) | ~22% | ~3,810 | Light curve quality |
| + BTS completeness | ~20.5% | ~3,550 | Spectroscopic classification |

## Requirements

- **GPU**: NVIDIA GPU with CUDA (for JAX-accelerated SALT3)
- **Python packages**: `jax`, `jaxlib`, `fiesta`, `jax_supernovae`, `sncosmo`
- **Data**: ZTF boom-pipeline HDF5 files (available from survey-sim-data HuggingFace repo)
- **Module loads** (HPC): `module load gcc/13.2.0 python/3.11.5 openmpi/4.1.6 hdf5/1.14.3 cuda/12.8.0`

!!! tip "Warm-up time"
    The SALT3 JIT compilation and GPU warm-up takes ~100s on first run. Subsequent
    `pipeline.run()` calls in the same Python session reuse the compiled kernels and
    complete Phase 2 in ~20s for 66K evaluations.
