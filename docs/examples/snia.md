# Type Ia Supernova Detection

This page summarizes Type Ia supernova detection simulations for both **ZTF** and **Rubin LSST**, using the SALT3 spectral template model via fiesta/JAX.

## Summary of results

| Survey | Duration | z_max | Injections | Efficiency | Expected | Observed |
|--------|----------|-------|-----------|------------|----------|----------|
| **ZTF** (DR2 validation) | 2.84 yr | 0.12 | 100K | 20.5% | 3,550 | 3,628 |
| **Rubin** (photometric) | 10 yr | 0.5 | 100K | 14.9% | 1,415,000 (142K/yr) | — |

The ZTF simulation reproduces the DR2 spectroscopic sample to **2.3% accuracy**. The Rubin prediction of ~142,000 photometric SNe Ia per year is consistent with LSST science book estimates.

---

## Background

Type Ia supernovae are standardizable candles used for precision cosmology. Their lightcurves are parameterized by the SALT2/3 model with stretch (x1), color (c), and overall amplitude (x0), related to peak luminosity via the Tripp standardization:

$$m_B = M_B + \mu(z) - \alpha x_1 + \beta c$$

We use the SALT3 spectral template model via [fiesta](https://github.com/nuclear-multimessenger-astronomy/fiesta)/JAX, which supports JIT compilation and GPU-accelerated batch evaluation via `vmap`.

### Population parameters

From Rigault et al. (2024), Perley (2020), Nicolas (2021), Ginolin (2024):

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Volumetric rate | 23,500 Gpc⁻³ yr⁻¹ | Perley (2020) |
| Stretch x1 | Bimodal Gaussian | Nicolas (2021) |
| Color c | Intrinsic + dust | Ginolin (2024) |
| Absolute magnitude M_B | −19.3 | Tripp relation |
| Standardization | α = −0.14, β = 3.15 | Tripp (1998) |

### SALT3 model

The `FiestaSALT3Model` wraps fiesta's JAX-native SALT3 implementation. It computes `log10(x0)` from the Tripp standardization relation:

$$\log_{10}(x_0) = -\frac{m_B - 10.682}{2.5}$$

For efficiency, the model uses three levels of optimization:

1. **Redshift binning**: Group transients by $\Delta z = 0.01$ bins, sharing the same SALT3 model instance per bin
2. **JAX vmap**: Vectorized evaluation within each bin using fixed-size padded arrays (avoids JIT recompilation on shape changes)
3. **Columnar array passing**: Rust passes flat numpy arrays across the PyO3 boundary instead of per-transient Python dicts, and receives flat results back

---

## ZTF DR2 validation

### Setup

Reproduces the ZTF Bright Transient Survey SN Ia DR2 sample from [Rigault et al. (2024)](https://arxiv.org/abs/2409.04650): **3,628 spectroscopically classified SNe Ia** observed March 2018 – December 2020.

```python
from survey_sim import SupernovaIaPopulation, DetectionCriteria, SimulationPipeline, load_ztf_survey
from survey_sim.salt3_model import FiestaSALT3Model

survey = load_ztf_survey(start="201803", end="202012", nside=64)
pop = SupernovaIaPopulation(rate=23500.0, z_max=0.12, peak_abs_mag=-19.3)
model = FiestaSALT3Model(filters=["g", "r", "i"])
model.warm_up(z_max=0.12, dz=0.01, batch_size=1024)
```

### Detection criteria

The DR2 selection includes both photometric quality cuts and BTS spectroscopic completeness:

| Criterion | Value |
|-----------|-------|
| SNR threshold | ≥ 5σ |
| Min detections | ≥ 7 total |
| Min bands | ≥ 2 (g + r) |
| Min per band | ≥ 3 |
| Pre-peak detections | ≥ 1 |
| Post-peak detections | ≥ 3 |
| Phase range | ≥ 30 days |
| Time baseline | ≥ 24 hours |
| Galactic latitude | \|b\| > 15° |
| **Spectroscopic completeness** | Logistic: k=2.378, m0=19.9 |

### BTS spectroscopic completeness

The ZTF DR2 sample is dominated by the Bright Transient Survey (BTS; Perley et al. 2020, Fremling et al. 2020), which has magnitude-dependent classification completeness. We model this as a logistic function applied probabilistically to each detected transient:

$$P(\text{classified}) = \frac{1}{1 + e^{k \cdot (m_\text{peak} - m_0)}}$$

with $k = 2.378$ and $m_0 = 19.9$:

| Peak mag | Model completeness |
|----------|--------------------|
| 18.0     | 99%                |
| 18.5     | 97%                |
| 19.0     | 89%                |
| 19.5     | 72%                |
| 20.0     | 44%                |

!!! note "BTS + non-BTS contributions"
    The $m_0 = 19.9$ midpoint is slightly fainter than a pure BTS fit ($m_0 \approx 19.5$)
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

### Results (100K injections)

| Quantity | Value |
|----------|-------|
| Comoving volume (z < 0.12) | 0.55 Gpc³ |
| ZTF sky fraction | 47% |
| Survey duration | 2.84 yr |
| Total SNe Ia in volume | ~17,300 |
| Detection efficiency | 20.5% |
| **Expected detections** | **~3,550** |
| **ZTF DR2 actual** | **3,628** |
| **Agreement** | **2.3%** |

### Effect of individual cuts

| Criteria | Efficiency | Expected |
|----------|-----------|----------|
| Photometric only (SNR ≥ 5, ≥ 2 det) | ~51% | ~8,800 |
| + Quality cuts (7 det, 2 bands, phase, etc.) | ~22% | ~3,810 |
| + BTS completeness | ~20.5% | ~3,550 |

---

## Rubin LSST prediction

### Setup

Predicts the number of SNe Ia that Rubin will photometrically detect over 10 years. **No spectroscopic completeness** is applied — Rubin will classify SNe Ia photometrically from multi-band lightcurves.

```python
from survey_sim import SurveyStore, SupernovaIaPopulation, SimulationPipeline
from survey_sim.salt3_model import FiestaSALT3Model, LSST_SALT3_BANDS

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db", nside=64)
pop = SupernovaIaPopulation(rate=23500.0, z_max=0.5, peak_abs_mag=-19.3)
model = FiestaSALT3Model(band_map=LSST_SALT3_BANDS)
model.warm_up(z_max=0.5, dz=0.01, batch_size=1024)
```

### Detection criteria

Photometric quality cuts only (no spectroscopic completeness):

| Criterion | Value |
|-----------|-------|
| SNR threshold | ≥ 5σ |
| Min detections | ≥ 7 total |
| Min bands | ≥ 2 |
| Min per band | ≥ 3 |
| Pre-peak detections | ≥ 1 |
| Post-peak detections | ≥ 3 |
| Phase range | ≥ 30 days |
| Time baseline | ≥ 24 hours |
| Galactic latitude | \|b\| > 15° |

### Results (100K injections)

| Quantity | Value |
|----------|-------|
| Comoving volume (z < 0.5) | 92.0 Gpc³ |
| Rubin sky fraction | 44% |
| Survey duration | 10 yr |
| Total SNe Ia in volume | ~9.5 million |
| Detection efficiency | 14.9% |
| **Expected detections** | **~1,415,000 in 10yr** |
| **Expected per year** | **~142,000** |

### Why is Rubin's efficiency lower than ZTF's?

Rubin's per-event efficiency (14.9%) is lower than ZTF's (20.5%) despite much greater depth. This is because:

1. **Higher z_max**: Rubin probes to z = 0.5 vs ZTF's z = 0.12. At higher redshift, SNe Ia are fainter and harder to get well-sampled lightcurves.
2. **Cadence**: Rubin WFD revisits each field every ~3 days vs ZTF's nightly cadence, making it harder to achieve 7+ detections with good phase coverage.
3. **No spectroscopic cut**: The ZTF efficiency includes spectroscopic completeness which preferentially selects bright, well-sampled events — this is a selection efficiency, not a pure detection efficiency.

Despite lower per-event efficiency, Rubin's enormous volume (92 vs 0.55 Gpc³) produces **~400× more SNe Ia** than ZTF.

---

## GPU requirements

The SALT3 model requires JAX with GPU support for practical performance:

```python
import survey_sim.gpu_setup  # auto-configures CUDA for JAX
```

The `warm_up()` call pre-compiles JAX JIT kernels for each redshift bin. This takes ~100s for z_max = 0.12 (12 bins) or ~10 minutes for z_max = 0.5 (50 bins), but eliminates first-call latency. Subsequent pipeline runs complete Phase 2 in ~20–30s for 60K evaluations.

!!! tip "Warm-up time"
    The SALT3 JIT compilation and GPU warm-up is a one-time cost per Python session.
    Subsequent `pipeline.run()` calls reuse the compiled kernels.

### Requirements

- **GPU**: NVIDIA GPU with CUDA (for JAX-accelerated SALT3)
- **Python packages**: `jax`, `jaxlib`, `fiesta`, `jax_supernovae`, `sncosmo`
- **Data**: ZTF boom-pipeline HDF5 files (available from survey-sim-data HuggingFace repo)
- **Module loads** (HPC): `module load gcc/13.2.0 python/3.11.5 openmpi/4.1.6 hdf5/1.14.3 cuda/12.8.0`

---

## Scripts

| Script | Description |
|--------|-------------|
| `python/scripts/run_snia_rubin.py` | Rubin SN Ia photometric detection (100K injections) |
| `python/scripts/run_snia_ztf.py` | ZTF DR2 validation (100K injections) |
