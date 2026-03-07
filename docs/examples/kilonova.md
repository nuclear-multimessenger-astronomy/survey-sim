# Kilonova Detection

This page summarizes kilonova detection simulations for both **ZTF** and **Rubin LSST**, using the Metzger 1-zone blackbody model and the Bu2026 two-component surrogate. We use AT2017gfo as the fiducial kilonova and adopt current LVK median merger rates (BNS = 45, NSBH = 25 Gpc⁻³ yr⁻¹).

## Summary of results

| Survey | Model | Inclination | Injections | Efficiency | Expected (BNS+NSBH, 10yr) |
|--------|-------|-------------|-----------|------------|---------------------------|
| **ZTF** (3.09yr) | Bu2026 | Fixed (26°) | 100K | 0.020% | — |
| **ZTF** (3.09yr) | Bu2026 | Varying | 100K | 0.008% | — |
| **ZTF** (3.09yr) | Metzger BB | N/A | 1M | 0.0066% | — |
| **Rubin** (10yr) | Metzger BB | N/A | 1M | 0.036% | 10.1 |

For ZTF, these efficiencies translate to rate upper limits (90% CL) of 721–2,185 Gpc⁻³ yr⁻¹ depending on model, consistent with the LVK BNS rate.

For Rubin, we predict **~10 kilonovae in 10 years** (6.5 BNS + 3.6 NSBH) using the Metzger BB model with LVK median rates. This is a blind-discovery rate from survey photometry alone, without gravitational-wave triggers.

---

## Background

Kilonovae are thermal transients powered by the radioactive decay of r-process elements synthesized in neutron star mergers. AT2017gfo, the counterpart to GW170817, is the only confirmed kilonova with extensive multi-band photometry and serves as our calibration benchmark.

We use two lightcurve models:

| Model | Description | Speed |
|-------|-------------|-------|
| **Metzger BB** | 1-zone ODE, blackbody at effective temperature. Pure Rust, rayon-parallel. | ~μs/eval |
| **Bu2026** | Two-component (dynamical + wind) neural network surrogate via fiestaEM/JAX. | ~ms/eval (GPU) |

The Metzger model solves a single-zone ODE for the thermal energy and photosphere evolution, then computes per-band AB magnitudes from a blackbody spectrum at the effective temperature:

$$T_\mathrm{eff}(t) = \left(\frac{L_\mathrm{rad}(t)}{4\pi R(t)^2 \sigma_\mathrm{SB}}\right)^{1/4}$$

It's much faster than the Bu2026 surrogate but limited to a single temperature — it cannot simultaneously reproduce the blue and red kilonova emission. The Bu2026 model uses a JAX-based neural network surrogate trained on numerical kilonova simulations with two ejecta components (dynamical + wind), producing per-band magnitudes that naturally capture the blue-to-red color evolution.

### AT2017gfo calibration

**Bu2026 best-fit** (grid search, ps1 g/r/i, t < 4 days):

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\log_{10}(M_\mathrm{ej,dyn}/M_\odot)$ | $-1.8$ | Dynamical ejecta mass |
| $v_\mathrm{ej,dyn}$ | $0.2c$ | Dynamical ejecta velocity |
| $Y_{e,\mathrm{dyn}}$ | 0.15 | Dynamical electron fraction |
| $\log_{10}(M_\mathrm{ej,wind}/M_\odot)$ | $-1.1$ | Wind ejecta mass |
| $v_\mathrm{ej,wind}$ | $0.1c$ | Wind ejecta velocity |
| $Y_{e,\mathrm{wind}}$ | 0.35 | Wind electron fraction |
| $\iota_\mathrm{EM}$ | 0.45 rad (26°) | Viewing angle |

**Metzger BB best-fit** (grid search, ps1 g/r/i, t < 3 days):

| Parameter | Value | Description |
|-----------|-------|-------------|
| $M_\mathrm{ej}$ | $0.00126$ $M_\odot$ | Ejecta mass |
| $v_\mathrm{ej}$ | $0.50c$ | Ejecta velocity |
| $\kappa$ | 398 cm²/g | Gray opacity |

---

## ZTF kilonova detection

### Survey and detection criteria

- **Period**: March 2018 – March 2021 (3.09 yr)
- **Observations**: ~504,000 in g/r/i
- **Sky fraction**: 47%

We adopt [ZTFReST](https://github.com/growth-astro/ztfrest)-like criteria tuned for fast transients:

| Criterion | Value |
|-----------|-------|
| SNR threshold (primary) | ≥ 5σ |
| SNR threshold (secondary) | ≥ 3σ |
| Min detections | ≥ 2 |
| Min primary detections | ≥ 1 |
| Max timespan | 14 days |
| Min time separation | ≥ 3 hours |
| Fast transient | Required (≥ 0.3 mag/day fading) |
| Galactic latitude | \|b\| > 15° |

```python
from survey_sim import DetectionCriteria

det = DetectionCriteria(
    snr_threshold=5.0, snr_threshold_secondary=3.0,
    min_detections=2, min_detections_primary=1,
    max_timespan_days=14.0, min_time_separation_hours=3.0,
    require_fast_transient=True, min_rise_rate=0.0, min_fade_rate=0.3,
    min_galactic_lat=15.0,
)
```

### Bu2026 model

```python
from survey_sim import FixedBu2026KilonovaPopulation, SimulationPipeline, load_ztf_survey
from survey_sim.fiesta_model import FiestaKNModel

survey = load_ztf_survey(nside=64)
pop = FixedBu2026KilonovaPopulation(
    log10_mej_dyn=-1.8, v_ej_dyn=0.2, ye_dyn=0.15,
    log10_mej_wind=-1.1, v_ej_wind=0.1, ye_wind=0.35,
    inclination_em=0.45,
    rate=1000.0, z_max=0.3,
)
model = FiestaKNModel()

pipe = SimulationPipeline(survey, [pop], {"Kilonova": model}, det, n_transients=100_000, seed=42)
result = pipe.run()
```

#### Results (100K injections)

| Inclination | Detected | Efficiency | R_upper (90% CL) |
|-------------|----------|------------|-------------------|
| Fixed (26°) | 20 / 100K | 0.020% | 721 Gpc⁻³ yr⁻¹ |
| Varying (isotropic) | 8 / 100K | 0.008% | 1,802 Gpc⁻³ yr⁻¹ |

!!! note "GPU acceleration"
    The Bu2026 model supports GPU batch evaluation via JAX. On a GPU node, 100K transients complete in ~45 seconds. On CPU, use a smaller sample.

### Metzger BB model

```python
from survey_sim import FixedMetzgerKilonovaPopulation, MetzgerKNModel

pop = FixedMetzgerKilonovaPopulation(
    mej=0.00126, vej=0.50, kappa=398.0,
    rate=1000.0, z_max=0.3,
)
model = MetzgerKNModel()

pipe = SimulationPipeline(survey, [pop], {"Kilonova": model}, det, n_transients=1_000_000, seed=42)
result = pipe.run()
```

#### Results (1M injections)

| Detected | Efficiency | R_upper (90% CL) |
|----------|------------|-------------------|
| ~66 / 1M | 0.0066% | 2,185 Gpc⁻³ yr⁻¹ |

!!! tip "Performance"
    The Metzger model is pure Rust and runs with rayon parallelism. 1M transients complete in ~45 seconds (29s lightcurve eval + 11s spatial match + 5s detection).

### Detection efficiency comparison

| Model | Inclination | N_sim | N_det | Efficiency |
|-------|-------------|-------|-------|-----------|
| Bu2026 | Fixed (26°) | 100,000 | 20 | 0.020% |
| Bu2026 | Varying (isotropic) | 100,000 | 8 | 0.008% |
| Metzger BB | N/A | 1,000,000 | ~66 | 0.0066% |

The Bu2026 model with varying inclination and the Metzger model give comparable efficiencies (~0.01%), while the fixed face-on case is ~2.5× more efficient.

---

## Rubin LSST kilonova detection

### Survey and detection criteria

- **Survey**: Rubin LSST baseline_v5.1.1, 10-year
- **Observations**: ~2.06M in ugrizy
- **Sky fraction**: 44%
- **Detection criteria**: 2 detections, ≥0.5h separation, ≥0.3 mag/day fading, |b| > 15°

### Metzger BB model

```python
from survey_sim import SurveyStore, FixedMetzgerKilonovaPopulation, MetzgerKNModel

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db", nside=64)
pop = FixedMetzgerKilonovaPopulation(
    mej=0.00126, vej=0.50, kappa=398.0,
    rate=45.0,    # BNS rate (LVK median)
    z_max=0.5,    # Rubin sees further than ZTF
)
model = MetzgerKNModel()
```

### Results (1M injections)

| Quantity | Value |
|----------|-------|
| Detected | 357 / 1M |
| Efficiency | 0.036% |
| V_eff | 40.4 Gpc³ (z < 0.5, f_sky = 0.44) |

#### Expected detections (10 years)

| Rate source | Rate (Gpc⁻³ yr⁻¹) | Expected (10yr) |
|-------------|-------------------|-----------------|
| BNS only | 45 | 6.5 |
| NSBH only | 25 | 3.6 |
| **BNS + NSBH** | **70** | **10.1** |

---

## Rate upper limits

The 90% CL upper limit on the volumetric rate (assuming 0 detections) is computed as:

$$R_\mathrm{upper} = \frac{-\ln(1 - \mathrm{CL})}{V_\mathrm{eff} \times T \times \epsilon}$$

```python
import math
R_upper_90 = -math.log(0.10) / (V_eff * duration * efficiency)
```

| Survey | Model | V_eff | Duration | Efficiency | VT_eff | R_upper (90%) | R_upper (95%) |
|--------|-------|-------|----------|------------|--------|---------------|---------------|
| ZTF | Bu2026 (fixed ι) | 5.17 Gpc³ | 3.09 yr | 0.020% | 0.0032 Gpc³ yr | **721** | 938 |
| ZTF | Bu2026 (varying ι) | 5.17 Gpc³ | 3.09 yr | 0.008% | 0.0013 Gpc³ yr | **1,802** | 2,345 |
| ZTF | Metzger BB | 5.17 Gpc³ | 3.09 yr | 0.0066% | 0.00105 Gpc³ yr | **2,185** | 2,842 |

All scenarios are consistent with the current BNS merger rate estimate of ~45 Gpc⁻³ yr⁻¹ (LVK O4 median).

!!! info "Interpretation"
    The Bu2026 model with fixed inclination gives the most constraining rate limit because AT2017gfo was observed near face-on. The varying-inclination case is more realistic for a population analysis and gives limits comparable to the Metzger BB model.

---

## Scripts

| Script | Description |
|--------|-------------|
| `python/scripts/run_metzger_bb_rubin.py` | Rubin kilonova detection with Metzger BB (1M injections) |
