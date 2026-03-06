# ZTF Kilonova Detection Efficiency

This example computes kilonova detection efficiency and rate upper limits with the ZTF Bright Transient Survey, using AT2017gfo as the fiducial kilonova. We compare two lightcurve models: the physics-based **Bu2026 two-component surrogate** and the analytic **Metzger 1-zone blackbody**.

## Background

AT2017gfo, the kilonova counterpart to GW170817, is the only confirmed kilonova with extensive multi-band photometry. It serves as our benchmark for calibrating kilonova lightcurve models. ZTF observes in three bands (g, r, i) with a 47 deg² field of view and typical 5σ limiting magnitudes of ~20.5 in g/r.

We use the [ZTFReST](https://github.com/growth-astro/ztfrest) detection criteria, which require a fast-fading transient detected at least twice with ≥3 hour separation within 14 days. The survey period is March 2018 through March 2021 (~3.09 years), consistent with the ZTFReST analysis.

## Load the ZTF survey

```python
import glob
from survey_sim import SurveyStore

boom_dir = "/path/to/ztf_boom"
boom_files = sorted(glob.glob(f"{boom_dir}/ztf_2018*.h5")
                  + glob.glob(f"{boom_dir}/ztf_2019*.h5")
                  + glob.glob(f"{boom_dir}/ztf_2020*.h5")
                  + [f"{boom_dir}/ztf_202101.h5",
                     f"{boom_dir}/ztf_202102.h5",
                     f"{boom_dir}/ztf_202103.h5"])

survey = SurveyStore.from_ztf_boom(boom_files, nside=64)
print(f"Observations: {survey.n_observations}")    # ~504,000
print(f"Duration: {survey.duration_years:.2f} yr")  # ~3.09 yr
print(f"Bands: {survey.bands}")                     # ['g', 'i', 'r']
```

## Detection criteria

We adopt ZTFReST-like criteria tuned for fast transients, with a galactic plane exclusion at \(\lvert b \rvert > 15°\):

```python
from survey_sim import DetectionCriteria

det = DetectionCriteria(
    snr_threshold=5.0,              # primary detection at 5σ
    snr_threshold_secondary=3.0,    # secondary at 3σ
    min_detections=2,               # at least 2 detections
    min_detections_primary=1,       # at least 1 at full depth
    max_timespan_days=14.0,         # within 14-day window
    min_time_separation_hours=3.0,  # ≥3h between detections
    require_fast_transient=True,    # must show rapid evolution
    min_rise_rate=0.0,              # mag/day (no rise cut)
    min_fade_rate=0.3,              # ≥0.3 mag/day fading
    min_galactic_lat=15.0,          # exclude galactic plane
)
```

!!! note "Galactic plane exclusion"
    Transient searches avoid the galactic plane due to high stellar density and extinction. We apply \(\lvert b \rvert > 15°\) to both kilonova and SN Ia simulations for consistency with ZTF operations.

---

## Option A: Bu2026 two-component model

The Bu2026 model uses a JAX-based neural network surrogate trained on numerical kilonova simulations with two ejecta components (dynamical + wind). It produces per-band magnitudes that naturally capture the blue-to-red color evolution.

### Tuning to AT2017gfo

We fit the Bu2026 parameters to AT2017gfo photometry (ps1 g/r/i, \(t < 4\) days) via grid search over ejecta mass and viewing angle. The best-fit parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| \(\log_{10}(M_\mathrm{ej,dyn}/M_\odot)\) | \(-1.8\) | Dynamical ejecta mass |
| \(v_\mathrm{ej,dyn}\) | \(0.2c\) | Dynamical ejecta velocity |
| \(Y_{e,\mathrm{dyn}}\) | 0.15 | Dynamical electron fraction |
| \(\log_{10}(M_\mathrm{ej,wind}/M_\odot)\) | \(-1.1\) | Wind ejecta mass |
| \(v_\mathrm{ej,wind}\) | \(0.1c\) | Wind ejecta velocity |
| \(Y_{e,\mathrm{wind}}\) | 0.35 | Wind electron fraction |
| \(\iota_\mathrm{EM}\) | 0.45 rad (26°) | Viewing angle |

These produce residuals \(\lesssim 0.5\) mag in all three bands for the first 4 days.

### Fixed inclination

```python
from survey_sim import (
    FixedBu2026KilonovaPopulation,
    SimulationPipeline,
)
from survey_sim.fiesta_model import FiestaKNModel

pop = FixedBu2026KilonovaPopulation(
    log10_mej_dyn=-1.8,
    v_ej_dyn=0.2,
    ye_dyn=0.15,
    log10_mej_wind=-1.1,
    v_ej_wind=0.1,
    ye_wind=0.35,
    inclination_em=0.45,
    rate=1000.0,
    z_max=0.3,
)

model = FiestaKNModel()

pipe = SimulationPipeline(
    survey, [pop], {"Kilonova": model}, det,
    n_transients=100_000, seed=42,
)
result = pipe.run()
```

Results (100K transients, Mar 2018 – Mar 2021, \(\lvert b \rvert > 15°\)):

| Quantity | Value |
|----------|-------|
| Detected | 20 / 100,000 |
| Efficiency | 0.02% |
| \(R_\mathrm{upper}\) (90% CL) | **721 Gpc\(^{-3}\) yr\(^{-1}\)** |
| \(R_\mathrm{upper}\) (95% CL) | 938 Gpc\(^{-3}\) yr\(^{-1}\) |

### Varying inclination

Real kilonovae are observed from random orientations. Edge-on viewing angles produce fainter, redder emission:

```python
pop = FixedBu2026KilonovaPopulation(
    log10_mej_dyn=-1.7,
    v_ej_dyn=0.2,
    ye_dyn=0.15,
    log10_mej_wind=-1.1,
    v_ej_wind=0.1,
    ye_wind=0.35,
    vary_inclination=True,      # sample cos(ι) uniformly
    rate=1000.0,
    z_max=0.3,
)
```

Results (100K transients, Mar 2018 – Mar 2021, \(\lvert b \rvert > 15°\)):

| Quantity | Value |
|----------|-------|
| Detected | 8 / 100,000 |
| Efficiency | 0.01% |
| \(R_\mathrm{upper}\) (90% CL) | **1,802 Gpc\(^{-3}\) yr\(^{-1}\)** |
| \(R_\mathrm{upper}\) (95% CL) | 2,345 Gpc\(^{-3}\) yr\(^{-1}\) |

!!! note "GPU acceleration"
    The Bu2026 model supports GPU batch evaluation via JAX. On a GPU node, 100K transients complete in ~45 seconds. On CPU, use a smaller sample.

---

## Option B: Metzger 1-zone blackbody model

The Metzger model solves a single-zone ODE for the thermal energy and photosphere evolution, then computes per-band AB magnitudes from a blackbody spectrum at the effective temperature:

$$T_\mathrm{eff}(t) = \left(\frac{L_\mathrm{rad}(t)}{4\pi R(t)^2 \sigma_\mathrm{SB}}\right)^{1/4}$$

It's much faster than the Bu2026 surrogate (pure Rust, rayon-parallel) but limited to a single temperature — it cannot simultaneously reproduce the blue and red kilonova emission.

### Tuning to AT2017gfo

Grid search over \((\log_{10} M_\mathrm{ej}, \log_{10} v_\mathrm{ej}, \log_{10} \kappa)\) fitted to AT2017gfo ps1 g/r/i at \(t < 3\) days:

| Parameter | Value | Description |
|-----------|-------|-------------|
| \(M_\mathrm{ej}\) | \(0.00126\) \(M_\odot\) | Ejecta mass |
| \(v_\mathrm{ej}\) | \(0.50c\) | Ejecta velocity |
| \(\kappa\) | 398 cm²/g | Gray opacity |

The high opacity drives a rapid \(g\)-band decline, mimicking the lanthanide-dominated dynamical ejecta. Typical residuals:

- **r-band**: ±0.2 mag for \(t < 2.5\) days (best fit)
- **i-band**: ±0.3 mag
- **g-band**: 0.3–0.6 mag too faint (single-temperature limitation)

### Run the simulation

```python
from survey_sim import (
    FixedMetzgerKilonovaPopulation,
    MetzgerKNModel,
    SimulationPipeline,
)

pop = FixedMetzgerKilonovaPopulation(
    mej=0.00126,
    vej=0.50,
    kappa=398.0,
    rate=1000.0,
    z_max=0.3,
)

model = MetzgerKNModel()

pipe = SimulationPipeline(
    survey, [pop], {"Kilonova": model}, det,
    n_transients=1_000_000, seed=42,
)
result = pipe.run()
```

!!! tip "Performance"
    The Metzger model is pure Rust and runs with rayon parallelism. 1M transients complete in ~45 seconds (29s lightcurve eval + 11s spatial match + 5s detection).

---

## Results comparison

### Detection efficiency

| Model | Inclination | \(N_\mathrm{sim}\) | \(N_\mathrm{det}\) | Efficiency |
|-------|-------------|---------------------|---------------------|-----------|
| Bu2026 | Fixed (26°) | 100,000 | 20 | 0.020% |
| Bu2026 | Varying (isotropic) | 100,000 | 8 | 0.008% |
| Metzger BB | N/A | 1,000,000 | ~66 | 0.0066% |

The Bu2026 model with varying inclination and the Metzger model give comparable efficiencies (~0.01%), while the fixed face-on Bu2026 case is ~2.5× more efficient.

### Rate upper limits

Computing the 90% and 95% confidence upper limits on the kilonova volumetric rate:

```python
import math

duration = survey.duration_years
eff = result.n_detected / result.n_simulated

z_max = 0.3
d_max = 1380.0   # Mpc (approx. for z=0.3)
V_max = (4/3) * math.pi * (d_max / 1000) ** 3  # Gpc³

f_sky = 0.47      # ZTF sky fraction
V_eff = V_max * f_sky
VT_eff = V_eff * duration * eff

for cl, label in [(0.90, "90%"), (0.95, "95%")]:
    R_upper = -math.log(1 - cl) / VT_eff
    print(f"R_upper ({label}) = {R_upper:.0f} Gpc^-3 yr^-1")
```

| Quantity | Bu2026 (fixed) | Bu2026 (varying) | Metzger BB |
|----------|----------------|------------------|------------|
| \(f_\mathrm{sky}\) | 0.47 | 0.47 | 0.47 |
| \(V_\mathrm{eff}\) | 5.17 Gpc³ | 5.17 Gpc³ | 5.17 Gpc³ |
| Duration | 3.09 yr | 3.09 yr | 3.09 yr |
| Efficiency | 0.020% | 0.008% | 0.0066% |
| \(VT_\mathrm{eff}\) | 0.0032 Gpc³ yr | 0.0013 Gpc³ yr | 0.00105 Gpc³ yr |
| \(R_\mathrm{upper}\) (90%) | **721** | **1,802** | 2,185 Gpc⁻³ yr⁻¹ |
| \(R_\mathrm{upper}\) (95%) | 938 | 2,345 | 2,842 Gpc⁻³ yr⁻¹ |

!!! info "Interpretation"
    The Bu2026 model with fixed inclination gives the most constraining rate limit because AT2017gfo was observed near face-on. The varying-inclination case is more realistic for a population analysis and gives limits comparable to the Metzger BB model. All three scenarios are consistent with the current best estimate of the BNS merger rate (~320 Gpc⁻³ yr⁻¹ from LIGO-Virgo-KAGRA O3).

---

## Full script

A complete end-to-end script for the Bu2026 analysis with galactic plane cut:

```python
#!/usr/bin/env python
"""ZTF kilonova detection efficiency with Bu2026 model."""
import glob
import math
from survey_sim import (
    SurveyStore,
    FixedBu2026KilonovaPopulation,
    DetectionCriteria,
    SimulationPipeline,
)
from survey_sim.fiesta_model import FiestaKNModel

# Survey: March 2018 – March 2021
boom_dir = "/path/to/ztf_boom"
boom_files = sorted(
    glob.glob(f"{boom_dir}/ztf_2018*.h5")
    + glob.glob(f"{boom_dir}/ztf_2019*.h5")
    + glob.glob(f"{boom_dir}/ztf_2020*.h5")
    + [f"{boom_dir}/ztf_202101.h5",
       f"{boom_dir}/ztf_202102.h5",
       f"{boom_dir}/ztf_202103.h5"]
)
survey = SurveyStore.from_ztf_boom(boom_files, nside=64)

# Population (AT2017gfo best-fit, varying inclination)
pop = FixedBu2026KilonovaPopulation(
    log10_mej_dyn=-1.7, v_ej_dyn=0.2, ye_dyn=0.15,
    log10_mej_wind=-1.1, v_ej_wind=0.1, ye_wind=0.35,
    vary_inclination=True,
    rate=1000.0, z_max=0.3,
)

# Model + detection
model = FiestaKNModel()
det = DetectionCriteria(
    snr_threshold=5.0, snr_threshold_secondary=3.0,
    min_detections=2, min_detections_primary=1,
    max_timespan_days=14.0, min_time_separation_hours=3.0,
    require_fast_transient=True, min_rise_rate=0.0, min_fade_rate=0.3,
    min_galactic_lat=15.0,
)

# Run
pipe = SimulationPipeline(
    survey, [pop], {"Kilonova": model}, det,
    n_transients=100_000, seed=42,
)
result = pipe.run()

# Results
eff = result.n_detected / result.n_simulated
print(f"Detected: {result.n_detected} / {result.n_simulated} = {eff:.6f}")

VT = 5.17 * survey.duration_years * eff  # Gpc³ yr
R90 = -math.log(0.10) / VT
print(f"R_upper (90% CL) = {R90:.0f} Gpc^-3 yr^-1")
```
