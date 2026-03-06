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

---

## ZTF kilonova detection

### Survey and detection criteria

- **Period**: March 2018 – March 2021 (3.09 yr)
- **Observations**: ~504,000 in g/r/i
- **Sky fraction**: 47%
- **Detection criteria**: ZTFReST-like (2 detections, ≥3h separation, ≥0.3 mag/day fading, |b| > 15°)

### Bu2026 model (AT2017gfo best-fit)

Best-fit parameters from grid search over AT2017gfo photometry (ps1 g/r/i, t < 4 days):

| Parameter | Value |
|-----------|-------|
| log₁₀(M_ej,dyn / M☉) | −1.8 |
| v_ej,dyn | 0.2c |
| Y_e,dyn | 0.15 |
| log₁₀(M_ej,wind / M☉) | −1.1 |
| v_ej,wind | 0.1c |
| Y_e,wind | 0.35 |
| ι_EM | 0.45 rad (26°) |

```python
from survey_sim import FixedBu2026KilonovaPopulation, SimulationPipeline
from survey_sim.fiesta_model import FiestaKNModel

pop = FixedBu2026KilonovaPopulation(
    log10_mej_dyn=-1.8, v_ej_dyn=0.2, ye_dyn=0.15,
    log10_mej_wind=-1.1, v_ej_wind=0.1, ye_wind=0.35,
    inclination_em=0.45,
    rate=1000.0, z_max=0.3,
)
model = FiestaKNModel()
```

#### Results (100K injections)

| Inclination | Detected | Efficiency | R_upper (90% CL) |
|-------------|----------|------------|-------------------|
| Fixed (26°) | 20 / 100K | 0.020% | 721 Gpc⁻³ yr⁻¹ |
| Varying (isotropic) | 8 / 100K | 0.008% | 1,802 Gpc⁻³ yr⁻¹ |

### Metzger BB model (AT2017gfo best-fit)

Best-fit parameters (grid search, ps1 g/r/i, t < 3 days):

| Parameter | Value |
|-----------|-------|
| M_ej | 0.00126 M☉ |
| v_ej | 0.50c |
| κ | 398 cm²/g |

```python
from survey_sim import FixedMetzgerKilonovaPopulation, MetzgerKNModel

pop = FixedMetzgerKilonovaPopulation(
    mej=0.00126, vej=0.50, kappa=398.0,
    rate=1000.0, z_max=0.3,
)
model = MetzgerKNModel()
```

#### Results (1M injections)

| Detected | Efficiency | R_upper (90% CL) |
|----------|------------|-------------------|
| ~66 / 1M | 0.0066% | 2,185 Gpc⁻³ yr⁻¹ |

The Bu2026 model with varying inclination and the Metzger BB give comparable efficiencies (~0.01%), while the fixed face-on case is ~2.5× more efficient.

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

```python
import math
R_upper_90 = -math.log(0.10) / (V_eff * duration * efficiency)
```

| Survey | Model | R_upper (90% CL) |
|--------|-------|-------------------|
| ZTF (3.09yr) | Bu2026, fixed ι | 721 Gpc⁻³ yr⁻¹ |
| ZTF (3.09yr) | Bu2026, varying ι | 1,802 Gpc⁻³ yr⁻¹ |
| ZTF (3.09yr) | Metzger BB | 2,185 Gpc⁻³ yr⁻¹ |

All scenarios are consistent with the current BNS merger rate estimate of ~45 Gpc⁻³ yr⁻¹ (LVK O4 median).

---

## Scripts

| Script | Description |
|--------|-------------|
| `python/scripts/run_metzger_bb_rubin.py` | Rubin kilonova detection with Metzger BB (1M injections) |
| See `docs/examples/ztf_kilonova.md` | ZTF kilonova detection with Bu2026 and Metzger BB models |
