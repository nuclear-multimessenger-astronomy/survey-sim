# Tidal Disruption Events

This page documents TDE detection simulations and rate forecasts, covering:

1. **ZTF validation** — reproducing the Yao et al. (2023) spectroscopic sample
2. **Rubin LSST injection predictions** — with and without host brightness cuts, with Karmen et al. (2026) rate evolution
3. **Semi-analytical rate forecasts** — reproducing Karmen et al. (2026) Table 1 via Rust
4. **Injection vs analytical comparison** — measuring real cadence efficiency

All injection simulations use the Yao et al. (2023) broken power-law luminosity function with a parametric TDE lightcurve model evaluated entirely in Rust.

---

## Background

Tidal disruption events (TDEs) occur when a star passes close enough to a supermassive black hole to be torn apart by tidal forces. The resulting accretion flare produces a luminous optical/UV transient lasting weeks to months.

### Lightcurve model

We use a parametric TDE model (sigmoid rise × power-law decay) from the `lightcurve-fitting` crate:

$$F(t) = e^{a} \cdot \sigma\!\left(\frac{t - t_0}{\tau_\mathrm{rise}}\right) \cdot \left(1 + \mathrm{softplus}\!\left(\frac{t - t_0}{\tau_\mathrm{fall}}\right)\right)^{-\alpha}$$

with 7 parameters: `log_a`, `b` (baseline), `t0`, `log_tau_rise`, `log_tau_fall`, `alpha`, and `sigma_extra`. This is a pure Rust model evaluated via `ParametricModel()` — no Python callback, no GPU required.

### Timescale parameters

Drawn from distributions calibrated to the Yao et al. (2023) Table 4 sample:

| Parameter | Distribution | Description |
|-----------|-------------|-------------|
| $\tau_\mathrm{rise}$ | $\mathcal{N}(18, 10)$, clamp [3, 60] days | Sigmoid rise timescale |
| $\tau_\mathrm{fall}$ | $\mathcal{N}(58, 25)$, clamp [10, 200] days | Power-law decay timescale |
| $\alpha$ | $\mathcal{N}(5/3, 0.3)$, clamp [0.8, 3.0] | Decay index (theoretical: 5/3) |

### Luminosity function

Peak absolute magnitudes are drawn from the **Yao et al. (2023) broken power-law luminosity function**:

$$\phi(L_g) = \frac{N_0}{\left(L_g / L_\mathrm{bk}\right)^{\gamma_1} + \left(L_g / L_\mathrm{bk}\right)^{\gamma_2}}$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $N_0$ | $2.87 \times 10^{-7}$ Mpc⁻³ yr⁻¹ dex⁻¹ | Normalization |
| $\log_{10} L_\mathrm{bk}$ | 43.13 | Break luminosity (erg/s) |
| $\gamma_1$ | 0.26 | Faint-end slope (shallow) |
| $\gamma_2$ | 2.58 | Bright-end slope (steep) |

Integrating from $M_g = -24$ to $-15$ gives a total local volumetric rate of **~829 Gpc⁻³ yr⁻¹**.

### Redshift-dependent rate evolution (Karmen et al. 2026)

The local TDE rate evolves with redshift due to several physical effects. The total annual detection rate is given by Karmen+2026 Eq. 1:

$$\Gamma_\mathrm{TDE} = \int_0^{z_\mathrm{Ly}(\lambda)} \epsilon(z)\, \mathcal{F}(z)\, N_\mathrm{BH}(z)\, R_0(z, \lambda)\, \mathcal{O}(z)\, dz$$

Five factors modify the rate at higher redshift:

**SMBH mass function $N_\mathrm{BH}(z)$** — two models:

- **Illustris/TNG** (hydro simulation): $N_\mathrm{BH}(z) = \exp(-0.82\, z)$ — slow decline
- **Shankar+09** (semi-empirical): $N_\mathrm{BH}(z) = \exp(-1.46\, z)$ — fast decline

**Galaxy-scale enhancement $\mathcal{F}(z) = \mathcal{M}(z)\, \mathcal{I}(z)\, \mathcal{D}(z)$:**

| Factor | Description | $z=0$ | $z=1$ | $z=2$ |
|--------|-------------|-------|-------|-------|
| $\mathcal{M}(z)$ | Merger enhancement | 1.0 | 2.8 | 4.5 |
| $\mathcal{D}(z)$ | Nuclear stellar density $(1+z)^{0.9\alpha}$ | 1.0 | 2.6 | 4.4 |
| $\mathcal{I}(z)$ | IMF evolution | 1.0 | 1.2 | 1.4 |
| $\mathcal{F}(z)$ | Combined | 1.0 | 8.5 | 27.1 |

**Dust obscuration $\mathcal{O}(z)$** — logistic function from $f_\mathrm{obsc} = 0.3$ at $z=0$ toward $f_\mathrm{max} = 0.9$ at high $z$.

The merger enhancement $E \sim \mathrm{U}(10, 100)$ and density slope $\alpha \sim \mathrm{U}(1, 2)$ are sampled via Monte Carlo (200 iterations) to propagate systematic uncertainties. The injection simulations use the median values ($E=30$, $\alpha=1.5$).

**Lyman-alpha cutoff** — maximum redshift before Ly$\alpha$ absorption blocks the filter:

| Filter | $\lambda_\mathrm{obs}$ | $z_\mathrm{Ly}$ |
|--------|----------------------|-----------------|
| LSST g | 482 nm | 2.96 |
| Roman F062 | 620 nm | 4.10 |

### Broadband K-corrections

The K-correction accounts for how apparent brightness changes with redshift due to the SED shape interacting with the filter bandpass. For TDEs, we use a blackbody at $T = 30{,}000$ K:

$$K(z) = -2.5 \log_{10}\!\left[\frac{\int B_\nu(\nu(1+z),\, T)\, R(\nu)\, d\nu}{(1+z)\,\int B_\nu(\nu,\, T)\, R(\nu)\, d\nu}\right]$$

| Redshift | LSST g-band (482 nm) | Roman F062 (620 nm) |
|----------|---------------------|---------------------|
| 0.5 | -0.11 | -0.18 |
| 1.0 | -0.07 | -0.24 |
| 1.5 | +0.06 | -0.20 |
| 2.0 | +0.24 | -0.08 |

```python
from survey_sim import blackbody_k_correction, blackbody_k_correction_instrument

# Direct: specify filter by wavelength and width
k = blackbody_k_correction(z=1.0, temperature_k=30000.0,
                            central_wavelength_nm=482.0, width_nm=140.0)
# -0.071

# Via built-in instrument definition
k = blackbody_k_correction_instrument(z=1.0, temperature_k=30000.0,
                                       instrument="rubin", band_name="g")
# -0.071

# Works with any built-in instrument: rubin, ztf, roman, ultrasat, uvex, argus
k = blackbody_k_correction_instrument(z=1.0, temperature_k=30000.0,
                                       instrument="roman", band_name="F062")
# -0.243
```

The `Sed` trait in Rust supports arbitrary SED models (blackbody, power-law, tabulated), so K-corrections can be computed for any transient type.

---

## ZTF validation (Yao et al. 2023)

Reproduces the flux-limited TDE sample from [Yao et al. (2023)](https://arxiv.org/abs/2303.06523): **33 spectroscopically classified TDEs** over 3 years of ZTF operation.

```python
from survey_sim import (
    TdePopulation, ParametricModel, DetectionCriteria,
    SimulationPipeline, load_ztf_survey,
)

survey = load_ztf_survey(start="201810", end="202109", nside=64)
pop = TdePopulation(z_max=0.3, use_luminosity_function=True)
model = ParametricModel()

det = DetectionCriteria(
    snr_threshold=5.0, snr_threshold_secondary=5.0,
    min_detections=3, min_detections_primary=3,
    min_bands=1, min_per_band=2,
    max_timespan_days=365.0, min_time_separation_hours=48.0,
    require_fast_transient=False, min_galactic_lat=15.0,
    spectroscopic_completeness_k=5.0, spectroscopic_completeness_m0=18.8,
)
```

### Results (100K injections)

| Quantity | Value |
|----------|-------|
| Comoving volume ($z < 0.3$) | 11.0 Gpc³ |
| ZTF sky fraction | 47% |
| Survey duration | 2.99 yr |
| Detection efficiency | 0.26% |
| **Expected detections** | **32.7** |
| **Yao+2023 actual** | **33** |
| **Agreement** | **1%** |

The 0.26% efficiency reflects the tiny fraction of the LF that produces TDEs bright enough ($m_\mathrm{peak} < 18.8$) to be identified against their host galaxies at ZTF depth.

---

## Rubin LSST injection predictions

Predicts TDE detections over Rubin's 10-year baseline survey using **identical population parameters** as ZTF. Five configurations probe the effects of host brightness cuts and rate evolution:

```python
from survey_sim import (
    SurveyStore, TdePopulation, ParametricModel,
    DetectionCriteria, SimulationPipeline, TdeRateForecast,
)

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db", nside=64)

# Constant local rate (no evolution)
pop = TdePopulation(z_max=0.8, use_luminosity_function=True)

# With Karmen+2026 rate evolution to Lyman-alpha cutoff
pop_evolved = TdePopulation(
    z_max=2.96, use_luminosity_function=True,
    use_rate_evolution=True, bhmf_model="illustris",
)

model = ParametricModel()
```

### Detection criteria

| Criterion | Value |
|-----------|-------|
| SNR threshold | $\geq 5\sigma$ |
| Min detections | $\geq 3$ total |
| Min per band | $\geq 2$ |
| Max timespan | 365 days |
| Min time separation | $\geq 48$ hours |
| Galactic latitude | $\|b\| > 15°$ |
| **Host brightness cut** (when enabled) | Logistic: $k = 5.0$, $m_0 = 22.5$ |

The host brightness cut requires $m_\mathrm{peak} \lesssim 22.5$, about 2 mag brighter than Rubin's single-visit 5σ limit (~24.5). This models the requirement that TDEs must outshine their host nucleus.

### Results (100K injections per configuration)

| Configuration | z_max | Eff% | N/yr | N/10yr |
|---------------|-------|------|------|--------|
| z<0.8, host cut, no evolution | 0.80 | 1.54 | 555 | 5,548 |
| z<0.8, no host cut, no evolution | 0.80 | 7.17 | 2,583 | 25,821 |
| z<2.96, no cut, Illustris evolution | 2.96 | 0.84 | 6,585 | 65,824 |
| z<2.96, no cut, Shankar evolution | 2.96 | 1.49 | 4,082 | 40,801 |
| z<2.96, host cut, Illustris evolution | 2.96 | 0.12 | 928 | 9,281 |

### What each configuration means

- **Host cut, no evolution** (555/yr): Conservative spectroscopic-like sample. Only TDEs bright enough to outshine their host nucleus ($m < 22.5$), constant local rate. Comparable to what a spectroscopic follow-up program would confirm.

- **No host cut, no evolution** (2,583/yr): All photometrically detectable TDEs at $z < 0.8$. The 4.7× increase over the host-cut case shows how many faint-end LF TDEs are lost to the brightness requirement.

- **No host cut, Illustris evolution** (6,585/yr): Full photometric sample with Karmen+2026 rate evolution out to $z_\mathrm{Ly} = 2.96$. Galaxy-scale enhancements ($\mathcal{F}(z)$ up to ~27× at $z=2$) boost the high-redshift rate, partially offset by BHMF decline.

- **No host cut, Shankar evolution** (4,082/yr): Same but with faster BHMF decline ($\alpha = -1.46$ vs $-0.82$), giving ~40% fewer detections.

- **Host cut, Illustris evolution** (928/yr): The realistic spectroscopic case — rate evolution extends the volume, but the host cut removes most faint TDEs. Only 928/yr survive both filters.

---

## Semi-analytical rate forecasts (Karmen et al. 2026)

The `TdeRateForecast` class reproduces the predictions from [Karmen et al. (2026, arXiv:2602.04947)](https://arxiv.org/abs/2602.04947) entirely in Rust. These assume perfect survey efficiency ($\epsilon = 1$) for time-domain surveys.

### Comparison with Karmen et al. (2026) Table 1

Computed with `TdeRateForecast(temperature_k=30000, n_mc=200, seed=42)`:

| Survey | BHMF | This work | Paper | Ratio |
|--------|------|-----------|-------|-------|
| **Rubin (LSST)** | Illustris | 21,383 | 26,873 | 0.80 |
| **Rubin (LSST)** | Shankar | 13,135 | 13,803 | 0.95 |
| **Rubin (DDF)** | Illustris | 66.7 | 61.9 | 1.08 |
| **Rubin (DDF)** | Shankar | 35.5 | 31.8 | 1.11 |
| **Roman wide** | Illustris | 58.9 | 72.3 | 0.81 |
| **Roman wide** | Shankar | 30.1 | 33.9 | 0.89 |
| **Roman deep** | Illustris | 35.3 | 36.0 | 0.98 |
| **Roman deep** | Shankar | 15.2 | 14.2 | 1.07 |

Six of eight values are within 15% of the paper. The remaining differences (Rubin Illustris at 0.80, Roman wide Illustris at 0.81) arise from:

1. **Broadband K-correction**: We use a top-hat filter approximation; the paper uses `sncosmo` with realistic filter curves
2. **Lightcurve parameters**: We average over the van Velzen (2021) parameter grid; the paper Monte Carlos over the full ZTF sample
3. **Cosmology**: We use Simpson's rule integration; the paper uses Romberg integration

### Survey specifications

| Survey | Area | Filter | Depth | $\epsilon$ | Seasonal |
|--------|------|--------|-------|-----------|----------|
| Rubin (LSST) | 18,000 deg² | g (482 nm) | 25.0 | 1 (time-domain) | 1.0 |
| Rubin (DDF) | 50 deg² | g (482 nm) | 26.0 | 1 (time-domain) | 0.5 |
| Roman wide | 19 deg² | F062 (620 nm) | 25.95 | 1 (time-domain) | 1.0 |
| Roman deep | 6 deg² | F062 (620 nm) | 26.95 | 1 (time-domain) | 1.0 |

### Usage

```python
from survey_sim import TdeRateForecast

forecast = TdeRateForecast(temperature_k=30000.0, n_mc=200, seed=42)

# Built-in survey names: rubin, rubin_ddf, roman_wide, roman_deep
result = forecast.compute_rate("rubin", bhmf_model="illustris")
print(f"Rubin Illustris: {result['N_median']:.0f} "
      f"[{result['N_16']:.0f}, {result['N_84']:.0f}] TDEs/yr")
# Rubin Illustris: 21383 [15276, 28091] TDEs/yr

# Custom survey
result = forecast.compute_rate_custom(
    name="ULTRASAT TDE survey",
    area_deg2=204.0, depth=22.5,
    central_wavelength_nm=260.0, width_nm=68.0,
    bhmf_model="illustris", is_time_domain=True,
)
print(f"{result['survey']}: {result['N_median']:.1f} TDEs/yr")
```

---

## Injection vs analytical: measuring cadence efficiency

The injection and analytical approaches answer different questions:

- **Analytical** ($\epsilon = 1$): How many TDEs are *above the depth limit* at any given time?
- **Injection**: How many TDEs actually *pass the detection criteria* through the real survey cadence?

### Rubin cadence efficiency

Comparing injection (no host cut, with evolution) to analytical at matched BHMF:

| BHMF | Injection (N/yr) | Analytical (N/yr) | Cadence efficiency |
|------|-------------------|-------------------|--------------------|
| Illustris | 6,585 | 21,383 | 0.31 |
| Shankar | 4,082 | 13,135 | 0.31 |

The cadence efficiency of **~31%** is consistent across both BHMF models, meaning Rubin's real cadence detects about one-third of TDEs that are above the depth limit. The remaining ~69% are lost to:

- **Cadence gaps**: TDE peaks falling between visits
- **Detection threshold**: Requiring $\geq 3$ detections at $\geq 5\sigma$ with $\geq 48$hr separation
- **Seasonal gaps**: Some sky regions have months-long gaps between observations
- **Band coverage**: Requiring detections in $\geq 1$ band with $\geq 2$ per band

This $\epsilon \approx 0.31$ is the key result that injection simulations provide beyond analytical forecasts. It can be used to calibrate analytical predictions for realistic survey performance.

### Effect of host brightness cut

The host brightness cut further reduces the sample by removing faint TDEs that cannot outshine their host nucleus:

| Cut | Injection (N/yr) | Factor |
|-----|-------------------|--------|
| No host cut, Illustris | 6,585 | 1.0× |
| Host cut ($m_0 = 22.5$), Illustris | 928 | 0.14× |

The host cut removes ~86% of photometrically detectable TDEs — primarily faint events near the LF break that are too dim to stand out against their host galaxies.

---

## Survey-independent population design

The population is **survey-independent** — both ZTF and Rubin use identical parameters:

```python
pop = TdePopulation(z_max=..., use_luminosity_function=True)
```

The `use_luminosity_function=True` flag activates:

1. **LF-based magnitude drawing**: Peak $M_g$ drawn from Yao+2023 via rejection sampling ($M_g = -24$ to $-15$)
2. **Auto-computed rate**: Volumetric rate from LF integration (~829 Gpc⁻³ yr⁻¹)

Adding `use_rate_evolution=True` further activates:

3. **Evolved redshift sampling**: $dN/dz \propto dV/dz \times \mathcal{F}(z) \times N_\mathrm{BH}(z) \times \mathcal{O}(z) / (1+z)$
4. **BHMF model selection**: `bhmf_model="illustris"` or `"shankar"`

The only survey-dependent parameter is the **host brightness cut** ($m_0$), which encodes how bright a TDE must be to outshine its host at the survey's depth.

---

## Architecture

The computation is split across Rust modules:

| Module | Purpose |
|--------|---------|
| `lightcurve::kcorrection` | General-purpose broadband K-corrections (`Sed` trait, `BlackbodySed`, `PowerLawSed`, `TopHatFilter`) |
| `efficiency::tde` | TDE rate forecast (`TdeLuminosityFunction`, `BhmfModel`, scaling factors, `compute_tde_rate`) |
| `population::generator` | `TdePopulation` with optional rate evolution (`sample_redshift_evolved`) |
| PyO3 bindings (`py_rates`) | `TdeRateForecast`, `blackbody_k_correction`, `blackbody_k_correction_instrument` |

### Performance

| Implementation | Time (8 survey×BHMF combos) | Speedup |
|----------------|---------------------------|---------|
| Python (astropy cosmology) | 4.3 s | 1× |
| **Rust (survey_sim)** | **0.07 s** | **60×** |

The Rust implementation pre-computes cosmology, K-corrections, and the LF integration grid once, then runs the MC loop with minimal overhead.

---

## Scripts

| Script | Description |
|--------|-------------|
| `python/scripts/run_tde_ztf.py` | ZTF Yao+2023 validation (100K injections) |
| `python/scripts/run_tde_rubin.py` | Rubin 10-year prediction (5 configs + analytical comparison) |
| `python/scripts/run_tde_rates_forecast.py` | Semi-analytical forecast for all surveys |

## References

- Karmen et al. (2026), [arXiv:2602.04947](https://arxiv.org/abs/2602.04947) — TDE rate forecasts for LSST, Roman, and JWST
- Yao et al. (2023), [arXiv:2303.06523](https://arxiv.org/abs/2303.06523) — ZTF TDE sample and g-band luminosity function
- van Velzen (2021), [ApJ 908 4](https://doi.org/10.3847/1538-4357/abd1df) — TDE lightcurve parameterization
- Shankar et al. (2009), [ApJ 690 20](https://doi.org/10.1088/0004-637X/690/1/20) — SMBH mass function evolution
- Bricman & Gomboc (2020), [ApJ 890 73](https://doi.org/10.3847/1538-4357/ab6989) — LSST TDE predictions
