# Tidal Disruption Events

This page summarizes TDE detection simulations for both **ZTF** and **Rubin LSST**, using the parametric TDE lightcurve model with peak magnitudes drawn from the Yao et al. (2023) broken power-law luminosity function.

## Summary of results

| Survey | Duration | z_max | Injections | Efficiency | Expected | Observed |
|--------|----------|-------|-----------|------------|----------|----------|
| **ZTF** (Yao+2023 validation) | 2.99 yr | 0.3 | 100K | 0.26% | 32.7 | 33 |
| **Rubin** (photometric) | 10 yr | 0.8 | 100K | 1.54% | 26,000 (2,600/yr) | — |

The ZTF simulation reproduces the Yao et al. (2023) spectroscopic sample to **1% accuracy**. Both surveys use identical population parameters — only the detection criteria differ.

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

Peak absolute magnitudes are drawn directly from the **Yao et al. (2023) broken power-law luminosity function** rather than a simple Gaussian. This ensures both ZTF and Rubin use identical population parameters — the survey depth naturally determines which TDEs are detected.

$$\phi(L_g) = \frac{N_0}{\left(L_g / L_\mathrm{bk}\right)^{\gamma_1} + \left(L_g / L_\mathrm{bk}\right)^{\gamma_2}}$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| $N_0$ | $2.87 \times 10^{-7}$ Mpc⁻³ yr⁻¹ dex⁻¹ | Normalization |
| $\log_{10} L_\mathrm{bk}$ | 43.13 | Break luminosity (erg/s) |
| $\gamma_1$ | 0.26 | Faint-end slope (shallow) |
| $\gamma_2$ | 2.58 | Bright-end slope (steep) |

The luminosity-to-magnitude conversion uses:

$$M_g = -2.5 \log_{10} L_g + 88.6$$

derived from $\nu L_\nu$ at $\nu_g = 6.3 \times 10^{14}$ Hz evaluated at 10 pc.

### Integrated rate

Integrating the LF from $M_g = -24$ to $-15$ gives a total volumetric rate of **~829 Gpc⁻³ yr⁻¹**. This is much higher than the effective ZTF-detectable rate (~8 Gpc⁻³ yr⁻¹) because the LF is dominated by faint TDEs ($\gamma_1 = 0.26$ is a shallow faint-end slope, meaning many low-luminosity events). The simulation draws from the full LF and lets the survey sensitivity reject the undetectable events.

Peak magnitudes are drawn via **rejection sampling** against the LF envelope, which peaks at the faint end.

---

## ZTF validation (Yao et al. 2023)

### Setup

Reproduces the flux-limited TDE sample from [Yao et al. (2023, arXiv:2303.06523)](https://arxiv.org/abs/2303.06523): **33 spectroscopically classified TDEs** over 3 years of ZTF operation (October 2018 – September 2021).

```python
from survey_sim import (
    TdePopulation, ParametricModel, DetectionCriteria,
    SimulationPipeline, load_ztf_survey,
)

survey = load_ztf_survey(start="201810", end="202109", nside=64)

# Peak magnitudes drawn from Yao+2023 broken power-law LF.
# Rate auto-computed from LF integration (~829 Gpc^-3/yr).
pop = TdePopulation(z_max=0.3, use_luminosity_function=True)
model = ParametricModel()  # built-in Rust TDE model, no GPU needed
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
| **Host brightness cut** | Logistic: $k = 5.0$, $m_0 = 18.8$ |

### Host brightness cut

TDEs must outshine their host galaxy nucleus to be identified. Yao et al. required peak apparent magnitude brighter than ~18.75 (ZTF-I) or ~19.1 (ZTF-II) for spectroscopic classification. We model this as a steep logistic function applied to each detected transient's peak magnitude:

$$P(\text{identified}) = \frac{1}{1 + e^{k \cdot (m_\mathrm{peak} - m_0)}}$$

with $k = 5.0$ and $m_0 = 18.8$, approximating a hard cut at peak mag ~18.8 — about **1.7 mag brighter** than ZTF's 5σ limit (~20.5).

```python
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=5.0,
    min_detections=3,
    min_detections_primary=3,
    min_bands=1,
    min_per_band=2,
    max_timespan_days=365.0,
    min_time_separation_hours=48.0,
    require_fast_transient=False,
    min_galactic_lat=15.0,
    spectroscopic_completeness_k=5.0,
    spectroscopic_completeness_m0=18.8,
)
```

### Results (100K injections)

| Quantity | Value |
|----------|-------|
| Comoving volume ($z < 0.3$) | 11.0 Gpc³ |
| ZTF sky fraction | 47% |
| Survey duration | 2.99 yr |
| LF-integrated rate | 829 Gpc⁻³ yr⁻¹ |
| Total TDEs in volume | ~12,800 |
| Detection efficiency | 0.26% |
| **Expected detections** | **32.7** |
| **Yao+2023 actual** | **33** |
| **Agreement** | **1%** |

!!! note "Why is the efficiency so low?"
    The LF-integrated rate includes all TDEs down to $M_g = -15$, most of which
    are far too faint for ZTF to detect even at low redshift. The 0.26% efficiency
    reflects the tiny fraction of the LF that produces TDEs bright enough
    ($m_\mathrm{peak} < 18.8$) to be identified against their host galaxies at
    ZTF depth. This is exactly the point — the population model is survey-independent,
    and the detection criteria encode what each survey can actually find.

---

## Rubin LSST prediction

### Setup

Predicts the number of TDEs that Rubin will detect over 10 years. Uses **identical population parameters** as ZTF — only the detection criteria change.

```python
from survey_sim import (
    SurveyStore, TdePopulation, ParametricModel,
    DetectionCriteria, SimulationPipeline,
)

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db", nside=64)

# Same LF-based population as ZTF — identical parameters.
pop = TdePopulation(z_max=0.8, use_luminosity_function=True)
model = ParametricModel()
```

### Detection criteria

Same photometric quality cuts as ZTF, but with a host brightness cut scaled to Rubin's depth:

| Criterion | Value |
|-----------|-------|
| SNR threshold | $\geq 5\sigma$ |
| Min detections | $\geq 3$ total |
| Min per band | $\geq 2$ |
| Max timespan | 365 days |
| Min time separation | $\geq 48$ hours |
| Galactic latitude | $\|b\| > 15°$ |
| **Host brightness cut** | Logistic: $k = 5.0$, $m_0 = 22.5$ |

The host brightness cut requires peak apparent magnitude ~2 mag brighter than Rubin's single-visit 5σ limit (~24.5), i.e., $m_\mathrm{peak} \lesssim 22.5$. This accounts for the requirement that TDEs must outshine their host nucleus in difference imaging.

```python
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=5.0,
    min_detections=3,
    min_detections_primary=3,
    min_bands=1,
    min_per_band=2,
    max_timespan_days=365.0,
    min_time_separation_hours=48.0,
    require_fast_transient=False,
    min_galactic_lat=15.0,
    spectroscopic_completeness_k=5.0,
    spectroscopic_completeness_m0=22.5,
)
```

### Results (100K injections)

| Quantity | Value |
|----------|-------|
| Comoving volume ($z < 0.8$) | 463 Gpc³ |
| Rubin sky fraction | 44% |
| Survey duration | 10 yr |
| LF-integrated rate | 829 Gpc⁻³ yr⁻¹ |
| Total TDEs in volume | ~1,689,000 |
| Detection efficiency | 1.54% |
| **Expected detections** | **~26,000 in 10yr** |
| **Expected per year** | **~2,600** |

### Comparison with literature

| Source | Prediction |
|--------|-----------|
| **This work** | ~2,600/yr |
| Bricman & Gomboc (2020) | ~1,000/yr |
| van Velzen et al. (2011) | ~$10^{3}$–$10^{4}$/yr |

Our prediction is 2.6× higher than Bricman & Gomboc (2020), who used a more conservative luminosity function. It falls within the range of van Velzen et al. (2011).

### Why does Rubin detect so many more?

Rubin's advantage is threefold:

1. **Depth**: Single-visit 5σ limit ~24.5 vs ZTF's ~20.5. The host brightness cut allows TDEs to $m \sim 22.5$ (vs ZTF's ~18.8), reaching intrinsically fainter events.
2. **Volume**: $z_\mathrm{max} = 0.8$ vs 0.3 → 42× larger comoving volume (× sky fraction).
3. **LF sampling**: The shallow faint-end slope ($\gamma_1 = 0.26$) means Rubin's extra 3.7 mag of depth probes significantly more of the LF.

---

## Key design: survey-independent population

The critical feature of this approach is that the **population is survey-independent**:

```python
# Both surveys use identical population parameters:
pop = TdePopulation(z_max=..., use_luminosity_function=True)
```

The `use_luminosity_function=True` flag activates:

1. **LF-based magnitude drawing**: Peak $M_g$ drawn from the Yao+2023 broken power-law LF via rejection sampling (range $M_g = -24$ to $-15$)
2. **Auto-computed rate**: Volumetric rate obtained by integrating the LF (~829 Gpc⁻³ yr⁻¹)

The only survey-dependent parameter is the **host brightness cut** ($m_0$), which encodes "how bright must a TDE be to outshine its host at this survey's depth?" This is physically motivated — the same TDE physics applies everywhere, but shallow surveys can only identify the brightest events.

!!! tip "Legacy mode"
    For backward compatibility, `TdePopulation(rate=8.0, z_max=0.3, peak_abs_mag=-20.0)` still
    works — it uses a Gaussian magnitude distribution with the specified rate. Set
    `use_luminosity_function=True` to use the LF-based approach.

---

## Scripts

| Script | Description |
|--------|-------------|
| `python/scripts/run_tde_ztf.py` | ZTF Yao+2023 validation (100K injections) |
| `python/scripts/run_tde_rubin.py` | Rubin 10-year TDE prediction (100K injections) |

## References

- Yao et al. (2023), [arXiv:2303.06523](https://arxiv.org/abs/2303.06523) — ZTF TDE sample (33 TDEs, broken power-law LF)
- Bricman & Gomboc (2020), [ApJ 890 73](https://doi.org/10.3847/1538-4357/ab6989) — LSST TDE predictions
- van Velzen et al. (2011), [ApJ 741 73](https://doi.org/10.1088/0004-637X/741/2/73) — TDE rate estimates for optical surveys
