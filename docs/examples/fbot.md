# Fast Blue Optical Transients

This page summarizes FBOT detection simulations for both **ZTF** and **Rubin LSST**, using the Bazin parametric model with timescales drawn from the Ho et al. (2021) ZTF Phase I sample.

## Summary of results

| Survey | Duration | z_max | Injections | Efficiency | Expected | Observed |
|--------|----------|-------|-----------|------------|----------|----------|
| **ZTF** (Ho+2021 validation) | 2.67 yr | 0.3 | 100K | 4.2% | 38.1 | 38 |
| **Rubin** (photometric) | 10 yr | 0.5 | 100K | 12.4% | 3,600 (360/yr) | — |

The ZTF simulation reproduces the Ho et al. (2021) sample to **< 1% accuracy**. Rubin is predicted to detect ~360 FBOTs per year, a ~25× increase over ZTF.

---

## Background

Fast Blue Optical Transients (FBOTs) are a class of rapidly evolving transients with rise times of ~1–5 days and half-light durations of 1–12 days. They are blue at peak (g−r < −0.2 mag) and span a wide luminosity range ($M_g$ = −16 to −21). The most luminous and exotic subclass — exemplified by AT2018cow ("the Cow") — shows featureless spectra, luminous radio/X-ray emission, and may be powered by central engine activity (accreting black hole or magnetar).

Ho et al. (2021, [arXiv:2105.08811](https://arxiv.org/abs/2105.08811)) conducted a systematic search of ZTF Phase I data (March 2018 – October 2020), identifying **38 FBOTs** with well-sampled lightcurves.

### Lightcurve model

We use the **Bazin model** (Bazin et al. 2009), a simple analytical rise/fall profile commonly used for fast transients:

$$F(t) = A \cdot \frac{e^{-(t-t_0)/\tau_\mathrm{fall}}}{1 + e^{-(t-t_0)/\tau_\mathrm{rise}}} + c$$

This is a built-in Rust model evaluated via `ParametricModel()` — no Python callback or GPU required.

### Timescale parameters

Drawn from distributions calibrated to the Ho et al. (2021) Table 10 sample (g-band):

| Parameter | Distribution | Ho+2021 range |
|-----------|-------------|---------------|
| $t_{1/2,\mathrm{rise}}$ | $\mathcal{N}(2.5, 1.2)$, clamp [0.3, 6.0] days | 0.5–4.6 days |
| $t_{1/2,\mathrm{fade}}$ | $\mathcal{N}(5.5, 2.0)$, clamp [1.0, 15.0] days | 1.5–8.0 days |

The Bazin e-folding times are related to observed half-light times by $\tau \approx t_{1/2} / \ln 2$.

### Peak magnitudes

From Ho+2021 Table 10, the 38 FBOTs span $M_g$ = −16.4 to −21.2 with a median of approximately −18.7. We draw peak absolute magnitudes from $\mathcal{N}(-18.7, 1.5)$, clamped to [−22.7, −14.7].

Selected events from Table 10:

| Event | $M_g$ | $t_\mathrm{rise}$ (d) | $t_\mathrm{fade}$ (d) | Type |
|-------|-------|----------------------|----------------------|------|
| AT2018lug | −21.17 | 1.12 | 2.92 | Exotic (cow-like) |
| AT2020xnd | −21.03 | 1.6–4.8 | 2.39 | Exotic (cow-like) |
| SN2018gep | −19.84 | 3.27 | 6.00 | SN Ic-BL |
| SN2019deh | −19.73 | 4.35 | 6.33 | SN II |
| SN2018bcc | −19.82 | 3.20 | 5.87 | SN Ib |
| SN2020rsc | −16.37 | 1.62 | 1.72 | SN Ibn |

### Volumetric rate

Ho et al. (2021) find that FBOTs represent approximately **0.1% of the local core-collapse supernova rate**. With a CC SN rate of ~70,000 Gpc⁻³ yr⁻¹ (Perley et al. 2020), this gives:

$$R_\mathrm{FBOT} \approx 65 \text{ Gpc}^{-3} \text{ yr}^{-1}$$

This rate encompasses all FBOTs with $1 < t_{1/2} < 12$ days and blue colors (g−r < −0.2), not just the exotic AT2018cow-like subset.

---

## ZTF validation (Ho et al. 2021)

### Setup

Reproduces the FBOT sample from [Ho et al. (2021)](https://arxiv.org/abs/2105.08811): **38 FBOTs** from ZTF Phase I (March 2018 – October 2020).

```python
from survey_sim import (
    FbotPopulation, ParametricModel, DetectionCriteria,
    SimulationPipeline, load_ztf_survey,
)

survey = load_ztf_survey(start="201803", end="202010", nside=64)
pop = FbotPopulation(rate=65.0, z_max=0.3, peak_abs_mag=-18.7)
model = ParametricModel()  # built-in Rust Bazin model
```

### Detection criteria

Ho+2021 applied strict selection criteria: fast-rising ($\geq 1$ mag in 6.5 days), well-sampled near peak in both g and r, blue color (g−r < −0.2), and spectroscopic classification. We approximate these with:

| Criterion | Value |
|-----------|-------|
| SNR threshold | $\geq 5\sigma$ |
| Min detections | $\geq 5$ |
| Min bands | $\geq 2$ (g + r) |
| Min per band | $\geq 2$ |
| Max timespan | 24 days |
| Min time separation | $\geq 24$ hours |
| Fast transient | Required |
| Min rise rate | 0.15 mag/day ($\geq 1$ mag in 6.5d) |
| Min fade rate | 0.1 mag/day |
| Pre-peak detections | $\geq 1$ |
| Post-peak detections | $\geq 1$ |
| Phase range | $\geq 3$ days |
| Galactic latitude | $\|b\| > 7°$ |
| **Host brightness cut** | Logistic: $k = 3.0$, $m_0 = 19.0$ |

```python
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=5.0,
    min_detections=5,
    min_detections_primary=5,
    min_bands=2,
    min_per_band=2,
    max_timespan_days=24.0,
    min_time_separation_hours=24.0,
    require_fast_transient=True,
    min_rise_rate=0.15,
    min_fade_rate=0.1,
    min_pre_peak_detections=1,
    min_post_peak_detections=1,
    min_phase_range_days=3.0,
    min_galactic_lat=7.0,
    spectroscopic_completeness_k=3.0,
    spectroscopic_completeness_m0=19.0,
)
```

### Results (100K injections)

| Quantity | Value |
|----------|-------|
| Comoving volume ($z < 0.3$) | 11.0 Gpc³ |
| ZTF sky fraction | 47% |
| Survey duration | 2.67 yr |
| Volumetric rate | 65 Gpc⁻³ yr⁻¹ |
| Total FBOTs in volume | ~900 |
| Detection efficiency | 4.2% |
| **Expected detections** | **38.1** |
| **Ho+2021 actual** | **38** |
| **Agreement** | **< 1%** |

---

## Rubin LSST prediction

### Setup

Predicts the number of FBOTs that Rubin will detect over 10 years. Uses the same population parameters as ZTF.

```python
from survey_sim import (
    SurveyStore, FbotPopulation, ParametricModel,
    DetectionCriteria, SimulationPipeline,
)

survey = SurveyStore.from_rubin("baseline_v5.1.1_10yrs.db", nside=64)
pop = FbotPopulation(rate=65.0, z_max=0.5, peak_abs_mag=-18.7)
model = ParametricModel()
```

### Detection criteria

Relaxed relative to ZTF — Rubin will identify fast transients photometrically from multi-band difference imaging without requiring spectroscopic classification. A host brightness cut at $m_0 = 22.5$ (2 mag brighter than the 5σ limit) models the requirement that the transient must outshine its host.

| Criterion | Value |
|-----------|-------|
| SNR threshold | $\geq 5\sigma$ |
| Min detections | $\geq 3$ |
| Min bands | $\geq 2$ |
| Min per band | $\geq 2$ |
| Max timespan | 24 days |
| Min time separation | $\geq 24$ hours |
| Fast transient | Required |
| Min rise rate | 0.15 mag/day |
| Min fade rate | 0.1 mag/day |
| Galactic latitude | $\|b\| > 15°$ |
| **Host brightness cut** | Logistic: $k = 5.0$, $m_0 = 22.5$ |

```python
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=5.0,
    min_detections=3,
    min_detections_primary=3,
    min_bands=2,
    min_per_band=2,
    max_timespan_days=24.0,
    min_time_separation_hours=24.0,
    require_fast_transient=True,
    min_rise_rate=0.15,
    min_fade_rate=0.1,
    min_galactic_lat=15.0,
    spectroscopic_completeness_k=5.0,
    spectroscopic_completeness_m0=22.5,
)
```

### Results (100K injections)

| Quantity | Value |
|----------|-------|
| Comoving volume ($z < 0.5$) | 102 Gpc³ |
| Rubin sky fraction | 44% |
| Survey duration | 10 yr |
| Volumetric rate | 65 Gpc⁻³ yr⁻¹ |
| Total FBOTs in volume | ~29,200 |
| Detection efficiency | 12.4% |
| **Expected detections** | **~3,600 in 10yr** |
| **Expected per year** | **~360** |

### Why does Rubin find so many more?

Rubin detects ~25× more FBOTs per year than ZTF (360 vs 14). The improvement comes from:

1. **Depth**: Rubin's 5σ limit (~24.5) is ~4 mag deeper than ZTF (~20.5). With a host brightness cut at 22.5 vs 19.0, Rubin can identify FBOTs 3.5 mag fainter.
2. **Volume**: $z_\mathrm{max} = 0.5$ vs 0.3 → ~9× larger effective volume.
3. **Higher efficiency**: 12.4% vs 4.2%, because Rubin's multi-band coverage and deeper imaging make it easier to satisfy the fast-transient detection criteria.

!!! note "Cadence challenge"
    FBOTs evolve on timescales of days, but Rubin's WFD revisits each field every
    ~3 days. Some FBOTs may peak and fade between visits. The 12.4% efficiency
    already accounts for this cadence limitation — events that fall entirely
    between visits are naturally rejected by the detection criteria.

---

## FBOT subclasses

Ho et al. (2021) classified their 38 FBOTs spectroscopically:

| Classification | Count | Fraction |
|----------------|-------|----------|
| Type II/IIb/Ib (H- or He-rich) | 11 | 29% |
| Type IIn/Ibn (interacting) | 6 | 16% |
| Type Ic/Ic-BL (stripped) | 2 | 5% |
| Exotic (AT2018cow-like) | 2 | 5% |
| Unclassified / no spectrum | 17 | 45% |

The exotic AT2018cow-like events (AT2018lug, AT2020xnd) are the rarest and most luminous ($M_g < -21$), with featureless spectra and luminous radio/X-ray emission. They represent a small fraction of the broader FBOT population, which is dominated by rapidly evolving core-collapse supernovae.

---

## Scripts

| Script | Description |
|--------|-------------|
| `python/scripts/run_fbot_ztf.py` | ZTF Ho+2021 validation (100K injections) |
| `python/scripts/run_fbot_rubin.py` | Rubin 10-year FBOT prediction (100K injections) |

## References

- Ho et al. (2021), [arXiv:2105.08811](https://arxiv.org/abs/2105.08811) — ZTF FBOT sample (38 events, rate measurement)
- Bazin et al. (2009), [A&A 499 653](https://doi.org/10.1051/0004-6361/200911847) — Bazin lightcurve model
- Perley et al. (2020), [ApJ 904 35](https://doi.org/10.3847/1538-4357/abbd98) — ZTF Bright Transient Survey, CC SN rate
