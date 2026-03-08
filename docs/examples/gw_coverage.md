# GW Event Coverage

This page demonstrates how to compute 2D and 3D probability coverage of a gravitational-wave skymap using survey-sim, with GW190425 as a worked example.

## Background

When a GW event is detected, the sky localization is distributed as a HEALPix probability map. For wide-field surveys like ZTF, the key metric is the **probability covered** — the fraction of the skymap's integrated probability enclosed by the survey's observed fields.

For 3D coverage, each pixel also carries a distance posterior. The **detectable probability** weights each observed pixel by the fraction of the distance posterior within the survey's detection horizon (set by limiting magnitude and kilonova absolute magnitude).

## GW190425 (S190425z)

GW190425 was a binary neutron star merger detected by LIGO Livingston on 2019-04-25 08:18:05 UTC. As a single-detector event, the localization was extremely broad (~10,000 deg²). ZTF conducted Target of Opportunity observations over two nights.

### ZTF CCD Geometry

We model the ZTF focal plane as 64 rectangular readout channels following [Bellm et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019PASP..131a8002B), matching the [m4opt](https://github.com/m4opt/m4opt) instrument definition:

| Parameter | Value |
|-----------|-------|
| CCD mosaic | 4 × 4 CCDs, 4 readout channels each |
| Plate scale | 1.01"/pixel |
| CCD size | 6160 × 6144 pixels (1.728° × 1.724°) |
| RC size | 3080 × 3072 pixels (0.864° × 0.862°) |
| NS chip gap | 0.205° |
| EW chip gap | 0.140° |
| Active area | 47.7 deg² |
| Fill factor | 86.6% |

The chip gaps reduce the effective covered area by ~7% compared to a simple circular field-of-view approximation.

### 2D Coverage Results

| Skymap | 90% Area | Night 1 Prob | Total Prob | Paper Value |
|--------|----------|-------------|------------|-------------|
| **BAYESTAR** | 10,183 deg² | 46.5% | 52.8% | 46% |
| **LALInference** | 7,461 deg² | 24.8% | 28.5% | 21% |
| **Publication (2020)** | 9,881 deg² | 36.1% | 40.6% | — |

Our values are systematically ~7% higher than [Coughlin et al. (2019)](https://arxiv.org/abs/1907.12645) because:

1. The boom data includes all ZTF observing programs (public survey + Caltech time + ToO), not just the dedicated GW follow-up
2. The paper accounts for per-CCD processing failures that reduce effective coverage
3. Night 1 BAYESTAR probability (46.5%) closely matches the paper's total (46%), consistent with the paper noting that Night 2 added little BAYESTAR probability

### 3D Distance-Weighted Coverage

#### Flat absolute magnitude cuts

The simplest 3D approach uses a flat detection horizon set by a fixed absolute magnitude and ZTF's median depth of 21.0 mag:

| M_abs | d_max (Mpc) | BAYESTAR 3D | LALInference 3D |
|-------|-------------|-------------|-----------------|
| -15.0 | 158 | 37.2% | 20.7% |
| -16.0 | 251 | 52.4% | 28.3% |
| -17.0 | 398 | 52.7% | 28.5% |
| -18.0 | 631 | 52.7% | 28.5% |

At M = -15 (fainter KN), the LALInference 3D probability is 20.7%, matching the paper's 21% for the total 2D coverage. This suggests the paper's effective coverage accounts for a detection horizon cut similar to this brightness level.

#### Time-dependent Bu2026 kilonova model

A more physical approach uses the full [Bu2026](https://arxiv.org/abs/2012.04810) kilonova model via [fiesta](https://github.com/nuclear-multimessenger-astronomy/fiesta) with AT2017gfo best-fit parameters:

| Parameter | Value |
|-----------|-------|
| log10(M_ej,dyn) | -1.8 |
| v_ej,dyn | 0.2 c |
| Y_e,dyn | 0.15 |
| log10(M_ej,wind) | -1.1 |
| v_ej,wind | 0.1 c |
| Y_e,wind | 0.35 |
| inclination | 0.45 rad |

The kilonova fades rapidly, so the detection horizon depends on when each pixel was actually observed relative to the GW trigger:

| Time post-event | r-band horizon (Mpc) |
|-----------------|---------------------|
| 0.3 days (7h) | 320 |
| 0.5 days (12h) | 310 |
| 1.0 days | 260 |
| 1.5 days | 220 |
| 2.0 days | 160 |

For each observed pixel, the script maps the actual ZTF observation time to a time-dependent detection horizon using 2D interpolation over (time, distance) → apparent magnitude. Pixels observed earlier have deeper effective 3D coverage.

| Skymap | Bu2026 3D Prob |
|--------|---------------|
| **BAYESTAR** | 48.0% |
| **LALInference** | 25.8% |
| **Publication (2020)** | 36.7% |

The Bu2026 time-dependent 3D coverage is lower than the flat M=-16 case because: (1) the KN fades significantly between Night 1 and Night 2, reducing the effective horizon for later observations; and (2) the proper distance-dependent weighting accounts for the shape of the luminosity distance posterior at each pixel.

### Depth Statistics

| Band | Night 1 | Night 2 | Combined |
|------|---------|---------|----------|
| g | 20.5 mag | 20.7 mag | 20.7 mag |
| r | 20.4 mag | 20.7 mag | 20.5 mag |

Paper values: 21.0 r-band, 20.9 g-band (5σ median). The ~0.3 mag difference is because the paper quotes depths for the ToO observations (30s Night 1, 90s Night 2), while the boom data averages over all programs.

## Running the Example

```python
python python/scripts/run_gw190425_coverage.py
```

The script requires:

- GW190425 skymaps (BAYESTAR, LALInference, and/or publication samples in multi-order FITS format)
- ZTF boom-pipeline monthly HDF5 files (auto-downloaded from HuggingFace)
- Python packages: `ligo.skymap`, `mocpy`, `healpy`, `astropy`, `fiesta`, `jax`
- For GPU-accelerated Bu2026 model evaluation: CUDA toolkit with `survey_sim.gpu_setup` (preloads cuSPARSE)

### Key Steps

1. **Load skymap**: `ligo.skymap.io.read_sky_map()` reads multi-order FITS with 3D distance posteriors
2. **Load ZTF observations**: boom HDF5 files contain per-CCD (ra, dec, depth, field, rcid) data
3. **Build coverage MOC**: `MOC.from_boxes()` creates rectangular MOCs for each readout channel, properly accounting for chip gaps
4. **Compute 2D probability**: `moc.probability_in_multiordermap(skymap)` integrates probability over the observed region
5. **Compute 3D probability**: For each observed pixel, sample from the distance posterior and check detectability
6. **Time-dependent 3D** (Bu2026): Build a 2D interpolation grid of (time_post_event, distance) → apparent magnitude using the Bu2026 model, then for each pixel use the actual observation time to determine the detection horizon

```python
from ligo.skymap.io import read_sky_map
from mocpy import MOC
import astropy.units as u

# Load multi-order skymap
skymap = read_sky_map("GW190425_bayestar.fits", moc=True, distances=True)

# Build rectangular MOC for one ZTF readout channel
moc = MOC.from_boxes(
    lon=[180.0] * u.deg,
    lat=[30.0] * u.deg,
    a=[0.432] * u.deg,   # EW half-width
    b=[0.431] * u.deg,   # NS half-width
    angle=[0.0] * u.deg,
    max_depth=12,
)[0]

# Compute probability in this MOC
prob = moc.probability_in_multiordermap(skymap)
```
