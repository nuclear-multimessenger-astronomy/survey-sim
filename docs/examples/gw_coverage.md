# GW Event Coverage

This page demonstrates how to compute 2D and 3D probability coverage of a gravitational-wave skymap using `survey_sim.Skymap` (Rust), with GW190425 as a worked example.

## Background

When a GW event is detected, the sky localization is distributed as a HEALPix probability map. For wide-field surveys like ZTF, the key metric is the **probability covered** — the fraction of the skymap's integrated probability enclosed by the survey's observed fields.

For 3D coverage, each pixel also carries a distance posterior. The **detectable probability** weights each observed pixel by the fraction of the distance posterior within the survey's detection horizon (set by limiting magnitude and kilonova model).

## GW190425 (S190425z)

GW190425 was a binary neutron star merger detected by LIGO Livingston on 2019-04-25 08:18:05 UTC. As a single-detector event, the localization was extremely broad (~10,000 deg²). ZTF conducted Target of Opportunity observations over two nights.

### ZTF CCD Geometry

We model the ZTF focal plane as 64 rectangular readout channels following [Bellm et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019PASP..131a8002B), matching the [m4opt](https://github.com/m4opt/m4opt) instrument definition. The ZTF field grid (`ZTF_Fields.txt`) is bundled in `survey_sim/data/`.

| Parameter | Value |
|-----------|-------|
| CCD mosaic | 4 × 4 CCDs, 4 readout channels each |
| Plate scale | 1.01"/pixel |
| CCD size | 6160 × 6144 pixels (1.728° × 1.724°) |
| RC size | 3080 × 3072 pixels (0.864° × 0.862°) |
| RC half-widths | 0.432° (RA) × 0.431° (Dec) |
| NS chip gap | 0.205° |
| EW chip gap | 0.140° |
| Active area | 47.7 deg² |

### 2D Coverage Results

Computed with `survey_sim.Skymap.coverage_2d()` (Rust, HEALPix NSIDE=256):

| Skymap | 90% Area | Night 1 Prob | Total Prob | Paper Value |
|--------|----------|-------------|------------|-------------|
| **BAYESTAR** | 10,183 deg² | 45.9% | 52.2% | 46% |
| **LALInference** | 7,461 deg² | 24.5% | 28.2% | 21% |
| **Publication (2020)** | 9,881 deg² | 35.6% | 40.2% | — |

Our values are systematically ~7% higher than [Coughlin et al. (2019)](https://arxiv.org/abs/1907.12645) because:

1. The boom data includes all ZTF observing programs (public survey + Caltech time + ToO), not just the dedicated GW follow-up
2. The paper accounts for per-CCD processing failures that reduce effective coverage
3. Night 1 BAYESTAR probability (45.9%) closely matches the paper's total (46%), consistent with the paper noting that Night 2 added little BAYESTAR probability

### 3D Distance-Weighted Coverage

#### Flat absolute magnitude cuts

The simplest 3D approach uses a flat detection horizon set by a fixed absolute magnitude and ZTF's median depth of 21.0 mag. Computed with `survey_sim.CoverageResult.coverage_3d()` (Rust Monte Carlo, 2000 samples/pixel):

| M_abs | d_max (Mpc) | BAYESTAR 3D | LALInference 3D | Publication 3D |
|-------|-------------|-------------|-----------------|----------------|
| -15.0 | 158 | 36.8% | 20.4% | 27.9% |
| -16.0 | 251 | 51.8% | 28.0% | 39.9% |
| -17.0 | 398 | 52.2% | 28.2% | 40.2% |

At M = -15 (fainter KN), the LALInference 3D probability is 20.4%, matching the paper's 21%. This suggests the paper's effective coverage accounts for a detection horizon cut similar to this brightness level.

#### Time-dependent Bu2026 kilonova model (per-RC)

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

Each ZTF readout channel observation gets its own detection horizon `d_max`, computed from the Bu2026 model at that observation's time-since-event and limiting magnitude. For pixels covered by multiple observations, the best (highest) `d_max` is kept. This is computed entirely in Rust via `survey_sim.Skymap.coverage_2d_3d()`.

| Time post-event | Detection horizon (Mpc) |
|-----------------|------------------------|
| 0.3 days (7h) | 320 |
| 0.5 days (12h) | 310 |
| 1.0 days | 260 |
| 1.5 days | 220 |
| 2.0 days | 160 |

Per-observation statistics: Night 1 median `d_max` = 220 Mpc, Night 2 median `d_max` = 170 Mpc.

| Skymap | Bu2026 3D (Total) | Bu2026 3D (Night 1 only) |
|--------|-------------------|--------------------------|
| **BAYESTAR** | 48.0% | 39.3% |
| **LALInference** | 25.8% | 20.4% |
| **Publication (2020)** | 36.9% | 30.2% |

The LALInference Night 1 Bu2026 3D (20.4%) matches the paper's reported 21% total coverage, consistent with the paper using only ToO data and accounting for processing failures.

### Depth Statistics

| Band | Night 1 | Night 2 | Combined |
|------|---------|---------|----------|
| g | 20.5 mag | 20.7 mag | 20.7 mag |
| r | 20.4 mag | 20.7 mag | 20.5 mag |

Paper values: 21.0 r-band, 20.9 g-band (5σ median). The ~0.3 mag difference is because the paper quotes depths for the ToO observations (30s Night 1, 90s Night 2), while the boom data averages over all programs.

## Running the Example

```bash
python python/scripts/run_gw190425_coverage.py
```

The script requires:

- GW190425 skymaps (BAYESTAR, LALInference, and/or publication samples in multi-order FITS format)
- ZTF boom-pipeline monthly HDF5 files (auto-downloaded from HuggingFace)
- Python packages: `ligo.skymap`, `healpy`, `astropy`, `fiesta`, `jax`
- For GPU-accelerated Bu2026 model evaluation: CUDA toolkit with `survey_sim.gpu_setup` (preloads cuSPARSE)

### Architecture

The coverage computation is split between Python (I/O, model evaluation) and Rust (HEALPix coverage, Monte Carlo integration):

1. **Load skymap** (Python): `ligo.skymap.io.read_sky_map()` reads multi-order FITS, `rasterize()` converts to flat NSIDE=256
2. **Construct Skymap** (Rust): `survey_sim.Skymap.from_arrays(nside, prob, distmu, distsigma, distnorm)`
3. **Load ZTF observations** (Python): boom HDF5 files provide per-RC (ra, dec, depth, mjd) data
4. **Compute per-RC d_max** (Python): Bu2026 model on a (time, distance) grid → `RegularGridInterpolator` → per-observation detection horizon
5. **Compute 2D+3D coverage** (Rust): `skymap.coverage_2d_3d(obs_ra, obs_dec, hw_ra, hw_dec, obs_d_max)` — resolves overlapping pixels by keeping the best d_max, then runs Monte Carlo 3D integration

```python
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize
import healpy as hp
import numpy as np
from survey_sim import Skymap

# Load and rasterize skymap
skymap_moc = read_sky_map("bayestar.fits.gz,0", moc=True, distances=True)
skymap_flat = rasterize(skymap_moc, order=hp.nside2order(256))

# Build Rust Skymap from arrays
skymap = Skymap.from_arrays(
    256,
    skymap_flat["PROB"].tolist(),
    skymap_flat["DISTMU"].tolist(),
    skymap_flat["DISTSIGMA"].tolist(),
    skymap_flat["DISTNORM"].tolist(),
)

# 2D coverage only
result_2d = skymap.coverage_2d(obs_ra, obs_dec, 0.432, 0.431)
print(f"2D: {result_2d.prob_2d:.1%}, {result_2d.area_deg2:.0f} deg²")

# Combined 2D+3D with per-observation detection horizons
# obs_d_max[k] = max detectable distance (Mpc) for observation k
result = skymap.coverage_2d_3d(
    obs_ra, obs_dec, 0.432, 0.431,
    obs_d_max,  # per-RC d_max from Bu2026 model
    n_samples=2000, seed=42,
)
print(f"2D: {result.prob_2d:.1%}, 3D: {result.prob_3d:.1%}")
```
