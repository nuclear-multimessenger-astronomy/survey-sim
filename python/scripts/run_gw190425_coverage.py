#!/usr/bin/env python
"""
Reproduce ZTF 2D and 3D coverage for GW190425 (S190425z).

Uses the survey_sim Rust library for skymap coverage computation.

GW190425 was a BNS merger detected by LIGO Livingston on 2019-04-25 08:18:05 UTC.
The single-detector localization spanned ~10,000 deg².
ZTF covered ~8,000 deg² over two nights, enclosing:
  - 46% of the BAYESTAR probability (2D)
  - 21% of the LALInference probability (2D)

ZTF CCD geometry follows Bellm et al. (2019) Table 1, matching the m4opt
representation: 4x4 CCD mosaic with 4 readout channels each (64 rcids).
Chip gaps: 0.205° NS, 0.140° EW. Plate scale 1.01"/pix.

Reference: Coughlin et al. 2019 (arXiv:1907.12645)
"""

import sys
sys.path.insert(0, "/fred/oz480/mcoughli/fiestaEM/src")

import survey_sim.gpu_setup  # noqa: F401 — preload CUDA libs for JAX

from pathlib import Path

import numpy as np
import h5py
import healpy as hp
import astropy.units as u
from astropy.time import Time
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize

from survey_sim import Skymap
from survey_sim.data import download_ztf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SKYMAP_DIR = "/fred/oz480/mcoughli/simulations/GW190425"
SKYMAPS = {
    "BAYESTAR": f"{SKYMAP_DIR}/bayestar.fits.gz,0",
    "LALInference": f"{SKYMAP_DIR}/LALInference.fits.gz,0",
    "Publication (Sep 2020)": f"{SKYMAP_DIR}/GW190425_PublicationSamples.multiorder.fits",
}

EVENT_TIME = Time("2019-04-25T08:18:05", scale="utc")
EVENT_MJD = EVENT_TIME.mjd

# Time windows
NIGHT1_END_MJD = EVENT_MJD + 1.0
NIGHT2_END_MJD = EVENT_MJD + 2.0

# HEALPix resolution for rasterization (nested ordering throughout)
NSIDE = 256

# ZTF field grid (from m4opt / ztf_information, bundled in survey_sim/data/)
ZTF_FIELDS_PATH = str(Path(__file__).resolve().parents[1] / "survey_sim" / "data" / "ZTF_Fields.txt")

# ---------------------------------------------------------------------------
# ZTF CCD Geometry (Bellm+2019 Table 1, matching m4opt)
# ---------------------------------------------------------------------------
PLATE_SCALE_ARCSEC = 1.01  # arcsec/pixel
NS_NPIX = 6144  # pixels per CCD (NS direction)
EW_NPIX = 6160  # pixels per CCD (EW direction)
NS_CHIP_GAP_DEG = 0.205  # degrees
EW_CHIP_GAP_DEG = 0.140  # degrees

# Readout channel half-widths (1 RC = half a CCD in each direction)
RC_HW_RA_DEG = (EW_NPIX / 2 * PLATE_SCALE_ARCSEC / 3600.0) / 2  # 0.432°
RC_HW_DEC_DEG = (NS_NPIX / 2 * PLATE_SCALE_ARCSEC / 3600.0) / 2  # 0.431°

# Active area
TOTAL_RC_AREA = 64 * (EW_NPIX / 2 * PLATE_SCALE_ARCSEC / 3600.0) * (NS_NPIX / 2 * PLATE_SCALE_ARCSEC / 3600.0)

print(f"ZTF CCD geometry (Bellm+2019, m4opt):")
print(f"  RC half-widths: {RC_HW_RA_DEG:.4f}° (RA) × {RC_HW_DEC_DEG:.4f}° (Dec)")
print(f"  Active area (64 RCs): {TOTAL_RC_AREA:.1f} deg²")

# ---------------------------------------------------------------------------
# 1. Load ZTF observations from boom HDF5
# ---------------------------------------------------------------------------
print("\nLoading ZTF observations (April 2019)...")
boom_paths = download_ztf(start="201904", end="201904")

obs_ra = []
obs_dec = []
obs_mjd = []
obs_band = []
obs_depth = []
obs_field = []
obs_rcid = []
obs_programid = []

for bp in boom_paths:
    with h5py.File(str(bp), "r") as f:
        data = f["observations"][:]
        mjd = data["obsjd"] - 2400000.5

        mask = (mjd >= EVENT_MJD) & (mjd <= NIGHT2_END_MJD)
        obs_ra.append(data["ra"][mask])
        obs_dec.append(data["dec"][mask])
        obs_mjd.append(mjd[mask])
        obs_band.append(data["fid"][mask])
        obs_depth.append(data["diffmaglim"][mask])
        obs_field.append(data["field"][mask])
        obs_rcid.append(data["rcid"][mask])
        obs_programid.append(data["programid"][mask])

obs_ra = np.concatenate(obs_ra)
obs_dec = np.concatenate(obs_dec)
obs_mjd = np.concatenate(obs_mjd)
obs_band = np.concatenate(obs_band)
obs_depth = np.concatenate(obs_depth)
obs_field = np.concatenate(obs_field)
obs_rcid = np.concatenate(obs_rcid)
obs_programid = np.concatenate(obs_programid)

print(f"  All observations in window: {len(obs_ra)}")
print(f"  By program: 1(public)={np.sum(obs_programid==1)}, "
      f"2(ToO)={np.sum(obs_programid==2)}, 3(Caltech)={np.sum(obs_programid==3)}")
print(f"  Night 1 (event to +1d): {np.sum(obs_mjd < NIGHT1_END_MJD)}")
print(f"  Night 2 (+1d to +2d): {np.sum(obs_mjd >= NIGHT1_END_MJD)}")

# ---------------------------------------------------------------------------
# 2. Deduplicate to unique RC pointings
# ---------------------------------------------------------------------------
# Round MJD to ~1 min to group same-exposure observations
obs_time_key = np.round(obs_mjd, 3)
pointing_key = (obs_field.astype(np.int64) * 1000000
                + obs_rcid.astype(np.int64) * 10000
                + (obs_time_key * 10 % 10000).astype(np.int64))
_, unique_idx = np.unique(pointing_key, return_index=True)

ura = obs_ra[unique_idx]
udec = obs_dec[unique_idx]
umjd = obs_mjd[unique_idx]

n1_mask = umjd < NIGHT1_END_MJD
print(f"\n  Unique RC pointings: {len(ura)} ({np.sum(n1_mask)} Night 1, {np.sum(~n1_mask)} Night 2)")

# ---------------------------------------------------------------------------
# 3. Rasterize skymaps and compute 2D coverage with survey_sim.Skymap
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"2D COVERAGE (survey_sim.Skymap.coverage_2d)")
print(f"{'='*60}")

skymap_objects = {}

for name, path in SKYMAPS.items():
    print(f"\n  Loading {name} skymap...")
    try:
        skymap_moc = read_sky_map(path, moc=True, distances=True)
    except Exception:
        try:
            skymap_moc = read_sky_map(path, moc=True, distances=False)
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    has_3d = "DISTMU" in skymap_moc.colnames

    # Rasterize to fixed NSIDE (nested ordering)
    skymap_flat = rasterize(skymap_moc, order=hp.nside2order(NSIDE))
    prob = np.asarray(skymap_flat["PROB"], dtype=np.float64)

    # Build survey_sim.Skymap from arrays
    if has_3d:
        distmu = np.asarray(skymap_flat["DISTMU"], dtype=np.float64)
        distsigma = np.asarray(skymap_flat["DISTSIGMA"], dtype=np.float64)
        distnorm = np.asarray(skymap_flat["DISTNORM"], dtype=np.float64)
        skymap = Skymap.from_arrays(
            int(NSIDE),
            prob.tolist(),
            distmu.tolist(),
            distsigma.tolist(),
            distnorm.tolist(),
        )
    else:
        skymap = Skymap.from_arrays(int(NSIDE), prob.tolist())

    # 2D coverage via Rust
    result_total = skymap.coverage_2d(ura.tolist(), udec.tolist(), RC_HW_RA_DEG, RC_HW_DEC_DEG)
    result_n1 = skymap.coverage_2d(
        ura[n1_mask].tolist(), udec[n1_mask].tolist(), RC_HW_RA_DEG, RC_HW_DEC_DEG
    )

    print(f"    90% area:   {skymap.area_90:.0f} deg²")
    print(f"    Night 1:    {result_n1.area_deg2:.0f} deg², {result_n1.prob_2d:.1%} prob")
    print(f"    Total:      {result_total.area_deg2:.0f} deg², {result_total.prob_2d:.1%} prob")
    print(f"    Pixels:     {result_total.n_pixels}")

    # 3D coverage with flat magnitude cuts
    if has_3d:
        for M_abs in [-15.0, -16.0, -17.0]:
            ZTF_DEPTH = 21.0
            d_max = 10 ** ((ZTF_DEPTH - M_abs + 5) / 5) / 1e6  # Mpc
            p3d = result_total.coverage_3d(skymap, d_max, n_samples=2000, seed=42)
            print(f"    3D (M={M_abs:.0f}, d_max={d_max:.0f} Mpc): {p3d:.1%}")

    skymap_objects[name] = {
        "skymap": skymap,
        "result": result_total,
        "prob": prob,
        "has_3d": has_3d,
    }

print(f"\n  Paper values (Coughlin+2019):")
print(f"    Night 1: 3250 deg², 36%/19% BAYESTAR/LALInference")
print(f"    Total:   ~8000 deg², 46%/21% BAYESTAR/LALInference")

# ---------------------------------------------------------------------------
# 4. 3D Time-Dependent Coverage with Bu2026 kilonova model
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"3D TIME-DEPENDENT COVERAGE (Bu2026 + survey_sim.coverage_3d_variable)")
print(f"{'='*60}")

# Map observations to HEALPix pixels for per-pixel timing
npix = 12 * NSIDE * NSIDE
obs_pix = hp.ang2pix(NSIDE, obs_ra, obs_dec, nest=True, lonlat=True)

pixel_first_mjd = np.full(npix, np.inf)
pixel_best_depth_g = np.full(npix, 0.0)
pixel_best_depth_r = np.full(npix, 0.0)

for k in range(len(obs_ra)):
    pix = obs_pix[k]
    if obs_mjd[k] < pixel_first_mjd[pix]:
        pixel_first_mjd[pix] = obs_mjd[k]
    if obs_band[k] == 1 and obs_depth[k] > pixel_best_depth_g[pix]:
        pixel_best_depth_g[pix] = obs_depth[k]
    elif obs_band[k] == 2 and obs_depth[k] > pixel_best_depth_r[pix]:
        pixel_best_depth_r[pix] = obs_depth[k]

pixel_dt = pixel_first_mjd - EVENT_MJD
pixel_best_depth = np.maximum(pixel_best_depth_g, pixel_best_depth_r)

# Fill in MOC-covered but unmatched pixels with median values
n1_median_depth = np.median(obs_depth[(obs_mjd < NIGHT1_END_MJD) & (obs_depth > 0)])
n1_median_dt = 0.3

# We need the covered mask from the total result — use any skymap's result
any_result = next(iter(skymap_objects.values()))["result"]
covered_mask = np.array(any_result.covered)

for i in range(npix):
    if covered_mask[i] and pixel_best_depth[i] == 0:
        pixel_best_depth[i] = n1_median_depth
        pixel_dt[i] = n1_median_dt

observed_dts = pixel_dt[covered_mask & (pixel_dt < np.inf)]
print(f"  Observation time distribution:")
print(f"    Median dt: {np.median(observed_dts):.2f} days ({np.median(observed_dts)*24:.1f} hours)")
print(f"    Night 1 pixels: {np.sum(observed_dts < 1.0)} ({np.sum(observed_dts < 1.0)/len(observed_dts)*100:.0f}%)")
print(f"    Night 2 pixels: {np.sum(observed_dts >= 1.0)} ({np.sum(observed_dts >= 1.0)/len(observed_dts)*100:.0f}%)")

# --- Bu2026 kilonova model: build detection horizon grid ---
from survey_sim.fiesta_model import FiestaKNModel
from astropy.cosmology import Planck18, z_at_value
from scipy.interpolate import RegularGridInterpolator

print("\n  Initializing Bu2026 kilonova model (AT2017gfo best-fit)...")
kn_model = FiestaKNModel()

BU2026_PARAMS = dict(
    log10_mej_dyn=-1.8,
    v_ej_dyn=0.2,
    Ye_dyn=0.15,
    log10_mej_wind=-1.1,
    v_ej_wind=0.1,
    Ye_wind=0.35,
    inclination_EM=0.45,
)

t_grid = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0])
d_grid = np.linspace(10, 500, 50)
z_grid = np.array([z_at_value(Planck18.luminosity_distance, d * u.Mpc).value for d in d_grid])

mag_grid_g = np.full((len(t_grid), len(d_grid)), 99.0)
mag_grid_r = np.full((len(t_grid), len(d_grid)), 99.0)

print(f"  Computing Bu2026 lightcurves on {len(t_grid)}×{len(d_grid)} grid...")
t_exp_mjd = EVENT_MJD
for j, (d_mpc, z) in enumerate(zip(d_grid, z_grid)):
    obs_times_g = t_exp_mjd + t_grid * (1.0 + z)
    obs_times_both = np.concatenate([obs_times_g, obs_times_g])
    obs_bands_list = ["g"] * len(t_grid) + ["r"] * len(t_grid)

    params = dict(BU2026_PARAMS)
    params["luminosity_distance"] = d_mpc
    params["redshift"] = z
    params["_obs_times_mjd"] = obs_times_both.tolist()
    params["_obs_bands"] = obs_bands_list
    params["_t_exp"] = t_exp_mjd

    times, mags = kn_model.predict(params)

    if "g" in mags:
        mag_grid_g[:, j] = np.array(mags["g"])[:len(t_grid)]
    if "r" in mags:
        mag_grid_r[:, j] = np.array(mags["r"])[:len(t_grid)]

mag_grid_best = np.minimum(mag_grid_g, mag_grid_r)

mag_interp = RegularGridInterpolator(
    (t_grid, d_grid), mag_grid_best,
    bounds_error=False, fill_value=99.0,
)

# Detection horizons
ZTF_DEPTH_MEDIAN = 21.0
print(f"\n  Detection horizon (mag < {ZTF_DEPTH_MEDIAN}) vs time:")
for t in [0.3, 0.5, 1.0, 1.5, 2.0]:
    i_t = np.argmin(np.abs(t_grid - t))
    mags_at_t = mag_grid_best[i_t, :]
    det_mask = mags_at_t < ZTF_DEPTH_MEDIAN
    if np.any(det_mask):
        d_hor = d_grid[det_mask][-1]
        print(f"    t={t:.1f}d ({t*24:.0f}h): {d_hor:.0f} Mpc")
    else:
        print(f"    t={t:.1f}d ({t*24:.0f}h): not detectable")

# --- Build per-pixel d_max and call Rust coverage_3d_variable ---
print(f"\n  Computing per-pixel d_max from Bu2026 model...")

d_max_per_pixel = np.zeros(npix)
for i in range(npix):
    if not covered_mask[i]:
        continue
    dt_i = pixel_dt[i]
    depth_i = pixel_best_depth[i]
    if dt_i <= 0 or dt_i > 14 or depth_i <= 0:
        continue
    # Find max distance where model is brighter than this pixel's depth
    mags_vs_d = mag_interp(np.column_stack([np.full(len(d_grid), dt_i), d_grid]))
    detectable = d_grid[mags_vs_d < depth_i]
    if len(detectable) > 0:
        d_max_per_pixel[i] = detectable[-1]

n_with_horizon = np.sum(d_max_per_pixel > 0)
print(f"  Pixels with d_max > 0: {n_with_horizon} of {np.sum(covered_mask)} covered")
print(f"  Median d_max (where > 0): {np.median(d_max_per_pixel[d_max_per_pixel > 0]):.0f} Mpc")

print(f"\n  Computing 3D coverage with survey_sim.coverage_3d_variable...")

for name, data in skymap_objects.items():
    if not data["has_3d"]:
        print(f"\n  {name}: no 3D distance info, skipping")
        continue

    skymap = data["skymap"]
    result = data["result"]

    # Time-dependent Bu2026 3D coverage (Rust)
    p3d_bu = result.coverage_3d_variable(
        skymap, d_max_per_pixel.tolist(), n_samples=2000, seed=42
    )

    print(f"\n  {name}:")
    print(f"    Bu2026 AT2017gfo 3D prob (time-dependent): {p3d_bu:.1%}")

# ---------------------------------------------------------------------------
# 5. Depth statistics
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"DEPTH STATISTICS")
print(f"{'='*60}")

for bname, fid_val in [("g-band", 1), ("r-band", 2), ("i-band", 3)]:
    mask = obs_band == fid_val
    if np.any(mask):
        print(f"  {bname}: {np.sum(mask)} obs, median depth {np.median(obs_depth[mask]):.1f} mag")

for night_name, night_mask in [("Night 1", obs_mjd < NIGHT1_END_MJD), ("Night 2", obs_mjd >= NIGHT1_END_MJD)]:
    print(f"  {night_name}:")
    for bname, fid_val in [("g", 1), ("r", 2)]:
        m = night_mask & (obs_band == fid_val)
        if np.any(m):
            print(f"    {bname}: {np.sum(m)} obs, median depth {np.median(obs_depth[m]):.1f} mag")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SUMMARY: GW190425 ZTF Coverage")
print(f"{'='*60}")
print(f"  Event: GW190425 (BNS), {EVENT_TIME.iso}")
print(f"  CCD model: 4×4 mosaic, 64 RCs, {NS_CHIP_GAP_DEG}° NS / {EW_CHIP_GAP_DEG}° EW gaps")
print(f"  Coverage engine: survey_sim.Skymap (Rust)")
for name, data in skymap_objects.items():
    print(f"\n  {name}:")
    print(f"    90% area:   {data['skymap'].area_90:.0f} deg²")
    print(f"    2D covered: {data['result'].prob_2d:.1%}")
print(f"\n  KN model: Bu2026 AT2017gfo best-fit (time-dependent)")
print(f"  Paper (Coughlin+2019): 46% BAYESTAR, 21% LALInference")
