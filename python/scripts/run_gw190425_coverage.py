#!/usr/bin/env python
"""
Reproduce ZTF 2D and 3D coverage for GW190425 (S190425z).

Uses the survey_sim Rust library for skymap coverage computation:
  - Skymap.from_arrays(): construct from rasterized arrays
  - Skymap.coverage_2d(): 2D probability via HEALPix rectangular footprints
  - Skymap.coverage_2d_3d(): combined 2D+3D with per-observation d_max

Each ZTF readout channel observation gets its own detection horizon d_max,
computed from the Bu2026 kilonova model at that observation's time and depth.
For pixels covered by multiple observations, the best (highest) d_max is kept.

ZTF CCD geometry follows Bellm et al. (2019) Table 1, matching m4opt.
ZTF field grid from survey_sim/data/ZTF_Fields.txt.

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
NIGHT1_END_MJD = EVENT_MJD + 1.0
NIGHT2_END_MJD = EVENT_MJD + 2.0

NSIDE = 256

# ZTF field grid (bundled in survey_sim/data/)
ZTF_FIELDS_PATH = str(Path(__file__).resolve().parents[1] / "survey_sim" / "data" / "ZTF_Fields.txt")

# ---------------------------------------------------------------------------
# ZTF CCD Geometry (Bellm+2019 Table 1, matching m4opt)
# ---------------------------------------------------------------------------
PLATE_SCALE_ARCSEC = 1.01
NS_NPIX = 6144
EW_NPIX = 6160
NS_CHIP_GAP_DEG = 0.205
EW_CHIP_GAP_DEG = 0.140

# Readout channel half-widths (1 RC = half a CCD in each direction)
RC_HW_RA_DEG = (EW_NPIX / 2 * PLATE_SCALE_ARCSEC / 3600.0) / 2   # 0.432°
RC_HW_DEC_DEG = (NS_NPIX / 2 * PLATE_SCALE_ARCSEC / 3600.0) / 2   # 0.431°
TOTAL_RC_AREA = 64 * (EW_NPIX / 2 * PLATE_SCALE_ARCSEC / 3600.0) * \
                     (NS_NPIX / 2 * PLATE_SCALE_ARCSEC / 3600.0)

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
obs_time_key = np.round(obs_mjd, 3)
pointing_key = (obs_field.astype(np.int64) * 1000000
                + obs_rcid.astype(np.int64) * 10000
                + (obs_time_key * 10 % 10000).astype(np.int64))
_, unique_idx = np.unique(pointing_key, return_index=True)

ura = obs_ra[unique_idx]
udec = obs_dec[unique_idx]
umjd = obs_mjd[unique_idx]
udepth = obs_depth[unique_idx]
uband = obs_band[unique_idx]

n1_mask = umjd < NIGHT1_END_MJD
print(f"\n  Unique RC pointings: {len(ura)} ({np.sum(n1_mask)} Night 1, {np.sum(~n1_mask)} Night 2)")

# Time since event for each observation (days)
u_dt = umjd - EVENT_MJD
print(f"  Observation dt range: {u_dt.min():.3f} – {u_dt.max():.3f} days")

# ---------------------------------------------------------------------------
# 3. Build Bu2026 detection horizon for each observation
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Bu2026 KILONOVA MODEL (AT2017gfo best-fit)")
print(f"{'='*60}")

from survey_sim.fiesta_model import FiestaKNModel
from astropy.cosmology import Planck18, z_at_value
from scipy.interpolate import RegularGridInterpolator

print("  Initializing Bu2026 kilonova model...")
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

# Detection horizons at reference depth
ZTF_DEPTH_MEDIAN = 21.0
print(f"\n  Detection horizon (mag < {ZTF_DEPTH_MEDIAN}) vs time:")
for t in [0.3, 0.5, 1.0, 1.5, 2.0]:
    i_t = np.argmin(np.abs(t_grid - t))
    mags_at_t = mag_grid_best[i_t, :]
    det_mask = mags_at_t < ZTF_DEPTH_MEDIAN
    if np.any(det_mask):
        print(f"    t={t:.1f}d ({t*24:.0f}h): {d_grid[det_mask][-1]:.0f} Mpc")
    else:
        print(f"    t={t:.1f}d ({t*24:.0f}h): not detectable")

# Compute per-observation d_max from Bu2026
print(f"\n  Computing per-observation d_max ({len(ura)} observations)...")
obs_d_max = np.zeros(len(ura))

for k in range(len(ura)):
    dt_k = u_dt[k]
    depth_k = udepth[k]
    if dt_k <= 0 or dt_k > 14 or depth_k <= 0:
        continue
    mags_vs_d = mag_interp(np.column_stack([np.full(len(d_grid), dt_k), d_grid]))
    detectable = d_grid[mags_vs_d < depth_k]
    if len(detectable) > 0:
        obs_d_max[k] = detectable[-1]

n_with_dmax = np.sum(obs_d_max > 0)
print(f"  Observations with d_max > 0: {n_with_dmax} of {len(ura)}")
print(f"  Median d_max (where > 0): {np.median(obs_d_max[obs_d_max > 0]):.0f} Mpc")
print(f"  Night 1 median d_max: {np.median(obs_d_max[n1_mask & (obs_d_max > 0)]):.0f} Mpc")
if np.any(~n1_mask & (obs_d_max > 0)):
    print(f"  Night 2 median d_max: {np.median(obs_d_max[~n1_mask & (obs_d_max > 0)]):.0f} Mpc")

# ---------------------------------------------------------------------------
# 4. Rasterize skymaps and compute coverage with survey_sim
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"COVERAGE RESULTS (survey_sim.Skymap)")
print(f"{'='*60}")

for name, path in SKYMAPS.items():
    print(f"\n  {name}:")
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

    # Build Rust Skymap
    if has_3d:
        distmu = np.asarray(skymap_flat["DISTMU"], dtype=np.float64)
        distsigma = np.asarray(skymap_flat["DISTSIGMA"], dtype=np.float64)
        distnorm = np.asarray(skymap_flat["DISTNORM"], dtype=np.float64)
        skymap = Skymap.from_arrays(
            int(NSIDE), prob.tolist(),
            distmu.tolist(), distsigma.tolist(), distnorm.tolist(),
        )
    else:
        skymap = Skymap.from_arrays(int(NSIDE), prob.tolist())

    # --- 2D coverage ---
    result_total = skymap.coverage_2d(ura.tolist(), udec.tolist(), RC_HW_RA_DEG, RC_HW_DEC_DEG)
    result_n1 = skymap.coverage_2d(
        ura[n1_mask].tolist(), udec[n1_mask].tolist(), RC_HW_RA_DEG, RC_HW_DEC_DEG
    )

    print(f"    90% area:    {skymap.area_90:.0f} deg²")
    print(f"    Night 1 2D:  {result_n1.area_deg2:.0f} deg², {result_n1.prob_2d:.1%}")
    print(f"    Total 2D:    {result_total.area_deg2:.0f} deg², {result_total.prob_2d:.1%}")

    # --- 3D with flat magnitude cuts ---
    if has_3d:
        for M_abs in [-15.0, -16.0, -17.0]:
            d_max = 10 ** ((ZTF_DEPTH_MEDIAN - M_abs + 5) / 5) / 1e6
            p3d = result_total.coverage_3d(skymap, d_max, n_samples=2000, seed=42)
            print(f"    3D flat M={M_abs:.0f} (d_max={d_max:.0f} Mpc): {p3d:.1%}")

    # --- 3D with per-observation Bu2026 horizons ---
    if has_3d:
        result_3d = skymap.coverage_2d_3d(
            ura.tolist(), udec.tolist(),
            RC_HW_RA_DEG, RC_HW_DEC_DEG,
            obs_d_max.tolist(),
            n_samples=2000, seed=42,
        )
        print(f"    Bu2026 3D (per-RC time-dependent): {result_3d.prob_3d:.1%}")

        # Night 1 only
        result_3d_n1 = skymap.coverage_2d_3d(
            ura[n1_mask].tolist(), udec[n1_mask].tolist(),
            RC_HW_RA_DEG, RC_HW_DEC_DEG,
            obs_d_max[n1_mask].tolist(),
            n_samples=2000, seed=42,
        )
        print(f"    Bu2026 3D Night 1 only:            {result_3d_n1.prob_3d:.1%}")

print(f"\n  Paper values (Coughlin+2019):")
print(f"    Night 1: 3250 deg², 36%/19% BAYESTAR/LALInference")
print(f"    Total:   ~8000 deg², 46%/21% BAYESTAR/LALInference")

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

for night_name, night_mask in [("Night 1", obs_mjd < NIGHT1_END_MJD),
                                ("Night 2", obs_mjd >= NIGHT1_END_MJD)]:
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
print(f"  3D method: per-RC d_max from Bu2026, best d_max kept per pixel")
print(f"  Paper (Coughlin+2019): 46% BAYESTAR, 21% LALInference")
