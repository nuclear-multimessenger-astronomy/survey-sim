#!/usr/bin/env python
"""
Reproduce ZTF 2D and 3D coverage for GW190425 (S190425z).

GW190425 was a BNS merger detected by LIGO Livingston on 2019-04-25 08:18:05 UTC.
The single-detector localization spanned ~10,000 deg².
ZTF covered ~8,000 deg² over two nights, enclosing:
  - 46% of the BAYESTAR probability (2D)
  - 21% of the LALInference probability (2D)

This script loads the skymaps and ZTF observations from survey-sim to compute
the 2D probability coverage and 3D distance-weighted coverage.

ZTF CCD geometry follows Bellm et al. (2019) Table 1, matching the m4opt
representation: 4x4 CCD mosaic with 4 readout channels each (64 rcids).
Chip gaps: 0.205° NS, 0.140° EW. Plate scale 1.01"/pix.

Reference: Coughlin et al. 2019 (arXiv:1907.12645)
"""

import sys
sys.path.insert(0, "/fred/oz480/mcoughli/fiestaEM/src")

import survey_sim.gpu_setup  # noqa: F401 — preload CUDA libs for JAX

import numpy as np
import h5py
import astropy.units as u
import healpy as hp
from astropy.time import Time
from astropy.coordinates import Longitude, Latitude
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize
from mocpy import MOC

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

# HEALPix resolution for rasterization
NSIDE = 256
MOC_DEPTH = 12

# ---------------------------------------------------------------------------
# ZTF CCD Geometry (Bellm+2019 Table 1, matching m4opt)
# ---------------------------------------------------------------------------
# 4x4 CCD mosaic, each CCD has 4 readout channels (quadrants)
# Plate scale: 1.01 arcsec/pixel
PLATE_SCALE = 1.01 * u.arcsec

# CCD pixel dimensions
NS_NPIX = 6144  # pixels per CCD (NS direction)
EW_NPIX = 6160  # pixels per CCD (EW direction)

# Chip gaps between CCDs
NS_CHIP_GAP = 0.205 * u.deg
EW_CHIP_GAP = 0.140 * u.deg

NS_NCHIPS = 4
EW_NCHIPS = 4


def compute_ztf_readout_channel_polygons():
    """Compute the 64 readout channel corner offsets from field center.

    Returns array of shape (64, 4, 2) with (dRA, dDec) corners in degrees
    for each readout channel, relative to the field center.
    Follows the m4opt ZTF geometry exactly.
    """
    rcids = np.arange(64)
    chipid, rc_in_chip_id = np.divmod(rcids, 4)
    ns_chip_index, ew_chip_index = np.divmod(chipid, EW_NCHIPS)

    # Readout channel position within chip (m4opt convention)
    # rc 0: ns=1, ew=0 (upper-left)
    # rc 1: ns=1, ew=1 (upper-right)
    # rc 2: ns=0, ew=1 (lower-right)
    # rc 3: ns=0, ew=0 (lower-left)
    ns_rc_in_chip_index = np.where(rc_in_chip_id <= 1, 1, 0)
    ew_rc_in_chip_index = np.where(
        (rc_in_chip_id == 0) | (rc_in_chip_id == 3), 0, 1
    )

    # CCD half-sizes
    ccd_ns = (NS_NPIX * PLATE_SCALE).to(u.deg).value
    ccd_ew = (EW_NPIX * PLATE_SCALE).to(u.deg).value
    rc_ns = ccd_ns / 2  # readout channel is half a CCD
    rc_ew = ccd_ew / 2

    ns_gap = NS_CHIP_GAP.to(u.deg).value
    ew_gap = EW_CHIP_GAP.to(u.deg).value

    polygons = np.zeros((64, 4, 2))  # (rcid, corner, (ew, ns))

    for i in range(64):
        # Chip center offsets from field center
        ew_chip_off = (
            ew_gap * (ew_chip_index[i] - (EW_NCHIPS - 1) / 2)
            + ccd_ew * (ew_chip_index[i] - EW_NCHIPS / 2)
            + 0.5 * ccd_ew  # shift so grid is centered
        )
        ns_chip_off = (
            ns_gap * (ns_chip_index[i] - (NS_NCHIPS - 1) / 2)
            + ccd_ns * (ns_chip_index[i] - NS_NCHIPS / 2)
            + 0.5 * ccd_ns
        )

        # Readout channel offset within chip
        ew_rc_off = ew_rc_in_chip_index[i] * rc_ew
        ns_rc_off = ns_rc_in_chip_index[i] * rc_ns

        # Bottom-left corner of this readout channel
        ew_bl = ew_chip_off + ew_rc_off - ccd_ew / 2  # chip goes from -ccd/2 to +ccd/2
        ns_bl = ns_chip_off + ns_rc_off - ccd_ns / 2

        # Four corners: BL, BR, TR, TL
        polygons[i, 0] = [ew_bl, ns_bl]
        polygons[i, 1] = [ew_bl + rc_ew, ns_bl]
        polygons[i, 2] = [ew_bl + rc_ew, ns_bl + rc_ns]
        polygons[i, 3] = [ew_bl, ns_bl + rc_ns]

    return polygons


# Pre-compute the 64 readout channel offset polygons
RC_POLYGONS = compute_ztf_readout_channel_polygons()

# Verify geometry
total_rc_area = 64 * (
    (NS_NPIX / 2 * PLATE_SCALE).to(u.deg).value
    * (EW_NPIX / 2 * PLATE_SCALE).to(u.deg).value
)
print(f"ZTF CCD geometry (Bellm+2019, m4opt):")
print(f"  Plate scale: {PLATE_SCALE}")
print(f"  CCD size: {(EW_NPIX * PLATE_SCALE).to(u.deg):.4f} x {(NS_NPIX * PLATE_SCALE).to(u.deg):.4f}")
print(f"  RC size:  {(EW_NPIX/2 * PLATE_SCALE).to(u.deg):.4f} x {(NS_NPIX/2 * PLATE_SCALE).to(u.deg):.4f}")
print(f"  Chip gaps: {NS_CHIP_GAP} (NS), {EW_CHIP_GAP} (EW)")
print(f"  Active area (no gaps): {total_rc_area:.1f} deg²")

# Full FoV extent including gaps
ew_extent = EW_NCHIPS * (EW_NPIX * PLATE_SCALE).to(u.deg).value + (EW_NCHIPS - 1) * EW_CHIP_GAP.to(u.deg).value
ns_extent = NS_NCHIPS * (NS_NPIX * PLATE_SCALE).to(u.deg).value + (NS_NCHIPS - 1) * NS_CHIP_GAP.to(u.deg).value
print(f"  Full FoV extent: {ew_extent:.3f}° x {ns_extent:.3f}° = {ew_extent * ns_extent:.1f} deg²")
print(f"  Fill factor: {total_rc_area / (ew_extent * ns_extent):.1%}")

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
# 2. Build ZTF coverage MOC with proper chip gaps
# ---------------------------------------------------------------------------
# The boom data gives per-rcid (ra, dec) which is the center of each readout
# channel. We use the m4opt-derived geometry to place rectangular MOCs at
# each rcid's reported center position.
#
# Each readout channel is a rectangle of size:
#   EW: 3080 pix × 1.01"/pix = 0.864°
#   NS: 3072 pix × 1.01"/pix = 0.862°
# The chip gaps (0.205° NS, 0.140° EW) are implicit in the spacing between
# rcid centers across different CCDs.

print("\nBuilding ZTF coverage MOC (rectangular CCDs with chip gaps)...")

rc_ew_half = (EW_NPIX / 2 * PLATE_SCALE).to(u.deg).value / 2  # half-width of a readout channel
rc_ns_half = (NS_NPIX / 2 * PLATE_SCALE).to(u.deg).value / 2

# Get unique (field, rcid, time) pointings
# Round MJD to ~1 min to group same-exposure observations
obs_time_key = np.round(obs_mjd, 3)
pointing_key = obs_field.astype(np.int64) * 1000000 + obs_rcid.astype(np.int64) * 10000 + (obs_time_key * 10 % 10000).astype(np.int64)
_, unique_idx = np.unique(pointing_key, return_index=True)

ura = obs_ra[unique_idx]
udec = obs_dec[unique_idx]
umjd = obs_mjd[unique_idx]
print(f"  Unique CCD pointings: {len(ura)}")

# Build rectangular MOCs for all readout channel pointings
# Each rcid observation gets a box centered at its (ra, dec)
print("  Building rectangular MOCs for each readout channel...")
all_mocs = MOC.from_boxes(
    lon=ura * u.deg,
    lat=udec * u.deg,
    a=np.full(len(ura), rc_ew_half) * u.deg,
    b=np.full(len(ura), rc_ns_half) * u.deg,
    angle=np.zeros(len(ura)) * u.deg,
    max_depth=MOC_DEPTH,
)

# Build night 1 and total union MOCs
night1_idx = np.where(umjd < NIGHT1_END_MJD)[0]
night2_idx = np.where(umjd >= NIGHT1_END_MJD)[0]

print(f"  Building Night 1 MOC ({len(night1_idx)} pointings)...")
moc_night1 = MOC.new_empty(max_depth=MOC_DEPTH)
for i in night1_idx:
    moc_night1 = moc_night1.union(all_mocs[i])

print(f"  Building total MOC (+{len(night2_idx)} Night 2 pointings)...")
moc_total = moc_night1
for i in night2_idx:
    moc_total = moc_total.union(all_mocs[i])

area_night1 = moc_night1.sky_fraction * 4 * 180**2 / np.pi
area_total = moc_total.sky_fraction * 4 * 180**2 / np.pi
print(f"  Night 1 area: {area_night1:.0f} deg²")
print(f"  Total area:   {area_total:.0f} deg²")

# Verify chip gap effect: compare with a no-gap version (full field circles)
unique_fields = np.unique(obs_field)
field_ra = np.array([np.mean(obs_ra[obs_field == fid]) for fid in unique_fields])
field_dec = np.array([np.mean(obs_dec[obs_field == fid]) for fid in unique_fields])
field_fov_radius = np.sqrt(total_rc_area / np.pi)  # effective circular radius from active area
no_gap_mocs = MOC.from_cones(
    lon=field_ra * u.deg,
    lat=field_dec * u.deg,
    radius=field_fov_radius * u.deg,
    max_depth=MOC_DEPTH,
)
moc_no_gap = MOC.new_empty(max_depth=MOC_DEPTH)
for i in range(len(field_ra)):
    moc_no_gap = moc_no_gap.union(no_gap_mocs[i])
area_no_gap = moc_no_gap.sky_fraction * 4 * 180**2 / np.pi
print(f"\n  Comparison:")
print(f"    With chip gaps (rectangular CCDs): {area_total:.0f} deg²")
print(f"    Without gaps (circular fields):    {area_no_gap:.0f} deg²")
print(f"    Area reduction from chip gaps:     {(1 - area_total/area_no_gap)*100:.1f}%")

# ---------------------------------------------------------------------------
# 3. Compute 2D probability coverage for each skymap
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"2D COVERAGE RESULTS")
print(f"{'='*60}")

skymap_data = {}
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

    # 90% credible area
    skymap_flat = rasterize(skymap_moc, order=hp.nside2order(NSIDE))
    prob = skymap_flat["PROB"]
    idx_sorted = np.argsort(-prob)
    cum_prob = np.cumsum(prob[idx_sorted])
    n90 = np.searchsorted(cum_prob, 0.9) + 1
    area_90 = n90 * hp.nside2pixarea(NSIDE, degrees=True)

    # 2D probability coverage
    prob_n1 = moc_night1.probability_in_multiordermap(skymap_moc)
    prob_tot = moc_total.probability_in_multiordermap(skymap_moc)

    # Also compute no-gap version for comparison
    prob_no_gap = moc_no_gap.probability_in_multiordermap(skymap_moc)

    print(f"    90% area: {area_90:.0f} deg²")
    print(f"    Night 1:        {area_night1:.0f} deg², {prob_n1:.1%} prob")
    print(f"    Total:          {area_total:.0f} deg², {prob_tot:.1%} prob")
    print(f"    No-gap circles: {area_no_gap:.0f} deg², {prob_no_gap:.1%} prob")

    skymap_data[name] = {
        "moc": skymap_moc,
        "flat": skymap_flat,
        "prob": prob,
        "has_3d": has_3d,
        "area_90": area_90,
        "prob_night1": prob_n1,
        "prob_total": prob_tot,
        "prob_no_gap": prob_no_gap,
    }

    if has_3d:
        skymap_data[name]["distmu"] = skymap_flat["DISTMU"]
        skymap_data[name]["distsigma"] = skymap_flat["DISTSIGMA"]
        skymap_data[name]["distnorm"] = skymap_flat["DISTNORM"]

print(f"\n  Paper values (Coughlin+2019):")
print(f"    Night 1: 3250 deg², 36%/19% BAYESTAR/LALInference")
print(f"    Total:   ~8000 deg², 46%/21% BAYESTAR/LALInference")

# ---------------------------------------------------------------------------
# 4. 3D Distance- and Time-Weighted Coverage (Bu2026 kilonova model)
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"3D DISTANCE- AND TIME-WEIGHTED COVERAGE (Bu2026)")
print(f"{'='*60}")

npix = len(skymap_data[list(skymap_data.keys())[0]]["prob"])
theta, phi = hp.pix2ang(NSIDE, np.arange(npix), nest=True)
ra_pix = np.degrees(phi)
dec_pix = 90.0 - np.degrees(theta)

print("  Building observed pixel mask with observation times...")
lon = Longitude(ra_pix, unit=u.deg)
lat = Latitude(dec_pix, unit=u.deg)
in_moc = moc_total.contains_lonlat(lon, lat)
print(f"  Observed pixels: {np.sum(in_moc)} of {npix}")

# For each observed pixel, find the earliest observation time and best depth
# by cross-matching pixel positions against the CCD observation catalog.
# We assign each observation to the HEALPix pixel at its center.
print("  Mapping observations to HEALPix pixels...")
import healpy as hp

# For each boom observation, compute which NSIDE=256 pixel it covers
obs_pix = hp.ang2pix(NSIDE, obs_ra, obs_dec, nest=True, lonlat=True)

# For each pixel, find the earliest observation time and best g/r depth
pixel_first_mjd = np.full(npix, np.inf)
pixel_best_depth_g = np.full(npix, 0.0)
pixel_best_depth_r = np.full(npix, 0.0)

for k in range(len(obs_ra)):
    pix = obs_pix[k]
    if obs_mjd[k] < pixel_first_mjd[pix]:
        pixel_first_mjd[pix] = obs_mjd[k]
    if obs_band[k] == 1:  # g
        if obs_depth[k] > pixel_best_depth_g[pix]:
            pixel_best_depth_g[pix] = obs_depth[k]
    elif obs_band[k] == 2:  # r
        if obs_depth[k] > pixel_best_depth_r[pix]:
            pixel_best_depth_r[pix] = obs_depth[k]

# Time since event for each pixel (days)
pixel_dt = pixel_first_mjd - EVENT_MJD

# Best depth across g and r
pixel_best_depth = np.maximum(pixel_best_depth_g, pixel_best_depth_r)

# Also mark pixels covered by the MOC but without a direct CCD match
# (MOC is built from CCD footprints, not just centers)
# For those, use median values from their night
n1_median_depth = np.median(obs_depth[(obs_mjd < NIGHT1_END_MJD) & (obs_depth > 0)])
n2_median_depth = np.median(obs_depth[(obs_mjd >= NIGHT1_END_MJD) & (obs_depth > 0)])
n1_median_dt = 0.3  # ~7 hours post-event
n2_median_dt = 1.3  # ~31 hours post-event

for i in range(npix):
    if in_moc[i] and pixel_best_depth[i] == 0:
        # This pixel is in the MOC but no boom observation lands on it
        # Use median values (most likely Night 1)
        pixel_best_depth[i] = n1_median_depth
        pixel_dt[i] = n1_median_dt

observed_dts = pixel_dt[in_moc & (pixel_dt < np.inf)]
print(f"  Observation time distribution:")
print(f"    Median dt: {np.median(observed_dts):.2f} days ({np.median(observed_dts)*24:.1f} hours)")
print(f"    Night 1 pixels: {np.sum(observed_dts < 1.0)} ({np.sum(observed_dts < 1.0)/len(observed_dts)*100:.0f}%)")
print(f"    Night 2 pixels: {np.sum(observed_dts >= 1.0)} ({np.sum(observed_dts >= 1.0)/len(observed_dts)*100:.0f}%)")

# --- Bu2026 kilonova model: magnitude(time, distance) grid ---
from survey_sim.fiesta_model import FiestaKNModel
from astropy.cosmology import Planck18, z_at_value
from scipy.interpolate import RegularGridInterpolator

print("\n  Initializing Bu2026 kilonova model (AT2017gfo best-fit)...")
kn_model = FiestaKNModel()

# AT2017gfo best-fit parameters (tuned for g/r/i at t < 4d)
BU2026_PARAMS = dict(
    log10_mej_dyn=-1.8,
    v_ej_dyn=0.2,
    Ye_dyn=0.15,
    log10_mej_wind=-1.1,
    v_ej_wind=0.1,
    Ye_wind=0.35,
    inclination_EM=0.45,  # 26° (GW170817 viewing angle)
)

# 2D grid: time (days post-event) × distance (Mpc)
t_grid = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0])
d_grid = np.linspace(10, 500, 50)
z_grid = np.array([z_at_value(Planck18.luminosity_distance, d * u.Mpc).value for d in d_grid])

# Compute apparent magnitudes on the full (t, d) grid
# mag_grid[i_t, i_d] = best apparent mag (min of g, r) at time t and distance d
mag_grid_g = np.full((len(t_grid), len(d_grid)), 99.0)
mag_grid_r = np.full((len(t_grid), len(d_grid)), 99.0)

print(f"  Computing Bu2026 lightcurves on {len(t_grid)}×{len(d_grid)} (time×distance) grid...")
t_exp_mjd = EVENT_MJD  # explosion time = event time
for j, (d_mpc, z) in enumerate(zip(d_grid, z_grid)):
    # Evaluate at all time grid points simultaneously
    obs_times_g = t_exp_mjd + t_grid * (1.0 + z)  # observer-frame times
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
        g_arr = np.array(mags["g"])[:len(t_grid)]
        mag_grid_g[:, j] = g_arr
    if "r" in mags:
        r_arr = np.array(mags["r"])[:len(t_grid)]
        mag_grid_r[:, j] = r_arr

# Best of g or r at each (t, d)
mag_grid_best = np.minimum(mag_grid_g, mag_grid_r)

# Build 2D interpolator: (time_days, distance_mpc) → apparent mag
mag_interp = RegularGridInterpolator(
    (t_grid, d_grid), mag_grid_best,
    bounds_error=False, fill_value=99.0,
)

# Show the lightcurve at a few distances
print(f"\n  Bu2026 apparent magnitudes (best of g, r):")
print(f"    {'t (days)':>10s}", end="")
for d_show in [50, 100, 150, 200, 300]:
    print(f"  {d_show:>5d} Mpc", end="")
print()
for i_t, t in enumerate(t_grid):
    if t > 5:
        break
    print(f"    {t:>10.2f}", end="")
    for d_show in [50, 100, 150, 200, 300]:
        j_d = np.argmin(np.abs(d_grid - d_show))
        print(f"  {mag_grid_best[i_t, j_d]:>9.1f}", end="")
    print()

# Detection horizons at different times
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

# --- 3D probability with time-dependent Bu2026 model ---
print(f"\n  Computing 3D coverage with time-dependent Bu2026 model...")
np.random.seed(42)
N_SAMPLES = 2000

for name, data in skymap_data.items():
    if not data["has_3d"]:
        print(f"\n  {name}: no 3D distance info, skipping")
        continue

    prob = data["prob"]
    distmu = data["distmu"]
    distsigma = data["distsigma"]

    p3d_bu2026 = 0.0
    p3d_flat = {M: 0.0 for M in [-15.0, -16.0, -17.0]}

    for i in range(npix):
        if not in_moc[i] or prob[i] < 1e-10:
            continue
        mu_i = distmu[i]
        sigma_i = distsigma[i]
        if sigma_i <= 0 or np.isnan(mu_i) or np.isinf(mu_i) or mu_i <= 0:
            continue

        dt_i = pixel_dt[i]
        depth_i = pixel_best_depth[i]

        # Skip pixels without valid observation time
        if dt_i <= 0 or dt_i > 14 or depth_i <= 0:
            continue

        d_samples = np.abs(np.random.normal(mu_i, sigma_i, N_SAMPLES))

        # Bu2026: evaluate mag at (dt_i, d) for each sampled distance
        query_pts = np.column_stack([
            np.full(len(d_samples), dt_i),
            d_samples,
        ])
        app_mags = mag_interp(query_pts)
        frac_bu2026 = np.mean(app_mags < depth_i)
        p3d_bu2026 += prob[i] * frac_bu2026

        # Flat M comparisons (use actual depth, not median)
        for M in p3d_flat:
            d_max = 10**((depth_i - M + 5) / 5) / 1e6
            p3d_flat[M] += prob[i] * np.mean(d_samples < d_max)

    print(f"\n  {name}:")
    print(f"    Bu2026 AT2017gfo 3D prob (time-dependent): {p3d_bu2026:.1%}")
    print(f"    Flat magnitude comparisons (actual pixel depths):")
    print(f"      {'M_abs':>6s}  {'3D prob':>10s}")
    for M in sorted(p3d_flat.keys()):
        print(f"      {M:>6.1f}  {p3d_flat[M]:>10.1%}")

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

print(f"\n  Paper: 21.0 r-band, 20.9 g-band (5σ median over both nights)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SUMMARY: GW190425 ZTF Coverage")
print(f"{'='*60}")
print(f"  Event: GW190425 (BNS), {EVENT_TIME.iso}")
print(f"  ZTF total area covered: {area_total:.0f} deg² (with chip gaps)")
print(f"  CCD model: 4×4 mosaic, 64 readout channels, {NS_CHIP_GAP} NS / {EW_CHIP_GAP} EW gaps")
for name, data in skymap_data.items():
    print(f"\n  {name}:")
    print(f"    90% area:           {data['area_90']:.0f} deg²")
    print(f"    2D covered:         {data['prob_total']:.1%}")
    print(f"    (without chip gaps: {data['prob_no_gap']:.1%})")
print(f"\n  KN model: Bu2026 AT2017gfo best-fit (time-dependent)")
print(f"  Paper (Coughlin+2019): 46% BAYESTAR, 21% LALInference")
