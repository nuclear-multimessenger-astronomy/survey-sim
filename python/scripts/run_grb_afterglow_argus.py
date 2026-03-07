#!/usr/bin/env python
"""Argus Array GRB afterglow detection: on-axis and off-axis.

Predicts the number of GRB afterglows that Argus will detect via
blind discovery (no gamma-ray trigger), for both on-axis and off-axis
(orphan) afterglows.

Uses the fiesta blastwave surrogate (FS+RS) and the same GRB catalog
as the ZTF/Rubin simulations.

Argus: 8000 deg^2 FoV, g-band, ~20 mag depth (stacked).
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import survey_sim.gpu_setup  # noqa: F401 — configure CUDA for JAX

import math
import time

from survey_sim import (
    SurveyStore,
    OnAxisGrbPopulation,
    OffAxisGrbPopulation,
    DetectionCriteria,
    SimulationPipeline,
)
from survey_sim.fiesta_afterglow_model import FiestaAfterglowModel

# Argus band map: single g-band
ARGUS_BAND_MAP = {"g": "ztfg"}

# Argus parquet files (single HEALPix pixel)
parquet_files = ["/fred/oz480/mcoughli/simulations/argus/argussim_hpx_6131.parquet"]

# GRB catalog
grb_csv = "/fred/oz480/mcoughli/simulations/argus/GRB_afterglows_argus.csv"

N_TRANSIENTS = 100_000
SEED = 42

# Detection criteria for Argus (fast-fading transient)
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=3.0,
    min_detections=2,
    min_detections_primary=1,
    min_bands=1,
    min_per_band=2,
    max_timespan_days=14.0,
    min_time_separation_hours=0.5,  # Argus has high cadence
    require_fast_transient=True,
    min_rise_rate=0.0,
    min_fade_rate=0.3,
    min_galactic_lat=15.0,
)

# Load fiesta afterglow model (FS + RS)
print("Loading fiesta blastwave_rs_gaussian_CVAE surrogate (FS+RS)...")
t0 = time.time()
model = FiestaAfterglowModel(
    name="blastwave_rs_gaussian_CVAE",
    band_map=ARGUS_BAND_MAP,
)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Has reverse shock: {model.has_rs}")
print()

# Load Argus survey (15-min stacked)
print("Loading Argus survey (15-min stacked)...")
t0 = time.time()
survey = SurveyStore.from_argus(
    parquet_files, band="g", nside=64, stack_window_s=900.0
)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Observations: {survey.n_observations:,}")
print(f"  Duration: {survey.duration_years:.2f} yr")
print()

duration = survey.duration_years
V_max_z6 = 1140.0   # Gpc^3 to z=6 (full sky)
V_max_z1 = 160.0    # Gpc^3 to z=1 (full sky)

# =====================================================================
# ON-AXIS
# =====================================================================
print("=" * 70)
print("ON-AXIS GRB AFTERGLOWS (blind discovery)")
print("=" * 70)

GRB_RATE_ONAXIS = 1.0  # Gpc^-3 yr^-1
grb_pop_on = OnAxisGrbPopulation(grb_csv, rate=GRB_RATE_ONAXIS, z_max=6.0)

print(f"Running pipeline with {N_TRANSIENTS:,} transients (on-axis, z_max=6.0)...")
t0 = time.time()
pipe = SimulationPipeline(
    survey=survey,
    populations=[grb_pop_on],
    models={"Afterglow": model},
    detection=det,
    n_transients=N_TRANSIENTS,
    seed=SEED,
)
result_on = pipe.run()
elapsed = time.time() - t0

eff_on = result_on.n_detected / max(result_on.n_simulated, 1)
print(f"  Detected: {result_on.n_detected} / {result_on.n_simulated}")
print(f"  Efficiency: {eff_on:.6f} ({eff_on*100:.4f}%)")
print(f"  Time: {elapsed:.1f}s")
print()

# Expected: eff already includes sky fraction via spatial matching
VT_on = V_max_z6 * duration * eff_on
expected_on = GRB_RATE_ONAXIS * VT_on

print(f"Expected on-axis detections:")
for rate in [0.5, 1.0, 1.3]:
    exp = rate * VT_on
    print(f"  R={rate:.1f} Gpc^-3/yr: {exp:.1f} in {duration:.1f}yr ({exp/duration:.1f}/yr)")
print()

# =====================================================================
# OFF-AXIS (ORPHAN)
# =====================================================================
print("=" * 70)
print("OFF-AXIS (ORPHAN) GRB AFTERGLOWS (blind discovery)")
print("=" * 70)

GRB_RATE_OFFAXIS = 100.0  # Gpc^-3 yr^-1
grb_pop_off = OffAxisGrbPopulation(grb_csv, rate=GRB_RATE_OFFAXIS, z_max=1.0)

print(f"Running pipeline with {N_TRANSIENTS:,} transients (off-axis, z_max=1.0)...")
t0 = time.time()
pipe = SimulationPipeline(
    survey=survey,
    populations=[grb_pop_off],
    models={"Afterglow": model},
    detection=det,
    n_transients=N_TRANSIENTS,
    seed=SEED,
)
result_off = pipe.run()
elapsed = time.time() - t0

eff_off = result_off.n_detected / max(result_off.n_simulated, 1)
print(f"  Detected: {result_off.n_detected} / {result_off.n_simulated}")
print(f"  Efficiency: {eff_off:.6f} ({eff_off*100:.4f}%)")
print(f"  Time: {elapsed:.1f}s")
print()

VT_off = V_max_z1 * duration * eff_off
expected_off = GRB_RATE_OFFAXIS * VT_off

print(f"Expected off-axis detections:")
for rate in [50.0, 100.0, 200.0]:
    exp = rate * VT_off
    print(f"  R={rate:.0f} Gpc^-3/yr: {exp:.1f} in {duration:.1f}yr ({exp/duration:.1f}/yr)")
print()

# =====================================================================
# SUMMARY
# =====================================================================
print("=" * 70)
print("SUMMARY: Argus GRB Afterglow Predictions")
print("=" * 70)
print(f"Survey duration: {duration:.2f} yr")
print()
print(f"{'Type':>12s}  {'Rate':>10s}  {'Efficiency':>12s}  {'Expected':>10s}  {'Per year':>10s}")
print("-" * 60)
print(f"{'On-axis':>12s}  {'1.0':>10s}  {eff_on*100:>11.4f}%  {expected_on:>10.1f}  {expected_on/duration:>10.1f}")
print(f"{'Off-axis':>12s}  {'100.0':>10s}  {eff_off*100:>11.4f}%  {expected_off:>10.1f}  {expected_off/duration:>10.1f}")
total = expected_on + expected_off
print(f"{'Total':>12s}  {'':>10s}  {'':>12s}  {total:>10.1f}  {total/duration:>10.1f}")
