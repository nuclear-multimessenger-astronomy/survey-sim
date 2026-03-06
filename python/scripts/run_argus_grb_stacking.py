#!/usr/bin/env python
"""Argus Array GRB afterglow detection with fiesta surrogate and flux-averaging stacking.

Strategy: load survey at 15-min stacking (finest window), evaluate lightcurves
at those times, then create coarser stacks (1-hr, 1-day) by flux-averaging
the 15-min evaluations. Multiple stacking scales are treated as independent
measurements combined for detection.

Uses a single HEALPix pixel (nside=64, pixel 6131) as a representative
pointing, scaled to full Argus FoV for rate estimates.
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import survey_sim.gpu_setup  # noqa: F401 — configure CUDA for JAX

import math
import time

from survey_sim import (
    SurveyStore,
    GrbPopulation,
    DetectionCriteria,
    SimulationPipeline,
)
from survey_sim.fiesta_afterglow_model import FiestaAfterglowModel

# Argus band map: single g-band (ztfg as closest match in sncosmo)
ARGUS_BAND_MAP = {"g": "ztfg"}

# Argus parquet files (single HEALPix pixel)
parquet_files = ["/fred/oz480/mcoughli/simulations/argus/argussim_hpx_6131.parquet"]

# GRB catalog
grb_csv = "/fred/oz480/mcoughli/simulations/argus/GRB_afterglows_argus.csv"

# On-axis GRB rate
GRB_RATE = 1.3  # Gpc^-3 yr^-1
N_TRANSIENTS = 10_000
SEED = 42

# Detection criteria for Argus (relaxed — wide field, high cadence)
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=3.0,
    min_detections=2,
    min_detections_primary=1,
    max_timespan_days=30.0,
    min_time_separation_hours=0.5,
    require_fast_transient=False,
    min_rise_rate=0.0,
    min_fade_rate=0.0,
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

# Load survey at 15-min stacking (finest resolution needed).
# The afterglow model doesn't vary on sub-15min timescales, so this
# is equivalent to evaluating at each raw exposure.
print("Loading Argus survey (15-min stacked)...")
t0 = time.time()
survey = SurveyStore.from_argus(
    parquet_files, band="g", nside=64, stack_window_s=900.0  # 15 min
)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  15-min stacked observations: {survey.n_observations:,}")
print(f"  Duration: {survey.duration_years:.2f} yr")
print()

# Stacking strategies:
# - None: use 15-min stacks directly (no further stacking)
# - [3600]: flux-average into 1-hr stacks, combine with 15-min as independent measurements
# - [86400]: flux-average into 1-day stacks, combine with 15-min
# - [3600, 86400]: 15-min + 1-hr + 1-day all as independent measurements
# Note: stack_windows_s=None means no pipeline stacking (just use the survey-level 15-min stacks)
STRATEGIES = [
    ("15min only",    None,                  "15-min stacks, no further stacking"),
    ("15m+1hr",       [3600.0],              "15-min + 1-hour as independent measurements"),
    ("15m+1hr+1day",  [3600.0, 86400.0],     "15-min + 1-hour + 1-day combined"),
]

print("=" * 70)
print("Argus Array GRB Afterglow — Stacking Comparison (FS+RS)")
print("=" * 70)
print(f"Rate: {GRB_RATE} Gpc^-3 yr^-1 (on-axis)")
print(f"N_transients: {N_TRANSIENTS:,}")
print()

results = []

for label, windows, description in STRATEGIES:
    print(f"--- {label}: {description} ---")
    t0 = time.time()

    grb_pop = GrbPopulation(grb_csv, rate=GRB_RATE, z_max=6.0)

    pipe = SimulationPipeline(
        survey=survey,
        populations=[grb_pop],
        models={"Afterglow": model},
        detection=det,
        n_transients=N_TRANSIENTS,
        seed=SEED,
        stack_windows_s=windows,
    )

    result = pipe.run()
    elapsed = time.time() - t0

    eff = result.n_detected / max(result.n_simulated, 1)
    print(f"  Detected: {result.n_detected} / {result.n_simulated}")
    print(f"  Efficiency: {eff:.6f} ({eff*100:.4f}%)")
    print(f"  Time: {elapsed:.1f}s")

    results.append({
        "label": label,
        "n_detected": result.n_detected,
        "n_simulated": result.n_simulated,
        "efficiency": eff,
        "elapsed_s": elapsed,
    })
    print()

# Summary table
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Strategy':>16s}  {'N_det':>6s}  {'N_sim':>8s}  {'Eff (%)':>10s}  {'Time':>7s}")
print("-" * 70)
for r in results:
    eff_pct = r["efficiency"] * 100
    print(f"{r['label']:>16s}  {r['n_detected']:>6d}  {r['n_simulated']:>8,d}  {eff_pct:>10.4f}  {r['elapsed_s']:>6.1f}s")

# Expected detections for full Argus.
# NOTE: The Argus instrument FoV (8000 deg^2) means the spatial cone query
# from even a single pixel already covers the full Argus sky fraction (~19%).
# So the MC efficiency epsilon = N_det/N_sim already includes sky coverage.
# We must NOT multiply by f_sky again.
print()
V_max = 1140.0  # Gpc^3 to z=6 (full sky)
duration = survey.duration_years

print("Expected detections (full Argus FoV, 8000 deg^2):")
print(f"{'Strategy':>16s}  {'Eff':>10s}  {'Expected/5yr':>12s}  {'R_90 (Gpc^-3/yr)':>18s}")
print("-" * 70)
for r in results:
    eff = r["efficiency"]
    # eff already includes sky fraction via spatial matching
    VT = V_max * duration * eff
    expected = GRB_RATE * VT
    if VT > 0:
        R90 = -math.log(0.10) / VT
        print(f"{r['label']:>16s}  {eff:>10.6f}  {expected:>12.1f}  {R90:>18.1f}")
    else:
        print(f"{r['label']:>16s}  {eff:>10.6f}  {'0.0':>12s}  {'inf':>18s}")

print()
print("For comparison, ZTF (FS+RS): ~5 detections / 3yr")
