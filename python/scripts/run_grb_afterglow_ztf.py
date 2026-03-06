#!/usr/bin/env python
"""ZTF GRB afterglow detection efficiency with fiesta blastwave surrogate.

Simulates GRB afterglow detections in ZTF over March 2018 – March 2021,
consistent with the ZTFReST analysis period. Compares with the observed
5 afterglows with GRB counterparts and 2 orphan afterglows.

Uses the on-axis long GRB rate (~1.3 Gpc^-3 yr^-1, Wanderman & Piran 2010)
and compares to 5 afterglows with GRB counterparts.
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import survey_sim.gpu_setup  # noqa: F401 — configure LD_LIBRARY_PATH for JAX GPU

import math
import time

from survey_sim import (
    SurveyStore,
    GrbPopulation,
    DetectionCriteria,
    SimulationPipeline,
    load_ztf_survey,
)
from survey_sim.fiesta_afterglow_model import FiestaAfterglowModel, ZTF_BAND_MAP

# --- Survey: March 2018 – March 2021 ---
print("Loading ZTF survey (March 2018 – March 2021)...")
t0 = time.time()
survey = load_ztf_survey(start="201803", end="202103", nside=64)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Observations: {survey.n_observations}")
print(f"  Duration: {survey.duration_years:.2f} yr")
print(f"  Bands: {survey.bands}")
print()

# --- GRB catalog ---
grb_csv = "/fred/oz480/mcoughli/simulations/argus/GRB_afterglows_argus.csv"

# --- Detection criteria (ZTFReST-like) ---
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=3.0,
    min_detections=2,
    min_detections_primary=1,
    max_timespan_days=14.0,
    min_time_separation_hours=3.0,
    require_fast_transient=True,
    min_rise_rate=0.0,
    min_fade_rate=0.3,
    min_galactic_lat=15.0,
)

# --- Model: fiesta blastwave surrogate (FS + RS) ---
print("Loading fiesta blastwave_rs_gaussian_CVAE surrogate (FS+RS)...")
t0 = time.time()
model = FiestaAfterglowModel(
    name="blastwave_rs_gaussian_CVAE",
    band_map=ZTF_BAND_MAP,
)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Parameters: {model.model.parameter_names}")
print(f"  Has reverse shock: {model.has_rs}")
print()

# --- Population ---
# On-axis (observed) long GRB rate: ~1.3 Gpc^-3 yr^-1 (Wanderman & Piran 2010)
# Range in literature: 0.5–2.0 Gpc^-3 yr^-1
# Note: catalog GRBs are already on-axis (thv <= thj), so use on-axis rate.
GRB_RATE = 1.3  # Gpc^-3 yr^-1 (on-axis, Wanderman & Piran 2010)
N_TRANSIENTS = 100_000
SEED = 42

grb_pop = GrbPopulation(grb_csv, rate=GRB_RATE, z_max=6.0)

# --- Pipeline ---
print(f"Running pipeline with {N_TRANSIENTS:,} transients (rate={GRB_RATE} Gpc^-3 yr^-1)...")
t0 = time.time()
pipe = SimulationPipeline(
    survey=survey,
    populations=[grb_pop],
    models={"Afterglow": model},
    detection=det,
    n_transients=N_TRANSIENTS,
    seed=SEED,
)
result = pipe.run()
elapsed = time.time() - t0

# --- Results ---
eff = result.n_detected / max(result.n_simulated, 1)
print()
print("=" * 70)
print("RESULTS: ZTF GRB Afterglow Detection (FS + RS)")
print("=" * 70)
print(f"Transients simulated: {result.n_simulated:,}")
print(f"Transients detected:  {result.n_detected:,}")
print(f"Efficiency:           {eff:.6f} ({eff*100:.4f}%)")
print(f"Elapsed:              {elapsed:.1f}s")
print()

# Rate upper limit calculation
# Comoving volume to z=6 ~ 1140 Gpc^3
# ZTF sky fraction ~ 0.47
z_max = 6.0
V_max = 1140.0  # Gpc^3 (approx comoving volume to z=6)
f_sky = 0.47
V_eff = V_max * f_sky
duration = survey.duration_years

print(f"Survey parameters:")
print(f"  Duration:  {duration:.2f} yr")
print(f"  f_sky:     {f_sky}")
print(f"  V_max:     {V_max:.0f} Gpc^3 (z < {z_max})")
print(f"  V_eff:     {V_eff:.1f} Gpc^3")
print()

# Expected detections for different on-axis rates
print(f"{'Rate (Gpc^-3 yr^-1)':>22s}  {'Expected detections':>20s}")
print("-" * 50)
for rate in [0.5, 1.0, 1.3, 2.0]:
    # Expected = rate * V_eff * duration * efficiency
    expected = rate * V_eff * duration * eff
    print(f"{rate:>22.1f}  {expected:>20.1f}")

print()
print("Observed in ZTFReST (Mar 2018 – Mar 2021):")
print("  5 afterglows with GRB counterpart (on-axis)")
print("  (2 orphan afterglows excluded from this comparison)")

# 90% CL upper limit on rate (if no detections or few detections)
if eff > 0:
    VT = V_eff * duration * eff
    R90 = -math.log(0.10) / VT
    R95 = -math.log(0.05) / VT
    print()
    print(f"Rate upper limits (assuming 0 detections):")
    print(f"  R_upper (90% CL) = {R90:.0f} Gpc^-3 yr^-1")
    print(f"  R_upper (95% CL) = {R95:.0f} Gpc^-3 yr^-1")

for rs in result.rate_summaries:
    print(f"\n  [{rs.transient_type}] efficiency = {rs.overall_efficiency:.6f}")
