#!/usr/bin/env python
"""Rubin LSST off-axis (orphan) GRB afterglow detection (blind discovery).

Estimates the rate of orphan afterglows (no associated gamma-ray trigger)
that Rubin can discover as fast-fading transients. These are GRBs viewed
off-axis (θ_v > θ_j) so the prompt gamma-ray emission is beamed away
from us, but the afterglow becomes visible as the jet decelerates.

This is distinct from the ~75 GRBs/yr that Rubin would image given
gamma-ray alerts (Burns+ 2025), which assumes the GRB time and position
are already known. Our blind-discovery rate requires the orphan afterglow
to be independently identified from survey photometry alone.

Uses OffAxisGrbPopulation: jet intrinsic properties from the catalog,
volumetric redshift (z_max = 1.0), isotropic viewing angles (θ_v > θ_j).

Rate = 100 Gpc⁻³/yr (off-axis).

Results (1M injections):
  Efficiency: 0.0071%
  Expected:   4.7 in 10yr (0.5/yr) at R = 100 Gpc⁻³ yr⁻¹
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import survey_sim.gpu_setup  # noqa: F401

import time

from survey_sim import (
    SurveyStore,
    OffAxisGrbPopulation,
    DetectionCriteria,
    SimulationPipeline,
)
from survey_sim.fiesta_afterglow_model import FiestaAfterglowModel, LSST_BAND_MAP

# --- Survey: Rubin LSST 10-year baseline ---
OPSIM_DB = "/fred/oz480/mcoughli/simulations/TESS-Rubin/baseline_v5.1.1_10yrs.db"
print("Loading Rubin OpSim survey (baseline_v5.1.1, 10yr)...")
t0 = time.time()
survey = SurveyStore.from_rubin(OPSIM_DB, nside=64)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Observations: {survey.n_observations}")
print(f"  Duration: {survey.duration_years:.2f} yr")
print(f"  Bands: {survey.bands}")
print()

# --- Detection criteria ---
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=3.0,
    min_detections=2,
    min_detections_primary=1,
    max_timespan_days=14.0,
    min_time_separation_hours=0.5,
    require_fast_transient=True,
    min_rise_rate=0.0,
    min_fade_rate=0.3,
    min_galactic_lat=15.0,
)

# --- Model: fiesta FS+RS ---
print("Loading fiesta blastwave_rs_gaussian_CVAE surrogate (FS+RS)...")
t0 = time.time()
model = FiestaAfterglowModel(
    name="blastwave_rs_gaussian_CVAE",
    band_map=LSST_BAND_MAP,
)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Has reverse shock: {model.has_rs}")
print()

# --- Population ---
grb_csv = "/fred/oz480/mcoughli/simulations/argus/GRB_afterglows_argus.csv"
GRB_RATE = 100.0  # Gpc^-3 yr^-1 (off-axis rate)
Z_MAX = 1.0       # off-axis afterglows beyond z~1 undetectable
N_TRANSIENTS = 100_000
SEED = 42

grb_pop = OffAxisGrbPopulation(grb_csv, rate=GRB_RATE, z_max=Z_MAX)

# --- Pipeline ---
print(f"Running pipeline with {N_TRANSIENTS:,} off-axis transients")
print(f"  Rate: {GRB_RATE} Gpc^-3 yr^-1, z_max={Z_MAX}")
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
print("RESULTS: Rubin LSST Off-Axis GRB Afterglow Detection (FS+RS)")
print("=" * 70)
print(f"Transients simulated: {result.n_simulated:,}")
print(f"Transients detected:  {result.n_detected:,}")
print(f"Efficiency:           {eff:.6f} ({eff*100:.4f}%)")
print(f"Elapsed:              {elapsed:.1f}s")
print()

for rs in result.rate_summaries:
    print(f"  [{rs.transient_type}] efficiency = {rs.overall_efficiency:.6f}")
print()

# Rate calculation
V_max = 150.0  # Gpc^3 (comoving volume to z=1)
f_sky = 0.44
V_eff = V_max * f_sky
duration = survey.duration_years

print(f"Survey parameters:")
print(f"  Duration:  {duration:.2f} yr")
print(f"  f_sky:     {f_sky}")
print(f"  V_max:     {V_max:.0f} Gpc^3 (z < {Z_MAX})")
print(f"  V_eff:     {V_eff:.1f} Gpc^3")
print()

expected = GRB_RATE * V_eff * duration * eff
print(f"Expected detections (R={GRB_RATE} Gpc^-3 yr^-1): {expected:.1f}")
print(f"Expected per year: {expected/duration:.1f}")
