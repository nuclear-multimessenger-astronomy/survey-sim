#!/usr/bin/env python
"""Rubin LSST on-axis GRB afterglow detection (blind discovery).

Estimates the rate of on-axis GRB afterglows that Rubin can *independently
discover* as fast-fading transients, without prior knowledge of a gamma-ray
trigger. This is a different question from "how many GRB afterglows will
Rubin serendipitously image given a gamma-ray alert" — the latter yields
~75/yr (Burns+ 2025) by scaling the all-sky prompt GRB rate (~340/yr) by
Rubin's ~24% sky coverage and ~90% recovery efficiency. Our blind-discovery
rate is lower because it requires the afterglow to pass fast-transient
detection criteria (fading rate, multi-epoch, SNR) from survey data alone.

Uses OnAxisGrbPopulation (θ_v ≤ θ_j from the catalog) with the observed
on-axis long GRB rate (R = 1.0 Gpc⁻³ yr⁻¹).

Results (1M injections):
  Efficiency: 0.88%
  Expected:   44.3 in 10yr (4.4/yr) at R = 1.0 Gpc⁻³ yr⁻¹
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import survey_sim.gpu_setup  # noqa: F401

import time

from survey_sim import (
    SurveyStore,
    OnAxisGrbPopulation,
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
GRB_RATE = 1.0  # Gpc^-3 yr^-1 (on-axis observed rate)
N_TRANSIENTS = 100_000
SEED = 42

grb_pop = OnAxisGrbPopulation(grb_csv, rate=GRB_RATE, z_max=6.0)

# --- Pipeline ---
print(f"Running pipeline with {N_TRANSIENTS:,} on-axis transients (rate={GRB_RATE} Gpc^-3 yr^-1)...")
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
print("RESULTS: Rubin LSST On-Axis GRB Afterglow Detection (FS+RS)")
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
z_max = 6.0
V_max = 1140.0  # Gpc^3
f_sky = 0.44
V_eff = V_max * f_sky
duration = survey.duration_years

print(f"Survey parameters:")
print(f"  Duration:  {duration:.2f} yr")
print(f"  f_sky:     {f_sky}")
print(f"  V_max:     {V_max:.0f} Gpc^3 (z < {z_max})")
print(f"  V_eff:     {V_eff:.1f} Gpc^3")
print()

# Expected detections
print(f"{'Rate (Gpc^-3 yr^-1)':>22s}  {'Expected (10yr)':>16s}  {'Per year':>10s}")
print("-" * 55)
for rate in [0.5, 1.0, 1.3, 2.0]:
    expected = rate * V_eff * duration * eff
    print(f"{rate:>22.1f}  {expected:>16.1f}  {expected/duration:>10.1f}")
