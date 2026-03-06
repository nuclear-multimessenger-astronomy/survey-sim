#!/usr/bin/env python
"""ZTF on-axis GRB afterglow detection (blind discovery).

Estimates the rate of on-axis GRB afterglows that ZTF can *independently
discover* as fast-fading transients, without prior knowledge of a gamma-ray
trigger. This is a different question from "how many GRB afterglows will a
survey serendipitously image given a gamma-ray alert" — the latter yields
much higher numbers (e.g. ~75/yr for Rubin, Burns+ 2025) because it assumes
the GRB position and time are already known from Swift/Fermi.

Uses OnAxisGrbPopulation (θ_v ≤ θ_j from the catalog) with the observed
on-axis long GRB rate (R = 1.0 Gpc⁻³ yr⁻¹). Time span matches the
ZTFReST analysis period: Feb 2020 – Mar 2021.

Results (1M injections):
  Efficiency: 1.52%
  Expected:   9.5  (R = 1.0 Gpc⁻³ yr⁻¹)
  Observed:   5    (ZTFReST on-axis afterglows)
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import survey_sim.gpu_setup  # noqa: F401

import math
import time

from survey_sim import (
    OnAxisGrbPopulation,
    DetectionCriteria,
    SimulationPipeline,
    load_ztf_survey,
)
from survey_sim.fiesta_afterglow_model import FiestaAfterglowModel, ZTF_BAND_MAP

# --- Survey: Feb 2020 – Mar 2021 (ZTFReST analysis period) ---
print("Loading ZTF survey (Feb 2020 – Mar 2021)...")
t0 = time.time()
survey = load_ztf_survey(start="202002", end="202103", nside=64)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Observations: {survey.n_observations}")
print(f"  Duration: {survey.duration_years:.2f} yr")
print(f"  Bands: {survey.bands}")
print()

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

# --- Model: fiesta FS+RS ---
print("Loading fiesta blastwave_rs_gaussian_CVAE surrogate (FS+RS)...")
t0 = time.time()
model = FiestaAfterglowModel(
    name="blastwave_rs_gaussian_CVAE",
    band_map=ZTF_BAND_MAP,
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
print("RESULTS: ZTF On-Axis GRB Afterglow Detection (FS+RS)")
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
f_sky = 0.47
V_eff = V_max * f_sky
duration = survey.duration_years

print(f"Survey parameters:")
print(f"  Duration:  {duration:.2f} yr")
print(f"  f_sky:     {f_sky}")
print(f"  V_max:     {V_max:.0f} Gpc^3 (z < {z_max})")
print(f"  V_eff:     {V_eff:.1f} Gpc^3")
print()

# Expected detections
print(f"{'Rate (Gpc^-3 yr^-1)':>22s}  {'Expected detections':>20s}")
print("-" * 50)
for rate in [0.5, 1.0, 1.3, 2.0]:
    expected = rate * V_eff * duration * eff
    print(f"{rate:>22.1f}  {expected:>20.1f}")

print()
print("Observed in ZTFReST (Mar 2018 – Mar 2021):")
print("  5 afterglows with GRB counterpart (on-axis)")

if eff > 0:
    VT = V_eff * duration * eff
    R90 = -math.log(0.10) / VT
    R95 = -math.log(0.05) / VT
    print()
    print(f"Rate upper limits (assuming 0 detections):")
    print(f"  R_upper (90% CL) = {R90:.0f} Gpc^-3 yr^-1")
    print(f"  R_upper (95% CL) = {R95:.0f} Gpc^-3 yr^-1")
