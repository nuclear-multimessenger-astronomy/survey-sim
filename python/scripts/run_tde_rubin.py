#!/usr/bin/env python
"""Rubin LSST TDE detection prediction.

Predicts the number of TDEs that Rubin will photometrically detect
over 10 years, using the Yao et al. (2023) broken power-law LF —
the same population model used for the ZTF validation.

Uses identical population parameters as the ZTF script: peak
magnitudes drawn from the LF, rate auto-computed from LF integration.
Rubin's greater depth (~24.5 mag vs ZTF's ~20.5) naturally detects
more of the faint TDEs that dominate the LF.

Key parameters:
  - LF: Yao+2023 broken power-law (same as ZTF)
  - Rate: auto-computed from LF integration (~829 Gpc^-3/yr)
  - z_max: 0.8

Result: ~2,600 TDEs/yr (~26,000 in 10yr).
Bricman & Gomboc (2020) predict ~1,000/yr (conservative LF).
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import math
import time

from survey_sim import (
    SurveyStore,
    TdePopulation,
    ParametricModel,
    DetectionCriteria,
    SimulationPipeline,
)

# --- Survey: Rubin LSST 10-year baseline ---
OPSIM_DB = "/fred/oz480/mcoughli/simulations/TESS-Rubin/baseline_v5.1.1_10yrs.db"
print("Loading Rubin survey...")
t0 = time.time()
survey = SurveyStore.from_rubin(OPSIM_DB, nside=64)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Observations: {survey.n_observations}")
print(f"  Duration: {survey.duration_years:.2f} yr")
print(f"  Bands: {survey.bands}")
print()

# --- Population ---
# Same Yao+2023 broken power-law LF as ZTF — identical population.
# Rubin's depth naturally selects more faint TDEs.
Z_MAX = 0.8
N_TRANSIENTS = 100_000
SEED = 42

pop = TdePopulation(z_max=Z_MAX, use_luminosity_function=True)

# --- Detection criteria ---
# Photometric quality cuts (no spectroscopic completeness for Rubin).
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=5.0,
    min_detections=3,
    min_detections_primary=3,
    min_bands=1,
    min_per_band=2,
    max_timespan_days=365.0,
    min_time_separation_hours=48.0,
    require_fast_transient=False,
    min_rise_rate=0.0,
    min_fade_rate=0.0,
    min_pre_peak_detections=0,
    min_post_peak_detections=0,
    min_phase_range_days=0.0,
    min_galactic_lat=15.0,
    # Require TDE to outshine host: peak ~2 mag brighter than the
    # single-visit 5-sigma limit (~24.5), i.e., peak < ~22.5.
    spectroscopic_completeness_k=5.0,
    spectroscopic_completeness_m0=22.5,
)

# --- Pipeline ---
print(f"Running pipeline with {N_TRANSIENTS:,} transients (z_max={Z_MAX})...")
t0 = time.time()
pipe = SimulationPipeline(
    survey=survey,
    populations=[pop],
    models={"TDE": ParametricModel()},
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
print("RESULTS: Rubin LSST TDE Detection Prediction")
print("=" * 70)
print(f"Transients simulated: {result.n_simulated:,}")
print(f"Transients detected:  {result.n_detected:,}")
print(f"Efficiency:           {eff:.6f} ({eff*100:.4f}%)")
print(f"Elapsed:              {elapsed:.1f}s")
print()

for rs in result.rate_summaries:
    print(f"  [{rs.transient_type}] efficiency = {rs.overall_efficiency:.6f}")
print()

# --- Rate calculation ---
RATE = 829.0  # LF-integrated rate (Yao+2023, M_g -24 to -15)
f_sky = 0.44  # Rubin WFD sky fraction
d_max_mpc = 4800.0  # ~z=0.8
V_max = (4.0 / 3.0) * math.pi * (d_max_mpc / 1000.0) ** 3  # Gpc^3
V_eff = V_max * f_sky
duration = survey.duration_years

N_total = RATE * V_eff * duration
N_detected = N_total * eff

print("Survey parameters:")
print(f"  Duration:  {duration:.2f} yr")
print(f"  f_sky:     {f_sky}")
print(f"  Rate:      {RATE:.0f} Gpc^-3 yr^-1 (LF-integrated)")
print(f"  V_max:     {V_max:.1f} Gpc^3 (z < {Z_MAX})")
print(f"  V_eff:     {V_eff:.1f} Gpc^3")
print()
print(f"Total TDEs in volume:     {N_total:,.0f}")
print(f"Detection efficiency:     {eff*100:.4f}%")
print(f"Expected detections:      {N_detected:,.0f} in {duration:.0f}yr")
print(f"Expected per year:        {N_detected/duration:,.0f}")
print()
print("Bricman & Gomboc (2020):  ~1,000/yr")
