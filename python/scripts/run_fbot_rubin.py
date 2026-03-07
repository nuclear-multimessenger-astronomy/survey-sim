#!/usr/bin/env python
"""Rubin LSST FBOT detection prediction.

Predicts the number of fast blue optical transients that Rubin will
photometrically detect over 10 years, using the same population model
calibrated against the Ho et al. (2021) ZTF sample.

Uses identical population parameters as the ZTF script (rate=65
Gpc^-3/yr, peak_abs_mag=-18.7). Rubin's greater depth enables
detection of fainter FBOTs at higher redshift.

Key parameters:
  - Rate: 65 Gpc^-3 yr^-1 (~0.1% of CC SN rate)
  - Peak abs mag: N(-18.7, 1.5)
  - z_max: 0.5
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import math
import time

from survey_sim import (
    SurveyStore,
    FbotPopulation,
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
# Same rate and peak mag as ZTF calibration.
RATE = 65.0     # Gpc^-3 yr^-1 (~0.1% of CC SN rate)
Z_MAX = 0.5
N_TRANSIENTS = 100_000
SEED = 42

pop = FbotPopulation(rate=RATE, z_max=Z_MAX, peak_abs_mag=-18.7)

# --- Detection criteria ---
# Photometric detection of fast transients. Relaxed relative to ZTF
# since Rubin identifies transients from difference imaging without
# requiring spectroscopic classification.
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=5.0,
    min_detections=3,
    min_detections_primary=3,
    min_bands=2,                 # multi-band for color
    min_per_band=2,
    max_timespan_days=24.0,
    min_time_separation_hours=24.0,
    require_fast_transient=True,
    min_rise_rate=0.15,
    min_fade_rate=0.1,
    min_pre_peak_detections=0,
    min_post_peak_detections=0,
    min_phase_range_days=0.0,
    min_galactic_lat=15.0,
    # Require ~2 mag brighter than limit to distinguish from host
    spectroscopic_completeness_k=5.0,
    spectroscopic_completeness_m0=22.5,
)

# --- Pipeline ---
print(f"Running pipeline with {N_TRANSIENTS:,} transients (z_max={Z_MAX})...")
t0 = time.time()
pipe = SimulationPipeline(
    survey=survey,
    populations=[pop],
    models={"FBOT": ParametricModel()},
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
print("RESULTS: Rubin LSST FBOT Detection Prediction")
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
f_sky = 0.44  # Rubin WFD sky fraction
d_max_mpc = 2900.0  # ~z=0.5
V_max = (4.0 / 3.0) * math.pi * (d_max_mpc / 1000.0) ** 3  # Gpc^3
V_eff = V_max * f_sky
duration = survey.duration_years

N_total = RATE * V_eff * duration
N_detected = N_total * eff

print("Survey parameters:")
print(f"  Duration:  {duration:.2f} yr")
print(f"  f_sky:     {f_sky}")
print(f"  Rate:      {RATE:.0f} Gpc^-3 yr^-1")
print(f"  V_max:     {V_max:.1f} Gpc^3 (z < {Z_MAX})")
print(f"  V_eff:     {V_eff:.1f} Gpc^3")
print()
print(f"Total FBOTs in volume:    {N_total:,.0f}")
print(f"Detection efficiency:     {eff*100:.4f}%")
print(f"Expected detections:      {N_detected:,.0f} in {duration:.0f}yr")
print(f"Expected per year:        {N_detected/duration:,.0f}")
