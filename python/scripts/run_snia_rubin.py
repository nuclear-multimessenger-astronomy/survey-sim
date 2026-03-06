#!/usr/bin/env python
"""Rubin LSST SN Ia detection (photometric, no spectroscopic completeness).

Predicts the number of SNe Ia that Rubin will photometrically detect
in the 10-year baseline survey. Unlike the ZTF DR2 validation which
includes BTS spectroscopic completeness, this is a pure photometric
detection estimate — every SN Ia that passes the light curve quality
cuts is counted.

Uses the SALT3 spectral template model via fiesta/JAX with population
parameters from skysurvey (Perley 2020 rate, Nicolas 2021 stretch,
Ginolin 2024 color).
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import survey_sim.gpu_setup  # noqa: F401

import math
import time

from survey_sim import (
    SurveyStore,
    SupernovaIaPopulation,
    DetectionCriteria,
    SimulationPipeline,
)
from survey_sim.salt3_model import FiestaSALT3Model, LSST_SALT3_BANDS

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

# --- Population ---
# Perley (2020) SN Ia volumetric rate
# z_max = 0.5 for well-sampled photometric SNe Ia with Rubin
RATE = 23500.0  # Gpc^-3 yr^-1
Z_MAX = 0.5
N_TRANSIENTS = 100_000
SEED = 42

pop = SupernovaIaPopulation(rate=RATE, z_max=Z_MAX, peak_abs_mag=-19.3)

# --- Detection criteria ---
# Photometric quality cuts for well-sampled SN Ia light curves.
# No spectroscopic completeness — Rubin is photometric classification.
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=5.0,
    min_detections=7,
    min_detections_primary=7,
    min_bands=2,
    min_per_band=3,
    max_timespan_days=100.0,
    min_time_separation_hours=24.0,
    require_fast_transient=False,
    min_rise_rate=0.0,
    min_fade_rate=0.0,
    min_pre_peak_detections=1,
    min_post_peak_detections=3,
    min_phase_range_days=30.0,
    min_galactic_lat=15.0,
)

# --- Model: SALT3 via fiesta/JAX ---
print("Loading SALT3 model for LSST bands...")
t0 = time.time()
model = FiestaSALT3Model(band_map=LSST_SALT3_BANDS)
model.warm_up(z_max=Z_MAX, dz=0.01, batch_size=1024)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Bands: {model.survey_filters}")
print()

# --- Pipeline ---
print(f"Running pipeline with {N_TRANSIENTS:,} transients (z_max={Z_MAX})...")
t0 = time.time()
pipe = SimulationPipeline(
    survey=survey,
    populations=[pop],
    models={"SNIa": model},
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
print("RESULTS: Rubin LSST SN Ia Detection (photometric, SALT3)")
print("=" * 70)
print(f"Transients simulated: {result.n_simulated:,}")
print(f"Transients detected:  {result.n_detected:,}")
print(f"Efficiency:           {eff:.6f} ({eff*100:.2f}%)")
print(f"Elapsed:              {elapsed:.1f}s")
print()

for rs in result.rate_summaries:
    print(f"  [{rs.transient_type}] efficiency = {rs.overall_efficiency:.6f}")
print()

# --- Rate calculation ---
f_sky = 0.44
d_max_mpc = 2800.0  # ~z=0.5
V_max = (4.0 / 3.0) * math.pi * (d_max_mpc / 1000.0) ** 3  # Gpc^3
V_eff = V_max * f_sky
duration = survey.duration_years

N_total = RATE * V_eff * duration
N_detected = N_total * eff

print(f"Survey parameters:")
print(f"  Duration:  {duration:.2f} yr")
print(f"  f_sky:     {f_sky}")
print(f"  V_max:     {V_max:.1f} Gpc^3 (z < {Z_MAX})")
print(f"  V_eff:     {V_eff:.1f} Gpc^3")
print()
print(f"Total SNe Ia in volume:  {N_total:,.0f}")
print(f"Detection efficiency:    {eff*100:.2f}%")
print(f"Expected detections:     {N_detected:,.0f} in {duration:.0f}yr")
print(f"Expected per year:       {N_detected/duration:,.0f}")
