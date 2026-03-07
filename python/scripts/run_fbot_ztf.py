#!/usr/bin/env python
"""ZTF FBOT detection rate (Ho et al. 2021 comparison).

Reproduces the fast blue optical transient sample from Ho et al. (2021,
arXiv:2105.08811), which identified 38 FBOTs in ZTF Phase I
(March 2018 – October 2020, ~2.6 years).

Uses the Bazin parametric model (exponential rise/fall) with timescales
drawn from the observed Ho+2021 Table 10 distributions.

Key parameters:
  - Rate: 65 Gpc^-3 yr^-1 (~0.1% of CC SN rate, Ho+2021)
  - Peak abs mag: N(-18.7, 1.5), range M_g ~ -16 to -22
  - z_max: 0.3
  - Fast transient: t_1/2 = 1–12 days

Detection criteria: fast-rising (>=1 mag in 6.5 days), well-sampled,
blue (g-r < -0.2), matching Ho+2021 selection.
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
    load_ztf_survey,
)

# --- Survey: ZTF Phase I (March 2018 – October 2020) ---
print("Loading ZTF survey (Mar 2018 – Oct 2020)...")
t0 = time.time()
survey = load_ztf_survey(start="201803", end="202010", nside=64)
print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  Observations: {survey.n_observations}")
print(f"  Duration: {survey.duration_years:.2f} yr")
print(f"  Bands: {survey.bands}")
print()

# --- Population ---
# Ho+2021: 38 FBOTs with M_g from -16.4 to -21.2.
# Median M_g ~ -18.7. Rate calibrated to match the observed sample.
# CC SN rate ~ 70,000 Gpc^-3/yr; FBOTs are a few % of CC SNe.
RATE = 65.0     # Gpc^-3 yr^-1 (~0.1% of CC SN rate, Ho+2021)
Z_MAX = 0.3
N_TRANSIENTS = 100_000
SEED = 42

pop = FbotPopulation(rate=RATE, z_max=Z_MAX, peak_abs_mag=-18.7)

# --- Detection criteria ---
# Ho+2021 selection: fast-rising, well-sampled, duration 1-12 days.
# We use fast transient criteria with rise rate >= 1 mag/6.5d ~ 0.15 mag/d
# and minimum fade rate to select genuine fast transients.
# Ho+2021 selection criteria:
#   1. Duration 1 < t_1/2 < 12 days (fast transient)
#   2. Fast-rising: >= 1 mag rise in preceding 6.5 days
#   3. Well-sampled: observations within 5.5 days of peak in g AND r
#   4. Blue color: g-r < -0.2 at peak
#   5. Spectroscopic classification
# We approximate criteria 1-2 with fast transient detection, 3 with
# multi-band coverage, and 4-5 with a strict brightness cut.
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=5.0,
    min_detections=5,            # well-sampled lightcurve
    min_detections_primary=5,
    min_bands=2,                 # g AND r required
    min_per_band=2,
    max_timespan_days=24.0,      # t_1/2 < 12d → visible for ~24d
    min_time_separation_hours=24.0,
    require_fast_transient=True,
    min_rise_rate=0.15,          # >= 1 mag in 6.5 days
    min_fade_rate=0.1,           # must show fading
    min_pre_peak_detections=1,   # need pre-peak coverage
    min_post_peak_detections=1,  # need post-peak coverage
    min_phase_range_days=3.0,    # coverage spanning peak
    min_galactic_lat=7.0,        # ZTF avoids |b| < 7
    # Bright enough for spectroscopic classification + host subtraction
    spectroscopic_completeness_k=3.0,
    spectroscopic_completeness_m0=19.0,
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
print("RESULTS: ZTF FBOT Detection (Ho et al. 2021 comparison)")
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
f_sky = 0.47
d_max_mpc = 1380.0  # ~z=0.3
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
print(f"Expected detections:      {N_detected:,.1f}")
print(f"Ho et al. (2021) actual:  38")
if N_detected > 0:
    print(f"Agreement factor:         {N_detected/38:.1f}x")
print()
# What rate would give exactly 38?
if eff > 0:
    R_needed = 38 / (V_eff * duration * eff)
    print(f"Rate to match 38:         {R_needed:,.0f} Gpc^-3 yr^-1")
    print(f"  (= {R_needed/70000*100:.1f}% of CC SN rate)")
