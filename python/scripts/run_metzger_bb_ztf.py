#!/usr/bin/env python
"""Run ZTF boom pipeline with best-fit AT2017gfo Metzger BB parameters."""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import glob
import math

from survey_sim import (
    SurveyStore,
    FixedMetzgerKilonovaPopulation,
    MetzgerKNModel,
    DetectionCriteria,
    SimulationPipeline,
)

# Load ZTF boom data: March 2018 through March 2021
boom_dir = "/fred/oz480/mcoughli/simulations/ztf_boom"
boom_files = sorted(
    glob.glob(f"{boom_dir}/ztf_2018*.h5")
    + glob.glob(f"{boom_dir}/ztf_2019*.h5")
    + glob.glob(f"{boom_dir}/ztf_2020*.h5")
    + [f"{boom_dir}/ztf_202101.h5", f"{boom_dir}/ztf_202102.h5", f"{boom_dir}/ztf_202103.h5"]
)
boom_files = [f for f in boom_files if __import__("os").path.isfile(f)]
print(f"Loading {len(boom_files)} ZTF boom HDF5 files (Mar 2018 – Mar 2021)...")
survey = SurveyStore.from_ztf_boom(boom_files, nside=64)
print(f"  Observations: {survey.n_observations}")
print(f"  MJD range: {survey.mjd_range}")
print(f"  Duration: {survey.duration_years:.2f} years")
print(f"  Bands: {survey.bands}")

# Best-fit Metzger BB parameters for AT2017gfo (tuned on g/r/i, t < 3d)
# mej = 0.00126 Msun, vej = 0.50c, kappa = 398 cm²/g
pop = FixedMetzgerKilonovaPopulation(
    mej=0.00126,
    vej=0.50,
    kappa=398.0,
    rate=1000.0,
    z_max=0.3,
)

# ZTFReST-like detection criteria
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
)

# MetzgerKN model (blackbody path)
model = MetzgerKNModel()

# Run pipeline
N = 1000000
print(f"\nRunning pipeline with {N} transients (Metzger BB, tuned AT2017gfo params)...")
print(f"  mej=0.00126 Msun, vej=0.50c, kappa=398 cm²/g")
pipeline = SimulationPipeline(
    survey=survey,
    populations=[pop],
    models={"Kilonova": model},
    detection=det,
    n_transients=N,
    seed=42,
)
result = pipeline.run()
print(f"\nResults:")
print(f"  Simulated: {result.n_simulated}")
print(f"  Detected:  {result.n_detected}")
eff = result.n_detected / max(result.n_simulated, 1)
print(f"  Efficiency: {eff:.6f} ({eff*100:.4f}%)")

for rs in result.rate_summaries:
    print(f"\n  {rs.transient_type}:")
    print(f"    Volumetric rate: {rs.volumetric_rate:.1f} Gpc^-3/yr")
    print(f"    Overall efficiency: {rs.overall_efficiency:.6f}")

# Rate upper limits
duration = survey.duration_years
z_max = 0.3
d_max = 1380.0  # Mpc for z=0.3 approx
V_max = (4.0 / 3.0) * math.pi * (d_max / 1000.0) ** 3  # Gpc^3

# Sky coverage: ZTF covers ~47% of sky in public survey
omega_ztf = 0.47 * 4 * math.pi  # sr
f_sky = omega_ztf / (4 * math.pi)
V_eff = V_max * f_sky

VT_eff = V_eff * duration * eff

print(f"\n--- Rate Upper Limits ---")
print(f"  z_max = {z_max}, V_max = {V_max:.3f} Gpc^3")
print(f"  f_sky = {f_sky:.2f}, V_eff = {V_eff:.3f} Gpc^3")
print(f"  Duration = {duration:.2f} yr")
print(f"  Efficiency = {eff:.6f}")
print(f"  VT_eff = {VT_eff:.6f} Gpc^3 yr")

for cl, label in [(0.90, "90%"), (0.95, "95%")]:
    R_upper = -math.log(1 - cl) / VT_eff if VT_eff > 0 else float("inf")
    print(f"  R_upper ({label} CL) = {R_upper:.0f} Gpc^-3 yr^-1")
