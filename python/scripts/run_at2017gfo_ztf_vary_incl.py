#!/usr/bin/env python
"""Run ZTF boom pipeline with best-fit AT2017gfo Bu2026 parameters,
varying inclination as flat in cos(iota)."""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")
sys.path.insert(0, "/fred/oz480/mcoughli/fiestaEM/src")
import survey_sim.gpu_setup  # noqa: F401 — configure LD_LIBRARY_PATH for JAX GPU

import glob
import math

from survey_sim import (
    SurveyStore,
    FixedBu2026KilonovaPopulation,
    DetectionCriteria,
    SimulationPipeline,
)
from survey_sim.fiesta_model import FiestaKNModel

# Load ZTF boom data
boom_files = sorted(glob.glob("/fred/oz480/mcoughli/simulations/ztf_boom/*.h5"))
print(f"Loading {len(boom_files)} ZTF boom HDF5 files...")
survey = SurveyStore.from_ztf_boom(boom_files, nside=64)
print(f"  Observations: {survey.n_observations}")
print(f"  MJD range: {survey.mjd_range}")
print(f"  Duration: {survey.duration_years:.2f} years")
print(f"  Bands: {survey.bands}")

# Best-fit AT2017gfo Bu2026 parameters, varying inclination
pop = FixedBu2026KilonovaPopulation(
    log10_mej_dyn=-1.7,
    v_ej_dyn=0.2,
    ye_dyn=0.15,
    log10_mej_wind=-1.1,
    v_ej_wind=0.1,
    ye_wind=0.35,
    vary_inclination=True,  # flat in cos(iota)
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

# Bu2026 model
model = FiestaKNModel()

# Run pipeline
N = 100000
print(f"\nRunning pipeline with {N} transients (varying inclination)...")
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
print(f"  Efficiency: {eff:.4f} ({eff*100:.2f}%)")

for rs in result.rate_summaries:
    print(f"\n  {rs.transient_type}:")
    print(f"    Volumetric rate: {rs.volumetric_rate:.1f} Gpc^-3/yr")
    print(f"    Overall efficiency: {rs.overall_efficiency:.4f}")

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
print(f"  Efficiency = {eff:.4f}")
print(f"  VT_eff = {VT_eff:.4f} Gpc^3 yr")

for cl, label in [(0.90, "90%"), (0.95, "95%")]:
    R_upper = -math.log(1 - cl) / VT_eff if VT_eff > 0 else float("inf")
    print(f"  R_upper ({label} CL) = {R_upper:.0f} Gpc^-3 yr^-1")
