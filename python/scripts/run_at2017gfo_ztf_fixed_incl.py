#!/usr/bin/env python
"""Run ZTF boom pipeline with best-fit AT2017gfo Bu2026 parameters,
fixed inclination, tuned ejecta mass for g/r/i match."""
import argparse
parser = argparse.ArgumentParser(description="Provide relevant paths")
parser.add_argument("--survey-sim", type=str, default="/fred/oz480/mcoughli/simulations/survey-sim/python", help="path to survey-sim python folder")
parser.add_argument("--fiesta", type=str, default="/fred/oz480/mcoughli/fiestaEM/src", help="path to fiestaEM")
parser.add_argument("--boom", type=str, default="/fred/oz480/mcoughli/simulations/ztf_boom", help="path to ZTF boom data")
args = parser.parse_args()

survey_sim_dir = args.survey_sim
fiestaEM_dir = args.fiesta

import sys
sys.path.insert(0, survey_sim_dir)
sys.path.insert(0, fiestaEM_dir)
import survey_sim.gpu_setup  # noqa: F401 — configure LD_LIBRARY_PATH for JAX GPU

import glob

from survey_sim import (
    SurveyStore,
    FixedBu2026KilonovaPopulation,
    DetectionCriteria,
    SimulationPipeline,
)
from survey_sim.fiesta_model import FiestaKNModel

# Load all available ZTF boom data (2018 through present).
boom_dir = args.boom
boom_files = sorted(glob.glob(f"{boom_dir}/ztf_*.h5"))
print(f"Loading {len(boom_files)} ZTF boom HDF5 files (full baseline)...")
survey = SurveyStore.from_ztf_boom(boom_files, nside=64)
print(f"  Observations: {survey.n_observations}")
print(f"  MJD range: {survey.mjd_range}")
print(f"  Duration: {survey.duration_years:.2f} years")
print(f"  Bands: {survey.bands}")

# Tuned AT2017gfo Bu2026 parameters (best g/r/i fit at t<4d)
# log10_mej_dyn=-1.8 (slightly less dynamical ejecta)
# inclination_em=0.45 rad (26 deg, consistent with GW170817 constraints)
pop = FixedBu2026KilonovaPopulation(
    log10_mej_dyn=-1.8,
    v_ej_dyn=0.2,
    ye_dyn=0.15,
    log10_mej_wind=-1.1,
    v_ej_wind=0.1,
    ye_wind=0.35,
    inclination_em=0.45,
    rate=1000.0,
    # z_max chosen well above the AT2017gfo-bright ZTF detection horizon.
    # compute_rate's integrand has already converged out here, so tightening
    # further would dilute MC stats without biasing VT_eff.
    z_max=0.15,
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
    min_galactic_lat=15.0,
)

# Bu2026 model
model = FiestaKNModel()

# Run pipeline
N = 1000000
print(f"\nRunning pipeline with {N} transients (fixed incl=0.45 rad, tuned ejecta)...")
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

print(f"\n--- Rate Upper Limits (projected, n_observed=0) ---")
for rs in result.rate_summaries:
    print(f"\n  {rs.transient_type}:")
    print(f"    Volumetric rate: {rs.volumetric_rate:.1f} Gpc^-3/yr")
    print(f"    Detected (MC):   {rs.n_detected} / {rs.n_simulated}")
    print(f"    Efficiency:      {rs.overall_efficiency:.4f}")
    print(f"    Duration:        {rs.survey_duration_years:.2f} yr")
    print(f"    VT_eff:          {rs.effective_vt_gpc3_yr:.4e} Gpc^3 yr")
    for cl in (0.90, 0.95, 0.99):
        ul = rs.upper_limit(confidence_level=cl, n_observed=0)
        print(f"    R_upper({cl:.0%}): {ul.rate_upper:8.1f} Gpc^-3 yr^-1")
