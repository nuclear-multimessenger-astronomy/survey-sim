#!/usr/bin/env python
"""Argus Array GRB afterglow detection with blastwave model across stacking strategies.

Loops over stacking windows (1s, 15min, 1hr, 1day, 5day) and computes
detection efficiency for GRB afterglows including reverse shocks.
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import math
import time

from survey_sim import (
    SurveyStore,
    GrbPopulation,
    BlastwaveModel,
    DetectionCriteria,
    SimulationPipeline,
)

# Argus parquet files
parquet_files = ["/fred/oz480/mcoughli/simulations/argus/argussim_hpx_6131.parquet"]

# GRB catalog
grb_csv = "/fred/oz480/mcoughli/simulations/argus/GRB_afterglows_argus.csv"

# Stacking strategies: (label, window_seconds, description)
STACKING_STRATEGIES = [
    ("1s",    None,      "No stacking (raw 1-second exposures)"),
    ("15min", 900.0,     "15-minute stacks"),
    ("1hr",   3600.0,    "1-hour stacks"),
    ("1day",  86400.0,   "1-day stacks"),
    ("5day",  432000.0,  "5-day stacks"),
]

# Argus band frequency: broad g-band centered at 445 nm
# ν = c / λ = 3e10 / 4.45e-5 = 6.74e14 Hz
ARGUS_BAND_FREQ = {"g": 6.74e14}

# GRB population
N_TRANSIENTS = 10000  # per stacking strategy
SEED = 42

# Detection criteria for Argus
det = DetectionCriteria(
    snr_threshold=5.0,
    snr_threshold_secondary=3.0,
    min_detections=2,
    min_detections_primary=1,
    max_timespan_days=30.0,
    min_time_separation_hours=0.5,
    require_fast_transient=False,
    min_rise_rate=0.0,
    min_fade_rate=0.0,
)

# Blastwave model with reverse shocks (default)
model = BlastwaveModel(
    radiation_model="sync_ssa_smooth",
    band_frequencies=ARGUS_BAND_FREQ,
)

print("=" * 70)
print("Argus Array GRB Afterglow Detection — Stacking Comparison")
print("=" * 70)
print(f"GRB catalog: {grb_csv}")
print(f"N_transients: {N_TRANSIENTS}")
print(f"Blastwave model: sync_ssa_smooth (with reverse shocks)")
print()

results = []

for label, window_s, description in STACKING_STRATEGIES:
    print(f"--- {label}: {description} ---")
    t0 = time.time()

    # Load survey with stacking
    survey = SurveyStore.from_argus(parquet_files, band="g", nside=64, stack_window_s=window_s)
    n_obs = survey.n_observations
    duration = survey.duration_years

    # Expected depth boost
    if window_s is not None:
        # Typical N exposures per window (1-second cadence, but not all seconds observed)
        approx_n_per_window = max(1, n_obs)  # rough
        boost_str = f"~1.25*log10(N) deeper"
    else:
        boost_str = "single-frame depth"

    print(f"  Observations: {n_obs}, Duration: {duration:.2f} yr, Depth: {boost_str}")

    # GRB population
    grb_pop = GrbPopulation(grb_csv, rate=1.0, z_max=6.0)

    # Pipeline
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

    eff = result.n_detected / max(result.n_simulated, 1)
    print(f"  Detected: {result.n_detected} / {result.n_simulated}")
    print(f"  Efficiency: {eff:.6f} ({eff*100:.4f}%)")
    print(f"  Time: {elapsed:.1f}s")

    # Rate calculation
    z_max = 6.0
    # Comoving volume out to z=6 ~ 1140 Gpc^3
    # For simplicity, use the efficiency and a nominal volume
    # The pipeline's rate_summary has more detail
    for rs in result.rate_summaries:
        print(f"  z_max used: {rs.transient_type}, eff={rs.overall_efficiency:.6f}")

    results.append({
        "label": label,
        "window_s": window_s,
        "n_obs": n_obs,
        "n_detected": result.n_detected,
        "n_simulated": result.n_simulated,
        "efficiency": eff,
        "duration_yr": duration,
        "elapsed_s": elapsed,
    })
    print()

# Summary table
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Stacking':>8s}  {'N_obs':>8s}  {'N_det':>6s}  {'N_sim':>7s}  {'Eff (%)':>9s}  {'Time':>7s}")
print("-" * 70)
for r in results:
    eff_pct = r["efficiency"] * 100
    print(f"{r['label']:>8s}  {r['n_obs']:>8d}  {r['n_detected']:>6d}  {r['n_simulated']:>7d}  {eff_pct:>9.4f}  {r['elapsed_s']:>6.1f}s")

# Rate upper limits (90% CL) for each strategy
print()
print(f"{'Stacking':>8s}  {'Eff':>10s}  {'VT_eff':>12s}  {'R_upper(90%)':>14s}")
print("-" * 70)
f_sky = 8000.0 / 41253.0  # Argus FoV / full sky
for r in results:
    eff = r["efficiency"]
    dur = r["duration_yr"]
    # Approximate comoving volume to z=6
    V_max = 1140.0  # Gpc^3 (approximate)
    V_eff = V_max * f_sky
    VT = V_eff * dur * eff
    if VT > 0:
        R90 = -math.log(0.10) / VT
        print(f"{r['label']:>8s}  {eff:>10.6f}  {VT:>10.4f}  {R90:>12.1f}")
    else:
        print(f"{r['label']:>8s}  {eff:>10.6f}  {'N/A':>12s}  {'inf':>14s}")
