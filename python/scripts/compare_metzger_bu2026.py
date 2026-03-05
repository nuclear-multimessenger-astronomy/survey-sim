#!/usr/bin/env python3
"""Compare Metzger analytic KN model vs Bu2026 fiesta surrogate.

Generates sample KN instances at a few redshifts, evaluates both models on
Rubin-like observation cadence, and prints per-band lightcurve comparisons.
Optionally runs both through the full pipeline to compare detection efficiencies.
"""

import sys
import numpy as np

from survey_sim import (
    SurveyStore,
    KilonovaPopulation,
    Bu2026KilonovaPopulation,
    MetzgerKNModel,
    DetectionCriteria,
    SimulationPipeline,
)
from survey_sim.fiesta_model import FiestaKNModel


OPSIM_DB = "/fred/oz480/mcoughli/simulations/TESS-Rubin/baseline_v5.1.1_10yrs.db"
BANDS = ["u", "g", "r", "i", "z", "y"]
REDSHIFTS = [0.05, 0.1, 0.2]


def print_lightcurve_comparison():
    """Print per-band lightcurves from both models at a few redshifts."""
    print("=" * 70)
    print("Lightcurve comparison: Metzger vs Bu2026")
    print("=" * 70)

    # Fiesta model
    fiesta = FiestaKNModel()

    # Observation times: 0.5 to 20 days post-explosion (observer frame), every 3 days
    obs_rest_days = np.arange(0.5, 20.0, 3.0)

    for z in REDSHIFTS:
        print(f"\n--- z = {z} ---")

        # Cosmology for luminosity distance (simple Hubble law approx for low z)
        d_l_mpc = z * 299792.458 / 70.0  # rough d_L in Mpc
        t_exp = 60000.0  # arbitrary MJD

        obs_days = obs_rest_days * (1.0 + z)
        obs_times_mjd = t_exp + obs_days
        obs_bands_repeated = BANDS * len(obs_rest_days)
        obs_times_repeated = np.repeat(obs_times_mjd, len(BANDS))

        # Bu2026 parameters (fiducial)
        bu_params = {
            "log10_mej_dyn": -2.5,
            "v_ej_dyn": 0.2,
            "Ye_dyn": 0.25,
            "log10_mej_wind": -1.5,
            "v_ej_wind": 0.1,
            "Ye_wind": 0.3,
            "inclination_EM": 0.5,
            "luminosity_distance": d_l_mpc,
            "redshift": z,
            "_obs_times_mjd": obs_times_repeated.tolist(),
            "_obs_bands": obs_bands_repeated,
            "_t_exp": t_exp,
        }

        times_bu, mags_bu = fiesta.predict(bu_params)

        print(f"  Bu2026 (d_L={d_l_mpc:.1f} Mpc):")
        for band in BANDS:
            if band in mags_bu:
                # Sample a few time points
                mags_arr = np.array(mags_bu[band])
                # Get per-epoch values (one per obs epoch, for this band)
                per_epoch = mags_arr[::len(BANDS)] if len(mags_arr) > len(BANDS) else mags_arr[:len(obs_rest_days)]
                vals = " ".join(f"{m:6.2f}" for m in per_epoch[:7])
                print(f"    {band}: {vals}")

        print(f"\n  (Metzger model is achromatic — single mag vs time, not per-band)")


def run_pipeline_comparison():
    """Run both models through the pipeline and compare detection efficiencies."""
    print("\n" + "=" * 70)
    print("Pipeline comparison: Metzger vs Bu2026 (Rubin Year 1)")
    print("=" * 70)

    survey = SurveyStore.from_opsim(OPSIM_DB, years=1)
    detection = DetectionCriteria(
        snr_threshold=5.0,
        n_detections=2,
        n_bands=1,
        time_window_days=14.0,
    )

    n_transients = 10000

    # --- Metzger ---
    print("\nRunning Metzger KN model...")
    metzger_pop = KilonovaPopulation(rate=1000.0, z_max=0.3, peak_abs_mag=-16.0)
    metzger_model = MetzgerKNModel(peak_abs_mag=-16.0)

    pipeline_metzger = SimulationPipeline(
        survey=survey,
        populations=[metzger_pop],
        models={"Kilonova": metzger_model},
        detection=detection,
        n_transients=n_transients,
        seed=42,
    )
    result_metzger = pipeline_metzger.run()
    print(f"  Metzger: {result_metzger}")
    for rs in result_metzger.rate_summaries:
        print(f"    {rs}")

    # --- Bu2026 ---
    print("\nRunning Bu2026 fiesta model...")
    bu_pop = Bu2026KilonovaPopulation(rate=1000.0, z_max=0.3)
    fiesta_model = FiestaKNModel()

    pipeline_bu = SimulationPipeline(
        survey=survey,
        populations=[bu_pop],
        models={"Kilonova": fiesta_model},
        detection=detection,
        n_transients=n_transients,
        seed=42,
    )
    result_bu = pipeline_bu.run()
    print(f"  Bu2026:  {result_bu}")
    for rs in result_bu.rate_summaries:
        print(f"    {rs}")

    # --- Summary ---
    print("\n--- Summary ---")
    eff_m = result_metzger.n_detected / max(result_metzger.n_simulated, 1)
    eff_b = result_bu.n_detected / max(result_bu.n_simulated, 1)
    print(f"  Metzger efficiency: {eff_m:.4f}")
    print(f"  Bu2026  efficiency: {eff_b:.4f}")


if __name__ == "__main__":
    print_lightcurve_comparison()

    if "--pipeline" in sys.argv:
        run_pipeline_comparison()
    else:
        print("\nSkipping pipeline comparison (pass --pipeline to enable).")
