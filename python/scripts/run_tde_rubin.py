#!/usr/bin/env python
"""Rubin LSST TDE detection prediction.

Predicts the number of TDEs that Rubin will photometrically detect
over 10 years, using the Yao et al. (2023) broken power-law LF.

Runs five configurations:
  1. With host brightness cut (m0=22.5): conservative, requires TDE to
     outshine host nucleus. Comparable to spectroscopically confirmed samples.
  2. Without host brightness cut: all photometrically detectable TDEs.
  3-4. With Karmen+2025 rate evolution F(z) × N_BH(z) × O(z), z_max=2.96
     (Lyman-alpha cutoff for g-band). One for each BHMF model.
  5. With host cut + rate evolution (Illustris): the "realistic spectroscopic" case.

Finally, compares injection results with the Rust semi-analytical forecast
from Karmen et al. (2025).
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import math
import time
import numpy as np

from survey_sim import (
    SurveyStore,
    TdePopulation,
    ParametricModel,
    DetectionCriteria,
    SimulationPipeline,
    TdeRateForecast,
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

N_TRANSIENTS = 100_000
SEED = 42
RATE = 829.0  # LF-integrated rate (Yao+2023, M_g -24 to -15), Gpc^-3 yr^-1
Z_LY_G = 2.96  # Lyman-alpha cutoff for g-band (482 nm)

# Evolution parameters (median of Karmen+2025 MC ranges)
E_FACTOR = 30.0   # median of U[10, 100]
DENSITY_ALPHA = 1.5  # median of U[1, 2]


def make_detection_criteria(host_cut=True):
    """Build detection criteria with or without the host brightness cut."""
    return DetectionCriteria(
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
        # Host brightness cut: peak ~2 mag above 5-sigma → m_peak < 22.5
        spectroscopic_completeness_k=5.0 if host_cut else 0.0,
        spectroscopic_completeness_m0=22.5,
    )


def compute_evolved_effective_volume(z_max, bhmf_model, duration):
    """Compute the evolution-weighted effective volume for rate scaling.

    For evolved populations, the number of TDEs is:
      N = rate_local × duration × ∫₀^z_max (1/(1+z)) × dV/dz × F(z) × N_BH(z) × O(z) dz × f_sky

    This function returns V_eff such that N = rate_local × V_eff × duration × f_sky.
    """
    from astropy.cosmology import Planck18 as cosmo
    from astropy import units as u

    # Simpson's rule integration
    n_steps = 500
    z_arr = np.linspace(0, z_max, n_steps + 1)
    dz = z_arr[1] - z_arr[0]

    # Evolution factors (matching Rust implementation)
    def merger_enhancement(z):
        f_pair = lambda z: 0.056 * (1+z)**5.910 * np.exp(-1.814*(1+z))
        return (1 + (E_FACTOR-1)*f_pair(z)) / (1 + (E_FACTOR-1)*f_pair(0))

    def density_evolution(z):
        return (1+z)**(0.9 * DENSITY_ALPHA)

    def dust_obscuration(z):
        f0, f_max, k = 0.3, 0.9, 0.7
        f_obsc = lambda z: f0 + (f_max - f0) / (1 + np.exp(-k * np.log(1+z)))
        return f_obsc(z) / f_obsc(0)

    def imf_evolution(z):
        m_min, m_max_m = 0.1, 100.0
        alpha_0, alpha_z8 = 2.35, 2.081
        alpha_z = alpha_0 + (alpha_z8 - alpha_0) * min(z, 8) / 8
        def mean_m2(alpha):
            num = (m_max_m**(3-alpha) - m_min**(3-alpha)) / (3-alpha)
            den = (m_max_m**(1-alpha) - m_min**(1-alpha)) / (1-alpha)
            return num / den
        return mean_m2(alpha_z) / mean_m2(alpha_0)

    def galaxy_effects(z):
        return merger_enhancement(z) * density_evolution(z) * imf_evolution(z)

    def bhmf_evolution(z):
        alpha = -0.82 if bhmf_model == "illustris" else -1.46
        return np.exp(alpha * z)

    # dV/dz in Gpc³/sr, times 4π for full sky
    integrand = np.zeros_like(z_arr)
    for i, z in enumerate(z_arr):
        if z == 0:
            # dV/dz → 0 at z=0
            integrand[i] = 0.0
            continue
        dv_dz = cosmo.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value * 4 * np.pi
        w = galaxy_effects(z) * bhmf_evolution(z) * dust_obscuration(z) / (1+z)
        integrand[i] = dv_dz * w

    # Simpson's rule
    from scipy.integrate import simpson
    V_eff = simpson(integrand, x=z_arr)
    return V_eff


def run_sim(survey, z_max, host_cut, label, use_rate_evolution=False, bhmf_model="illustris"):
    """Run injection simulation and return summary dict."""
    pop = TdePopulation(
        z_max=z_max,
        use_luminosity_function=True,
        use_rate_evolution=use_rate_evolution,
        bhmf_model=bhmf_model,
    )
    det = make_detection_criteria(host_cut=host_cut)
    model = ParametricModel()

    print(f"--- {label} ---")
    print(f"  z_max={z_max}, host_cut={host_cut}, evolved={use_rate_evolution}, "
          f"bhmf={bhmf_model}, N={N_TRANSIENTS:,}")
    t0 = time.time()
    pipe = SimulationPipeline(
        survey=survey,
        populations=[pop],
        models={"TDE": model},
        detection=det,
        n_transients=N_TRANSIENTS,
        seed=SEED,
    )
    result = pipe.run()
    elapsed = time.time() - t0

    eff = result.n_detected / max(result.n_simulated, 1)
    duration = survey.duration_years
    f_sky = 0.44  # Rubin sky fraction

    if use_rate_evolution:
        # Evolution-weighted effective volume
        V_eff = compute_evolved_effective_volume(z_max, bhmf_model, duration)
        N_total = RATE * V_eff * f_sky * duration
    else:
        # Simple comoving volume (constant rate)
        from astropy.cosmology import Planck18 as cosmo
        from astropy import units as u
        V_max = cosmo.comoving_volume(z_max).to(u.Gpc**3).value
        V_eff = V_max * f_sky
        N_total = RATE * V_eff * duration

    N_detected = N_total * eff
    N_per_yr = N_detected / duration

    print(f"  Detected: {result.n_detected:,}/{result.n_simulated:,}")
    print(f"  Efficiency: {eff*100:.4f}%")
    print(f"  Expected: {N_detected:,.0f} in {duration:.0f}yr ({N_per_yr:,.0f}/yr)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()

    return {
        "label": label,
        "z_max": z_max,
        "host_cut": host_cut,
        "evolved": use_rate_evolution,
        "bhmf": bhmf_model,
        "n_sim": result.n_simulated,
        "n_det": result.n_detected,
        "efficiency": eff,
        "N_per_yr": N_per_yr,
        "N_total": N_detected,
        "elapsed": elapsed,
    }


# =========================================================================
# Run injection simulations
# =========================================================================
results = []

# 1. z<0.8, with host cut, no evolution (spectroscopic-like baseline)
results.append(run_sim(survey, z_max=0.8, host_cut=True,
                       label="z<0.8, host cut, no evol"))

# 2. z<0.8, no host cut, no evolution (photometric-only baseline)
results.append(run_sim(survey, z_max=0.8, host_cut=False,
                       label="z<0.8, no host cut, no evol"))

# 3. z<2.96, no host cut, Illustris evolution (full analytical comparison)
results.append(run_sim(survey, z_max=Z_LY_G, host_cut=False,
                       use_rate_evolution=True, bhmf_model="illustris",
                       label="z<2.96, no cut, Illustris"))

# 4. z<2.96, no host cut, Shankar evolution
results.append(run_sim(survey, z_max=Z_LY_G, host_cut=False,
                       use_rate_evolution=True, bhmf_model="shankar",
                       label="z<2.96, no cut, Shankar"))

# 5. z<2.96, with host cut, Illustris evolution (realistic spectroscopic)
results.append(run_sim(survey, z_max=Z_LY_G, host_cut=True,
                       use_rate_evolution=True, bhmf_model="illustris",
                       label="z<2.96, host cut, Illustris"))

# =========================================================================
# Semi-analytical comparison (Karmen+2025, Rust)
# =========================================================================
print("--- Semi-analytical forecast (Karmen+2025, Rust) ---")
forecast = TdeRateForecast(temperature_k=30000.0, n_mc=200, seed=42)
t0 = time.time()
r_ill = forecast.compute_rate("rubin", "illustris")
r_sha = forecast.compute_rate("rubin", "shankar")
print(f"  Illustris: {r_ill['N_median']:,.0f}/yr [{r_ill['N_16']:,.0f}, {r_ill['N_84']:,.0f}]"
      f"  z_med={r_ill['z_median']:.2f}")
print(f"  Shankar:   {r_sha['N_median']:,.0f}/yr [{r_sha['N_16']:,.0f}, {r_sha['N_84']:,.0f}]"
      f"  z_med={r_sha['z_median']:.2f}")
print(f"  Paper:     Illustris 26,873/yr  |  Shankar 13,803/yr")
print(f"  Elapsed:   {time.time()-t0:.3f}s")
print()

# =========================================================================
# Summary table
# =========================================================================
duration = survey.duration_years

print("=" * 85)
print("SUMMARY: Rubin LSST TDE Predictions")
print("=" * 85)
print(f"{'Configuration':<38s} {'z_max':>5s} {'Eff%':>8s} {'N/yr':>10s} {'N/10yr':>10s}")
print("-" * 85)
for r in results:
    print(f"{r['label']:<38s} {r['z_max']:>5.2f} {r['efficiency']*100:>8.4f}"
          f" {r['N_per_yr']:>10,.0f} {r['N_total']:>10,.0f}")

# Add analytical forecasts
print("-" * 85)
print(f"{'Analytical (Illustris)':<38s} {'2.96':>5s} {'(ε=1)':>8s}"
      f" {r_ill['N_median']:>10,.0f} {r_ill['N_median']*duration:>10,.0f}")
print(f"{'Analytical (Shankar)':<38s} {'2.96':>5s} {'(ε=1)':>8s}"
      f" {r_sha['N_median']:>10,.0f} {r_sha['N_median']*duration:>10,.0f}")
print(f"{'Karmen+2025 (Illustris)':<38s} {'2.96':>5s} {'(ε=1)':>8s}"
      f" {'26,873':>10s} {'268,730':>10s}")
print(f"{'Karmen+2025 (Shankar)':<38s} {'2.96':>5s} {'(ε=1)':>8s}"
      f" {'13,803':>10s} {'138,030':>10s}")
print()

print("Notes:")
print("  - 'no evol' = constant local rate (829 Gpc^-3/yr)")
print("  - 'Illustris/Shankar' = Karmen+2025 F(z)×N_BH(z)×O(z) rate evolution")
print("  - 'host cut' = logistic brightness cut requiring TDE to outshine host (m0=22.5)")
print("  - Analytical assumes ε=1 (perfect cadence); injection uses real Rubin cadence")
print("  - Remaining gap: injection has cadence losses, analytical does not")
