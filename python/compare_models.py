#!/usr/bin/env python3
"""Compare Metzger (lightcurve-fitting) vs Bulla (fiestaEM) kilonova models."""

import sys
sys.path.insert(0, "/home/mcoughli/fiestaEM/src")

import numpy as np

# === Bulla model (fiestaEM) ===
from fiesta.inference.lightcurve_model import BullaFlux

FILTERS = ["ztf::g", "ztf::r", "ztf::i"]
bulla = BullaFlux(name="Bu2025_MLP", filters=FILTERS)

# GW170817-like parameters
bulla_params = {
    "inclination_EM": 0.5,        # ~30 deg viewing angle
    "log10_mej_dyn": -2.0,        # 0.01 Msun dynamical ejecta
    "v_ej_dyn": 0.2,              # 0.2c
    "Ye_dyn": 0.25,               # electron fraction
    "log10_mej_wind": -1.5,       # 0.032 Msun wind ejecta
    "v_ej_wind": 0.1,             # 0.1c
    "Ye_wind": 0.3,               # electron fraction
    "luminosity_distance": 1e-5,  # ~0 for absolute mags
    "redshift": 0.0,
}

times_bulla, mag_bulla = bulla.predict(bulla_params)
times_bulla = np.asarray(times_bulla)

print("=== Bulla Bu2025_MLP (GW170817-like) ===")
print(f"Time grid: {len(times_bulla)} points, {times_bulla[0]:.2f} to {times_bulla[-1]:.2f} days")
for filt in FILTERS:
    m = np.asarray(mag_bulla[filt])
    peak_idx = np.argmin(m)
    print(f"  {filt}: peak = {m[peak_idx]:.2f} mag at t = {times_bulla[peak_idx]:.2f} d, "
          f"range [{m.min():.1f}, {m[m < 90].max():.1f}]")

# === Metzger model (lightcurve-fitting) ===
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/lightcurve-fitting")

# The lightcurve-fitting crate is Rust, but we can compare by reading the model output
# from our Rust code. For now, let's use the Python interface if available,
# or compute manually.
#
# Metzger model: params = [log10(mej), log10(vej), log10(kappa), t0_offset]
# With mej=0.01, vej=0.2, kappa=1.0 (default for our population)
# Peak abs mag = -16.0 (set by population generator)

# Let's compute what our parametric model produces:
# The Metzger model gives normalized flux (peak=1), then we add peak_abs_mag.
# For comparison, let's just show the Bulla absolute mags.

print("\n=== Comparison at key epochs ===")
print(f"{'t (days)':>10} | {'Bulla g':>10} {'Bulla r':>10} {'Bulla i':>10}")
print("-" * 55)

# Pick specific times
check_times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0]
for t in check_times:
    idx = np.argmin(np.abs(times_bulla - t))
    g = np.asarray(mag_bulla["ztf::g"])[idx]
    r = np.asarray(mag_bulla["ztf::r"])[idx]
    i = np.asarray(mag_bulla["ztf::i"])[idx]
    print(f"{times_bulla[idx]:10.2f} | {g:10.2f} {r:10.2f} {i:10.2f}")

# Now vary viewing angle
print("\n=== Bulla: Effect of viewing angle ===")
for inc in [0.0, 0.3, 0.6, 0.9, 1.2, 1.57]:
    params = dict(bulla_params)
    params["inclination_EM"] = inc
    _, mag = bulla.predict(params)
    g = np.asarray(mag["ztf::g"])
    r = np.asarray(mag["ztf::r"])
    peak_g = g.min()
    peak_r = r.min()
    t_peak_g = times_bulla[np.argmin(g)]
    t_peak_r = times_bulla[np.argmin(r)]
    print(f"  inc={np.degrees(inc):5.1f} deg: peak g={peak_g:+.2f} (t={t_peak_g:.1f}d), "
          f"peak r={peak_r:+.2f} (t={t_peak_r:.1f}d)")

# Compare fade rates
print("\n=== Bulla: Fade rates (mag/day) in g and r ===")
params = dict(bulla_params)
params["inclination_EM"] = 0.5
_, mag = bulla.predict(params)
g = np.asarray(mag["ztf::g"])
r = np.asarray(mag["ztf::r"])

# Find peak, then measure fade over next few days
peak_g_idx = np.argmin(g)
peak_r_idx = np.argmin(r)

for band_name, m, peak_idx in [("g", g, peak_g_idx), ("r", r, peak_r_idx)]:
    t_peak = times_bulla[peak_idx]
    # Measure fade at 1, 2, 3 days post-peak
    for dt in [1.0, 2.0, 3.0, 5.0]:
        idx2 = np.argmin(np.abs(times_bulla - (t_peak + dt)))
        if idx2 > peak_idx and m[idx2] < 90:
            fade = (m[idx2] - m[peak_idx]) / (times_bulla[idx2] - t_peak)
            print(f"  {band_name}-band: Δt={dt:.0f}d, fade = {fade:.3f} mag/day")

# Distribution of peak absolute mags for different parameter draws
print("\n=== Bulla: Peak absolute mag distribution (random params) ===")
rng = np.random.default_rng(42)
peak_mags_g = []
peak_mags_r = []
for _ in range(50):
    params = {
        "inclination_EM": rng.uniform(0, np.pi/2),
        "log10_mej_dyn": rng.uniform(-3.0, -1.3),
        "v_ej_dyn": rng.uniform(0.12, 0.28),
        "Ye_dyn": rng.uniform(0.15, 0.35),
        "log10_mej_wind": rng.uniform(-2.0, -0.886),
        "v_ej_wind": rng.uniform(0.05, 0.15),
        "Ye_wind": rng.uniform(0.2, 0.4),
        "luminosity_distance": 1e-5,
        "redshift": 0.0,
    }
    _, mag = bulla.predict(params)
    g = np.asarray(mag["ztf::g"])
    r = np.asarray(mag["ztf::r"])
    peak_mags_g.append(g.min())
    peak_mags_r.append(r.min())

peak_mags_g = np.array(peak_mags_g)
peak_mags_r = np.array(peak_mags_r)
print(f"  g-band peak: mean={peak_mags_g.mean():.2f}, std={peak_mags_g.std():.2f}, "
      f"range=[{peak_mags_g.min():.1f}, {peak_mags_g.max():.1f}]")
print(f"  r-band peak: mean={peak_mags_r.mean():.2f}, std={peak_mags_r.std():.2f}, "
      f"range=[{peak_mags_r.min():.1f}, {peak_mags_r.max():.1f}]")
print(f"\n  (Our Metzger model uses fixed M_peak = -16.0, bolometric)")
