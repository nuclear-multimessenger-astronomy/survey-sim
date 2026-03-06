#!/usr/bin/env python3
"""Tune Metzger 1-zone blackbody kilonova model against AT2017gfo g/r/i data."""

import numpy as np
from datetime import datetime
import itertools

# Physical constants (CGS)
MSUN = 1.989e33
C_CGS = 2.998e10
SECS_PER_DAY = 86400.0
H_PLANCK = 6.62607015e-27  # erg·s
K_BOLTZ = 1.380649e-16     # erg/K
SIGMA_SB = 5.670374419e-5  # erg/s/cm²/K⁴

# AT2017gfo distance
D_L_CM = 40.7 * 3.086e24   # 40.7 Mpc in cm
Z = 0.0098

# Band effective frequencies (Hz) - from wavelengths
# ps1::g = 4866 Å, ps1::r = 6215 Å, ps1::i = 7545 Å
BAND_FREQS = {
    "ps1::g": C_CGS / (4866e-8),
    "ps1::r": C_CGS / (6215e-8),
    "ps1::i": C_CGS / (7545e-8),
}

# ZTF effective frequencies for reference
ZTF_FREQS = {
    "ztf_g": C_CGS / (4770e-8),
    "ztf_r": C_CGS / (6231e-8),
    "ztf_i": C_CGS / (7625e-8),
}


def parse_at2017gfo(path, bands=("ps1::g", "ps1::r", "ps1::i"), t_max_days=10.0):
    """Parse AT2017gfo data, return dict of {band: (t_days, mag, err)}."""
    # GW170817 merger time: 2017-08-17 12:41:04 UTC
    t_merger = datetime(2017, 8, 17, 12, 41, 4)

    data = {b: ([], [], []) for b in bands}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            ts, band, mag, err = parts[0], parts[1], float(parts[2]), float(parts[3])
            if band not in bands:
                continue
            if not np.isfinite(err) or err <= 0:
                continue
            # Parse timestamp
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")
            t_days = (dt - t_merger).total_seconds() / 86400.0
            if t_days <= 0 or t_days > t_max_days:
                continue
            data[band][0].append(t_days)
            data[band][1].append(mag)
            data[band][2].append(err)

    return {b: (np.array(t), np.array(m), np.array(e))
            for b, (t, m, e) in data.items() if len(t) > 0}


def metzger_bb_mags(log10_mej, log10_vej, log10_kappa, obs_times, band_freqs, d_l_cm):
    """
    Metzger 1-zone kilonova with blackbody emission.
    Returns dict of {band_name: apparent_mags}.
    """
    m_ej = 10**log10_mej * MSUN
    v_ej = 10**log10_vej * C_CGS
    kappa_r = 10**log10_kappa

    phase_max = max(obs_times) * 1.05
    if phase_max < 0.02:
        return {b: np.full(len(obs_times), 99.0) for b in band_freqs}

    # Log-spaced time grid
    n_grid = 300
    grid_t = np.logspace(np.log10(0.01), np.log10(phase_max), n_grid)

    ye = 0.1
    xn0 = 1.0 - 2.0 * ye

    scale = 1e40
    e0 = 0.5 * m_ej * v_ej**2
    e_th = e0 / scale
    e_kin = e0 / scale
    v = v_ej
    r = grid_t[0] * SECS_PER_DAY * v

    grid_lrad = np.zeros(n_grid)
    grid_r = np.zeros(n_grid)

    for i in range(n_grid):
        t_day = grid_t[i]
        t_sec = t_day * SECS_PER_DAY

        eth_factor = 0.34 * t_day**0.74
        if eth_factor > 1e-10:
            eth = 0.36 * (np.exp(-0.56 * t_day) + np.log(1.0 + eth_factor) / eth_factor)
        else:
            eth = 0.36 * (np.exp(-0.56 * t_day) + 1.0)

        xn = xn0 * np.exp(-t_sec / 900.0)
        eps_neutron = 3.2e14 * xn
        time_term = max(0.5 - np.arctan((t_sec - 1.3) / 0.11) / np.pi, 1e-30)
        eps_rp = 2e18 * eth * time_term**1.3
        l_heat = m_ej * (eps_neutron + eps_rp) / scale

        xr = 1.0 - xn0
        xn_decayed = xn0 - xn
        kappa_eff = 0.4 * xn_decayed + kappa_r * xr

        t_diff = 3.0 * kappa_eff * m_ej / (4.0 * np.pi * C_CGS * v * t_sec) + r / C_CGS

        l_rad = e_th / t_diff if (e_th > 0 and t_diff > 0) else 0.0
        grid_lrad[i] = l_rad
        grid_r[i] = r

        l_pdv = e_th * v / r if r > 0 else 0.0

        if i < n_grid - 1:
            dt_sec = (grid_t[i + 1] - grid_t[i]) * SECS_PER_DAY
            e_th += (l_heat - l_pdv - l_rad) * dt_sec
            e_th = max(e_th, 0.0)
            e_kin += l_pdv * dt_sec
            v = min(np.sqrt(2.0 * e_kin * scale / m_ej), C_CGS)
            r += v * dt_sec

    # Effective temperature on grid
    l_real = grid_lrad * scale  # erg/s
    grid_temp = np.zeros(n_grid)
    for i in range(n_grid):
        if l_real[i] > 0 and grid_r[i] > 0:
            grid_temp[i] = (l_real[i] / (4 * np.pi * grid_r[i]**2 * SIGMA_SB))**0.25

    # Interpolate and compute BB magnitudes
    dist_sq = d_l_cm**2
    result = {}
    for band_name, nu in band_freqs.items():
        mags = []
        for t in obs_times:
            if t <= 0:
                mags.append(99.0)
                continue
            # Interpolate T_eff and R
            temp = np.interp(t, grid_t, grid_temp)
            r_ph = np.interp(t, grid_t, grid_r)

            if temp <= 0 or r_ph <= 0:
                mags.append(99.0)
                continue

            # Planck BB flux density
            x = H_PLANCK * nu / (K_BOLTZ * temp)
            x = min(x, 700.0)
            bb = 2.0 * H_PLANCK * nu**3 / C_CGS**2
            f_nu = bb / np.expm1(x) * (r_ph**2 / dist_sq)

            if f_nu <= 0 or not np.isfinite(f_nu):
                mags.append(99.0)
                continue

            mags.append(-2.5 * np.log10(f_nu) - 48.6)
        result[band_name] = np.array(mags)
    return result


def chi2_fit(obs_data, log10_mej, log10_vej, log10_kappa, t_max=5.0):
    """Compute chi² between model and observations for t < t_max days."""
    # Gather all unique observation times
    all_times = set()
    for band, (t, m, e) in obs_data.items():
        mask = t <= t_max
        for ti in t[mask]:
            all_times.add(ti)
    all_times = sorted(all_times)
    if not all_times:
        return 1e30, 0

    model = metzger_bb_mags(log10_mej, log10_vej, log10_kappa,
                            all_times, BAND_FREQS, D_L_CM)

    chi2 = 0.0
    n_pts = 0
    for band, (t_obs, m_obs, e_obs) in obs_data.items():
        mask = t_obs <= t_max
        for ti, mi, ei in zip(t_obs[mask], m_obs[mask], e_obs[mask]):
            idx = all_times.index(ti)
            m_model = model[band][idx]
            if m_model >= 90:
                chi2 += 100.0
            else:
                chi2 += ((mi - m_model) / ei)**2
            n_pts += 1

    return chi2, n_pts


def main():
    data_path = "/home/mcoughli/nmma/example_files/lightcurves/AT2017gfo_corrected.dat"
    obs_data = parse_at2017gfo(data_path, t_max_days=10.0)

    print("AT2017gfo data loaded:")
    for band, (t, m, e) in obs_data.items():
        print(f"  {band}: {len(t)} points, t=[{t.min():.2f}, {t.max():.2f}] days")

    # ---- Coarse grid scan ----
    t_max_fit = 3.0  # Only fit first 3 days (1-zone most reliable)
    print(f"\n=== Coarse grid scan (t < {t_max_fit}d) ===")
    log10_mej_vals = np.arange(-3.0, -0.8, 0.1)    # 0.001 to 0.16 Msun
    log10_vej_vals = np.arange(-1.0, -0.1, 0.1)    # 0.1c to 0.79c
    log10_kappa_vals = np.arange(-1.0, 2.6, 0.2)   # 0.1 to 398 cm²/g

    best_chi2 = 1e30
    best_params = None
    results = []

    total = len(log10_mej_vals) * len(log10_vej_vals) * len(log10_kappa_vals)
    count = 0
    for lm, lv, lk in itertools.product(log10_mej_vals, log10_vej_vals, log10_kappa_vals):
        count += 1
        c2, npts = chi2_fit(obs_data, lm, lv, lk, t_max=t_max_fit)
        rchi2 = c2 / max(npts - 3, 1)
        results.append((rchi2, lm, lv, lk, npts))
        if c2 < best_chi2:
            best_chi2 = c2
            best_params = (lm, lv, lk, npts)
            if count % 200 == 0:
                print(f"  [{count}/{total}] best so far: mej={lm:.1f} vej={lv:.1f} kappa={lk:.1f} "
                      f"chi2={c2:.1f} rchi2={rchi2:.2f}")

    results.sort()
    lm, lv, lk, npts = best_params
    print(f"\nCoarse best: log10_mej={lm:.1f}, log10_vej={lv:.1f}, log10_kappa={lk:.1f}")
    print(f"  chi2={best_chi2:.1f}, rchi2={best_chi2/max(npts-3,1):.2f} ({npts} pts)")
    print(f"  mej={10**lm:.4f} Msun, vej={10**lv:.3f}c, kappa={10**lk:.1f} cm²/g")

    print("\nTop 10:")
    for rchi2, lm, lv, lk, npts in results[:10]:
        print(f"  mej={lm:.1f} vej={lv:.1f} kappa={lk:.1f} -> rchi2={rchi2:.2f}")

    # ---- Fine grid scan around coarse best ----
    print("\n=== Fine grid scan ===")
    bm, bv, bk = best_params[:3]
    log10_mej_fine = np.arange(bm - 0.3, bm + 0.35, 0.05)
    log10_vej_fine = np.arange(bv - 0.2, bv + 0.25, 0.05)
    log10_kappa_fine = np.arange(max(bk - 0.5, -1.0), bk + 1.1, 0.1)

    best_chi2_fine = 1e30
    best_fine = None
    fine_results = []

    for lm, lv, lk in itertools.product(log10_mej_fine, log10_vej_fine, log10_kappa_fine):
        c2, npts = chi2_fit(obs_data, lm, lv, lk, t_max=t_max_fit)
        rchi2 = c2 / max(npts - 3, 1)
        fine_results.append((rchi2, lm, lv, lk, npts))
        if c2 < best_chi2_fine:
            best_chi2_fine = c2
            best_fine = (lm, lv, lk, npts)

    fine_results.sort()
    lm, lv, lk, npts = best_fine
    print(f"Fine best: log10_mej={lm:.2f}, log10_vej={lv:.2f}, log10_kappa={lk:.1f}")
    print(f"  chi2={best_chi2_fine:.1f}, rchi2={best_chi2_fine/max(npts-3,1):.2f} ({npts} pts)")
    print(f"  mej={10**lm:.5f} Msun, vej={10**lv:.4f}c, kappa={10**lk:.2f} cm²/g")

    print("\nTop 10 fine:")
    for rchi2, lm, lv, lk, npts in fine_results[:10]:
        print(f"  mej={lm:.2f} vej={lv:.2f} kappa={lk:.1f} -> rchi2={rchi2:.2f}")

    # ---- Print model vs data comparison for best fit ----
    lm, lv, lk, _ = best_fine
    print(f"\n=== Best-fit model vs AT2017gfo (t < {t_max_fit}d) ===")
    print(f"Parameters: log10_mej={lm:.2f}, log10_vej={lv:.2f}, log10_kappa={lk:.1f}")
    print(f"  mej={10**lm:.5f} Msun, vej={10**lv:.4f}c, kappa={10**lk:.2f} cm²/g\n")

    all_times_sorted = sorted(set(
        t for band in obs_data for t in obs_data[band][0] if t <= t_max_fit
    ))
    model = metzger_bb_mags(lm, lv, lk, all_times_sorted, BAND_FREQS, D_L_CM)

    for band in ["ps1::g", "ps1::r", "ps1::i"]:
        if band not in obs_data:
            continue
        t_obs, m_obs, e_obs = obs_data[band]
        mask = t_obs <= t_max_fit
        print(f"{band}:")
        print(f"  {'t(d)':>6s}  {'obs':>7s}  {'model':>7s}  {'resid':>7s}")
        for ti, mi, ei in zip(t_obs[mask], m_obs[mask], e_obs[mask]):
            idx = all_times_sorted.index(ti)
            mm = model[band][idx]
            resid = mi - mm
            print(f"  {ti:6.2f}  {mi:7.3f}  {mm:7.3f}  {resid:+7.3f}")
        print()

    # Also evaluate at ZTF frequencies
    print("=== ZTF band predictions (best-fit) ===")
    t_ztf = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    model_ztf = metzger_bb_mags(lm, lv, lk, t_ztf, ZTF_FREQS, D_L_CM)
    print(f"{'t(d)':>5s}  {'ztf_g':>7s}  {'ztf_r':>7s}  {'ztf_i':>7s}")
    for i, t in enumerate(t_ztf):
        g = model_ztf["ztf_g"][i]
        r = model_ztf["ztf_r"][i]
        ii = model_ztf["ztf_i"][i]
        print(f"{t:5.1f}  {g:7.2f}  {r:7.2f}  {ii:7.2f}")


if __name__ == "__main__":
    main()
