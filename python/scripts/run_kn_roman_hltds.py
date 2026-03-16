#!/usr/bin/env python
"""Roman HLTDS serendipitous kilonova rate predictions.

Computes expected serendipitous kilonova detections in the Roman High
Latitude Time Domain Survey (HLTDS) using:
  1. Rate rescaling of Andreoni et al. (2024, arXiv:2307.09511) predictions
  2. Semi-analytical volume × rate calculation

Uses current LVK median merger rates:
  BNS:  45^{+53}_{-24} Gpc^-3 yr^-1
  NSBH: 25^{+29}_{-14} Gpc^-3 yr^-1
  (https://emfollow.docs.ligo.org/userguide/capabilities.html)

The Andreoni et al. result of 5-40 KNe/5yr used R_BNS = 210 Gpc^-3/yr
from GWTC-3 (Petrov+2022). The combined BNS+NSBH rate has decreased by
3.1x (70 vs 218.6 Gpc^-3/yr), reducing the prediction proportionally.
"""
import sys
sys.path.insert(0, "/fred/oz480/mcoughli/simulations/survey-sim/python")

import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

# =========================================================================
# LVK rates (latest: emfollow userguide)
# =========================================================================
BNS_RATE = 45.0    # Gpc^-3 yr^-1 (median)
NSBH_RATE = 25.0   # Gpc^-3 yr^-1 (median)
BNS_RATE_LO = 21.0   # 90% lower
BNS_RATE_HI = 98.0   # 90% upper
NSBH_RATE_LO = 11.0
NSBH_RATE_HI = 54.0

# Old rates from GWTC-3 / Petrov+2022 (used in Andreoni+2024)
OLD_BNS_RATE = 210.0
OLD_NSBH_RATE = 8.6

# =========================================================================
# Part 1: Rate rescaling of Andreoni+2024
# =========================================================================
print("=" * 78)
print("PART 1: Rescaling Andreoni et al. (2024) predictions")
print("=" * 78)

old_combined = OLD_BNS_RATE + OLD_NSBH_RATE  # 218.6
new_combined = BNS_RATE + NSBH_RATE  # 70
rate_ratio = new_combined / old_combined
bns_ratio = BNS_RATE / OLD_BNS_RATE

print(f"\nRate comparison (Gpc^-3 yr^-1):")
print(f"  {'Source':<30s} {'BNS':>8s} {'NSBH':>8s} {'Combined':>10s}")
print(f"  {'GWTC-3 (Andreoni+2024)':<30s} {OLD_BNS_RATE:>8.1f} {OLD_NSBH_RATE:>8.1f} {old_combined:>10.1f}")
print(f"  {'LVK current (median)':<30s} {BNS_RATE:>8.1f} {NSBH_RATE:>8.1f} {new_combined:>10.1f}")
print(f"  {'LVK current (90% lower)':<30s} {BNS_RATE_LO:>8.1f} {NSBH_RATE_LO:>8.1f} {BNS_RATE_LO+NSBH_RATE_LO:>10.1f}")
print(f"  {'LVK current (90% upper)':<30s} {BNS_RATE_HI:>8.1f} {NSBH_RATE_HI:>8.1f} {BNS_RATE_HI+NSBH_RATE_HI:>10.1f}")
print(f"  {'Ratio (new/old)':<30s} {bns_ratio:>8.2f} {NSBH_RATE/OLD_NSBH_RATE:>8.2f} {rate_ratio:>10.2f}")

# Andreoni+2024 predictions (5yr, serendipitous):
# Table 8/11: range across kilonova models and detection strategies
# Conservative (N=2 det, M=2 filters): 3.3 for GW170817_pol, up to ~7 for bright models
# Optimistic (N=3 det, M=2 filters): similar range
# The "5-40" range spans models, strategies, and rate uncertainties.
print(f"\nAndreoni+2024 predictions (5yr, wide tier 18 deg^2):")
print(f"  Original (R_BNS={OLD_BNS_RATE:.0f}): 5-40 KNe")

# Rescale: Andreoni's predictions were dominated by BNS (NSBH rate was tiny)
# The prediction scales linearly with rate.
# But the NSBH rate has increased 2.9x, adding ~1.0 NSBH KNe where before
# there were ~0.2. So we rescale BNS and NSBH contributions separately.
# Assume Andreoni's 5-40 was ~95% BNS, ~5% NSBH
andreoni_lo = 5.0
andreoni_hi = 40.0
bns_frac = OLD_BNS_RATE / old_combined  # 0.96

rescaled_lo = andreoni_lo * (bns_frac * bns_ratio + (1 - bns_frac) * NSBH_RATE / OLD_NSBH_RATE)
rescaled_hi = andreoni_hi * (bns_frac * bns_ratio + (1 - bns_frac) * NSBH_RATE / OLD_NSBH_RATE)

print(f"  Rescaled (R_BNS={BNS_RATE:.0f}, R_NSBH={NSBH_RATE:.0f}): "
      f"{rescaled_lo:.1f}-{rescaled_hi:.1f} KNe")
print(f"    (BNS contribution: ×{bns_ratio:.2f}, NSBH: ×{NSBH_RATE/OLD_NSBH_RATE:.1f})")

# Rate uncertainty range
lo_ratio = (BNS_RATE_LO + NSBH_RATE_LO) / old_combined
hi_ratio = (BNS_RATE_HI + NSBH_RATE_HI) / old_combined
rescaled_lo_lo = andreoni_lo * lo_ratio
rescaled_hi_hi = andreoni_hi * hi_ratio

print(f"\n  With rate uncertainties:")
print(f"    Median rates: {rescaled_lo:.1f}-{rescaled_hi:.1f} KNe/5yr")
print(f"    90% lower:    {rescaled_lo_lo:.1f}-{andreoni_hi*lo_ratio:.1f} KNe/5yr")
print(f"    90% upper:    {andreoni_lo*hi_ratio:.1f}-{rescaled_hi_hi:.1f} KNe/5yr")
print(f"    Full range:   {rescaled_lo_lo:.1f}-{rescaled_hi_hi:.1f} KNe/5yr")

# =========================================================================
# Part 2: Semi-analytical volume calculation
# =========================================================================
print(f"\n{'='*78}")
print("PART 2: Semi-analytical estimate")
print("=" * 78)

# Kilonova peak NIR magnitudes from possis models (Andreoni+2024 Table 2):
# GW170817_pol (face-on, 26°): M_F158 ~ -16.5
# GW170817_eq (edge-on, 90°): M_F158 ~ -15.5
# low-all (pessimistic):      M_F158 ~ -14.5
# high-wind (optimistic):     M_F158 ~ -17.5

print(f"\nKilonova peak NIR magnitudes (from possis models):")
models = [
    ("GW170817_pol (face-on)", -16.5),
    ("GW170817_eq (edge-on)", -15.5),
    ("Low-all (pessimistic)", -14.5),
    ("High-wind (optimistic)", -17.5),
]

# Roman F158 depths
depths = [
    ("Wide (1hr, F158)", 25.3),
    ("Deep (1hr, F158)", 26.0),
]

print(f"\n{'Model':<30s} {'M_F158':>7s} ", end="")
for dname, _ in depths:
    print(f" {'z_max('+dname+')':>25s}", end="")
print()
print("-" * 95)

z_grid = np.linspace(0.01, 3.0, 10000)
dl_grid = np.array([cosmo.luminosity_distance(z).to(u.Mpc).value for z in z_grid])
dm_grid = 5 * np.log10(dl_grid) + 25

survey_configs = []

for mname, M_peak in models:
    print(f"{mname:<30s} {M_peak:>7.1f} ", end="")
    for dname, depth in depths:
        m_app = M_peak + dm_grid
        idx = np.searchsorted(m_app, depth)
        z_max_det = z_grid[idx-1] if idx > 0 else 0.0
        print(f" {z_max_det:>25.2f}", end="")
    print()

# Full calculation for each model × tier
print(f"\n{'='*90}")
print("Expected serendipitous KN detections (combined BNS+NSBH)")
print(f"{'='*90}")
print(f"  Rates: BNS={BNS_RATE} [{BNS_RATE_LO},{BNS_RATE_HI}], "
      f"NSBH={NSBH_RATE} [{NSBH_RATE_LO},{NSBH_RATE_HI}] Gpc^-3 yr^-1")

# Cadence efficiency: fraction of KNe that produce >=2 detections
# with 5-day cadence. KN visible for ~T_vis days.
# P(>=2 epochs) = 1 - P(0) - P(1)
# P(0) = (1 - T_vis/cadence) for random phase, approximately
# For T_vis ~ 7-14 days (NIR) and cadence = 5 days: ~2-3 epochs
# Conservative eps_cadence ~ 0.6
eps_cadence = 0.6

# Viewing angle factor: isotropic angles. Face-on KNe are ~2 mag
# brighter than edge-on. Roughly half of viewing angles give
# detectable KNe at a given distance.
# From possis: GW170817_pol is 1 mag brighter than GW170817_eq
eps_angle = 0.5

# Duration: 2 years of HLTDS observations
duration_yr = 2.0

print(f"  eps_cadence={eps_cadence}, eps_angle={eps_angle}, T_survey={duration_yr} yr")
print()

tiers = [
    ("Wide (18 deg^2)", 18.0, 25.3),
    ("Deep (6.5 deg^2)", 6.5, 26.0),
    ("Wide+Deep (24.5 deg^2)", 24.5, 25.3),  # combined footprint
]

print(f"{'Model':<25s} {'Tier':<22s} {'z_max':>5s} {'V_eff':>10s} "
      f"{'N/2yr':>8s} {'N/5yr':>8s} {'90% range':>18s}")
print("-" * 100)

for mname, M_peak in models:
    for tname, area, depth in tiers:
        # Find z_max
        m_app = M_peak + dm_grid
        idx = np.searchsorted(m_app, depth)
        z_max_det = z_grid[idx-1] if idx > 0 else 0.0

        V_max_val = cosmo.comoving_volume(z_max_det).to(u.Gpc**3).value
        f_sky = area / 41253.0
        V_eff = V_max_val * f_sky

        # N = R × V_eff × T × eps_cadence × eps_angle / (1+z_med)
        # The (1+z) time dilation is already in the comoving volume
        n_2yr = new_combined * V_eff * duration_yr * eps_cadence * eps_angle
        n_5yr = n_2yr * 5.0 / duration_yr
        n_lo = (BNS_RATE_LO + NSBH_RATE_LO) * V_eff * 5.0 * eps_cadence * eps_angle
        n_hi = (BNS_RATE_HI + NSBH_RATE_HI) * V_eff * 5.0 * eps_cadence * eps_angle

        print(f"{mname:<25s} {tname:<22s} {z_max_det:>5.2f} {V_eff:>10.5f} "
              f"{n_2yr:>8.2f} {n_5yr:>8.2f} [{n_lo:.1f}, {n_hi:.1f}]")

# =========================================================================
# Summary table for proposal
# =========================================================================
print(f"\n{'='*78}")
print("SUMMARY FOR PROPOSAL")
print(f"{'='*78}")

# Use GW170817_pol as the fiducial model
M_fid = -16.5
for tname, area, depth in tiers:
    m_app = M_fid + dm_grid
    idx = np.searchsorted(m_app, depth)
    z_max_det = z_grid[idx-1] if idx > 0 else 0.0
    V_max_val = cosmo.comoving_volume(z_max_det).to(u.Gpc**3).value
    f_sky = area / 41253.0
    V_eff = V_max_val * f_sky
    n_5yr_med = new_combined * V_eff * 5.0 * eps_cadence * eps_angle
    n_5yr_lo = (BNS_RATE_LO + NSBH_RATE_LO) * V_eff * 5.0 * eps_cadence * eps_angle
    n_5yr_hi = (BNS_RATE_HI + NSBH_RATE_HI) * V_eff * 5.0 * eps_cadence * eps_angle
    print(f"\n  {tname}:")
    print(f"    Fiducial model: GW170817-like (M_F158 = {M_fid})")
    print(f"    Depth: {depth} (F158, 1hr)")
    print(f"    Max detectable z: {z_max_det:.2f}")
    print(f"    Effective volume: {V_eff:.5f} Gpc^3")
    print(f"    Expected KNe (5yr): {n_5yr_med:.1f} [{n_5yr_lo:.1f}, {n_5yr_hi:.1f}]")

print(f"\nComparison:")
print(f"  Andreoni+2024 (R_BNS=210): 5-40 KNe/5yr")
print(f"  This work (R_BNS={BNS_RATE:.0f}, R_NSBH={NSBH_RATE:.0f}): "
      f"{rescaled_lo:.1f}-{rescaled_hi:.1f} KNe/5yr (rescaled)")

print(f"\nNote: These predictions use the Metzger BB single-zone model.")
print(f"Andreoni+2024 used the possis 3D radiative transfer code, which")
print(f"produces brighter and longer-lived NIR emission due to explicit")
print(f"lanthanide opacity modeling. The possis predictions should be")
print(f"considered more reliable for NIR bands.")
print(f"\nFor the proposal, we recommend citing:")
print(f"  'Using updated LVK merger rates (BNS = 45, NSBH = 25 Gpc^-3 yr^-1),")
print(f"   we rescale the Andreoni et al. (2024) prediction of 5-40 serendipitous")
print(f"   kilonova detections in the HLTDS to {rescaled_lo:.0f}-{rescaled_hi:.0f} over 5 years.")
print(f"   The decrease is driven by the BNS rate revision from 210 to 45 Gpc^-3 yr^-1,")
print(f"   partially offset by the NSBH rate increase from 8.6 to 25 Gpc^-3 yr^-1.'")
