#!/usr/bin/env python
"""Generate a synthetic Argus Array pointing database CSV.

Produces a CSV with columns (expMJD, _ra, _dec, filter, fiveSigmaDepth)
matching the Freeburn et al. Argus Array specifications:

  - Northern sky (dec > -20 deg), 8000 deg^2 instantaneous FoV
  - Baseline cadence: 60s exposures in dark/grey time, 1s in bright time
  - Two filters: g and r, with 2/3 g-band and 1/3 r-band telescopes
  - Each 15-min tracking window: 30 min g then 15 min r (cycling)
  - Stacked depths: 15min g~22.1, 1hr g~22.9, 1night g~23.9, 1week g~24.7
  - Single-exposure depths: dark/grey g~20.5 r~20.2, bright g~17.1 r~16.9

The script simulates observations at a single sky position and saves as CSV
compatible with redback's SimulateOpticalTransient format.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skewnorm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Argus Array pointing database CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/fred/oz480/mcoughli/simulations/argus/argus_synthetic_pointings.csv",
        help="Output CSV path",
    )
    parser.add_argument("--ra", type=float, default=89.84, help="RA in degrees")
    parser.add_argument("--dec", type=float, default=34.27, help="Dec in degrees")
    parser.add_argument(
        "--mjd-start", type=float, default=51544.0, help="Start MJD"
    )
    parser.add_argument(
        "--mjd-end", type=float, default=53370.0, help="End MJD"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# Depth parameters (single-exposure 5-sigma depths)
# Dark/grey time
DEPTH_DARK_G = 20.5
DEPTH_DARK_R = 20.2
# Bright time
DEPTH_BRIGHT_G = 17.1
DEPTH_BRIGHT_R = 16.9

# Depth scatter (mag)
DEPTH_SCATTER = 0.3

# Skewnorm shape parameter (slight negative skew: more scatter toward shallower)
SKEW_ALPHA = -2.0

# Exposure times (seconds)
EXP_DARK = 60.0
EXP_BRIGHT = 1.0

# Observing night parameters
NIGHT_HOURS = 8.0  # hours of potential observing per night
DARK_HOURS = 4.0   # hours of dark time
GREY_HOURS = 2.0   # hours of grey time
BRIGHT_HOURS = 2.0 # hours of bright time

# Weather loss fraction
WEATHER_LOSS = 0.20

# Fraction of each night the target is visible (Argus pointing cycle)
VISIBILITY_FRACTION = 0.40

# Tracking window duration (minutes)
TRACKING_WINDOW_MIN = 15.0

# Filter split within a 45-min cycle: 30 min g, 15 min r
G_WINDOW_MIN = 30.0
R_WINDOW_MIN = 15.0


def generate_night_observations(
    night_mjd: float,
    ra: float,
    dec: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate all observations for a single night.

    A night is split into dark (~4h), grey (~2h), and bright (~2h) segments.
    Within each segment, the target is visible for VISIBILITY_FRACTION of the
    time. Observations are grouped into 15-min tracking windows, cycling
    through 30 min g then 15 min r (a 45-min super-cycle).

    Parameters
    ----------
    night_mjd : float
        MJD at the start of the night (sunset).
    ra, dec : float
        Sky position in degrees.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    list of dict
        Each dict has keys: expMJD, _ra, _dec, filter, fiveSigmaDepth.
    """
    rows = []

    # Total observable time for this target tonight (hours)
    total_obs_hours = NIGHT_HOURS * VISIBILITY_FRACTION

    # Distribute across dark/grey/bright proportionally
    dark_frac = DARK_HOURS / NIGHT_HOURS
    grey_frac = GREY_HOURS / NIGHT_HOURS
    bright_frac = BRIGHT_HOURS / NIGHT_HOURS

    dark_obs_hours = total_obs_hours * dark_frac
    grey_obs_hours = total_obs_hours * grey_frac
    bright_obs_hours = total_obs_hours * bright_frac

    # Convert to minutes
    dark_obs_min = dark_obs_hours * 60.0
    grey_obs_min = grey_obs_hours * 60.0
    bright_obs_min = bright_obs_hours * 60.0

    # Time offset from start of night (in days)
    # Dark time starts at sunset, grey follows, then bright
    t_offset_dark_start = 0.0
    t_offset_grey_start = DARK_HOURS / 24.0
    t_offset_bright_start = (DARK_HOURS + GREY_HOURS) / 24.0

    # Generate dark/grey observations (same cadence: 60s exposures)
    darkgrey_min = dark_obs_min + grey_obs_min
    rows.extend(
        _generate_segment_obs(
            night_mjd + t_offset_dark_start,
            darkgrey_min,
            EXP_DARK,
            DEPTH_DARK_G,
            DEPTH_DARK_R,
            ra,
            dec,
            rng,
        )
    )

    # Generate bright-time observations (1s exposures)
    rows.extend(
        _generate_segment_obs(
            night_mjd + t_offset_bright_start,
            bright_obs_min,
            EXP_BRIGHT,
            DEPTH_BRIGHT_G,
            DEPTH_BRIGHT_R,
            ra,
            dec,
            rng,
        )
    )

    return rows


def _generate_segment_obs(
    mjd_start: float,
    total_minutes: float,
    exp_time_sec: float,
    depth_g: float,
    depth_r: float,
    ra: float,
    dec: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate observations for a time segment using the g/r cycling pattern.

    Within each 45-min super-cycle (30 min g + 15 min r), observations are
    placed at intervals of exp_time_sec. Depths are drawn from a skew-normal
    distribution centered on the nominal depth.
    """
    rows = []
    cycle_min = G_WINDOW_MIN + R_WINDOW_MIN  # 45 min

    elapsed = 0.0  # minutes elapsed in this segment
    exp_time_min = exp_time_sec / 60.0

    while elapsed < total_minutes:
        # Position within the 45-min super-cycle
        cycle_pos = elapsed % cycle_min

        if cycle_pos < G_WINDOW_MIN:
            band = "g"
            nominal_depth = depth_g
        else:
            band = "r"
            nominal_depth = depth_r

        # Draw depth from skew-normal (negative skew = tail toward shallower)
        depth = skewnorm.rvs(SKEW_ALPHA, loc=nominal_depth, scale=DEPTH_SCATTER, random_state=rng)

        mjd = mjd_start + elapsed / (24.0 * 60.0)

        rows.append(
            {
                "expMJD": mjd,
                "_ra": ra,
                "_dec": dec,
                "filter": band,
                "fiveSigmaDepth": round(float(depth), 3),
            }
        )

        elapsed += exp_time_min

    return rows


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    mjd_start = args.mjd_start
    mjd_end = args.mjd_end
    ra = args.ra
    dec = args.dec

    n_nights = int(mjd_end - mjd_start)
    print(f"Generating Argus pointings for {n_nights} nights")
    print(f"  MJD range: {mjd_start} - {mjd_end}")
    print(f"  Position: RA={ra}, Dec={dec}")

    # Determine which nights are clear (weather loss)
    clear_mask = rng.random(n_nights) > WEATHER_LOSS
    clear_nights = np.where(clear_mask)[0]
    print(f"  Clear nights: {len(clear_nights)} / {n_nights} ({100*len(clear_nights)/n_nights:.1f}%)")

    all_rows = []
    for i, night_idx in enumerate(clear_nights):
        night_mjd = mjd_start + night_idx + 0.5  # sunset ~ 0.5 day offset
        rows = generate_night_observations(night_mjd, ra, dec, rng)
        all_rows.extend(rows)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(clear_nights)} nights ({len(all_rows)} exposures so far)")

    df = pd.DataFrame(all_rows)
    df.sort_values("expMJD", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Summary statistics
    n_g = (df["filter"] == "g").sum()
    n_r = (df["filter"] == "r").sum()
    print(f"\nGenerated {len(df)} total exposures")
    print(f"  g-band: {n_g} ({100*n_g/len(df):.1f}%)")
    print(f"  r-band: {n_r} ({100*n_r/len(df):.1f}%)")
    print(f"  Median depth (g, dark/grey): {df.loc[df['filter']=='g']['fiveSigmaDepth'].median():.2f}")
    print(f"  Median depth (r, dark/grey): {df.loc[df['filter']=='r']['fiveSigmaDepth'].median():.2f}")

    # Verify stacking depths (informational)
    # 15-min stack of 60s exposures = 15 exposures, gain = 2.5*log10(sqrt(15)) ~ 1.47 mag
    # Nominal: 20.5 + 1.47 = 21.97 ~ 22.0 (paper says 22.1)
    n_15min = int(15 * 60 / EXP_DARK)
    stack_gain_15 = 2.5 * np.log10(np.sqrt(n_15min))
    print(f"\n  Stacking check (g-band dark/grey):")
    print(f"    15-min stack ({n_15min} exp): {DEPTH_DARK_G:.1f} + {stack_gain_15:.2f} = {DEPTH_DARK_G + stack_gain_15:.1f} (paper: 22.1)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
