#!/usr/bin/env python3
"""Convert LIGO/Virgo observing scenario data files to HDF5.

Reads injections.dat, coincs.dat, and allsky.dat from each observing run
directory and merges them into a single HDF5 file per run.

Usage:
    python build_observing_scenario_hdf5.py [--runs-dir DIR] [--output-dir DIR] [--runs RUN1 RUN2 ...]
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

RUNS_DIR = Path("/fred/oz480/mcoughli/observing-scenarios/runs")
OUTPUT_DIR = Path("/fred/oz480/mcoughli/simulations/survey-sim/data/observing_scenarios")

# Runs to process
DEFAULT_RUNS = ["O4HL", "O4HLV", "O5a", "O5b", "O5c"]


def load_dat(path, skip_comments=True):
    """Load a whitespace-delimited .dat file, skipping comment lines."""
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skip_comments and line.startswith("#"):
                continue
            if header is None:
                header = line.split("\t") if "\t" in line else line.split()
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            rows.append(fields)
    return header, rows


def build_hdf5(run_name, run_dir, output_path):
    """Build HDF5 file from a single observing run directory."""
    inj_path = run_dir / "injections.dat"
    coinc_path = run_dir / "coincs.dat"
    allsky_path = run_dir / "allsky.dat"

    for p in [inj_path, coinc_path, allsky_path]:
        if not p.exists():
            print(f"  Skipping {run_name}: missing {p.name}")
            return

    # Load injections (truth parameters)
    inj_hdr, inj_rows = load_dat(inj_path)
    n_events = len(inj_rows)
    print(f"  {run_name}: {n_events} events")

    # Load coincs (detection info)
    coinc_hdr, coinc_rows = load_dat(coinc_path)

    # Load allsky (skymap statistics)
    allsky_hdr, allsky_rows = load_dat(allsky_path)

    # Index coincs and allsky by their IDs for merging
    # coincs: coinc_event_id, ifos, snr
    coinc_by_id = {}
    coinc_id_col = coinc_hdr.index("coinc_event_id")
    for row in coinc_rows:
        cid = int(row[coinc_id_col])
        coinc_by_id[cid] = row

    # allsky: coinc_event_id, simulation_id, ...
    allsky_by_simid = {}
    allsky_simid_col = allsky_hdr.index("simulation_id")
    for row in allsky_rows:
        sid = int(row[allsky_simid_col])
        allsky_by_simid[sid] = row

    # Build merged arrays
    inj_simid_col = inj_hdr.index("simulation_id")

    # Injection columns
    sim_ids = np.array([int(r[inj_simid_col]) for r in inj_rows], dtype=np.int64)
    ra = np.array([float(r[inj_hdr.index("longitude")]) for r in inj_rows])  # radians
    dec = np.array([float(r[inj_hdr.index("latitude")]) for r in inj_rows])  # radians
    distance = np.array([float(r[inj_hdr.index("distance")]) for r in inj_rows])  # Mpc
    inclination = np.array([float(r[inj_hdr.index("inclination")]) for r in inj_rows])  # rad
    mass1 = np.array([float(r[inj_hdr.index("mass1")]) for r in inj_rows])
    mass2 = np.array([float(r[inj_hdr.index("mass2")]) for r in inj_rows])
    spin1z = np.array([float(r[inj_hdr.index("spin1z")]) for r in inj_rows])
    spin2z = np.array([float(r[inj_hdr.index("spin2z")]) for r in inj_rows])

    # Convert ra/dec from radians to degrees
    ra_deg = np.degrees(ra) % 360.0
    dec_deg = np.degrees(dec)

    # Coinc columns (matched by index — coinc_event_id == row index)
    snr = np.full(n_events, np.nan)
    far = np.full(n_events, np.nan)
    ifos = np.full(n_events, "", dtype="U32")
    coinc_ids = np.full(n_events, -1, dtype=np.int64)

    for i in range(n_events):
        if i in coinc_by_id:
            row = coinc_by_id[i]
            coinc_ids[i] = int(row[coinc_hdr.index("coinc_event_id")])
            ifos_col = coinc_hdr.index("ifos")
            snr_col = coinc_hdr.index("snr")
            ifos[i] = row[ifos_col]
            snr[i] = float(row[snr_col])

    # Allsky columns (matched by simulation_id)
    area_90 = np.full(n_events, np.nan)
    area_50 = np.full(n_events, np.nan)
    area_20 = np.full(n_events, np.nan)
    distmean = np.full(n_events, np.nan)
    diststd = np.full(n_events, np.nan)
    searched_area = np.full(n_events, np.nan)

    for i, sid in enumerate(sim_ids):
        if sid in allsky_by_simid:
            row = allsky_by_simid[sid]
            try:
                area_90[i] = float(row[allsky_hdr.index("area(90)")])
                area_50[i] = float(row[allsky_hdr.index("area(50)")])
                area_20[i] = float(row[allsky_hdr.index("area(20)")])
                distmean[i] = float(row[allsky_hdr.index("distmean")])
                diststd[i] = float(row[allsky_hdr.index("diststd")])
                searched_area[i] = float(row[allsky_hdr.index("searched_area")])
                far[i] = float(row[allsky_hdr.index("far")])
            except (ValueError, IndexError):
                pass

    # Write HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["run"] = run_name
        f.attrs["n_events"] = n_events
        f.attrs["source"] = str(run_dir)

        # Truth parameters
        truth = f.create_group("truth")
        truth.create_dataset("simulation_id", data=sim_ids)
        truth.create_dataset("ra_deg", data=ra_deg)
        truth.create_dataset("dec_deg", data=dec_deg)
        truth.create_dataset("distance_mpc", data=distance)
        truth.create_dataset("inclination_rad", data=inclination)
        truth.create_dataset("mass1", data=mass1)
        truth.create_dataset("mass2", data=mass2)
        truth.create_dataset("spin1z", data=spin1z)
        truth.create_dataset("spin2z", data=spin2z)

        # Detection info
        det = f.create_group("detection")
        det.create_dataset("coinc_event_id", data=coinc_ids)
        det.create_dataset("snr", data=snr)
        det.create_dataset("far", data=far)
        # Store ifos as variable-length strings
        dt = h5py.string_dtype()
        ds = det.create_dataset("ifos", shape=(n_events,), dtype=dt)
        for i in range(n_events):
            ds[i] = ifos[i]

        # Skymap statistics
        sky = f.create_group("skymap")
        sky.create_dataset("area_90", data=area_90)
        sky.create_dataset("area_50", data=area_50)
        sky.create_dataset("area_20", data=area_20)
        sky.create_dataset("distmean", data=distmean)
        sky.create_dataset("diststd", data=diststd)
        sky.create_dataset("searched_area", data=searched_area)

    print(f"  Wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Convert observing scenarios to HDF5")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    args = parser.parse_args()

    print(f"Runs directory: {args.runs_dir}")
    print(f"Output directory: {args.output_dir}")

    for run_name in args.runs:
        run_dir = args.runs_dir / run_name / "bgp"
        if not run_dir.exists():
            print(f"  Skipping {run_name}: {run_dir} not found")
            continue
        output_path = args.output_dir / f"{run_name}.hdf5"
        build_hdf5(run_name, run_dir, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
