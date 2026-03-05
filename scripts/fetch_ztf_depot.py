#!/usr/bin/env python
"""
Download ZTF observation metadata from IRSA TAP or the ZTF depot, store in HDF5.

Usage:
    # Download from IRSA (default), one HDF5 per year
    python fetch_ztf_depot.py --start 20180301 --end 20260303 --year-split --outdir ztf_data/

    # Download from the ZTF depot (goodsubs files)
    python fetch_ztf_depot.py --source depot --start 20190601 --end 20220101 --year-split --outdir ztf_depot/

    # Export per-exposure CSV for survey-sim ZtfLoader (accepts multiple HDF5 files)
    python fetch_ztf_depot.py --export-csv ztf_data/ztf_*.h5 -o ztf_obs.csv

    # Cross-check IRSA vs depot data
    python fetch_ztf_depot.py --compare ztf_data/ ztf_depot/
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime, timedelta

import h5py
import numpy as np
import requests

def _load_config():
    """Load config from config.yaml next to this script, fall back to defaults."""
    import yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("config.yaml", "config.defaults.yaml"):
        path = os.path.join(script_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise RuntimeError("No config.yaml or config.defaults.yaml found next to script")

_CFG = _load_config()

TAP_URL = _CFG["irsa"]["tap_url"]
DEPOT_URL = _CFG["depot"]["url"]
DEPOT_USERNAME = _CFG["depot"]["username"]
DEPOT_PASSWORD = _CFG["depot"]["password"]
BOOM_URL = _CFG["boom"]["url"]
BOOM_USERNAME = _CFG["boom"]["username"]
BOOM_PASSWORD = _CFG["boom"]["password"]

FILTER_MAP = {1: "g", 2: "r", 3: "i"}

# ---------- IRSA dtype ----------
IRSA_DTYPE = np.dtype([
    ("obsjd", "f8"),
    ("field", "i2"),
    ("rcid", "i1"),
    ("ra", "f8"),
    ("dec", "f8"),
    ("programid", "i1"),
    ("expid", "i4"),
    ("fid", "i1"),
    ("maglimit", "f4"),
    ("seeing", "f4"),
    ("airmass", "f4"),
    ("exptime", "f4"),
])

# ---------- Depot dtype (extra columns from goodsubs) ----------
DEPOT_DTYPE = np.dtype([
    ("obsjd", "f8"),
    ("field", "i2"),
    ("rcid", "i1"),
    ("ra", "f8"),
    ("dec", "f8"),
    ("nalertpackets", "i2"),
    ("programid", "i1"),
    ("expid", "i4"),
    ("fid", "i1"),
    ("scimaglim", "f4"),
    ("diffmaglim", "f4"),
    ("sciinpseeing", "f4"),
    ("difffwhm", "f4"),
    ("exptime", "i2"),
])

# Depot column indices (subtractionstatus is col 14)
DEPOT_SUBSTATUS_COL = 14

# ---------- IRSA TAP ----------

TAP_QUERY = """
SELECT field, rcid, fid, expid, obsjd, exptime, maglimit, seeing, airmass,
       ra, dec, ipac_gid
FROM ztf.ztf_current_meta_sci
WHERE (obsjd BETWEEN {jd_start} AND {jd_end})
"""


def table_to_irsa_array(table) -> np.ndarray:
    """Convert astropy table from IRSA to structured NumPy array."""
    if len(table) == 0:
        return np.array([], dtype=IRSA_DTYPE)
    data = np.empty(len(table), dtype=IRSA_DTYPE)
    data["obsjd"] = table["obsjd"]
    data["field"] = table["field"]
    data["rcid"] = table["rcid"]
    data["ra"] = table["ra"]
    data["dec"] = table["dec"]
    data["programid"] = table["ipac_gid"]
    data["expid"] = table["expid"]
    data["fid"] = table["fid"]
    data["maglimit"] = table["maglimit"]
    data["seeing"] = table["seeing"]
    data["airmass"] = table["airmass"]
    data["exptime"] = table["exptime"]
    return data


def query_irsa_range(client, jd_start: float, jd_end: float) -> np.ndarray:
    """Query IRSA TAP for a JD range."""
    result = client.search(TAP_QUERY.format(jd_start=jd_start, jd_end=jd_end))
    return table_to_irsa_array(result.to_table())


# ---------- Depot ----------

def parse_depot_text(text: str) -> np.ndarray:
    """Parse pipe-delimited depot goodsubs text into a structured NumPy array.

    The depot format has spaces around pipes, a header row, a separator row,
    and a footer row. Filters to subtractionstatus == 1.
    """
    rows = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("jd") or line.startswith("-") or line.startswith("("):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 16:
            continue
        try:
            substatus = int(parts[DEPOT_SUBSTATUS_COL])
        except ValueError:
            continue
        if substatus != 1:
            continue
        try:
            row = (
                float(parts[0]),   # jd
                int(parts[1]),     # field
                int(parts[2]),     # rcid
                float(parts[3]),   # ra0
                float(parts[4]),   # dec0
                int(parts[5]),     # nalertpackets
                int(parts[6]),     # programid
                int(parts[7]),     # expid
                int(parts[8]),     # fid
                float(parts[9]),   # scimaglim
                float(parts[10]),  # diffmaglim
                float(parts[11]),  # sciinpseeing
                float(parts[12]),  # difffwhm
                int(parts[13]),    # exptime
            )
            rows.append(row)
        except (ValueError, IndexError):
            continue
    if not rows:
        return np.array([], dtype=DEPOT_DTYPE)
    return np.array(rows, dtype=DEPOT_DTYPE)


def fetch_depot_day(session, date_str: str) -> np.ndarray:
    """Fetch one day of goodsubs data from the depot. Returns DEPOT_DTYPE array."""
    import requests
    url = f"{DEPOT_URL}/{date_str}/goodsubs_{date_str}.txt"
    try:
        resp = session.get(url, timeout=120)
    except requests.RequestException as e:
        raise RuntimeError(f"connection error: {e}")

    if resp.status_code == 404 or resp.text.strip().startswith("<!"):
        return np.array([], dtype=DEPOT_DTYPE)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")

    return parse_depot_text(resp.text)


# ---------- BOOM ----------

class BoomClient:
    """Minimal BOOM API client with bearer-token auth and retry logic."""

    def __init__(self):
        self.session = requests.Session()
        self.auth_time = 0
        self._authenticate()

    def _authenticate(self):
        resp = self.session.post(
            f"{BOOM_URL}/auth",
            data={"username": BOOM_USERNAME, "password": BOOM_PASSWORD},
            timeout=30,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token") or resp.json().get("token")
        if not token:
            raise RuntimeError(f"BOOM auth failed: {resp.text[:200]}")
        self.session.headers["Authorization"] = f"Bearer {token}"
        self.auth_time = time.time()
        print("  BOOM authenticated.", flush=True)

    def _maybe_reauth(self):
        if time.time() - self.auth_time > 1200:  # 20 min
            print("  Re-authenticating with BOOM...", flush=True)
            self._authenticate()

    def pipeline_query(self, catalog_name: str, pipeline: list) -> list:
        self._maybe_reauth()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = self.session.post(
                    f"{BOOM_URL}/queries/pipeline",
                    json={"catalog_name": catalog_name, "pipeline": pipeline},
                    timeout=600,
                )
                if resp.status_code in (502, 503, 504, 429):
                    wait = 2 ** attempt
                    print(f"    BOOM retry ({resp.status_code}) in {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json().get("data", [])
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"    BOOM connection error, retry in {wait}s: {e}", flush=True)
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("BOOM API failed after retries")


def fetch_boom_day(client: BoomClient, jd_start: float, jd_end: float) -> np.ndarray:
    """Query BOOM for per-CCD diffmaglim in a 1-day JD range.

    Groups alerts by (jd, field, rcid, fid) to get one row per CCD observation.
    Returns DEPOT_DTYPE array.
    """
    pipeline = [
        {"$match": {
            "candidate.jd": {"$gte": jd_start, "$lt": jd_end},
        }},
        {"$group": {
            "_id": {
                "jd": "$candidate.jd",
                "field": "$candidate.field",
                "rcid": "$candidate.rcid",
                "fid": "$candidate.fid",
            },
            "diffmaglim": {"$first": "$candidate.diffmaglim"},
            "ra": {"$avg": "$candidate.ra"},
            "dec": {"$avg": "$candidate.dec"},
            "programid": {"$first": "$candidate.programid"},
            "fwhm": {"$first": "$candidate.fwhm"},
            "n_alerts": {"$sum": 1},
            "exptime": {"$first": "$candidate.exptime"},
        }},
    ]

    data = client.pipeline_query("ZTF_alerts", pipeline)

    if not data:
        return np.array([], dtype=DEPOT_DTYPE)

    rows = []
    for doc in data:
        _id = doc["_id"]
        try:
            exptime_raw = doc.get("exptime")
            exptime = int(float(exptime_raw)) if exptime_raw is not None else 30
            fwhm_raw = doc.get("fwhm")
            fwhm = float(fwhm_raw) if fwhm_raw is not None else 0.0
            diffmaglim_raw = doc.get("diffmaglim")
            diffmaglim = float(diffmaglim_raw) if diffmaglim_raw is not None else 0.0
            programid_raw = doc.get("programid")
            programid = int(programid_raw) if programid_raw is not None else 0

            row = (
                float(_id["jd"]),                         # obsjd
                int(_id["field"]),                         # field
                int(_id["rcid"]),                          # rcid
                float(doc.get("ra") or 0),                 # ra (avg of source positions)
                float(doc.get("dec") or 0),                # dec
                int(doc.get("n_alerts") or 0),             # nalertpackets
                programid,                                 # programid
                0,                                         # expid (not in alert schema)
                int(_id["fid"]),                           # fid
                0.0,                                       # scimaglim (N/A)
                diffmaglim,                                # diffmaglim
                fwhm,                                      # sciinpseeing (sci PSF FWHM)
                fwhm,                                      # difffwhm (approx from sci FWHM)
                exptime,                                   # exptime
            )
            rows.append(row)
        except (ValueError, KeyError, TypeError) as e:
            continue

    if not rows:
        return np.array([], dtype=DEPOT_DTYPE)

    return np.array(rows, dtype=DEPOT_DTYPE)


# ---------- Generic HDF5 helpers ----------

def create_hdf5(path: str, dtype, source_label: str):
    """Create a new HDF5 file with the observations dataset."""
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "observations",
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(10000,),
            compression="gzip",
            compression_opts=4,
        )
        f.attrs["source"] = source_label
        f.attrs["date_range"] = ""
        f.attrs["months_downloaded"] = ""
        f.attrs["created"] = datetime.utcnow().isoformat()
        f.attrs["last_updated"] = datetime.utcnow().isoformat()


def get_downloaded_months(f: h5py.File) -> set:
    raw = f.attrs.get("months_downloaded", "")
    if not raw:
        return set()
    return set(raw.split(","))


def add_downloaded_month(f: h5py.File, month_str: str):
    existing = get_downloaded_months(f)
    existing.add(month_str)
    f.attrs["months_downloaded"] = ",".join(sorted(existing))
    f.attrs["last_updated"] = datetime.utcnow().isoformat()


def append_observations(f: h5py.File, data: np.ndarray):
    ds = f["observations"]
    old_len = ds.shape[0]
    new_len = old_len + len(data)
    ds.resize((new_len,))
    ds[old_len:new_len] = data


# ---------- Date helpers ----------

def month_ranges(start_date: str, end_date: str):
    """Yield (year_month_str, jd_start, jd_end) for each calendar month in range."""
    from astropy.time import Time
    from calendar import monthrange

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    current = start_dt.replace(day=1)
    while current <= end_dt:
        year, month = current.year, current.month
        _, ndays = monthrange(year, month)
        month_start = datetime(year, month, 1)
        month_end = datetime(year, month, ndays, 23, 59, 59)

        actual_start = max(month_start, start_dt)
        actual_end = min(month_end, end_dt)

        from astropy.time import Time as T
        jd_start = T(actual_start).jd
        jd_end = T(actual_end).jd + 0.5

        ym_str = f"{year}{month:02d}"
        yield ym_str, actual_start, actual_end, jd_start, jd_end

        if month == 12:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, month + 1, 1)


def date_range_days(start_dt, end_dt):
    """Yield YYYYMMDD strings for each day in [start_dt, end_dt]."""
    current = start_dt
    while current <= end_dt:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


# ---------- Download logic ----------

def download_to_file(h5_path: str, months: list[tuple], fetch_fn, dtype, source_label: str):
    """Download months into a single HDF5 file using fetch_fn for each month.

    fetch_fn(ym_str, dt_start, dt_end, jd_start, jd_end) -> np.ndarray

    Uses /tmp as staging area to avoid Lustre HDF5 write issues.
    """
    import shutil
    import tempfile

    already = set()
    if os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            already = get_downloaded_months(f)

    to_fetch = [m for m in months if m[0] not in already]
    if not to_fetch:
        print(f"  All {len(months)} months already downloaded, skipping.", flush=True)
        return
    print(f"  To fetch: {len(to_fetch)} months (skipping {len(months) - len(to_fetch)} already done)", flush=True)

    tmpdir = tempfile.mkdtemp(prefix="ztf_")
    tmp_h5 = os.path.join(tmpdir, os.path.basename(h5_path))

    if os.path.exists(h5_path):
        shutil.copy2(h5_path, tmp_h5)
    else:
        print(f"  Creating new HDF5 file (staging in {tmpdir})", flush=True)
        create_hdf5(tmp_h5, dtype, source_label)

    try:
        with h5py.File(tmp_h5, "a") as f:
            total_rows = 0
            for i, (ym_str, dt_start, dt_end, jd_start, jd_end) in enumerate(to_fetch):
                data = None
                for attempt in range(3):
                    try:
                        data = fetch_fn(ym_str, dt_start, dt_end, jd_start, jd_end)
                        break
                    except Exception as e:
                        if attempt < 2:
                            wait = 10 * (attempt + 1)
                            print(f"    [{i+1}/{len(to_fetch)}] {ym_str}: error (attempt {attempt+1}/3): {e}, retrying in {wait}s", flush=True)
                            time.sleep(wait)
                        else:
                            print(f"    [{i+1}/{len(to_fetch)}] {ym_str}: FAILED after 3 attempts: {e}", flush=True)

                if data is None:
                    continue

                if len(data) == 0:
                    print(f"    [{i+1}/{len(to_fetch)}] {ym_str}: no data", flush=True)
                    add_downloaded_month(f, ym_str)
                    continue

                append_observations(f, data)
                add_downloaded_month(f, ym_str)
                total_rows += len(data)
                print(f"    [{i+1}/{len(to_fetch)}] {ym_str}: {len(data)} rows", flush=True)
                f.flush()

                time.sleep(2)

            f.flush()
            final_size = f["observations"].shape[0]
            print(f"  Done. Added {total_rows} rows. Total in file: {final_size}", flush=True)

        shutil.copy2(tmp_h5, h5_path)
        print(f"  Copied to {h5_path}", flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def download_dates(args):
    """Download ZTF observation metadata for a date range."""
    source = args.source

    all_months = list(month_ranges(args.start, args.end))
    print(f"Source: {source}", flush=True)
    print(f"Date range: {args.start} to {args.end} ({len(all_months)} months)", flush=True)

    if source == "irsa":
        import pyvo.dal
        client = pyvo.dal.TAPService(TAP_URL)
        dtype = IRSA_DTYPE
        source_label = TAP_URL

        def fetch_fn(ym_str, dt_start, dt_end, jd_start, jd_end):
            return query_irsa_range(client, jd_start, jd_end)

    elif source == "depot":
        session = requests.Session()
        session.auth = (DEPOT_USERNAME, DEPOT_PASSWORD)
        dtype = DEPOT_DTYPE
        source_label = DEPOT_URL

        def fetch_fn(ym_str, dt_start, dt_end, jd_start, jd_end):
            # Fetch day-by-day and concatenate
            all_data = []
            for date_str in date_range_days(dt_start, dt_end):
                data = fetch_depot_day(session, date_str)
                if len(data) > 0:
                    all_data.append(data)
                time.sleep(0.5)
            if not all_data:
                return np.array([], dtype=DEPOT_DTYPE)
            return np.concatenate(all_data)

    elif source == "boom":
        boom_client = BoomClient()
        dtype = DEPOT_DTYPE
        source_label = BOOM_URL

        def fetch_fn(ym_str, dt_start, dt_end, jd_start, jd_end):
            # Try full month first; if empty, fall back to weekly chunks
            # (BOOM can silently return empty on large aggregations)
            data = fetch_boom_day(boom_client, jd_start, jd_end)
            if len(data) > 0:
                return data

            # Weekly fallback
            all_data = []
            week_start = jd_start
            while week_start < jd_end:
                week_end = min(week_start + 7.0, jd_end)
                chunk = fetch_boom_day(boom_client, week_start, week_end)
                if len(chunk) > 0:
                    all_data.append(chunk)
                    print(f"      JD {week_start:.1f}-{week_end:.1f}: {len(chunk)} rows", flush=True)
                week_start = week_end
                time.sleep(1)
            if not all_data:
                return np.array([], dtype=DEPOT_DTYPE)
            return np.concatenate(all_data)

    else:
        print(f"Unknown source: {source}")
        sys.exit(1)

    if args.month_split:
        import shutil
        import tempfile

        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)

        total_months = len(all_months)
        for i, (ym_str, dt_start, dt_end, jd_start, jd_end) in enumerate(all_months):
            h5_path = os.path.join(outdir, f"ztf_{ym_str}.h5")

            # Skip if file already exists with data
            if os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, "r") as f:
                        n = f["observations"].shape[0]
                    if n > 0:
                        print(f"[{i+1}/{total_months}] {ym_str}: already done ({n:,} rows), skipping.", flush=True)
                        continue
                except Exception:
                    pass  # corrupted file, re-download

            print(f"[{i+1}/{total_months}] {ym_str}: fetching...", flush=True)

            data = None
            for attempt in range(3):
                try:
                    data = fetch_fn(ym_str, dt_start, dt_end, jd_start, jd_end)
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = 10 * (attempt + 1)
                        print(f"  error (attempt {attempt+1}/3): {e}, retrying in {wait}s", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"  FAILED after 3 attempts: {e}", flush=True)

            if data is None:
                continue

            if len(data) == 0:
                print(f"  {ym_str}: no data (writing empty marker)", flush=True)

            # Write to /tmp then copy to Lustre
            tmpdir = tempfile.mkdtemp(prefix="ztf_")
            tmp_h5 = os.path.join(tmpdir, os.path.basename(h5_path))
            try:
                with h5py.File(tmp_h5, "w") as f:
                    if len(data) == 0:
                        f.create_dataset("observations", shape=(0,), maxshape=(None,),
                                         dtype=dtype, chunks=(1,))
                    else:
                        f.create_dataset(
                            "observations", data=data,
                            chunks=(min(10000, len(data)),),
                            compression="gzip", compression_opts=4,
                        )
                    f.attrs["source"] = source_label
                    f.attrs["month"] = ym_str
                    f.attrs["created"] = datetime.utcnow().isoformat()
                shutil.copy2(tmp_h5, h5_path)
                print(f"  {ym_str}: {len(data):,} rows → {h5_path}", flush=True)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            time.sleep(2)

    elif args.year_split:
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)

        year_months: dict[str, list[tuple]] = {}
        for entry in all_months:
            year = entry[0][:4]
            year_months.setdefault(year, []).append(entry)

        for year in sorted(year_months):
            h5_path = os.path.join(outdir, f"ztf_{year}.h5")
            print(f"\n=== {year} ({len(year_months[year])} months) → {h5_path} ===", flush=True)
            download_to_file(h5_path, year_months[year], fetch_fn, dtype, source_label)
    else:
        download_to_file(args.output, all_months, fetch_fn, dtype, source_label)


# ---------- CSV export ----------

def export_csv(args):
    """Read HDF5 file(s), aggregate per-exposure, write CSV for ZtfLoader."""
    import pandas as pd

    h5_paths = []
    for pattern in args.export_csv:
        expanded = sorted(glob.glob(pattern))
        if not expanded:
            print(f"Warning: no files match '{pattern}'")
        h5_paths.extend(expanded)

    if not h5_paths:
        print("Error: no HDF5 files found.")
        sys.exit(1)

    csv_path = args.o

    all_obs = []
    for h5_path in h5_paths:
        print(f"Reading {h5_path}...", flush=True)
        with h5py.File(h5_path, "r") as f:
            source = f.attrs.get("source", "unknown")
            obs = f["observations"][:]
            print(f"  {len(obs)} rows (source: {source})", flush=True)
            all_obs.append((obs, source))

    # Determine source type from first file
    first_obs, first_source = all_obs[0]
    is_depot = "depot" in first_source

    # Normalize column names across sources
    frames = []
    for obs, source in all_obs:
        if "depot" in source:
            frames.append(pd.DataFrame({
                "expid": obs["expid"],
                "ra": obs["ra"],
                "dec": obs["dec"],
                "obsjd": obs["obsjd"],
                "fid": obs["fid"],
                "programid": obs["programid"],
                "maglimit": obs["diffmaglim"],   # use diff maglim
                "seeing": obs["difffwhm"],        # use diff FWHM
                "airmass": np.nan,                # not in depot
                "exptime": obs["exptime"],
            }))
        else:
            frames.append(pd.DataFrame({
                "expid": obs["expid"],
                "ra": obs["ra"],
                "dec": obs["dec"],
                "obsjd": obs["obsjd"],
                "fid": obs["fid"],
                "programid": obs["programid"],
                "maglimit": obs["maglimit"],
                "seeing": obs["seeing"],
                "airmass": obs["airmass"],
                "exptime": obs["exptime"],
            }))

    df = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(df)} raw CCD-level rows", flush=True)

    print("Aggregating per-exposure...", flush=True)
    agg = df.groupby("expid").agg(
        ra=("ra", "median"),
        dec=("dec", "median"),
        obsjd=("obsjd", "first"),
        fid=("fid", "first"),
        programid=("programid", "first"),
        maglim=("maglimit", "median"),
        seeing=("seeing", "median"),
        airmass=("airmass", "median"),
        exptime=("exptime", "first"),
        n_ccds=("expid", "count"),
    ).reset_index()

    agg["mjd"] = agg["obsjd"] - 2400000.5
    agg["filter"] = agg["fid"].map(FILTER_MAP)
    agg["night"] = np.floor(agg["obsjd"] - 0.5).astype(int) - 2400000
    agg["skymag"] = 21.0

    # If no airmass (depot), compute it
    if agg["airmass"].isna().any():
        print("Computing airmass for depot data...", flush=True)
        mask = agg["airmass"].isna()
        agg.loc[mask, "airmass"] = compute_airmass(
            agg.loc[mask, "ra"].values,
            agg.loc[mask, "dec"].values,
            agg.loc[mask, "obsjd"].values,
        )

    out = agg[["expid", "ra", "dec", "mjd", "filter", "programid", "maglim",
               "seeing", "exptime", "airmass", "skymag", "night"]].copy()
    out.rename(columns={"expid": "obsid"}, inplace=True)
    out.sort_values("mjd", inplace=True)

    out.to_csv(csv_path, index=False)
    print(f"Wrote {len(out)} exposures to {csv_path}", flush=True)
    print(f"  Date range: MJD {out['mjd'].min():.2f} to {out['mjd'].max():.2f}", flush=True)
    print(f"  Filters: {out['filter'].value_counts().to_dict()}", flush=True)
    print(f"  Programs: {out['programid'].value_counts().to_dict()}", flush=True)


def compute_airmass(ra_deg, dec_deg, jd):
    """Compute airmass from RA/Dec/JD at Palomar using astropy."""
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u

    palomar = EarthLocation(lat=33.3564*u.deg, lon=-116.8650*u.deg, height=1712.0*u.m)
    times = Time(jd, format="jd")
    coords = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
    altaz = coords.transform_to(AltAz(obstime=times, location=palomar))
    alt_rad = np.radians(altaz.alt.deg)
    arg = alt_rad + np.radians(244.0 / (165.0 + 47.0 * np.degrees(alt_rad) ** 1.1))
    airmass = 1.0 / np.sin(arg)
    return np.clip(airmass, 1.0, 10.0)


# ---------- Compare ----------

def compare_sources(args):
    """Compare IRSA and depot HDF5 files on overlapping time ranges."""
    import pandas as pd

    irsa_dir, depot_dir = args.compare

    # Find overlapping years
    irsa_files = sorted(glob.glob(os.path.join(irsa_dir, "ztf_*.h5")))
    depot_files = sorted(glob.glob(os.path.join(depot_dir, "ztf_*.h5")))

    irsa_years = {os.path.basename(f).replace("ztf_", "").replace(".h5", ""): f for f in irsa_files}
    depot_years = {os.path.basename(f).replace("ztf_", "").replace(".h5", ""): f for f in depot_files}

    common = sorted(set(irsa_years) & set(depot_years))
    if not common:
        print("No overlapping years found.")
        return

    print(f"Comparing years: {', '.join(common)}\n", flush=True)

    for year in common:
        with h5py.File(irsa_years[year], "r") as f:
            irsa = f["observations"][:]
        with h5py.File(depot_years[year], "r") as f:
            depot = f["observations"][:]

        if len(irsa) == 0 or len(depot) == 0:
            print(f"=== {year}: skipping (IRSA={len(irsa)}, Depot={len(depot)} rows) ===", flush=True)
            continue

        print(f"=== {year}: IRSA={len(irsa):,} rows, Depot={len(depot):,} rows ===", flush=True)

        # Compare by expid+rcid (unique CCD-level identifier)
        irsa_keys = set(zip(irsa["expid"].tolist(), irsa["rcid"].tolist()))
        depot_keys = set(zip(depot["expid"].tolist(), depot["rcid"].tolist()))

        common_keys = irsa_keys & depot_keys
        only_irsa = irsa_keys - depot_keys
        only_depot = depot_keys - irsa_keys

        print(f"  Common CCD-obs:    {len(common_keys):>10,}", flush=True)
        print(f"  Only in IRSA:      {len(only_irsa):>10,}", flush=True)
        print(f"  Only in Depot:     {len(only_depot):>10,}", flush=True)

        if not common_keys:
            continue

        # Compare values for matching rows
        # Index by (expid, rcid)
        irsa_df = pd.DataFrame({
            "expid": irsa["expid"], "rcid": irsa["rcid"],
            "ra_irsa": irsa["ra"], "dec_irsa": irsa["dec"],
            "maglim_irsa": irsa["maglimit"], "seeing_irsa": irsa["seeing"],
        })
        depot_df = pd.DataFrame({
            "expid": depot["expid"], "rcid": depot["rcid"],
            "ra_depot": depot["ra"], "dec_depot": depot["dec"],
            "diffmaglim_depot": depot["diffmaglim"], "difffwhm_depot": depot["difffwhm"],
            "scimaglim_depot": depot["scimaglim"], "sciinpseeing_depot": depot["sciinpseeing"],
        })

        merged = irsa_df.merge(depot_df, on=["expid", "rcid"])

        # Position comparison
        ra_diff = np.abs(merged["ra_irsa"] - merged["ra_depot"])
        dec_diff = np.abs(merged["dec_irsa"] - merged["dec_depot"])
        print(f"  RA diff (arcsec):  median={np.median(ra_diff)*3600:.3f}, max={np.max(ra_diff)*3600:.3f}", flush=True)
        print(f"  Dec diff (arcsec): median={np.median(dec_diff)*3600:.3f}, max={np.max(dec_diff)*3600:.3f}", flush=True)

        # Maglim: IRSA maglimit vs depot scimaglim and diffmaglim
        sci_diff = merged["maglim_irsa"] - merged["scimaglim_depot"]
        diff_diff = merged["maglim_irsa"] - merged["diffmaglim_depot"]
        print(f"  IRSA maglimit vs depot scimaglim:  median diff={np.median(sci_diff):.4f}, std={np.std(sci_diff):.4f}", flush=True)
        print(f"  IRSA maglimit vs depot diffmaglim: median diff={np.median(diff_diff):.4f}, std={np.std(diff_diff):.4f}", flush=True)

        # Seeing: IRSA seeing vs depot sciinpseeing and difffwhm
        sci_see = merged["seeing_irsa"] - merged["sciinpseeing_depot"]
        diff_see = merged["seeing_irsa"] - merged["difffwhm_depot"]
        print(f"  IRSA seeing vs depot sciinpseeing: median diff={np.median(sci_see):.4f}, std={np.std(sci_see):.4f}", flush=True)
        print(f"  IRSA seeing vs depot difffwhm:     median diff={np.median(diff_see):.4f}, std={np.std(diff_see):.4f}", flush=True)

        print(flush=True)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Download ZTF observation metadata from IRSA or depot, store in HDF5."
    )
    parser.add_argument("--start", type=str, help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, help="End date YYYYMMDD")
    parser.add_argument("--source", type=str, default="irsa", choices=["irsa", "depot", "boom"],
                        help="Data source: irsa (default), depot, or boom")
    parser.add_argument("--output", type=str, default="ztf_observations.h5",
                        help="Output HDF5 file path (default: ztf_observations.h5)")
    parser.add_argument("--year-split", action="store_true",
                        help="Save one HDF5 file per year")
    parser.add_argument("--month-split", action="store_true",
                        help="Save one HDF5 file per month (one query = one file)")
    parser.add_argument("--outdir", type=str, default="ztf_data",
                        help="Output directory for split files (default: ztf_data/)")
    parser.add_argument("--export-csv", type=str, nargs="+", metavar="H5_FILE",
                        help="Export CSV from HDF5 file(s) (accepts multiple files/globs)")
    parser.add_argument("-o", type=str, default="ztf_obs.csv",
                        help="Output CSV path (default: ztf_obs.csv)")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("IRSA_DIR", "DEPOT_DIR"),
                        help="Compare IRSA vs depot HDF5 directories")

    args = parser.parse_args()

    if args.compare:
        compare_sources(args)
    elif args.export_csv:
        export_csv(args)
    elif args.start and args.end:
        download_dates(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
