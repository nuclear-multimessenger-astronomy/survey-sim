"""Data management for survey-sim: download survey data from Hugging Face."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import EntryNotFoundError

HF_REPO_ID = "nuclear-multimessenger-astronomy/survey-sim-data"
HF_REVISION = "main"
HF_REPO_TYPE = "dataset"

def data_dir() -> Path:
    """Return the local data directory.

    Respects, in order:
    1. SURVEY_SIM_DATA_DIR (explicit override)
    2. XDG_CACHE_HOME/survey-sim (standard cache location)
    3. ~/.cache/survey-sim (fallback)
    """
    if "SURVEY_SIM_DATA_DIR" in os.environ:
        d = Path(os.environ["SURVEY_SIM_DATA_DIR"])
    else:
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        d = cache_home / "survey-sim"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_file(filename: str, force: bool = False) -> Path:
    """Download a single file from the Hugging Face dataset repo.

    Parameters
    ----------
    filename : str
        File path on HF (e.g. "ztf_boom/ztf_201803.h5").
    force : bool
        Re-download even if file exists locally.

    Returns
    -------
    Path to the downloaded local file.
    """
    local_dir = data_dir()
    local_path = local_dir / filename

    if local_path.exists() and not force:
        return local_path

    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        repo_type=HF_REPO_TYPE,
        filename=filename,
        local_dir=local_dir,
    )
    return Path(downloaded)


def download_ztf(
    start: str = "201803", end: str = "202603", force: bool = False
) -> list[Path]:
    """Download ZTF boom-pipeline monthly observation HDF5 files.

    Parameters
    ----------
    start : str
        First month to download, YYYYMM format (default: "201803").
    end : str
        Last month to download, YYYYMM format (default: "202603").
    force : bool
        Re-download even if files exist locally.

    Returns
    -------
    List of paths to the downloaded HDF5 files.
    """
    # Generate YYYYMM range
    start_y, start_m = int(start[:4]), int(start[4:])
    end_y, end_m = int(end[:4]), int(end[4:])

    months = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        months.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1

    paths = []
    for ym in months:
        filename = f"ztf_boom/ztf_{ym}.h5"
        try:
            p = download_file(filename, force=force)
            paths.append(p)
            print(f"  ztf_{ym}.h5 -> {p}")
        except EntryNotFoundError:
            pass  # month may not exist yet
    print(f"Downloaded {len(paths)} ZTF monthly files.")
    return paths


def download_observing_scenarios(
    runs: list[str] | None = None, force: bool = False
) -> list[Path]:
    """Download GW observing scenario HDF5 files.

    Parameters
    ----------
    runs : list of str, optional
        Which runs to download (default: all available).
    force : bool
        Re-download even if files exist locally.

    Returns
    -------
    List of paths to the downloaded HDF5 files.
    """
    if runs is None:
        runs = ["O4HL", "O4HLV", "O5a", "O5b", "O5c"]

    paths = []
    for run in runs:
        filename = f"observing_scenarios/{run}.hdf5"
        try:
            p = download_file(filename, force=force)
            paths.append(p)
            print(f"  {run}.hdf5 -> {p}")
        except EntryNotFoundError:
            print(f"  {run}.hdf5 not found on HF, skipping")
    return paths


def list_available() -> dict[str, list[str]]:
    """List all files available on the HF dataset repo."""
    api = HfApi()
    files = api.list_repo_files(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
        repo_type=HF_REPO_TYPE,
    )

    result = {}
    for f in files:
        if f.startswith("."):
            continue
        parts = f.split("/")
        category = parts[0] if len(parts) > 1 else "root"
        result.setdefault(category, []).append(f)
    return result
