#!/usr/bin/env python3
"""Upload survey-sim data files to Hugging Face dataset repo.

Usage:
    python upload_hf_data.py --ztf          # upload ZTF HDF5s
    python upload_hf_data.py --scenarios     # upload observing scenario HDF5s
    python upload_hf_data.py --all           # upload everything

Requires: huggingface_hub, and `huggingface-cli login` for write access.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

HF_REPO_ID = "nuclear-multimessenger-astronomy/survey-sim-data"
HF_REPO_TYPE = "dataset"

ZTF_DIR = Path("/fred/oz480/mcoughli/simulations/ztf_data")
SCENARIO_DIR = Path("/fred/oz480/mcoughli/simulations/survey-sim/data/observing_scenarios")


def upload_ztf(api: HfApi):
    """Upload ZTF observation HDF5 files."""
    for h5 in sorted(ZTF_DIR.glob("ztf_*.h5")):
        size_mb = h5.stat().st_size / 1e6
        print(f"  Uploading {h5.name} ({size_mb:.0f} MB)...")
        api.upload_file(
            path_or_fileobj=str(h5),
            path_in_repo=f"ztf/{h5.name}",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )
        print(f"    Done.")


def upload_scenarios(api: HfApi):
    """Upload observing scenario HDF5 files."""
    for h5 in sorted(SCENARIO_DIR.glob("*.hdf5")):
        size_mb = h5.stat().st_size / 1e6
        print(f"  Uploading {h5.name} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(h5),
            path_in_repo=f"observing_scenarios/{h5.name}",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
        )
        print(f"    Done.")


def main():
    parser = argparse.ArgumentParser(description="Upload survey-sim data to Hugging Face")
    parser.add_argument("--ztf", action="store_true", help="Upload ZTF HDF5 files")
    parser.add_argument("--scenarios", action="store_true", help="Upload observing scenario HDF5 files")
    parser.add_argument("--all", action="store_true", help="Upload everything")
    parser.add_argument("--create-repo", action="store_true", help="Create the HF repo if it doesn't exist")
    args = parser.parse_args()

    if not (args.ztf or args.scenarios or args.all):
        parser.print_help()
        return

    api = HfApi()

    if args.create_repo:
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            exist_ok=True,
        )
        print(f"Repo {HF_REPO_ID} ready.")

    if args.ztf or args.all:
        print("Uploading ZTF data...")
        upload_ztf(api)

    if args.scenarios or args.all:
        print("Uploading observing scenarios...")
        upload_scenarios(api)

    print("All uploads complete.")
    print(f"View at: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
