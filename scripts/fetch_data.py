#!/usr/bin/env python
"""
Download external data files needed by survey-sim.

Usage:
    python fetch_data.py rubin          # Download Rubin OpSim baseline database
    python fetch_data.py rubin --list   # List available OpSim runs
    python fetch_data.py all            # Download everything
"""

import argparse
import hashlib
import os
import sys
import urllib.request
import shutil

import yaml


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("config.yaml", "config.defaults.yaml"):
        path = os.path.join(script_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f), script_dir
    raise RuntimeError("No config file found")


def download_file(url, dest, chunk_size=1 << 20):
    """Download a file with progress reporting. Skips if file already exists."""
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"

    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1e6
                        total_mb = total / 1e6
                        print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r  {downloaded / 1e6:.0f} MB", end="", flush=True)

            print()  # newline after progress
        shutil.move(tmp, dest)
        print(f"  Done: {os.path.getsize(dest) / 1e6:.1f} MB")
    except Exception as e:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise RuntimeError(f"Download failed: {e}")


def fetch_rubin(cfg, base_dir):
    """Download the Rubin OpSim baseline database."""
    rubin_cfg = cfg.get("rubin", {})
    url = rubin_cfg.get("opsim_url")
    data_dir = rubin_cfg.get("data_dir", "data/rubin")

    if not url:
        print("Error: no rubin.opsim_url in config")
        sys.exit(1)

    filename = url.rsplit("/", 1)[-1]
    dest = os.path.join(base_dir, data_dir, filename)

    print(f"Rubin OpSim: {filename}")
    download_file(url, dest)

    # Verify it's a valid SQLite file
    if os.path.exists(dest):
        with open(dest, "rb") as f:
            header = f.read(16)
        if header[:6] != b"SQLite":
            print(f"  WARNING: {dest} does not look like a SQLite database!")
        else:
            print(f"  Verified: valid SQLite database")


def main():
    parser = argparse.ArgumentParser(description="Download survey-sim data files")
    parser.add_argument("target", choices=["rubin", "all"],
                        help="What to download")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if file exists")
    args = parser.parse_args()

    cfg, base_dir = load_config()

    if args.target in ("rubin", "all"):
        if args.force:
            rubin_cfg = cfg.get("rubin", {})
            data_dir = rubin_cfg.get("data_dir", "data/rubin")
            url = rubin_cfg.get("opsim_url", "")
            filename = url.rsplit("/", 1)[-1]
            dest = os.path.join(base_dir, data_dir, filename)
            if os.path.exists(dest):
                os.unlink(dest)
        fetch_rubin(cfg, base_dir)


if __name__ == "__main__":
    main()
