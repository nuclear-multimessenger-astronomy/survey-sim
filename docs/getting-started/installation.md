# Installation

## Prerequisites

- [Rust toolchain](https://rustup.rs/) (stable, 1.70+)
- HDF5 library (for OpSim/HDF5 survey loading)

For Python bindings:

- Python >= 3.9
- [maturin](https://github.com/PyO3/maturin)
- numpy

## Rust library

```bash
cargo build --release
cargo test -p survey-sim --lib
```

## Python bindings

```bash
pip install maturin numpy
cd python
maturin develop --release
```

This builds and installs the `survey_sim` package into the active virtualenv.

!!! tip "Virtual environment"
    Maturin requires a virtualenv or conda environment. Create one with:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

!!! note "HPC / module systems"
    On HPC clusters you may need to load compiler, HDF5, and Python modules first:
    ```bash
    module load gcc/13.2.0 openmpi/4.1.6 hdf5/1.14.3 python/3.11.5
    source /path/to/your/venv/bin/activate
    cd python && maturin develop --release
    ```

## Dependencies

survey-sim depends on the following local crates:

| Crate | Path | Purpose |
|-------|------|---------|
| [`lightcurve-fitting`](https://github.com/boom-astro/lightcurve-fitting) | `../lightcurve-fitting` | Parametric lightcurve templates (Metzger, Bazin, Villar) |
| [`blastwave`](https://github.com/nuclear-multimessenger-astronomy/blastwave) | `/home/mcoughli/blastwave` | Relativistic blast wave hydrodynamics + afterglow radiation |

Ensure these are available at the expected paths, or update `Cargo.toml` accordingly.

## Run tests

```bash
# Unit tests (no external data required)
cargo test -p survey-sim --lib

# Integration tests (requires OpSim database)
cargo test -p survey-sim --test test_survey -- --ignored
```

## Controlling parallelism

survey-sim uses [Rayon](https://github.com/rayon-rs/rayon) for multi-core parallelism. By default it uses all available cores:

```bash
export RAYON_NUM_THREADS=4  # limit to 4 threads
```
