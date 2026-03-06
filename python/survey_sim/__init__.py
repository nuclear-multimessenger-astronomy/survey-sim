"""survey-sim: Survey simulation framework for transient astronomy."""

from survey_sim.survey_sim import (
    PySurveyStore as SurveyStore,
    Instrument,
    GwEvent,
    load_gw_events as _load_gw_events_raw,
    KilonovaPopulation,
    FixedMetzgerKilonovaPopulation,
    Bu2026KilonovaPopulation,
    FixedBu2026KilonovaPopulation,
    SupernovaIaPopulation,
    SupernovaIIPopulation,
    TdePopulation,
    GrbPopulation,
    OnAxisGrbPopulation,
    OffAxisGrbPopulation,
    MetzgerKNModel,
    BlastwaveModel,
    DetectionCriteria,
    DetectionResult,
    SimulationPipeline,
    SimulationResult,
    RateSummary,
    TooSimulationResult,
    run_too_simulation,
)


def _ensure_ztf_boom(start="201803", end="202603"):
    """Download ZTF boom files from HF if not available locally, return paths."""
    from survey_sim.data import download_ztf
    return [str(p) for p in download_ztf(start=start, end=end)]


def _ensure_observing_scenario(run):
    """Download observing scenario HDF5 from HF if not available locally."""
    from survey_sim.data import download_file

    filename = f"observing_scenarios/{run}.hdf5"
    p = download_file(filename)
    return str(p)


def load_ztf_survey(start="201803", end="202603", nside=64):
    """Load ZTF boom-pipeline survey, auto-downloading from HF if needed.

    Parameters
    ----------
    start : str
        First month (YYYYMM). Default: "201803".
    end : str
        Last month (YYYYMM). Default: "202603".
    nside : int
        HEALPix NSIDE for spatial indexing. Default: 64.

    Returns
    -------
    SurveyStore
    """
    paths = _ensure_ztf_boom(start=start, end=end)
    return SurveyStore.from_ztf_boom(paths, nside=nside)


def load_gw_events(run="O5a"):
    """Load GW events from an observing scenario, auto-downloading from HF if needed.

    Parameters
    ----------
    run : str
        Observing run name (e.g. "O5a", "O5b", "O4HL"), a path to a
        directory with injections.dat/coincs.dat/allsky.dat, or a path
        to an HDF5 file.

    Returns
    -------
    list of GwEvent
    """
    import os

    # If it's an existing directory, load directly via Rust
    if os.path.isdir(run):
        return _load_gw_events_raw(run)

    # If it's an existing HDF5 file, load via Python
    if os.path.isfile(run) and (run.endswith(".hdf5") or run.endswith(".h5")):
        return _load_gw_events_hdf5(run)

    # Otherwise treat as a run name — download HDF5 from HF
    hdf5_path = _ensure_observing_scenario(run)
    return _load_gw_events_hdf5(hdf5_path)


def _load_gw_events_hdf5(path):
    """Load GW events from a merged HDF5 file."""
    import h5py
    import numpy as np
    from survey_sim.survey_sim import GwEvent

    with h5py.File(path, "r") as f:
        n = f.attrs["n_events"]
        ra = f["truth/ra_deg"][:]
        dec = f["truth/dec_deg"][:]
        distance = f["truth/distance_mpc"][:]
        inclination = f["truth/inclination_rad"][:]
        mass1 = f["truth/mass1"][:]
        mass2 = f["truth/mass2"][:]
        snr = f["detection/snr"][:]
        far = f["detection/far"][:]
        area_90 = f["skymap/area_90"][:]
        area_50 = f["skymap/area_50"][:]
        distmean = f["skymap/distmean"][:]
        diststd = f["skymap/diststd"][:]
        sim_ids = f["truth/simulation_id"][:]

    events = []
    for i in range(n):
        if np.isnan(area_90[i]):
            continue
        events.append(GwEvent._from_values(
            simulation_id=int(sim_ids[i]),
            ra=float(ra[i]),
            dec=float(dec[i]),
            distance_mpc=float(distance[i]),
            mass1=float(mass1[i]),
            mass2=float(mass2[i]),
            inclination=float(inclination[i]),
            snr=float(snr[i]),
            far=float(far[i]),
            area_90=float(area_90[i]),
            area_50=float(area_50[i]),
            dist_mean=float(distmean[i]),
            dist_std=float(diststd[i]),
        ))
    return events


__all__ = [
    "SurveyStore",
    "Instrument",
    "GwEvent",
    "load_gw_events",
    "load_ztf_survey",
    "KilonovaPopulation",
    "FixedMetzgerKilonovaPopulation",
    "Bu2026KilonovaPopulation",
    "FixedBu2026KilonovaPopulation",
    "SupernovaIaPopulation",
    "SupernovaIIPopulation",
    "TdePopulation",
    "GrbPopulation",
    "OnAxisGrbPopulation",
    "OffAxisGrbPopulation",
    "MetzgerKNModel",
    "BlastwaveModel",
    "DetectionCriteria",
    "DetectionResult",
    "SimulationPipeline",
    "SimulationResult",
    "RateSummary",
    "TooSimulationResult",
    "run_too_simulation",
]
