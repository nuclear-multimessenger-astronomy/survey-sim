"""Fiesta Bu2026 kilonova model adapter for survey-sim.

Wraps fiestaEM's FluxModel surrogate to produce per-band apparent magnitudes
interpolated to actual survey observation times.
"""

import numpy as np
import jax.numpy as jnp

from fiesta.inference.lightcurve_model import FluxModel

# Band name mapping: survey-sim short names <-> sncosmo/fiesta filter names.
_BAND_TO_FIESTA = {
    "u": "lsstu",
    "g": "lsstg",
    "r": "lsstr",
    "i": "lssti",
    "z": "lsstz",
    "y": "lssty",
}
_FIESTA_TO_BAND = {v: k for k, v in _BAND_TO_FIESTA.items()}

FIESTA_FILTERS = list(_BAND_TO_FIESTA.values())


class FiestaKNModel:
    """Adapter wrapping fiesta FluxModel('Bu2026_MLP') for survey-sim.

    The ``predict(params)`` interface matches what ``PythonCallbackModel``
    expects: it receives a dict of model parameters plus observation context
    keys (``_obs_times_mjd``, ``_obs_bands``, ``_t_exp``), and returns
    ``(times, {band: mags})`` where times and mags are aligned to the
    observation schedule.

    Parameters
    ----------
    name : str
        Fiesta surrogate name (default ``"Bu2026_MLP"``).
    filters : list[str] or None
        Fiesta filter names. Defaults to all six LSST bands.
    """

    def __init__(self, name="Bu2026_MLP", filters=None):
        if filters is None:
            filters = FIESTA_FILTERS
        self.model = FluxModel(name, filters=filters)
        self.filters = filters

    def predict(self, params):
        """Evaluate the Bu2026 kilonova model at observation times.

        Parameters
        ----------
        params : dict
            Must contain the Bu2026 physical parameters
            (``log10_mej_dyn``, ``v_ej_dyn``, ``Ye_dyn``,
            ``log10_mej_wind``, ``v_ej_wind``, ``Ye_wind``,
            ``inclination_EM``), plus ``luminosity_distance``,
            ``redshift``, and the observation context keys
            ``_obs_times_mjd``, ``_obs_bands``, ``_t_exp``.

        Returns
        -------
        (list[float], dict[str, list[float]])
            Observation times (MJD) and per-band apparent magnitudes
            interpolated to those times. Bands use survey-sim short names.
        """
        obs_times_mjd = np.asarray(params["_obs_times_mjd"], dtype=np.float64)
        obs_bands = list(params["_obs_bands"])
        t_exp = float(params["_t_exp"])
        redshift = float(params["redshift"])

        # Rest-frame time grid from fiesta (days since explosion).
        # Observations in rest frame: (t_obs - t_exp) / (1 + z).
        obs_rest_days = (obs_times_mjd - t_exp) / (1.0 + redshift)

        # Build fiesta input dict (jax scalars).
        fiesta_params = {}
        for key in self.model.parameter_names:
            fiesta_params[key] = jnp.float32(params[key])
        fiesta_params["luminosity_distance"] = jnp.float32(params["luminosity_distance"])
        fiesta_params["redshift"] = jnp.float32(redshift)

        # Call fiesta — returns (times_obs_days, {fiesta_filter: mags_array}).
        model_times, model_mags = self.model.predict(fiesta_params)
        model_times = np.asarray(model_times, dtype=np.float64)

        # Interpolate fiesta grid to each observation time, per band.
        # Extrapolate outside fiesta range by holding the boundary values.
        result_mags = {}
        for fiesta_filt, short_band in _FIESTA_TO_BAND.items():
            if fiesta_filt not in model_mags:
                continue
            grid_mags = np.asarray(model_mags[fiesta_filt], dtype=np.float64)
            interp_all = np.interp(obs_rest_days, model_times, grid_mags)
            result_mags[short_band] = interp_all.tolist()

        # For bands not covered by fiesta, fill with 99.
        unique_bands = set(obs_bands)
        for band in unique_bands:
            if band not in result_mags:
                result_mags[band] = [99.0] * len(obs_times_mjd)

        return (obs_times_mjd.tolist(), result_mags)

    def batch_predict(self, params_list, chunk_size=2048):
        """Batch evaluate multiple transients using JAX vmap on GPU.

        Parameters
        ----------
        params_list : list[dict]
            Each dict has Bu2026 physical parameters, ``luminosity_distance``,
            ``redshift``, and observation context keys (``_obs_times_mjd``,
            ``_obs_bands``, ``_t_exp``).
        chunk_size : int
            Maximum number of transients per GPU vmap call, to avoid OOM.
            Default 2048 fits comfortably in ~10 GB GPU memory.

        Returns
        -------
        list[tuple]
            List of ``(times_mjd, {band: mags})`` tuples, one per transient.
        """
        if not params_list:
            return []

        results = []
        for start in range(0, len(params_list), chunk_size):
            chunk = params_list[start : start + chunk_size]
            chunk_results = self._vpredict_chunk(chunk)
            results.extend(chunk_results)

        return results

    def _vpredict_chunk(self, params_list):
        """Evaluate a single chunk via vpredict with vectorized interpolation."""
        N = len(params_list)
        param_names = list(self.model.parameter_names)
        X = {}
        for name in param_names:
            X[name] = jnp.array([float(p[name]) for p in params_list], dtype=jnp.float32)
        X["luminosity_distance"] = jnp.array(
            [float(p["luminosity_distance"]) for p in params_list], dtype=jnp.float32
        )
        X["redshift"] = jnp.array(
            [float(p["redshift"]) for p in params_list], dtype=jnp.float32
        )

        # GPU call via JAX vmap
        model_times_batch, model_mags_batch = self.model.vpredict(X)

        # Convert to numpy — model_times_batch: (N, T), model_mags: {filt: (N, T)}
        model_times_batch = np.asarray(model_times_batch, dtype=np.float64)
        model_mags_np = {k: np.asarray(v, dtype=np.float64) for k, v in model_mags_batch.items()}

        # Pre-extract obs context arrays for all transients (avoid repeated Python dict access)
        obs_times_list = [np.asarray(p["_obs_times_mjd"], dtype=np.float64) for p in params_list]
        t_exps = np.array([float(p["_t_exp"]) for p in params_list])
        redshifts = np.array([float(p["redshift"]) for p in params_list])
        obs_bands_list = [list(p["_obs_bands"]) for p in params_list]

        # Pre-compute rest-frame observation times for each transient
        obs_rest_list = [
            (obs_times_list[i] - t_exps[i]) / (1.0 + redshifts[i]) for i in range(N)
        ]

        # Vectorized interpolation: for each band, interpolate all transients at once.
        # Since each transient has a different number of obs times, we pad to a uniform
        # length, do vectorized interp, then slice back.
        n_obs = np.array([len(t) for t in obs_rest_list])
        max_obs = int(n_obs.max()) if N > 0 else 0

        # Pad obs rest-frame times to (N, max_obs)
        obs_rest_padded = np.full((N, max_obs), np.nan, dtype=np.float64)
        for i in range(N):
            obs_rest_padded[i, : n_obs[i]] = obs_rest_list[i]

        # Vectorized interp: for each filter, compute interpolated mags for all (N, max_obs)
        # model_times_batch is (N, T) — each row is a sorted time grid
        T = model_times_batch.shape[1]

        # Clamp obs times to grid range for each transient (shared across bands)
        t_min = model_times_batch[:, 0:1]   # (N, 1)
        t_max = model_times_batch[:, -1:]   # (N, 1)
        obs_clamped = np.clip(obs_rest_padded, t_min, t_max)

        # Vectorized per-row searchsorted: flatten rows, offset by row index, searchsorted once.
        # Each row's grid is offset by row_idx * T in a flattened 1D array.
        flat_grid = model_times_batch.ravel()  # (N*T,)
        row_offsets = (np.arange(N) * T)[:, None]  # (N, 1)
        flat_obs = obs_clamped + 0  # copy
        # For each (i, j), we want searchsorted(model_times_batch[i], obs_clamped[i, j]).
        # Trick: offset each row's obs values to make them searchable in the flat grid.
        # Instead, just use a row-wise loop with numpy (faster than broadcast for large T).
        idx = np.empty((N, max_obs), dtype=np.int64)
        for i in range(N):
            idx[i] = np.clip(
                np.searchsorted(model_times_batch[i], obs_clamped[i]) - 1, 0, T - 2
            )

        # Gather grid values at idx and idx+1 for linear interpolation
        t0 = np.take_along_axis(model_times_batch, idx, axis=1)  # (N, max_obs)
        t1 = np.take_along_axis(model_times_batch, np.minimum(idx + 1, T - 1), axis=1)

        # Linear interpolation weight (shared across bands)
        dt = t1 - t0
        dt = np.where(dt == 0, 1.0, dt)
        w = (obs_clamped - t0) / dt

        interp_results = {}
        for fiesta_filt, short_band in _FIESTA_TO_BAND.items():
            if fiesta_filt not in model_mags_np:
                continue
            grid_mags = model_mags_np[fiesta_filt]  # (N, T)
            m0 = np.take_along_axis(grid_mags, idx, axis=1)
            m1 = np.take_along_axis(grid_mags, np.minimum(idx + 1, T - 1), axis=1)
            interp_results[short_band] = m0 + w * (m1 - m0)  # (N, max_obs)

        # Build results list — slice each transient's actual obs count
        results = []
        for i in range(N):
            ni = n_obs[i]
            result_mags = {}
            for band, mags_arr in interp_results.items():
                result_mags[band] = mags_arr[i, :ni].tolist()

            unique_bands = set(obs_bands_list[i])
            for band in unique_bands:
                if band not in result_mags:
                    result_mags[band] = [99.0] * ni

            results.append((obs_times_list[i].tolist(), result_mags))

        return results
