"""Fiesta Bu2026 kilonova model adapter for survey-sim.

Wraps fiestaEM's FluxModel surrogate to produce per-band apparent magnitudes
interpolated to actual survey observation times.
"""

import numpy as np
import jax.numpy as jnp

from fiesta.inference.lightcurve_model import FluxModel
from survey_sim.survey_config import ZTF_BAND_MAP, LSST_BAND_MAP, resolve_band_map

# Backward compatibility alias
_BAND_TO_FIESTA = LSST_BAND_MAP
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
    survey : str, optional
        Survey identifier (e.g. ``"ztf"`` or ``"lsst"``/``"rubin"``). Defaults to ``"ztf"``.
    filters : list[str] or None
        Explicit list of Fiesta filter names (e.g. ``['ztfg', 'ztfr']``).
    band_map : dict[str, str] or None
        Custom mapping from survey short band names to fiesta filter names.
    """

    def __init__(self, name="Bu2026_MLP", survey="ztf", filters=None, band_map=None, max_phase_days=25.0):
        self.band_map = resolve_band_map(
            survey=survey, band_map=band_map, filters=filters, default_survey="ztf"
        )
        self.reverse_map = {v: k for k, v in self.band_map.items()}
        fiesta_filters = list(self.band_map.values())
        self.model = FluxModel(name, filters=fiesta_filters)
        self.filters = fiesta_filters
        self.max_phase_days = float(max_phase_days)

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
        t_upper = min(self.max_phase_days, float(model_times[-1]))
        t_lower = max(0.0, float(model_times[0]))
        out_of_bounds = (obs_rest_days < t_lower) | (obs_rest_days > t_upper)
        result_mags = {}
        for fiesta_filt, short_band in self.reverse_map.items():
            if fiesta_filt not in model_mags:
                continue
            grid_mags = np.asarray(model_mags[fiesta_filt], dtype=np.float64)
            interp_all = np.interp(obs_rest_days, model_times, grid_mags, left=99.0, right=99.0)
            interp_all = np.where(out_of_bounds, 99.0, interp_all)
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

        # Clamp obs times to grid range for interpolation, but mask out-of-bounds to 99.0
        t_min = model_times_batch[:, 0:1]   # (N, 1)
        t_max = model_times_batch[:, -1:]   # (N, 1)
        t_upper = np.minimum(t_max, self.max_phase_days)
        t_lower = np.maximum(0.0, t_min)
        out_of_bounds = (
            np.isnan(obs_rest_padded)
            | (obs_rest_padded < t_lower)
            | (obs_rest_padded > t_upper)
        )
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
        for fiesta_filt, short_band in self.reverse_map.items():
            if fiesta_filt not in model_mags_np:
                continue
            grid_mags = model_mags_np[fiesta_filt]  # (N, T)
            m0 = np.take_along_axis(grid_mags, idx, axis=1)
            m1 = np.take_along_axis(grid_mags, np.minimum(idx + 1, T - 1), axis=1)
            interp_mags = m0 + w * (m1 - m0)  # (N, max_obs)
            interp_results[short_band] = np.where(out_of_bounds, 99.0, interp_mags)

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

    def batch_evaluate_arrays(self, redshifts, d_ls, peak_mags,
                              param_names, param_arrays,
                              obs_times_flat, obs_bands_flat, obs_counts,
                              t_exps):
        """Fast columnar batch evaluation — avoids per-transient Python dicts.

        Called by PythonCallbackModel.batch_evaluate_columnar() on the Rust side.

        Returns
        -------
        (obs_times_flat, {band: flat_mags}, obs_counts)
        """
        N = len(redshifts)
        if N == 0:
            return ([], {}, [])

        redshifts = np.asarray(redshifts, dtype=np.float64)
        d_ls = np.asarray(d_ls, dtype=np.float64)
        t_exps = np.asarray(t_exps, dtype=np.float64)
        obs_times_flat = np.asarray(obs_times_flat, dtype=np.float64)
        obs_counts = np.asarray(obs_counts, dtype=np.int64)
        param_names = list(param_names)
        param_dict = {name: np.asarray(arr, dtype=np.float64) for name, arr in zip(param_names, param_arrays)}

        # Vectorized parameter mapping
        X_all = {}
        for name in self.model.parameter_names:
            if name in param_dict:
                X_all[name] = jnp.array(param_dict[name], dtype=jnp.float32)
            else:
                X_all[name] = jnp.zeros(N, dtype=jnp.float32)
        X_all["luminosity_distance"] = jnp.array(d_ls, dtype=jnp.float32)
        X_all["redshift"] = jnp.array(redshifts, dtype=jnp.float32)

        # GPU vmap call in chunks
        CHUNK = 2048
        time_chunks = []
        mag_chunks = {filt: [] for filt in self.reverse_map}
        for c0 in range(0, N, CHUNK):
            c1 = min(c0 + CHUNK, N)
            X_chunk = {k: v[c0:c1] for k, v in X_all.items()}
            t_chunk, m_chunk = self.model.vpredict(X_chunk)
            time_chunks.append(np.asarray(t_chunk, dtype=np.float64))
            for filt in self.reverse_map:
                if filt in m_chunk:
                    mag_chunks[filt].append(np.asarray(m_chunk[filt], dtype=np.float64))

        model_times_batch = np.concatenate(time_chunks, axis=0)  # (N, T)
        del time_chunks
        model_mags_np = {filt: np.concatenate(chunks, axis=0)
                         for filt, chunks in mag_chunks.items() if chunks}
        del mag_chunks, X_all

        T = model_times_batch.shape[1]

        # Build flat rest-frame times for all observations.
        offsets = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(obs_counts, out=offsets[1:])
        total_obs = int(offsets[-1])

        trans_idx = np.repeat(np.arange(N), obs_counts)
        z_per_obs = redshifts[trans_idx]
        texp_per_obs = t_exps[trans_idx]
        obs_rest_flat = (obs_times_flat - texp_per_obs) / (1.0 + z_per_obs)

        t_min_per_obs = model_times_batch[trans_idx, 0]
        t_max_per_obs = model_times_batch[trans_idx, -1]
        t_upper_per_obs = np.minimum(t_max_per_obs, self.max_phase_days)
        t_lower_per_obs = np.maximum(0.0, t_min_per_obs)
        out_of_bounds_flat = (obs_rest_flat < t_lower_per_obs) | (obs_rest_flat > t_upper_per_obs)
        obs_clamped = np.clip(obs_rest_flat, t_min_per_obs, t_max_per_obs)
        del t_min_per_obs, t_max_per_obs, t_upper_per_obs, t_lower_per_obs

        idx = np.empty(total_obs, dtype=np.int64)
        for i in range(N):
            s, e = offsets[i], offsets[i + 1]
            if s < e:
                idx[s:e] = np.clip(
                    np.searchsorted(model_times_batch[i], obs_clamped[s:e]) - 1, 0, T - 2
                )

        idx1 = np.minimum(idx + 1, T - 1)

        t0 = model_times_batch[trans_idx, idx]
        t1 = model_times_batch[trans_idx, idx1]
        dt = np.where((t1 - t0) == 0, 1.0, t1 - t0)
        w = (obs_clamped - t0) / dt
        del t0, t1, dt, obs_clamped

        result_mags = {}
        for fiesta_filt, short_band in self.reverse_map.items():
            if fiesta_filt not in model_mags_np:
                continue
            m_grid = model_mags_np[fiesta_filt]
            m0 = m_grid[trans_idx, idx]
            m1 = m_grid[trans_idx, idx1]
            interp_vals = m0 + w * (m1 - m0)
            interp_vals = np.where(out_of_bounds_flat, 99.0, interp_vals)
            result_mags[short_band] = interp_vals.tolist()

        return (obs_times_flat.tolist(), result_mags, obs_counts.tolist())
