"""Fiesta GRB afterglow model adapter for survey-sim.

Wraps fiestaEM's FluxModel surrogate (blastwave_gaussian_CVAE or
blastwave_rs_gaussian_CVAE) to produce per-band apparent magnitudes
interpolated to actual survey observation times.
"""

import numpy as np
import jax
import jax.numpy as jnp

from fiesta.inference.lightcurve_model import FluxModel

# Default band mappings for common surveys.
ZTF_BAND_MAP = {"g": "ztfg", "r": "ztfr", "i": "ztfi"}
LSST_BAND_MAP = {"u": "lsstu", "g": "lsstg", "r": "lsstr", "i": "lssti", "z": "lsstz", "y": "lssty"}

# Gamma-ray efficiency for converting E_gamma_iso -> E_kinetic_iso.
ETA_GAMMA = 0.15


class FiestaAfterglowModel:
    """Adapter wrapping fiesta FluxModel for GRB afterglow surrogates.

    Parameters
    ----------
    name : str
        Fiesta surrogate name. Default: ``"blastwave_gaussian_CVAE"`` (FS only).
    band_map : dict[str, str]
        Mapping from survey-sim short band names to sncosmo/fiesta filter names.
    eta : float
        Gamma-ray efficiency for E_kinetic = E_gamma * (1-eta)/eta.
    default_duration : float
        Default log10(duration) in seconds for RS model.
    """

    # CSV column name -> fiesta parameter name mapping
    _CSV_TO_FIESTA = {
        "theta_v": "inclination_EM",
        "logn0": "log10_n0",
        "p": "p",
        "logepse": "log10_epsilon_e",
        "logepsB": "log10_epsilon_B",
    }

    def __init__(self, name="blastwave_gaussian_CVAE", band_map=None,
                 eta=ETA_GAMMA, default_duration=4.0):
        if band_map is None:
            band_map = ZTF_BAND_MAP
        self.band_map = band_map
        self.reverse_map = {v: k for k, v in band_map.items()}
        fiesta_filters = list(band_map.values())

        self.model = FluxModel(name, filters=fiesta_filters)
        self.eta = eta
        self.default_duration = default_duration
        self.has_rs = "log10_RB" in self.model.parameter_names

        # Store parameter bounds for vectorized clamping.
        self.param_bounds = {}
        for k, v in self.model.parameter_distributions.items():
            self.param_bounds[k] = (v[0], v[1])

        # Warm up JIT with a dummy call.
        self._warmup()

    def _warmup(self):
        """Pre-compile JIT for scalar and vmap paths."""
        dummy = {}
        for name in self.model.parameter_names:
            lo, hi = self.param_bounds.get(name, (0.0, 1.0))
            dummy[name] = jnp.float32((lo + hi) / 2)
        dummy["luminosity_distance"] = jnp.float32(100.0)
        dummy["redshift"] = jnp.float32(0.1)
        _ = self.model.predict(dummy)

        # Warmup vmap with batch of 2.
        X = {}
        for name in self.model.parameter_names:
            lo, hi = self.param_bounds.get(name, (0.0, 1.0))
            X[name] = jnp.array([(lo + hi) / 2, (lo + hi) / 2], dtype=jnp.float32)
        X["luminosity_distance"] = jnp.array([100.0, 100.0], dtype=jnp.float32)
        X["redshift"] = jnp.array([0.1, 0.1], dtype=jnp.float32)
        _ = self.model.vpredict(X)

    def _map_params_vectorized(self, param_names, param_arrays, N):
        """Vectorized parameter mapping from CSV columns to fiesta params.

        Parameters
        ----------
        param_names : list[str]
            CSV column names.
        param_arrays : list[array]
            Parallel arrays of parameter values, each length N.
        N : int
            Number of transients.

        Returns
        -------
        dict[str, jnp.array]
            Fiesta parameter arrays ready for vpredict.
        """
        # Build lookup: csv_name -> numpy array
        csv = {name: np.asarray(arr, dtype=np.float64) for name, arr in zip(param_names, param_arrays)}
        eta = self.eta

        # Derived parameters
        eiso = csv.get("Eiso", np.ones(N))
        log10_E0 = np.log10(eiso * (1.0 - eta) / eta)
        gamma_0 = csv.get("Gamma_0", np.full(N, 100.0))
        log10_lf = np.log10(gamma_0)
        logthc = csv.get("logthc", np.zeros(N))
        theta_core = np.power(10.0, logthc)

        fiesta = {}
        fiesta["inclination_EM"] = csv.get("theta_v", np.zeros(N))
        fiesta["log10_E0"] = log10_E0
        fiesta["thetaCore"] = theta_core
        fiesta["log10_n0"] = csv.get("logn0", np.zeros(N))
        fiesta["p"] = csv.get("p", np.full(N, 2.3))
        fiesta["log10_epsilon_e"] = csv.get("logepse", np.full(N, -1.0))
        fiesta["log10_epsilon_B"] = csv.get("logepsB", np.full(N, -2.0))
        fiesta["log10_lf"] = log10_lf

        if self.has_rs:
            logepsB_rvs = csv.get("logepsB_rvs", csv.get("logepsB", np.full(N, -2.0)))
            logepsB = csv.get("logepsB", np.full(N, -2.0))
            fiesta["log10_RB"] = logepsB_rvs - logepsB
            fiesta["log10_duration"] = csv.get(
                "log10_duration", np.full(N, self.default_duration)
            )

        # Clamp to surrogate bounds
        for name in fiesta:
            if name in self.param_bounds:
                lo, hi = self.param_bounds[name]
                fiesta[name] = np.clip(fiesta[name], lo, hi)

        # Convert to jnp float32
        return {k: jnp.array(v, dtype=jnp.float32) for k, v in fiesta.items()}

    def predict(self, params):
        """Single-transient evaluation (for PythonCallbackModel fallback)."""
        obs_times_mjd = np.asarray(params["_obs_times_mjd"], dtype=np.float64)
        t_exp = float(params["_t_exp"])
        redshift = float(params["redshift"])
        obs_rest_days = (obs_times_mjd - t_exp) / (1.0 + redshift)

        # Map params
        eta = self.eta
        eiso = float(params["Eiso"])
        fiesta_params = {
            "inclination_EM": jnp.float32(float(params.get("theta_v", 0.0))),
            "log10_E0": jnp.float32(np.log10(eiso * (1.0 - eta) / eta)),
            "thetaCore": jnp.float32(10.0 ** float(params["logthc"])),
            "log10_n0": jnp.float32(float(params["logn0"])),
            "p": jnp.float32(float(params["p"])),
            "log10_epsilon_e": jnp.float32(float(params["logepse"])),
            "log10_epsilon_B": jnp.float32(float(params["logepsB"])),
            "log10_lf": jnp.float32(np.log10(float(params["Gamma_0"]))),
            "luminosity_distance": jnp.float32(params["luminosity_distance"]),
            "redshift": jnp.float32(redshift),
        }
        # Clamp
        for name in fiesta_params:
            if name in self.param_bounds:
                lo, hi = self.param_bounds[name]
                fiesta_params[name] = jnp.clip(fiesta_params[name], lo, hi)

        if self.has_rs:
            log10_RB = float(params.get("logepsB_rvs", params["logepsB"])) - float(params["logepsB"])
            fiesta_params["log10_RB"] = jnp.float32(np.clip(log10_RB, *self.param_bounds.get("log10_RB", (-10, 10))))
            fiesta_params["log10_duration"] = jnp.float32(np.clip(
                float(params.get("log10_duration", self.default_duration)),
                *self.param_bounds.get("log10_duration", (0, 10))
            ))

        model_times, model_mags = self.model.predict(fiesta_params)
        model_times = np.asarray(model_times, dtype=np.float64)

        result_mags = {}
        for fiesta_filt, short_band in self.reverse_map.items():
            if fiesta_filt not in model_mags:
                continue
            grid_mags = np.asarray(model_mags[fiesta_filt], dtype=np.float64)
            result_mags[short_band] = np.interp(obs_rest_days, model_times, grid_mags).tolist()

        for band in set(params["_obs_bands"]):
            if band not in result_mags:
                result_mags[band] = [99.0] * len(obs_times_mjd)

        return (obs_times_mjd.tolist(), result_mags)

    def batch_predict(self, params_list, chunk_size=4096):
        """Batch evaluate using JAX vmap (legacy dict-based path)."""
        if not params_list:
            return []
        results = []
        for start in range(0, len(params_list), chunk_size):
            chunk = params_list[start : start + chunk_size]
            chunk_results = self._vpredict_chunk(chunk)
            results.extend(chunk_results)
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
        redshifts = np.asarray(redshifts, dtype=np.float64)
        d_ls = np.asarray(d_ls, dtype=np.float64)
        t_exps = np.asarray(t_exps, dtype=np.float64)
        obs_times_flat = np.asarray(obs_times_flat, dtype=np.float64)
        obs_counts = np.asarray(obs_counts, dtype=np.int64)
        param_names = list(param_names)
        param_arrays = [np.asarray(a, dtype=np.float64) for a in param_arrays]

        # Vectorized parameter mapping (no Python loops over N).
        X_all = self._map_params_vectorized(param_names, param_arrays, N)
        X_all["luminosity_distance"] = jnp.array(d_ls, dtype=jnp.float32)
        X_all["redshift"] = jnp.array(redshifts, dtype=jnp.float32)

        # GPU vmap call in chunks to avoid OOM.
        # Scale chunk size by number of filters (6 LSST bands needs smaller chunks).
        CHUNK = max(512, 4096 // max(len(self.reverse_map), 1))
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
        # obs_times_flat has sum(obs_counts) entries.
        # For each transient i, obs indices are [offset:offset+count].
        offsets = np.zeros(N + 1, dtype=np.int64)
        np.cumsum(obs_counts, out=offsets[1:])
        total_obs = int(offsets[-1])

        # Repeat redshift/t_exp per observation for vectorized rest-frame conversion.
        trans_idx = np.repeat(np.arange(N), obs_counts)  # which transient each obs belongs to
        z_per_obs = redshifts[trans_idx]
        texp_per_obs = t_exps[trans_idx]
        obs_rest_flat = (obs_times_flat - texp_per_obs) / (1.0 + z_per_obs)

        # For each observation, interpolate from the model grid of its transient.
        # model_times_batch[i] is the time grid for transient i.
        # Use trans_idx to gather the right grid row.

        # Clamp obs times to grid range.
        t_min_per_obs = model_times_batch[trans_idx, 0]
        t_max_per_obs = model_times_batch[trans_idx, -1]
        obs_clamped = np.clip(obs_rest_flat, t_min_per_obs, t_max_per_obs)
        del t_min_per_obs, t_max_per_obs

        # Searchsorted per-transient to find interpolation index.
        idx = np.empty(total_obs, dtype=np.int64)
        for i in range(N):
            s, e = offsets[i], offsets[i + 1]
            if s < e:
                idx[s:e] = np.clip(
                    np.searchsorted(model_times_batch[i], obs_clamped[s:e]) - 1, 0, T - 2
                )

        idx1 = np.minimum(idx + 1, T - 1)

        # Gather only the two grid values needed (avoids (total_obs, T) expansion).
        t0 = model_times_batch[trans_idx, idx]
        t1 = model_times_batch[trans_idx, idx1]
        dt = np.where((t1 - t0) == 0, 1.0, t1 - t0)
        w = (obs_clamped - t0) / dt
        del t0, t1, dt, obs_clamped

        # Interpolate per band using direct 2D indexing.
        result_mags = {}
        for fiesta_filt, short_band in self.reverse_map.items():
            if fiesta_filt not in model_mags_np:
                continue
            grid_mags = model_mags_np[fiesta_filt]  # (N, T)
            m0 = grid_mags[trans_idx, idx]
            m1 = grid_mags[trans_idx, idx1]
            result_mags[short_band] = (m0 + w * (m1 - m0)).tolist()

        # Fill missing bands with 99.
        all_bands = set(obs_bands_flat)
        for band in all_bands:
            if band not in result_mags:
                result_mags[band] = [99.0] * total_obs

        return (obs_times_flat.tolist(), result_mags, obs_counts.tolist())

    def _vpredict_chunk(self, params_list):
        """Legacy dict-based chunk evaluation via vpredict."""
        N = len(params_list)

        # Extract CSV param names/values from first dict, build arrays.
        csv_keys = [k for k in params_list[0] if k not in (
            "luminosity_distance", "redshift", "peak_abs_mag",
            "_obs_times_mjd", "_obs_bands", "_t_exp")]
        param_names = csv_keys
        param_arrays = [np.array([float(p.get(k, 0.0)) for p in params_list]) for k in csv_keys]

        X = self._map_params_vectorized(param_names, param_arrays, N)
        X["luminosity_distance"] = jnp.array(
            [float(p["luminosity_distance"]) for p in params_list], dtype=jnp.float32
        )
        X["redshift"] = jnp.array(
            [float(p["redshift"]) for p in params_list], dtype=jnp.float32
        )

        model_times_batch, model_mags_batch = self.model.vpredict(X)
        model_times_batch = np.asarray(model_times_batch, dtype=np.float64)
        model_mags_np = {k: np.asarray(v, dtype=np.float64) for k, v in model_mags_batch.items()}

        obs_times_list = [np.asarray(p["_obs_times_mjd"], dtype=np.float64) for p in params_list]
        t_exps = np.array([float(p["_t_exp"]) for p in params_list])
        redshifts = np.array([float(p["redshift"]) for p in params_list])
        obs_bands_list = [list(p["_obs_bands"]) for p in params_list]

        n_obs = np.array([len(t) for t in obs_times_list])
        max_obs = int(n_obs.max()) if N > 0 else 0
        T = model_times_batch.shape[1]

        obs_rest_padded = np.full((N, max_obs), np.nan, dtype=np.float64)
        for i in range(N):
            obs_rest_padded[i, : n_obs[i]] = (obs_times_list[i] - t_exps[i]) / (1.0 + redshifts[i])

        t_min = model_times_batch[:, 0:1]
        t_max = model_times_batch[:, -1:]
        obs_clamped = np.clip(obs_rest_padded, t_min, t_max)

        idx = np.empty((N, max_obs), dtype=np.int64)
        for i in range(N):
            idx[i] = np.clip(np.searchsorted(model_times_batch[i], obs_clamped[i]) - 1, 0, T - 2)

        t0 = np.take_along_axis(model_times_batch, idx, axis=1)
        t1 = np.take_along_axis(model_times_batch, np.minimum(idx + 1, T - 1), axis=1)
        dt = np.where((t1 - t0) == 0, 1.0, t1 - t0)
        w = (obs_clamped - t0) / dt

        interp_results = {}
        for fiesta_filt, short_band in self.reverse_map.items():
            if fiesta_filt not in model_mags_np:
                continue
            grid_mags = model_mags_np[fiesta_filt]
            m0 = np.take_along_axis(grid_mags, idx, axis=1)
            m1 = np.take_along_axis(grid_mags, np.minimum(idx + 1, T - 1), axis=1)
            interp_results[short_band] = m0 + w * (m1 - m0)

        results = []
        for i in range(N):
            ni = n_obs[i]
            result_mags = {band: arr[i, :ni].tolist() for band, arr in interp_results.items()}
            for band in set(obs_bands_list[i]):
                if band not in result_mags:
                    result_mags[band] = [99.0] * ni
            results.append((obs_times_list[i].tolist(), result_mags))

        return results
