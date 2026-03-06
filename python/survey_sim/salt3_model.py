"""SALT3 SN Ia model adapter for survey-sim.

Wraps fiesta's SALT3Model (JAX-native, JIT-compiled) to produce per-band
apparent magnitudes for use with PythonCallbackModel.
"""

import numpy as np
import jax.numpy as jnp

# Default band name mappings: survey-sim short names -> jax_supernovae filter names.
ZTF_SALT3_BANDS = {"g": "ztfg", "r": "ztfr", "i": "ztfi"}
LSST_SALT3_BANDS = {
    "u": "lsstu", "g": "lsstg", "r": "lsstr",
    "i": "lssti", "z": "lsstz", "y": "lssty",
}


def _ensure_bandpass_registered(name):
    """Register a bandpass from sncosmo if not already in jax_supernovae."""
    from jax_supernovae.bandpasses import get_bandpass, register_bandpass, Bandpass
    try:
        get_bandpass(name)
    except (ValueError, KeyError):
        import sncosmo
        b = sncosmo.get_bandpass(name)
        bp = Bandpass(wave=jnp.array(b.wave), trans=jnp.array(b.trans), name=name)
        register_bandpass(name, bp)


class FiestaSALT3Model:
    """Adapter wrapping fiesta SALT3Model for survey-sim.

    Parameters
    ----------
    filters : list[str] or None
        Survey-sim band names to support. Default: ["g", "r", "i"] (ZTF).
    times_grid : array-like or None
        Rest-frame time grid in days since peak. Default: -20 to +50, 200 points.
    """

    def __init__(self, filters=None, band_map=None, times_grid=None):
        if band_map is None:
            band_map = ZTF_SALT3_BANDS
        if filters is None:
            filters = list(band_map.keys())
        self.band_map = band_map
        self.survey_filters = filters
        self.salt3_filters = [band_map[f] for f in filters]

        # Register any missing bandpasses from sncosmo
        for filt in self.salt3_filters:
            _ensure_bandpass_registered(filt)

        if times_grid is None:
            times_grid = jnp.linspace(-20, 50, 200)
        self.times_grid = jnp.array(times_grid)

        # Will be lazily initialized per redshift
        self._models = {}

    def _get_model(self, z):
        """Get or create a SALT3Model for a given redshift (rounded to 0.01)."""
        z_key = round(float(z), 2)
        if z_key not in self._models:
            from fiesta.inference.analytical_models.salt3_models import SALT3Model
            self._models[z_key] = SALT3Model(
                filters=self.salt3_filters,
                times=self.times_grid,
                redshift=z_key,
            )
        return self._models[z_key]

    def warm_up(self, z_max=0.12, dz=0.01, batch_size=1024):
        """Pre-compile SALT3 models (scalar + vmap) for a grid of redshifts.

        The vmap path is compiled with a fixed batch_size so subsequent calls
        with different counts don't trigger recompilation (we pad to this size).
        """
        import jax
        import time
        zs = np.arange(dz, z_max + dz, dz)
        self._batch_size = batch_size
        print(f"Pre-compiling SALT3 for {len(zs)} redshift bins (batch={batch_size})...",
              end=" ", flush=True)
        t0 = time.time()
        dummy_params = {"log10_x0": -3.5, "x1": 0.0, "c": 0.0, "t0": 0.0}
        dummy_batch = {k: jnp.full(batch_size, v) for k, v in dummy_params.items()}
        self._vpredicts = {}
        for z in zs:
            m = self._get_model(z)
            m.predict(dummy_params)  # scalar JIT
            vp = jax.vmap(m.predict)
            vp(dummy_batch)  # compile with fixed batch size
            self._vpredicts[round(float(z), 2)] = vp

        # Force results to materialize (JAX dispatch is async)
        for z_key, vp in self._vpredicts.items():
            out = vp(dummy_batch)
            jax.block_until_ready(out)

        # Warm the full batch_predict path with a dummy call
        dummy_list = [{
            "x1": 0.5, "c": 0.02, "redshift": float(z), "peak_abs_mag": -19.3,
            "luminosity_distance": float(z) * 4500.0,
            "_obs_times_mjd": [58200.0 + j for j in range(20)],
            "_obs_bands": ["g", "r"] * 10, "_t_exp": 58200.0,
        } for z in zs for _ in range(10)]
        self.batch_predict(dummy_list)
        print(f"done ({time.time()-t0:.1f}s)")

    @staticmethod
    def _compute_log10_x0(params, d_l_mpc, x1, c,
                           M_B=-19.3, alpha=-0.14, beta=3.15):
        """Compute log10(x0) from the Tripp standardization relation.

        m_B = M_B + mu - alpha*x1 + beta*c
        m_B = -2.5 * log10(x0) + 10.682  (SALT2/3 convention)
        => log10(x0) = -(m_B - 10.682) / 2.5
        """
        # Use per-instance peak_abs_mag if available, else default M_B
        M = float(params.get("peak_abs_mag", M_B))
        mu = 5.0 * np.log10(d_l_mpc * 1e6 / 10.0)  # distance modulus
        m_B = M + mu - alpha * x1 + beta * c
        return -(m_B - 10.682) / 2.5

    def predict(self, params):
        """Evaluate SALT3 model at observation times.

        Parameters
        ----------
        params : dict
            Must contain SALT3 parameters (``log10_x0``, ``x1``, ``c``, ``t0``),
            plus ``redshift`` and observation context keys
            ``_obs_times_mjd``, ``_obs_bands``, ``_t_exp``.

        Returns
        -------
        (list[float], dict[str, list[float]])
            Observation times (MJD) and per-band apparent magnitudes.
        """
        obs_times_mjd = np.asarray(params["_obs_times_mjd"], dtype=np.float64)
        obs_bands = list(params["_obs_bands"])
        t_exp = float(params["_t_exp"])
        redshift = float(params["redshift"])
        d_l = float(params["luminosity_distance"])  # Mpc

        x1 = float(params.get("x1", 0.0))
        c = float(params.get("c", 0.0))

        # Compute log10_x0 from absolute mag + distance + standardization
        # if not provided directly.
        if "log10_x0" in params:
            log10_x0 = float(params["log10_x0"])
        else:
            log10_x0 = self._compute_log10_x0(params, d_l, x1, c)

        # Rest-frame days since peak. t0_offset is rest-frame days from
        # explosion to B-band peak (typically ~17 days for SNe Ia).
        t0_offset = float(params.get("t0_offset", 17.0))
        obs_rest_days = (obs_times_mjd - t_exp) / (1.0 + redshift) - t0_offset

        model = self._get_model(redshift)
        salt3_params = {
            "log10_x0": log10_x0,
            "x1": x1,
            "c": c,
            "t0": 0.0,  # grid is centered on peak
        }

        # Call fiesta SALT3 — returns (times, {salt3_filter: mags_array})
        model_times, model_mags = model.predict(salt3_params)
        model_times = np.asarray(model_times, dtype=np.float64)

        # Interpolate to observation times per band
        result_mags = {}
        for salt3_filt, survey_band in zip(self.salt3_filters, self.survey_filters):
            if salt3_filt not in model_mags:
                continue
            grid_mags = np.asarray(model_mags[salt3_filt], dtype=np.float64)
            interp_mags = np.interp(obs_rest_days, model_times, grid_mags,
                                     left=99.0, right=99.0)
            result_mags[survey_band] = interp_mags.tolist()

        # Fill missing bands with 99
        for band in set(obs_bands):
            if band not in result_mags:
                result_mags[band] = [99.0] * len(obs_times_mjd)

        return (obs_times_mjd.tolist(), result_mags)

    def batch_evaluate_arrays(self, redshifts, d_ls, peak_mags,
                               model_param_names, model_param_arrays,
                               obs_times_flat, obs_bands_flat,
                               obs_counts, t_exps):
        """Fast columnar batch evaluation — avoids per-transient dict overhead.

        Parameters
        ----------
        redshifts : 1D array (N,)
        d_ls : 1D array (N,) — luminosity distances in Mpc
        peak_mags : 1D array (N,) — peak absolute magnitudes
        model_param_names : list of str — parameter names
        model_param_arrays : list of 1D arrays (N,) — one per param name
        obs_times_flat : 1D array — all obs times concatenated
        obs_bands_flat : list of str — all obs bands concatenated
        obs_counts : 1D int array (N,) — number of obs per transient
        t_exps : 1D array (N,) — explosion times

        Returns
        -------
        list of (times_list, {band: mags_list}) — one per transient
        """
        import jax

        N = len(redshifts)
        redshifts = np.asarray(redshifts, dtype=np.float64)
        d_ls = np.asarray(d_ls, dtype=np.float64)
        peak_mags = np.asarray(peak_mags, dtype=np.float64)
        t_exps = np.asarray(t_exps, dtype=np.float64)
        obs_counts = np.asarray(obs_counts, dtype=np.int64)
        obs_times_flat = np.asarray(obs_times_flat, dtype=np.float64)

        # Build param lookup
        params = {}
        for name, arr in zip(model_param_names, model_param_arrays):
            params[name] = np.asarray(arr, dtype=np.float64)

        x1s = params.get("x1", np.zeros(N))
        cs = params.get("c", np.zeros(N))

        # Compute log10_x0 vectorized
        mus = 5.0 * np.log10(d_ls * 1e6 / 10.0)
        m_Bs = peak_mags + mus - (-0.14) * x1s + 3.15 * cs
        log10_x0s = -(m_Bs - 10.682) / 2.5

        # Pre-compute offsets into flat obs arrays
        offsets = np.empty(N + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(obs_counts, out=offsets[1:])

        # Group by redshift bin and evaluate
        z_keys = np.round(redshifts, 2)
        unique_z = np.unique(z_keys)
        batch_size = getattr(self, '_batch_size', 1024)
        total_obs_all = int(obs_counts.sum())
        # Pre-allocate flat magnitude arrays per band (filled with 99 = undetected)
        all_mags_flat = {band: np.full(total_obs_all, 99.0, dtype=np.float64)
                         for band in self.survey_filters}

        for z_bin in unique_z:
            mask = z_keys == z_bin
            indices = np.where(mask)[0]
            n_bin = len(indices)

            model = self._get_model(z_bin)
            vpredict = getattr(self, '_vpredicts', {}).get(z_bin)
            if vpredict is None:
                vpredict = jax.vmap(model.predict)

            for chunk_start in range(0, n_bin, batch_size):
                chunk_end = min(chunk_start + batch_size, n_bin)
                chunk_indices = indices[chunk_start:chunk_end]
                n_chunk = len(chunk_indices)

                def _pad(arr):
                    a = jnp.array(arr, dtype=jnp.float32)
                    if len(a) >= batch_size:
                        return a
                    return jnp.concatenate([a, jnp.zeros(batch_size - len(a), dtype=jnp.float32)])

                batched_params = {
                    "log10_x0": _pad(log10_x0s[chunk_indices]),
                    "x1": _pad(x1s[chunk_indices]),
                    "c": _pad(cs[chunk_indices]),
                    "t0": jnp.zeros(batch_size, dtype=jnp.float32),
                }

                model_times_batch, model_mags_batch = vpredict(batched_params)
                grid_t = np.asarray(model_times_batch[0], dtype=np.float64)
                T = len(grid_t)
                model_mags_np = {k: np.asarray(v[:n_chunk], dtype=np.float64)
                                 for k, v in model_mags_batch.items()}

                # Flat vectorized interpolation — no per-transient Python loop
                n_obs_chunk = obs_counts[chunk_indices]
                total_obs = int(n_obs_chunk.sum())

                # Build flat obs times for this chunk
                chunk_obs_slices = [slice(offsets[gi], offsets[gi] + obs_counts[gi])
                                    for gi in chunk_indices]
                obs_flat = np.concatenate([obs_times_flat[s] for s in chunk_obs_slices])

                # Transient index for each flat obs (vectorized via np.repeat)
                local_trans_idx = np.repeat(np.arange(n_chunk), n_obs_chunk)
                global_trans_idx = chunk_indices[local_trans_idx]

                # Rest-frame times (fully vectorized)
                rest_flat = (obs_flat - t_exps[global_trans_idx]) / (1.0 + redshifts[global_trans_idx]) - 17.0

                # Clamp and interpolate on flat arrays
                rest_clamped = np.clip(rest_flat, grid_t[0], grid_t[-1])
                grid_idx = np.clip(np.searchsorted(grid_t, rest_clamped) - 1, 0, T - 2)
                t0v = grid_t[grid_idx]
                t1v = grid_t[np.minimum(grid_idx + 1, T - 1)]
                dt = np.where((t1v - t0v) == 0, 1.0, t1v - t0v)
                w = (rest_clamped - t0v) / dt

                # Interpolate per band (flat indexing into 2D grid_mags)
                # Clamped interpolation: values outside grid get edge values
                interp_flat = {}
                for salt3_filt, survey_band in zip(self.salt3_filters, self.survey_filters):
                    if salt3_filt not in model_mags_np:
                        continue
                    gm = model_mags_np[salt3_filt]  # (n_chunk, T)
                    m0 = gm[local_trans_idx, grid_idx]
                    m1 = gm[local_trans_idx, np.minimum(grid_idx + 1, T - 1)]
                    mags = m0 + w * (m1 - m0)
                    interp_flat[survey_band] = mags

                # Scatter flat results into global flat arrays
                # Build scatter indices: global flat positions for this chunk's obs
                chunk_global_slices = np.concatenate([
                    np.arange(offsets[gi], offsets[gi] + obs_counts[gi])
                    for gi in chunk_indices
                ])
                for band, mags in interp_flat.items():
                    all_mags_flat[band][chunk_global_slices] = mags

        # Return flat arrays: (obs_times_flat, {band: flat_mags_array}, obs_counts)
        # The Rust side splits using obs_counts.
        return (obs_times_flat, all_mags_flat, obs_counts)

    def batch_predict(self, params_list, chunk_size=4096):
        """Batch evaluate multiple SNe Ia using vectorized JAX evaluation.

        Groups transients by redshift bin, evaluates each group with vmap,
        then interpolates to per-transient observation times.
        """
        if not params_list:
            return []

        import jax

        N = len(params_list)

        # Pre-extract all parameters
        redshifts = np.array([float(p["redshift"]) for p in params_list])
        d_ls = np.array([float(p["luminosity_distance"]) for p in params_list])
        x1s = np.array([float(p.get("x1", 0.0)) for p in params_list])
        cs = np.array([float(p.get("c", 0.0)) for p in params_list])
        t_exps = np.array([float(p["_t_exp"]) for p in params_list])
        obs_times_list = [np.asarray(p["_obs_times_mjd"], dtype=np.float64) for p in params_list]
        obs_bands_list = [list(p["_obs_bands"]) for p in params_list]

        # Compute log10_x0 for all transients (vectorized numpy)
        peak_mags = np.array([float(p.get("peak_abs_mag", -19.3)) for p in params_list])
        mus = 5.0 * np.log10(d_ls * 1e6 / 10.0)
        m_Bs = peak_mags + mus - (-0.14) * x1s + 3.15 * cs
        log10_x0s = -(m_Bs - 10.682) / 2.5

        # Group by redshift bin
        z_keys = np.round(redshifts, 2)
        unique_z = np.unique(z_keys)

        # Pre-allocate results
        results = [None] * N

        batch_size = getattr(self, '_batch_size', 1024)

        for z_bin in unique_z:
            mask = z_keys == z_bin
            indices = np.where(mask)[0]
            n_bin = len(indices)

            model = self._get_model(z_bin)
            vpredict = getattr(self, '_vpredicts', {}).get(z_bin)
            if vpredict is None:
                vpredict = jax.vmap(model.predict)

            # Process in chunks of batch_size
            for chunk_start in range(0, n_bin, batch_size):
                chunk_end = min(chunk_start + batch_size, n_bin)
                chunk_indices = indices[chunk_start:chunk_end]
                n_chunk = len(chunk_indices)

                # Pad to batch_size for stable JIT shapes
                def _pad(arr):
                    a = jnp.array(arr, dtype=jnp.float32)
                    if len(a) >= batch_size:
                        return a
                    return jnp.concatenate([a, jnp.zeros(batch_size - len(a), dtype=jnp.float32)])

                batched_params = {
                    "log10_x0": _pad(log10_x0s[chunk_indices]),
                    "x1": _pad(x1s[chunk_indices]),
                    "c": _pad(cs[chunk_indices]),
                    "t0": jnp.zeros(batch_size, dtype=jnp.float32),
                }

                model_times_batch, model_mags_batch = vpredict(batched_params)

                grid_t = np.asarray(model_times_batch[0], dtype=np.float64)
                T = len(grid_t)

                model_mags_np = {k: np.asarray(v[:n_chunk], dtype=np.float64)
                                 for k, v in model_mags_batch.items()}

                # Vectorized interpolation for this chunk
                n_obs_chunk = np.array([len(obs_times_list[gi]) for gi in chunk_indices])
                max_obs = int(n_obs_chunk.max()) if n_chunk > 0 else 0

                obs_rest_padded = np.full((n_chunk, max_obs), np.nan, dtype=np.float64)
                for local_idx, global_idx in enumerate(chunk_indices):
                    obs_t = obs_times_list[global_idx]
                    ni = len(obs_t)
                    obs_rest_padded[local_idx, :ni] = (
                        (obs_t - t_exps[global_idx]) / (1.0 + redshifts[global_idx]) - 17.0
                    )

                obs_clamped = np.clip(obs_rest_padded, grid_t[0], grid_t[-1])
                idx = np.clip(np.searchsorted(grid_t, obs_clamped) - 1, 0, T - 2)
                t0_vals = grid_t[idx]
                t1_vals = grid_t[np.minimum(idx + 1, T - 1)]
                dt = np.where((t1_vals - t0_vals) == 0, 1.0, t1_vals - t0_vals)
                w = (obs_clamped - t0_vals) / dt
                out_of_range = np.isnan(obs_rest_padded)

                interp_results = {}
                for salt3_filt, survey_band in zip(self.salt3_filters, self.survey_filters):
                    if salt3_filt not in model_mags_np:
                        continue
                    grid_mags = model_mags_np[salt3_filt]
                    m0 = np.take_along_axis(grid_mags, idx, axis=1)
                    m1 = np.take_along_axis(grid_mags, np.minimum(idx + 1, T - 1), axis=1)
                    interp_m = m0 + w * (m1 - m0)
                    interp_m[out_of_range] = 99.0
                    interp_results[survey_band] = interp_m

                for local_idx, global_idx in enumerate(chunk_indices):
                    ni = n_obs_chunk[local_idx]
                    result_mags = {}
                    for band, mags_arr in interp_results.items():
                        result_mags[band] = mags_arr[local_idx, :ni].tolist()
                    for band in set(obs_bands_list[global_idx]):
                        if band not in result_mags:
                            result_mags[band] = [99.0] * ni
                    results[global_idx] = (obs_times_list[global_idx].tolist(), result_mags)

        return results
