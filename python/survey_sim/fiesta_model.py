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
