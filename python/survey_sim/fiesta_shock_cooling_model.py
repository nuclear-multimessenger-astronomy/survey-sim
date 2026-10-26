"""Fiesta Type IIb Shock Cooling model adapter for survey-sim.

Wraps fiestaEM's ShockCoolingModel to produce per-band apparent magnitudes
interpolated to actual survey observation times.
"""

import numpy as np
import jax.numpy as jnp

try:
    from fiesta.inference.analytical_models import ShockCoolingModel
except ImportError as err:
    ShockCoolingModel = None
    _FIESTA_IMPORT_ERROR = err

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


class FiestaShockCoolingModel:
    """Type IIb supernova early shock-cooling light-curve model adapter.

    Wraps fiestaEM's ``ShockCoolingModel`` for survey-sim. Fails with an
    ``ImportError`` if ``fiesta`` is not installed.

    Parameters
    ----------
    filters : list[str] or None
        List of filter names (e.g. ``["lsstg", "lsstr"]``).
        Defaults to all six LSST bands.
    """

    def __init__(self, filters=None):
        if ShockCoolingModel is None:
            raise ImportError(
                "FiestaShockCoolingModel requires fiesta to be installed. "
                f"Original import error: {_FIESTA_IMPORT_ERROR}"
            )
        if filters is None:
            filters = FIESTA_FILTERS
        self.filters = filters
        self.model = ShockCoolingModel(filters=filters)

    def predict(self, params):
        """Evaluate the shock-cooling model at observation times.

        Parameters
        ----------
        params : dict
            Contains model physical parameters (``log10_Menv``, ``log10_Renv``,
            ``log10_Ee``) plus observation context keys (``_obs_times_mjd``,
            ``_obs_bands``, ``_t_exp``, ``redshift``, ``luminosity_distance``).

        Returns
        -------
        (list[float], dict[str, list[float]])
            Observation times (MJD) and per-band apparent magnitudes.
        """
        raw_obs_times = list(params["_obs_times_mjd"])
        obs_times_mjd = np.asarray(raw_obs_times, dtype=np.float64)
        obs_bands = list(params["_obs_bands"])
        t_exp = float(params["_t_exp"])
        redshift = float(params["redshift"])
        d_l = float(params["luminosity_distance"])

        log10_menv = float(params.get("log10_Menv", -1.0))
        log10_renv = float(params.get("log10_Renv", 2.0))
        log10_ee = float(params.get("log10_Ee", 51.0))

        # Rest-frame days post-explosion
        obs_rest_days = (obs_times_mjd - t_exp) / (1.0 + redshift)

        fiesta_params = {
            "log10_Menv": jnp.float32(log10_menv),
            "log10_Renv": jnp.float32(log10_renv),
            "log10_Ee": jnp.float32(log10_ee),
            "luminosity_distance": jnp.float32(d_l),
            "redshift": jnp.float32(redshift),
        }

        model_times, model_mags = self.model.predict(fiesta_params)
        model_times = np.asarray(model_times, dtype=np.float64)

        result_mags = {}
        for fiesta_filt, short_band in _FIESTA_TO_BAND.items():
            if fiesta_filt in model_mags:
                grid_m = np.asarray(model_mags[fiesta_filt], dtype=np.float64)
                interp_all = np.interp(obs_rest_days, model_times, grid_m, left=99.0, right=99.0)
                interp_all = np.where(obs_rest_days <= 0.0, 99.0, interp_all)
                result_mags[short_band] = interp_all.tolist()

        # For bands not covered, fill with 99.0
        for band in set(obs_bands):
            if band not in result_mags:
                result_mags[band] = [99.0] * len(obs_times_mjd)

        return [float(t) for t in raw_obs_times], result_mags
