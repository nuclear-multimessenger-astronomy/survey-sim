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

from survey_sim.survey_config import ZTF_BAND_MAP, LSST_BAND_MAP, resolve_band_map

_BAND_TO_FIESTA = LSST_BAND_MAP
_FIESTA_TO_BAND = {v: k for k, v in _BAND_TO_FIESTA.items()}
FIESTA_FILTERS = list(_BAND_TO_FIESTA.values())


class FiestaShockCoolingModel:
    """Type IIb supernova early shock-cooling light-curve model adapter.

    Wraps fiestaEM's ``ShockCoolingModel`` for survey-sim. Fails with an
    ``ImportError`` if ``fiesta`` is not installed.

    Parameters
    ----------
    survey : str, optional
        Survey identifier (e.g. ``"ztf"`` or ``"lsst"``/``"rubin"``). Defaults to ``"ztf"``.
    filters : list[str] or None
        List of filter names (e.g. ``["ztfg", "ztfr"]``).
    band_map : dict[str, str] or None
        Custom mapping from short band name to fiesta filter name.
    """

    def __init__(self, survey="ztf", filters=None, band_map=None):
        if ShockCoolingModel is None:
            raise ImportError(
                "FiestaShockCoolingModel requires fiesta to be installed. "
                f"Original import error: {_FIESTA_IMPORT_ERROR}"
            )
        self.band_map = resolve_band_map(
            survey=survey, band_map=band_map, filters=filters, default_survey="ztf"
        )
        self.reverse_map = {v: k for k, v in self.band_map.items()}
        fiesta_filters = list(self.band_map.values())
        self.filters = fiesta_filters
        self.model = ShockCoolingModel(filters=fiesta_filters)

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
        for fiesta_filt, short_band in self.reverse_map.items():
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
