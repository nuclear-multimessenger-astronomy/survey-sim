"""Relativistic shocked cocoon model adapter for survey-sim.

Wraps fiestaEM's ShockedCocoonModel to produce per-band apparent magnitudes
interpolated to actual survey observation times.
"""

import numpy as np
import jax.numpy as jnp

try:
    from fiesta.inference.analytical_models import ShockedCocoonModel
except ImportError as err:
    ShockedCocoonModel = None
    _FIESTA_IMPORT_ERROR = err

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


class FiestaCocoonModel:
    """Relativistic shocked cocoon light-curve model adapter.

    Wraps fiestaEM's ``ShockedCocoonModel`` for survey-sim. Fails with an
    ``ImportError`` if ``fiesta`` is not installed.

    Parameters
    ----------
    filters : list[str] or None
        List of filter names (e.g. ``["lsstg", "lsstr"]``).
        Defaults to all six LSST bands.
    """

    def __init__(self, filters=None):
        if ShockedCocoonModel is None:
            raise ImportError(
                "FiestaCocoonModel requires fiesta to be installed. "
                f"Original import error: {_FIESTA_IMPORT_ERROR}"
            )
        if filters is None:
            filters = FIESTA_FILTERS
        else:
            filters = [_BAND_TO_FIESTA.get(f, f) for f in filters]
        self.filters = filters
        self.model = ShockedCocoonModel(filters=filters)

    def predict(self, params):
        """Evaluate the shocked cocoon model at observation times.

        Parameters
        ----------
        params : dict
            Contains model physical parameters (``log10_mej``, ``log10_vej``,
            ``eta``, ``log10_tshock``, ``shocked_fraction``, ``cos_theta_cocoon``,
            ``log10_kappa``) plus observation context keys (``_obs_times_mjd``,
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
        d_l = float(params.get("luminosity_distance", params.get("d_l", 100.0)))

        log10_mej = float(params.get("log10_mej", -2.0))
        log10_vej = float(params.get("log10_vej", -0.5))
        eta = float(params.get("eta", 1.5))
        log10_tshock = float(params.get("log10_tshock", 2.2))
        shocked_fraction = float(params.get("shocked_fraction", 0.15))
        cos_theta_cocoon = float(params.get("cos_theta_cocoon", 0.85))
        log10_kappa = float(params.get("log10_kappa", -0.47))

        obs_rest_days = (obs_times_mjd - t_exp) / (1.0 + redshift)

        fiesta_params = {
            "log10_mej": jnp.float32(log10_mej),
            "log10_vej": jnp.float32(log10_vej),
            "eta": jnp.float32(eta),
            "log10_tshock": jnp.float32(log10_tshock),
            "shocked_fraction": jnp.float32(shocked_fraction),
            "cos_theta_cocoon": jnp.float32(cos_theta_cocoon),
            "log10_kappa": jnp.float32(log10_kappa),
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

        for band in set(obs_bands):
            if band not in result_mags:
                result_mags[band] = [99.0] * len(obs_times_mjd)

        return [float(t) for t in raw_obs_times], result_mags
