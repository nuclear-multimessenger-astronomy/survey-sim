"""Dense CSM interaction model adapter for survey-sim.

Wraps fiestaEM's CSMInteractionModel to produce per-band apparent magnitudes
interpolated to actual survey observation times.
"""

import numpy as np
import jax.numpy as jnp

try:
    from fiesta.inference.analytical_models import CSMInteractionModel
except ImportError as err:
    CSMInteractionModel = None
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


class FiestaCsmModel:
    """Dense CSM interaction light-curve model adapter.

    Wraps fiestaEM's ``CSMInteractionModel`` for survey-sim. Fails with an
    ``ImportError`` if ``fiesta`` is not installed.

    Parameters
    ----------
    filters : list[str] or None
        List of filter names (e.g. ``["lsstg", "lsstr"]``).
        Defaults to all six LSST bands.
    """

    def __init__(self, filters=None):
        if CSMInteractionModel is None:
            raise ImportError(
                "FiestaCsmModel requires fiesta to be installed. "
                f"Original import error: {_FIESTA_IMPORT_ERROR}"
            )
        if filters is None:
            filters = FIESTA_FILTERS
        else:
            filters = [_BAND_TO_FIESTA.get(f, f) for f in filters]
        self.filters = filters
        self.model = CSMInteractionModel(filters=filters)

    def predict(self, params):
        """Evaluate the CSM interaction model at observation times.

        Parameters
        ----------
        params : dict
            Contains model physical parameters (``log10_mej``, ``log10_csm_mass``,
            ``log10_vej``, ``eta``, ``log10_rho``, ``log10_kappa``, ``log10_r0``)
            plus observation context keys (``_obs_times_mjd``, ``_obs_bands``,
            ``_t_exp``, ``redshift``, ``luminosity_distance``).

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

        log10_mej = float(params.get("log10_mej", 0.5))
        log10_csm_mass = float(params.get("log10_csm_mass", -0.5))
        log10_vej = float(params.get("log10_vej", 3.8))
        eta = float(params.get("eta", 1.5))
        log10_rho = float(params.get("log10_rho", -12.0))
        log10_kappa = float(params.get("log10_kappa", -0.47))
        log10_r0 = float(params.get("log10_r0", 14.5))

        obs_rest_days = (obs_times_mjd - t_exp) / (1.0 + redshift)

        fiesta_params = {
            "log10_mej": jnp.float32(log10_mej),
            "log10_csm_mass": jnp.float32(log10_csm_mass),
            "log10_vej": jnp.float32(log10_vej),
            "eta": jnp.float32(eta),
            "log10_rho": jnp.float32(log10_rho),
            "log10_kappa": jnp.float32(log10_kappa),
            "log10_r0": jnp.float32(log10_r0),
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
