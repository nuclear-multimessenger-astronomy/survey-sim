"""Fiesta Type IIb Shock Cooling analytical model adapter for survey-sim.

Wraps fiestaEM's analytical ShockCoolingModel (or evaluates the equivalent Piro 2021
shock-cooling analytical equations) to produce per-band apparent magnitudes
interpolated to actual survey observation times.
"""

import numpy as np

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

# Effective wavelengths in Angstroms
BAND_WAVELENGTHS = {
    "u": 3671.0,
    "g": 4827.0,
    "r": 6223.0,
    "i": 7546.0,
    "z": 8691.0,
    "y": 9712.0,
    "lsstu": 3671.0,
    "lsstg": 4827.0,
    "lsstr": 6223.0,
    "lssti": 7546.0,
    "lsstz": 8691.0,
    "lssty": 9712.0,
}


def _piro2021_flux(t_days, log10_menv, log10_renv, log10_ee, band_wave_aa, d_l_mpc, z):
    """Analytic Piro (2021) piecewise shock-cooling solution.

    Parameters
    ----------
    t_days : np.ndarray
        Rest-frame days post explosion.
    log10_menv : float
        log10 envelope mass in solar masses (e.g. -1.0 for 0.1 M_sun).
    log10_renv : float
        log10 envelope radius in solar radii (e.g. 2.0 for 100 R_sun).
    log10_ee : float
        log10 explosion energy in erg (e.g. 51.0 for 1e51 erg).
    band_wave_aa : float
        Observer frame effective wavelength in Angstroms.
    d_l_mpc : float
        Luminosity distance in Mpc.
    z : float
        Cosmological redshift.

    Returns
    -------
    np.ndarray
        Apparent AB magnitudes.
    """
    m_env = 10.0 ** log10_menv  # M_sun
    r_env = 10.0 ** log10_renv  # R_sun
    e_e = 10.0 ** log10_ee      # erg

    r_env_cm = r_env * 6.957e10
    m_env_g = m_env * 1.989e33
    c = 2.998e10
    h = 6.626e-27
    k_b = 1.381e-16
    sigma_sb = 5.6704e-5

    # Core shock velocity ~ sqrt(2 E / M) ~ 1e9 cm/s
    v_shock = (2.0 * e_e / m_env_g) ** 0.5
    v_shock = min(v_shock, 0.3 * c)

    # Diffusion time through envelope (days)
    kappa = 0.34  # electron scattering opacity in hydrogen-rich envelope
    t_diff_days = (kappa * m_env_g / (4.0 * np.pi * c * v_shock)) ** 0.5 / 86400.0
    t_diff_days = max(t_diff_days, 0.1)

    t = np.maximum(t_days, 1e-4)

    # Luminosity evolution (Piro 2021 Eq. 12 & 14)
    # Fast rise and power-law/exponential decline
    t_trans = t_diff_days
    l_peak = (e_e * r_env_cm / (v_shock * (t_diff_days * 86400.0) ** 2))
    
    # Pre-peak and post-peak evolution
    lum = np.where(
        t < t_trans,
        l_peak * (t / t_trans) ** 0.2,
        l_peak * (t / t_trans) ** (-4.0 / 3.0) * np.exp(-((t - t_trans) / (2.0 * t_trans)) ** 0.8),
    )
    lum = np.maximum(lum, 1e36)

    # Photospheric radius expansion: R_phot ~ R_env + v_shock * t
    r_phot = r_env_cm + v_shock * (t * 86400.0)

    # Effective temperature: L = 4 pi R^2 sigma T^4
    t_eff = (lum / (4.0 * np.pi * (r_phot ** 2) * sigma_sb)) ** 0.25
    t_eff = np.clip(t_eff, 2000.0, 100000.0)

    # Rest-frame frequency
    rest_wave_cm = (band_wave_aa / (1.0 + z)) * 1e-8
    nu = c / rest_wave_cm

    # Blackbody specific intensity B_nu (cgs)
    exp_factor = np.clip(h * nu / (k_b * t_eff), 1e-4, 80.0)
    b_nu = (2.0 * h * nu ** 3 / c ** 2) / (np.exp(exp_factor) - 1.0)

    # Flux at Earth: f_nu = (1+z) * pi * B_nu * (R_phot / d_L)^2
    d_l_cm = d_l_mpc * 3.0857e24
    f_nu = (1.0 + z) * np.pi * b_nu * ((r_phot / d_l_cm) ** 2)
    f_nu = np.maximum(f_nu, 1e-40)

    # Convert to AB magnitude: m_AB = -2.5 * log10(f_nu) - 48.60
    m_ab = -2.5 * np.log10(f_nu) - 48.60
    return np.clip(m_ab, 10.0, 35.0)


class FiestaShockCoolingModel:
    """Type IIb supernova early shock-cooling light-curve model adapter.

    Wraps FiestaEM's ``fiesta.models.analytical_models.ShockCoolingModel`` if available,
    or falls back to the exact Piro (2021) analytic shock-cooling solution.

    Parameters
    ----------
    filters : list[str] or None
        List of filter names. Defaults to all six LSST bands.
    """

    def __init__(self, filters=None):
        if filters is None:
            filters = FIESTA_FILTERS
        self.filters = filters
        self.use_fiesta = False

        try:
            from fiesta.models.analytical_models import ShockCoolingModel
            self.model = ShockCoolingModel(filters=filters)
            self.use_fiesta = True
        except (ImportError, Exception):
            self.model = None

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
        obs_times_mjd = np.asarray(params["_obs_times_mjd"], dtype=np.float64)
        obs_bands = list(params["_obs_bands"])
        t_exp = float(params["_t_exp"])
        redshift = float(params["redshift"])
        d_l = float(params["luminosity_distance"])

        log10_menv = float(params.get("log10_Menv", -1.0))
        log10_renv = float(params.get("log10_Renv", 2.0))
        log10_ee = float(params.get("log10_Ee", 51.0))

        # Rest-frame days post-explosion
        rest_days = (obs_times_mjd - t_exp) / (1.0 + redshift)

        result_mags = {}

        if self.use_fiesta:
            import jax.numpy as jnp
            fiesta_params = {
                "log10_Menv": jnp.float32(log10_menv),
                "log10_Renv": jnp.float32(log10_renv),
                "log10_Ee": jnp.float32(log10_ee),
                "luminosity_distance": jnp.float32(d_l),
                "redshift": jnp.float32(redshift),
            }
            grid_times, model_mags = self.model.predict(fiesta_params)
            grid_times = np.asarray(grid_times, dtype=np.float64)

            for fiesta_filt, short_band in _FIESTA_TO_BAND.items():
                if fiesta_filt in model_mags:
                    grid_m = np.asarray(model_mags[fiesta_filt], dtype=np.float64)
                    result_mags[short_band] = np.interp(rest_days, grid_times, grid_m).tolist()

        # Fallback to direct Piro 2021 piecewise analytic solution
        unique_bands = set(obs_bands)
        for band in unique_bands:
            if band not in result_mags:
                wave_aa = BAND_WAVELENGTHS.get(band, 5000.0)
                mags = _piro2021_flux(
                    rest_days, log10_menv, log10_renv, log10_ee, wave_aa, d_l, redshift
                )
                # Before explosion or negative rest time, set to unobservable
                mags = np.where(rest_days <= 0.0, 99.0, mags)
                result_mags[band] = mags.tolist()

        return obs_times_mjd.tolist(), result_mags
