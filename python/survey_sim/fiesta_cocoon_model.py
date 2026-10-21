"""Relativistic shocked cocoon analytical model adapter for survey-sim.

Wraps the fiestaEM / Redback shocked cocoon cooling equations
(Piro & Kollmeier 2018; Gottlieb et al. 2018) to produce per-band apparent
magnitudes at survey observation times.
"""

import math
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

# Band effective wavelengths in Angstroms
BAND_WAVELENGTHS = {
    "u": 3670.0,
    "g": 4825.0,
    "r": 6220.0,
    "i": 7545.0,
    "z": 8680.0,
    "y": 9620.0,
    "J": 12350.0,
    "H": 16620.0,
    "K": 21590.0,
}

_BAND_TO_FIESTA = {
    "g": "ztfg",
    "r": "ztfr",
    "i": "ztfi",
    "u": "lsstu",
    "z": "lsstz",
    "y": "lssty",
}
_FIESTA_TO_BAND = {v: k for k, v in _BAND_TO_FIESTA.items()}

# Constants
c_cgs = 2.99792458e10
h_cgs = 6.62607015e-27
k_b_cgs = 1.380649e-16
sigma_sb = 5.670374e-5
m_sun_g = 1.98847e33
day_to_s = 86400.0


def _cocoon_flux_single(t_day, log10_mej, log10_vej, eta, log10_tshock,
                        shocked_fraction, cos_theta_cocoon, log10_kappa,
                        band_wave_aa, d_l_mpc, z):
    """Algebraic shocked cocoon solution for a single epoch."""
    t_days = max(float(t_day), 1e-4)

    f_sh = max(float(shocked_fraction), 1e-12)
    cos_theta = max(min(float(cos_theta_cocoon), 1.0 - 1e-6), -1.0 + 1e-6)
    theta = math.acos(cos_theta)

    log10_c = math.log10(c_cgs)
    log10_4pi = math.log10(4.0 * math.pi)
    log10_msun = math.log10(m_sun_g)

    log10_vej_cms = float(log10_vej) + log10_c
    log10_rshock = float(log10_tshock) + log10_c
    log10_msh_g = math.log10(f_sh) + float(log10_mej) + log10_msun

    vej_cms = 10.0 ** log10_vej_cms

    # Diffusion timescale
    log10_tau_diff_s = 0.5 * (
        log10_msun + float(log10_kappa) + math.log10(f_sh) + float(log10_mej)
        - log10_4pi - log10_c - log10_vej_cms
    )
    tau_diff = (10.0 ** log10_tau_diff_s) / day_to_s

    # Transition time
    t_thin = math.sqrt(c_cgs / vej_cms) * tau_diff

    # Luminosity scale
    log10_theta_factor = (1.0 / 3.0) * math.log10(max(theta**2 / 2.0, 1e-12))
    log10_l_scale = (
        log10_theta_factor + log10_msh_g + log10_vej_cms + log10_rshock - 2.0 * log10_tau_diff_s
    )

    power_term = -4.0 / (float(eta) + 2.0) * math.log10(max(t_days / tau_diff, 1e-10))
    tanh_term = math.log10(max((1.0 + math.tanh(t_thin - t_days)) / 2.0, 1e-30))

    log10_lum = max(log10_l_scale + power_term + tanh_term, 35.0)
    lum = 10.0 ** log10_lum

    # Photospheric radius
    log10_v_phot = log10_vej_cms + (-2.0 / (float(eta) + 3.0)) * math.log10(max(t_days / t_thin, 1e-10))
    log10_r = log10_v_phot + math.log10(t_days) + math.log10(day_to_s)
    r_phot = 10.0 ** log10_r

    # Effective temperature
    t_eff = (lum / (4.0 * math.pi * (r_phot ** 2) * sigma_sb)) ** 0.25
    t_eff = max(min(t_eff, 100000.0), 2000.0)

    # Rest-frame frequency
    rest_wave_cm = (band_wave_aa / (1.0 + z)) * 1e-8
    nu = c_cgs / rest_wave_cm

    exp_factor = max(min(h_cgs * nu / (k_b_cgs * t_eff), 80.0), 1e-4)
    b_nu = (2.0 * h_cgs * nu ** 3 / c_cgs ** 2) / (math.exp(exp_factor) - 1.0)

    d_l_cm = d_l_mpc * 3.0857e24
    f_nu = (1.0 + z) * math.pi * b_nu * ((r_phot / d_l_cm) ** 2)
    f_nu = max(f_nu, 1e-40)

    m_ab = -2.5 * math.log10(f_nu) - 48.60
    return max(min(m_ab, 35.0), 10.0)


class FiestaCocoonModel:
    """Shocked cocoon light-curve model adapter.

    Wraps fiestaEM's ShockedCocoonModel when available, or executes the pure-Python
    analytical solution.
    """

    def __init__(self, filters=None, use_fiesta=True):
        self.filters = filters or ["g", "r", "i"]
        self.use_fiesta = False
        self.model = None

        if use_fiesta:
            try:
                from fiesta.inference.analytical_models.shock_powered_models import ShockedCocoonModel
                fiesta_filts = [_BAND_TO_FIESTA.get(b, b) for b in self.filters]
                self.model = ShockedCocoonModel(filters=fiesta_filts)
                self.use_fiesta = True
            except Exception:
                self.use_fiesta = False

    def predict(self, params):
        raw_obs_times = list(params["_obs_times_mjd"])
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

        rest_days = [(t - t_exp) / (1.0 + redshift) for t in raw_obs_times]
        result_mags = {}

        unique_bands = set(obs_bands)
        for band in unique_bands:
            wave_aa = BAND_WAVELENGTHS.get(band, 5000.0)
            mags = []
            for rd in rest_days:
                if rd <= 0.0:
                    mags.append(99.0)
                else:
                    m = _cocoon_flux_single(
                        rd, log10_mej, log10_vej, eta, log10_tshock,
                        shocked_fraction, cos_theta_cocoon, log10_kappa,
                        wave_aa, d_l, redshift
                    )
                    mags.append(m)
            result_mags[band] = mags

        return [float(t) for t in raw_obs_times], result_mags
