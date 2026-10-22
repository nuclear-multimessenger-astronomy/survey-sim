"""Dense CSM interaction analytical model adapter for survey-sim.

Wraps the fiestaEM / Chevalier (1982) / Chatzopoulos et al. (2012)
circumstellar interaction equations to produce per-band apparent
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

# Physical constants (CGS)
c_cgs = 2.99792458e10
h_cgs = 6.62607015e-27
k_b_cgs = 1.380649e-16
sigma_sb = 5.670374e-5
m_sun_g = 1.98847e33
day_to_s = 86400.0


def _csm_flux_single(t_day, log10_mej, log10_csm_mass, log10_vej,
                     eta, log10_rho, log10_r0, log10_kappa,
                     band_wave_aa, d_l_mpc, z):
    """Algebraic fallback for CSM interaction emission at a single epoch."""
    t_days = max(float(t_day), 1e-4)

    mej = 10.0 ** (float(log10_mej) + math.log10(m_sun_g))
    mcsm = 10.0 ** (float(log10_csm_mass) + math.log10(m_sun_g))
    vej = 10.0 ** (float(log10_vej) + 5.0)  # km/s -> cm/s
    kappa = 10.0 ** float(log10_kappa)

    # Effective diffusion timescale
    tau_diff = math.sqrt(2.0 * kappa * mcsm / (13.8 * c_cgs * vej)) / day_to_s
    tau_diff = max(tau_diff, 1.0)

    # Peak shock kinetic luminosity
    e_kin = 0.5 * mcsm * (vej ** 2)
    l_peak = 0.5 * e_kin / (tau_diff * day_to_s)
    l_peak = max(min(l_peak, 1e45), 1e40)

    # Arnett/Chatzopoulos diffusion profile
    x = t_days / tau_diff
    if x < 1.0:
        lum = l_peak * (x ** 1.5)
    else:
        lum = l_peak * math.exp(-0.5 * (x - 1.0))
    lum = max(lum, 1e38)

    # Expanding photosphere radius
    r_phot = vej * (t_days * day_to_s) * 0.5 + 10.0 ** (float(log10_r0) + math.log10(1.496e13))
    r_phot = max(r_phot, 1e14)

    # Effective temperature
    t_eff = (lum / (4.0 * math.pi * (r_phot ** 2) * sigma_sb)) ** 0.25
    t_eff = max(min(t_eff, 60000.0), 3000.0)

    # Rest-frame frequency
    rest_wave_cm = (band_wave_aa / (1.0 + z)) * 1e-8
    nu_rest = c_cgs / rest_wave_cm

    # Planck blackbody function B_nu
    exp_arg = (h_cgs * nu_rest) / (k_b_cgs * t_eff)
    if exp_arg > 100.0:
        b_nu = 0.0
    else:
        b_nu = (2.0 * h_cgs * (nu_rest ** 3) / (c_cgs ** 2)) / (math.exp(exp_arg) - 1.0)

    # Observed flux density
    d_l_cm = d_l_mpc * 3.08567758128e24
    flux_nu = (1.0 + z) * math.pi * (r_phot / d_l_cm) ** 2 * b_nu

    if flux_nu <= 0.0 or not math.isfinite(flux_nu):
        return 99.0

    mag_ab = -2.5 * math.log10(flux_nu) - 48.60
    return max(min(mag_ab, 40.0), 5.0)


class FiestaCsmModel:
    """CSM interaction lightcurve model adapter for survey-sim.

    Evaluates dense circumstellar interaction light curves using fiestaEM
    when available, or the analytical diffusion solution.
    """

    def __init__(self, filters=None):
        self.filters = filters or ["g", "r", "i"]
        self._fiesta_model = None

        try:
            from fiesta.inference.analytical_models.supernova_models import CSMInteractionModel
            fiesta_filts = [_BAND_TO_FIESTA.get(f, f) for f in self.filters]
            self._fiesta_model = CSMInteractionModel(filters=fiesta_filts)
        except Exception:
            self._fiesta_model = None

    def predict(self, params):
        """Callback signature matching survey_sim PythonCallbackModel."""
        times = params.get("_obs_times_mjd", [])
        bands = params.get("_obs_bands", [])
        t_exp = params.get("_t_exp", 0.0)
        z = params.get("redshift", 0.0)
        d_l = params.get("luminosity_distance", 100.0)

        log10_mej = params.get("log10_mej", 0.0)
        log10_csm_mass = params.get("log10_csm_mass", -0.5)
        log10_vej = params.get("log10_vej", 3.9)
        eta = params.get("eta", 1.5)
        log10_rho = params.get("log10_rho", -13.0)
        log10_r0 = params.get("log10_r0", 1.0)
        log10_kappa = params.get("log10_kappa", -0.47)

        band_results = {b: [] for b in self.filters}

        if len(times) == 0:
            return (times, band_results)

        for t_obs, band in zip(times, bands):
            if band not in band_results:
                continue

            dt_rest = (t_obs - t_exp) / (1.0 + z)
            if dt_rest <= 0.0:
                band_results[band].append(99.0)
                continue

            wave = BAND_WAVELENGTHS.get(band, 4825.0)

            # Try fiesta if loaded
            mag = None
            if self._fiesta_model is not None:
                try:
                    f_band = _BAND_TO_FIESTA.get(band, band)
                    fiesta_params = {
                        "log10_mej": log10_mej,
                        "log10_csm_mass": log10_csm_mass,
                        "log10_vej": log10_vej,
                        "eta": eta,
                        "log10_rho": log10_rho,
                        "log10_r0": log10_r0,
                        "log10_kappa": log10_kappa,
                        "redshift": z,
                        "luminosity_distance": d_l,
                    }
                    pred = self._fiesta_model.predict(fiesta_params, times=np.array([dt_rest]))
                    if f_band in pred and len(pred[f_band]) > 0:
                        m_val = float(pred[f_band][0])
                        if math.isfinite(m_val) and m_val < 50.0:
                            mag = m_val
                except Exception:
                    pass

            if mag is None:
                mag = _csm_flux_single(
                    dt_rest, log10_mej, log10_csm_mass, log10_vej,
                    eta, log10_rho, log10_r0, log10_kappa,
                    wave, d_l, z,
                )

            band_results[band].append(mag)

        return (times, band_results)
