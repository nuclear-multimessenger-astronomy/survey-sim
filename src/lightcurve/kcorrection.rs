//! Broadband K-corrections for spectral energy distributions through filter bandpasses.
//!
//! Provides general-purpose K-correction computation for arbitrary SEDs (blackbody,
//! power-law, tabulated) observed through instrument filter bandpasses. This replaces
//! the bolometric approximation in [`super::cosmology::k_correction_bolometric`] with
//! physically motivated SED-through-bandpass integration.

use crate::instrument::BandConfig;

/// Physical constants (SI).
const H_PLANCK: f64 = 6.626e-34; // J·s
const K_BOLTZMANN: f64 = 1.381e-23; // J/K
const C_LIGHT: f64 = 3.0e8; // m/s

/// Spectral energy distribution trait.
///
/// Only the relative shape matters for K-corrections — absolute normalization cancels.
pub trait Sed {
    /// Spectral radiance (or flux density) at frequency `nu_hz` in arbitrary units.
    fn spectral_radiance(&self, nu_hz: f64) -> f64;
}

/// Planck blackbody at a fixed temperature.
#[derive(Clone, Debug)]
pub struct BlackbodySed {
    pub temperature_k: f64,
}

impl Sed for BlackbodySed {
    fn spectral_radiance(&self, nu_hz: f64) -> f64 {
        let x = (H_PLANCK * nu_hz / (K_BOLTZMANN * self.temperature_k)).min(500.0);
        if x <= 0.0 {
            return 0.0;
        }
        2.0 * H_PLANCK * nu_hz.powi(3) / (C_LIGHT * C_LIGHT) / (x.exp() - 1.0)
    }
}

/// Power-law SED: F_nu proportional to nu^alpha.
#[derive(Clone, Debug)]
pub struct PowerLawSed {
    pub spectral_index: f64,
}

impl Sed for PowerLawSed {
    fn spectral_radiance(&self, nu_hz: f64) -> f64 {
        nu_hz.powf(self.spectral_index)
    }
}

/// Top-hat filter bandpass defined by wavelength range.
#[derive(Clone, Debug)]
pub struct TopHatFilter {
    /// Minimum wavelength in meters.
    pub lambda_min_m: f64,
    /// Maximum wavelength in meters.
    pub lambda_max_m: f64,
}

impl TopHatFilter {
    /// Construct from a [`BandConfig`], using central_wavelength ± width/2.
    pub fn from_band_config(band: &BandConfig) -> Self {
        let center = band.central_wavelength_nm * 1e-9;
        let half_width = band.width_nm * 1e-9 / 2.0;
        Self {
            lambda_min_m: center - half_width,
            lambda_max_m: center + half_width,
        }
    }

    /// Construct from wavelength bounds in nanometers.
    pub fn from_nm(lambda_min_nm: f64, lambda_max_nm: f64) -> Self {
        Self {
            lambda_min_m: lambda_min_nm * 1e-9,
            lambda_max_m: lambda_max_nm * 1e-9,
        }
    }
}

/// Broadband K-correction for an SED observed through a filter at redshift z.
///
/// K(z) = -2.5 × log₁₀[ ∫ S(ν(1+z)) R(ν) dν / (1+z) / ∫ S(ν) R(ν) dν ]
///
/// where S(ν) is the SED and R(ν) is the filter response (top-hat).
/// The (1+z) factor in the denominator accounts for the compression of the
/// frequency interval when transforming from observer to rest frame.
pub fn k_correction<S: Sed>(
    sed: &S,
    filter: &TopHatFilter,
    z: f64,
    n_points: usize,
) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }

    // Build frequency grid (from lambda_max to lambda_min → nu increasing)
    let n = n_points.max(10);
    let dlam = (filter.lambda_max_m - filter.lambda_min_m) / n as f64;

    let mut f_rest = 0.0;
    let mut f_obs = 0.0;

    for i in 0..=n {
        let lam = filter.lambda_min_m + i as f64 * dlam;
        let nu = C_LIGHT / lam;
        let nu_rest = nu * (1.0 + z);

        let s_rest = sed.spectral_radiance(nu);
        let s_obs = sed.spectral_radiance(nu_rest);

        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        f_rest += w * s_rest;
        f_obs += w * s_obs;
    }

    if f_rest <= 0.0 || f_obs <= 0.0 {
        return 2.5 * (1.0 + z).log10(); // fallback to bolometric
    }

    -2.5 * (f_obs / f_rest / (1.0 + z)).log10()
}

/// Convenience: blackbody K-correction through a [`BandConfig`].
pub fn k_correction_blackbody(temperature_k: f64, band: &BandConfig, z: f64) -> f64 {
    let sed = BlackbodySed { temperature_k };
    let filter = TopHatFilter::from_band_config(band);
    k_correction(&sed, &filter, z, 200)
}

/// Convenience: blackbody K-correction through a named band of an instrument.
pub fn k_correction_blackbody_named(
    temperature_k: f64,
    instrument: &crate::instrument::InstrumentConfig,
    band_name: &str,
    z: f64,
) -> Option<f64> {
    instrument
        .bands
        .get(band_name)
        .map(|band| k_correction_blackbody(temperature_k, band, z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_correction_zero_redshift() {
        let sed = BlackbodySed {
            temperature_k: 30000.0,
        };
        let filter = TopHatFilter::from_nm(400.0, 552.0);
        let k = k_correction(&sed, &filter, 0.0, 200);
        assert!(k.abs() < 1e-10, "K(z=0) should be 0, got {}", k);
    }

    #[test]
    fn test_k_correction_blackbody_g_band() {
        // At z=0.5, T=30000K through LSST g-band (400-552nm):
        // Python reference: K ≈ -0.10
        let sed = BlackbodySed {
            temperature_k: 30000.0,
        };
        let filter = TopHatFilter::from_nm(400.0, 552.0);
        let k = k_correction(&sed, &filter, 0.5, 200);
        assert!(
            (k - (-0.10)).abs() < 0.05,
            "K(z=0.5, T=30kK, g) = {:.3}, expected ~-0.10",
            k
        );
    }

    #[test]
    fn test_k_correction_blackbody_z1() {
        // At z=1.0, T=30000K through g-band: K ≈ -0.04
        let sed = BlackbodySed {
            temperature_k: 30000.0,
        };
        let filter = TopHatFilter::from_nm(400.0, 552.0);
        let k = k_correction(&sed, &filter, 1.0, 200);
        assert!(
            (k - (-0.04)).abs() < 0.10,
            "K(z=1.0, T=30kK, g) = {:.3}, expected ~-0.04",
            k
        );
    }

    #[test]
    fn test_k_correction_power_law() {
        // For a flat spectrum (alpha=0): K = -2.5*log10(1/(1+z)) = 2.5*log10(1+z)
        // This matches the bolometric K-correction
        let sed = PowerLawSed {
            spectral_index: 0.0,
        };
        let filter = TopHatFilter::from_nm(400.0, 700.0);
        let z = 1.0;
        let k = k_correction(&sed, &filter, z, 200);
        let k_expected = 2.5 * (1.0 + z).log10();
        assert!(
            (k - k_expected).abs() < 0.01,
            "Flat SED K = {:.3}, expected {:.3}",
            k,
            k_expected
        );
    }

    #[test]
    fn test_k_correction_from_band_config() {
        let band = BandConfig {
            central_wavelength_nm: 482.0,
            width_nm: 140.0,
            typical_seeing_arcsec: 0.87,
            single_visit_depth: 25.0,
            sky_brightness: 22.24,
        };
        let k = k_correction_blackbody(30000.0, &band, 0.5);
        // Should be close to the manual top-hat result
        assert!(k.abs() < 1.0, "K should be moderate, got {}", k);
    }

    #[test]
    fn test_k_correction_hot_vs_cool() {
        // Hotter BB should have more negative K (more brightening) because
        // more UV flux gets redshifted into the optical
        let filter = TopHatFilter::from_nm(400.0, 552.0);
        let k_hot = k_correction(
            &BlackbodySed {
                temperature_k: 40000.0,
            },
            &filter,
            1.0,
            200,
        );
        let k_cool = k_correction(
            &BlackbodySed {
                temperature_k: 10000.0,
            },
            &filter,
            1.0,
            200,
        );
        assert!(
            k_hot < k_cool,
            "Hot BB K={:.3} should be less than cool BB K={:.3}",
            k_hot,
            k_cool
        );
    }
}
