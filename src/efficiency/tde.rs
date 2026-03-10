//! Semi-analytical TDE rate forecast following Karmen et al. (2025, arXiv:2602.04947).
//!
//! Computes annual TDE detection rates for survey configurations by integrating:
//!
//! Γ_TDE = ∫₀^{z_Ly} ε(z) F(z) N_BH(z) R₀(z,λ) O(z) dz
//!
//! where R₀ uses the Yao+2023 g-band luminosity function, N_BH is the SMBH mass
//! function evolution, F(z) captures galaxy-scale enhancements, O(z) is dust
//! obscuration, and ε(z) is the survey efficiency.

use crate::instrument::BandConfig;
use crate::lightcurve::kcorrection::{BlackbodySed, TopHatFilter, k_correction};
use crate::types::Cosmology;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};

// ============================================================================
// Luminosity Function (Yao+2023)
// ============================================================================

/// Broken power-law TDE g-band luminosity function parameters.
#[derive(Clone, Debug)]
pub struct TdeLuminosityFunction {
    /// Break luminosity log₁₀(L_g / erg s⁻¹).
    pub log_l_break: f64,
    /// Faint-end slope.
    pub gamma1: f64,
    /// Bright-end slope.
    pub gamma2: f64,
    /// Normalization in Mpc⁻³ yr⁻¹ dex⁻¹.
    pub n0: f64,
    /// Minimum observed log₁₀(L_g).
    pub log_l_min: f64,
    /// Maximum observed log₁₀(L_g).
    pub log_l_max: f64,
}

impl Default for TdeLuminosityFunction {
    /// Yao+2023 best-fit values.
    fn default() -> Self {
        Self {
            log_l_break: 43.13,
            gamma1: 0.26,
            gamma2: 2.58,
            n0: 2.87e-7,
            log_l_min: 42.68,
            log_l_max: 44.68,
        }
    }
}

impl TdeLuminosityFunction {
    /// Rate density φ(log L_g) in Mpc⁻³ yr⁻¹ dex⁻¹ at a given luminosity.
    pub fn phi(&self, log_lg: f64) -> f64 {
        let x = log_lg - self.log_l_break;
        self.n0 / (10f64.powf(self.gamma1 * x) + 10f64.powf(self.gamma2 * x))
    }

    /// Integrate LF over [log_l_min, log_l_max] → Mpc⁻³ yr⁻¹.
    pub fn integrate(&self, log_l_min: f64, log_l_max: f64, n_steps: usize) -> f64 {
        let n = n_steps.max(2);
        let d_log = (log_l_max - log_l_min) / n as f64;
        let mut sum = 0.0;
        for i in 0..=n {
            let ll = log_l_min + i as f64 * d_log;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            sum += w * self.phi(ll);
        }
        sum * d_log
    }

    /// Total integrated rate over the observed range.
    pub fn total_rate(&self) -> f64 {
        self.integrate(self.log_l_min, self.log_l_max, 1000)
    }
}

// ============================================================================
// BHMF Models
// ============================================================================

/// Black hole mass function evolution model.
#[derive(Clone, Debug)]
pub enum BhmfModel {
    /// Shankar+09 semi-empirical: α = -1.46 (fast decline).
    Shankar,
    /// Illustris/TNG hydro simulation: α = -0.82 (slow decline).
    Illustris,
    /// Custom exponential with given α.
    Custom(f64),
}

impl BhmfModel {
    /// N_BH(z) / N_BH(0): SMBH number density evolution.
    /// Parameterized as exp(α × z).
    pub fn evolution(&self, z: f64) -> f64 {
        let alpha = match self {
            BhmfModel::Shankar => -1.46,
            BhmfModel::Illustris => -0.82,
            BhmfModel::Custom(a) => *a,
        };
        (alpha * z).exp()
    }
}

// ============================================================================
// Galaxy-Scale Evolution Factors
// ============================================================================

/// Galaxy merger rate enhancement M(z).
///
/// f_pair(z) = 0.056 × (1+z)^5.910 × exp(-1.814×(1+z))
/// M(z) = [1 + (E-1)×f_pair(z)] / [1 + (E-1)×f_pair(0)]
pub fn merger_enhancement(z: f64, e_factor: f64) -> f64 {
    let f_pair = |z: f64| 0.056 * (1.0 + z).powf(5.910) * (-1.814 * (1.0 + z)).exp();
    (1.0 + (e_factor - 1.0) * f_pair(z)) / (1.0 + (e_factor - 1.0) * f_pair(0.0))
}

/// Nuclear stellar density evolution D(z) = (1+z)^(0.9α).
pub fn density_evolution(z: f64, alpha: f64) -> f64 {
    (1.0 + z).powf(0.9 * alpha)
}

/// Dust obscuration factor O(z).
///
/// f_obsc(z) = f₀ + (f_max - f₀) / (1 + exp(-k × ln(1+z)))
/// O(z) = f_obsc(z) / f_obsc(0)
pub fn dust_obscuration(z: f64) -> f64 {
    let f0 = 0.3;
    let f_max = 0.9;
    let k = 0.7;
    let f_obsc = |z: f64| f0 + (f_max - f0) / (1.0 + (-k * (1.0 + z).ln()).exp());
    f_obsc(z) / f_obsc(0.0)
}

/// IMF evolution I(z).
///
/// α(z) interpolates from 2.35 (Salpeter, z=0) to 2.081 (z=8).
/// I(z) = ⟨M²⟩_{α(z)} / ⟨M²⟩_{α₀}
pub fn imf_evolution(z: f64) -> f64 {
    let m_min = 0.1f64;
    let m_max = 100.0f64;
    let alpha_0 = 2.35;
    let alpha_z8 = 2.081;
    let alpha_z = alpha_0 + (alpha_z8 - alpha_0) * z.min(8.0) / 8.0;

    let mean_m2 = |alpha: f64| -> f64 {
        let num = if (alpha - 3.0).abs() < 1e-6 {
            (m_max / m_min).ln()
        } else {
            (m_max.powf(3.0 - alpha) - m_min.powf(3.0 - alpha)) / (3.0 - alpha)
        };
        let den = if (alpha - 1.0).abs() < 1e-6 {
            (m_max / m_min).ln()
        } else {
            (m_max.powf(1.0 - alpha) - m_min.powf(1.0 - alpha)) / (1.0 - alpha)
        };
        num / den
    };

    mean_m2(alpha_z) / mean_m2(alpha_0)
}

/// Combined galaxy-scale enhancement F(z) = M(z) × I(z) × D(z).
pub fn galaxy_effects(z: f64, e_factor: f64, density_alpha: f64) -> f64 {
    merger_enhancement(z, e_factor)
        * imf_evolution(z)
        * density_evolution(z, density_alpha)
}

// ============================================================================
// Lyman-alpha Cutoff
// ============================================================================

const LYMAN_ALPHA_NM: f64 = 121.567;

/// Maximum redshift before Lyman-alpha absorption blocks a filter.
pub fn z_lyman(lambda_obs_nm: f64) -> f64 {
    lambda_obs_nm / LYMAN_ALPHA_NM - 1.0
}

// ============================================================================
// TDE Lightcurve Visibility
// ============================================================================

/// Average visibility coefficients from van Velzen (2021) lightcurve parameters.
///
/// log₁₀(σ/days) ∈ [0.4, 1.3], log₁₀(τ/days) ∈ [1.2, 2.3]
/// Rise: t_rise = σ × √(2 × Δm / 1.086)
/// Decay: t_decay = τ × Δm / 1.086
struct VisibilityCoeffs {
    /// ⟨σ⟩ × √(2/1.086)
    rise_coeff: f64,
    /// ⟨τ⟩ / 1.086
    decay_coeff: f64,
}

fn visibility_coeffs() -> VisibilityCoeffs {
    // Average over log-uniform grid of (σ, τ)
    let n = 10;
    let mut sigma_sum = 0.0;
    let mut tau_sum = 0.0;
    for i in 0..n {
        let ls = 0.4 + (1.3 - 0.4) * i as f64 / (n - 1) as f64;
        sigma_sum += 10f64.powf(ls);
        let lt = 1.2 + (2.3 - 1.2) * i as f64 / (n - 1) as f64;
        tau_sum += 10f64.powf(lt);
    }
    let mean_sigma = sigma_sum / n as f64;
    let mean_tau = tau_sum / n as f64;
    VisibilityCoeffs {
        rise_coeff: mean_sigma * (2.0 / 1.086f64).sqrt(),
        decay_coeff: mean_tau / 1.086,
    }
}

/// Minimum detectable log₁₀(L_g) at a given redshift and survey depth.
///
/// m_peak = -2.5×log₁₀(L_g) + 88.6 + μ + K < m_limit
/// → log₁₀(L_g) > (88.6 + μ + K - m_limit) / 2.5
fn log_lg_min(m_limit: f64, mu: f64, k_corr: f64) -> f64 {
    (88.6 + mu + k_corr - m_limit) / 2.5
}

// ============================================================================
// Survey Specification
// ============================================================================

/// Survey parameters for TDE rate computation.
#[derive(Clone, Debug)]
pub struct TdeRateSurvey {
    pub name: String,
    pub area_deg2: f64,
    pub best_filter: BandConfig,
    pub m_limit: f64,
    pub is_time_domain: bool,
    /// Fraction of the year the survey actively observes (e.g., 0.5 for DDFs).
    pub seasonal_coverage: f64,
}

// ============================================================================
// Rate Result
// ============================================================================

/// Result of a TDE rate forecast.
#[derive(Clone, Debug)]
pub struct TdeRateResult {
    /// Median annual detection count.
    pub n_median: f64,
    /// 16th percentile.
    pub n_16: f64,
    /// 84th percentile.
    pub n_84: f64,
    /// Median redshift of detected TDEs.
    pub z_median: f64,
    /// Mean redshift of detected TDEs.
    pub z_mean: f64,
    /// Maximum redshift with significant rate.
    pub z_max: f64,
}

// ============================================================================
// Main Rate Integral
// ============================================================================

/// Compute annual TDE detection rate for a survey configuration.
///
/// Implements Karmen+2025 Eq. 1 with Monte Carlo over uncertain parameters:
/// - E: merger enhancement factor, U(10, 100)
/// - density_alpha: density evolution slope, U(1, 2)
pub fn compute_tde_rate(
    survey: &TdeRateSurvey,
    lf: &TdeLuminosityFunction,
    bhmf: &BhmfModel,
    cosmo: &Cosmology,
    temperature_k: f64,
    n_mc: usize,
    seed: u64,
) -> TdeRateResult {
    let f_sky = survey.area_deg2 / 41253.0;
    let z_ly = z_lyman(survey.best_filter.central_wavelength_nm);
    let z_max_grid = z_ly.min(5.0);

    // Redshift grid
    let n_z = 200;
    let z_edges: Vec<f64> = (0..=n_z).map(|i| 0.01 + (z_max_grid - 0.01) * i as f64 / n_z as f64).collect();
    let z_centers: Vec<f64> = (0..n_z).map(|i| 0.5 * (z_edges[i] + z_edges[i + 1])).collect();
    let dz: Vec<f64> = (0..n_z).map(|i| z_edges[i + 1] - z_edges[i]).collect();

    // Pre-compute cosmology (the expensive part)
    let dv_dz_sky: Vec<f64> = z_centers
        .iter()
        .map(|&z| cosmo.dv_dz(z) * 1e9 * 4.0 * std::f64::consts::PI * f_sky) // Gpc³→Mpc³
        .collect();
    let d_l_pc: Vec<f64> = z_centers
        .iter()
        .map(|&z| cosmo.luminosity_distance(z) * 1e6) // Mpc→pc
        .collect();

    // Pre-compute BHMF and obscuration
    let nbh: Vec<f64> = z_centers.iter().map(|&z| bhmf.evolution(z)).collect();
    let oz: Vec<f64> = z_centers.iter().map(|&z| dust_obscuration(z)).collect();

    // Pre-compute K-corrections
    let sed = BlackbodySed { temperature_k };
    let filter = TopHatFilter::from_band_config(&survey.best_filter);
    let k_corr: Vec<f64> = z_centers
        .iter()
        .map(|&z| k_correction(&sed, &filter, z, 200))
        .collect();

    // Luminosity grid
    let n_lg = 200;
    let d_log_lg = (lf.log_l_max - lf.log_l_min) / n_lg as f64;
    let log_lgs: Vec<f64> = (0..=n_lg).map(|i| lf.log_l_min + i as f64 * d_log_lg).collect();
    let phi: Vec<f64> = log_lgs.iter().map(|&ll| lf.phi(ll)).collect();

    // Pre-compute visibility coefficients
    let vc = visibility_coeffs();

    // Pre-compute rate integrand per z bin (independent of MC params)
    let rate_integrand: Vec<f64> = (0..n_z)
        .map(|i| {
            let z = z_centers[i];
            let mu = 5.0 * (d_l_pc[i] / 10.0).log10();
            let llg_min = log_lg_min(survey.m_limit, mu, k_corr[i]).max(lf.log_l_min);

            let mut sum = 0.0;
            for j in 0..=n_lg {
                if log_lgs[j] < llg_min {
                    continue;
                }
                let eps = if survey.is_time_domain {
                    1.0
                } else {
                    // Visibility time for this luminosity at this z
                    let m_peak = -2.5 * log_lgs[j] + 88.6 + mu + k_corr[i];
                    let delta_mag = survey.m_limit - m_peak;
                    if delta_mag <= 0.0 {
                        0.0
                    } else {
                        let t_vis = vc.rise_coeff * delta_mag.sqrt()
                            + vc.decay_coeff * delta_mag;
                        (t_vis * (1.0 + z) / 365.0).min(1.0)
                    }
                };
                let w = if j == 0 || j == n_lg { 0.5 } else { 1.0 };
                sum += w * phi[j] * eps;
            }
            sum * d_log_lg
        })
        .collect();

    // Monte Carlo over uncertain parameters
    let mut rng = StdRng::seed_from_u64(seed);
    let e_dist = Uniform::new(10.0, 100.0).unwrap();
    let alpha_dist = Uniform::new(1.0, 2.0).unwrap();

    let mut n_samples = Vec::with_capacity(n_mc);
    let mut z_med_samples = Vec::with_capacity(n_mc);
    let mut z_mean_samples = Vec::with_capacity(n_mc);
    let mut z_max_samples = Vec::with_capacity(n_mc);

    for _ in 0..n_mc {
        let e = e_dist.sample(&mut rng);
        let density_alpha = alpha_dist.sample(&mut rng);

        // Galaxy effects vary per MC draw
        let fz: Vec<f64> = z_centers
            .iter()
            .map(|&z| galaxy_effects(z, e, density_alpha))
            .collect();

        // dN/dz = rate_integrand × F(z) × N_BH(z) × O(z) × dV/dz_sky / (1+z)
        let dn_dz: Vec<f64> = (0..n_z)
            .map(|i| {
                rate_integrand[i] * fz[i] * nbh[i] * oz[i] * dv_dz_sky[i]
                    / (1.0 + z_centers[i])
                    * survey.seasonal_coverage
            })
            .collect();

        let n_total: f64 = dn_dz.iter().zip(dz.iter()).map(|(dn, dz)| dn * dz).sum();
        n_samples.push(n_total);

        if n_total > 0.0 {
            let mut cum = 0.0;
            let mut z_med = 0.0;
            let mut z_mean_val = 0.0;
            for i in 0..n_z {
                let contrib = dn_dz[i] * dz[i];
                cum += contrib;
                z_mean_val += z_centers[i] * contrib;
                if z_med == 0.0 && cum >= 0.5 * n_total {
                    // Linear interpolation
                    z_med = z_centers[i];
                }
            }
            z_mean_val /= n_total;

            let peak = dn_dz.iter().cloned().fold(0.0f64, f64::max);
            let z_max_val = z_centers
                .iter()
                .zip(dn_dz.iter())
                .rev()
                .find(|(_, &dn)| dn > 0.001 * peak)
                .map(|(&z, _)| z)
                .unwrap_or(0.0);

            z_med_samples.push(z_med);
            z_mean_samples.push(z_mean_val);
            z_max_samples.push(z_max_val);
        } else {
            z_med_samples.push(0.0);
            z_mean_samples.push(0.0);
            z_max_samples.push(0.0);
        }
    }

    n_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    z_med_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    z_mean_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    z_max_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = |v: &[f64]| {
        let n = v.len();
        if n % 2 == 0 {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        } else {
            v[n / 2]
        }
    };
    let percentile = |v: &[f64], p: f64| {
        let idx = (p / 100.0 * (v.len() - 1) as f64).round() as usize;
        v[idx.min(v.len() - 1)]
    };

    TdeRateResult {
        n_median: median(&n_samples),
        n_16: percentile(&n_samples, 16.0),
        n_84: percentile(&n_samples, 84.0),
        z_median: median(&z_med_samples),
        z_mean: median(&z_mean_samples),
        z_max: median(&z_max_samples),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luminosity_function_break() {
        let lf = TdeLuminosityFunction::default();
        // At the break: phi = N0 / (1 + 1) = N0/2
        let phi_break = lf.phi(lf.log_l_break);
        assert!(
            (phi_break - lf.n0 / 2.0).abs() < 1e-15,
            "phi at break = {}, expected {}",
            phi_break,
            lf.n0 / 2.0
        );
    }

    #[test]
    fn test_luminosity_function_total() {
        let lf = TdeLuminosityFunction::default();
        let rate = lf.total_rate();
        // Should be ~1.45e-7 Mpc^-3 yr^-1 over observed range
        assert!(
            (rate - 1.45e-7).abs() < 0.1e-7,
            "Total LF rate = {:.2e}, expected ~1.45e-7",
            rate
        );
    }

    #[test]
    fn test_bhmf_z0() {
        assert!((BhmfModel::Shankar.evolution(0.0) - 1.0).abs() < 1e-10);
        assert!((BhmfModel::Illustris.evolution(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bhmf_shankar_declines() {
        // Shankar declines faster than Illustris
        assert!(BhmfModel::Shankar.evolution(1.0) < BhmfModel::Illustris.evolution(1.0));
    }

    #[test]
    fn test_galaxy_effects_z0() {
        let f = galaxy_effects(0.0, 30.0, 1.5);
        assert!((f - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_galaxy_effects_increases() {
        // Galaxy effects should increase with z
        assert!(galaxy_effects(1.0, 30.0, 1.5) > galaxy_effects(0.0, 30.0, 1.5));
    }

    #[test]
    fn test_z_lyman() {
        // g-band (482nm): z_Ly ≈ 2.96
        assert!((z_lyman(482.0) - 2.96).abs() < 0.1);
        // F062 (620nm): z_Ly ≈ 4.10
        assert!((z_lyman(620.0) - 4.10).abs() < 0.1);
    }

    #[test]
    fn test_dust_obscuration_z0() {
        assert!((dust_obscuration(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_imf_evolution_z0() {
        assert!((imf_evolution(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_tde_rate_rubin_illustris() {
        let cosmo = Cosmology::default();
        let lf = TdeLuminosityFunction::default();
        let survey = TdeRateSurvey {
            name: "Rubin (LSST)".to_string(),
            area_deg2: 18000.0,
            best_filter: BandConfig {
                central_wavelength_nm: 482.0,
                width_nm: 140.0,
                typical_seeing_arcsec: 0.87,
                single_visit_depth: 25.0,
                sky_brightness: 22.24,
            },
            m_limit: 25.0,
            is_time_domain: true,
            seasonal_coverage: 1.0,
        };
        let result = compute_tde_rate(&survey, &lf, &BhmfModel::Illustris, &cosmo, 30000.0, 100, 42);

        // Paper: 26,873. Our Python gives ~20,811. Should be in the right ballpark.
        assert!(
            result.n_median > 10000.0 && result.n_median < 50000.0,
            "Rubin Illustris N = {:.0}, expected ~20000-30000",
            result.n_median
        );
    }

    #[test]
    fn test_compute_tde_rate_roman_deep() {
        let cosmo = Cosmology::default();
        let lf = TdeLuminosityFunction::default();
        let survey = TdeRateSurvey {
            name: "Roman HLTDS (deep)".to_string(),
            area_deg2: 6.0,
            best_filter: BandConfig {
                central_wavelength_nm: 620.0,
                width_nm: 180.0,
                typical_seeing_arcsec: 0.11,
                single_visit_depth: 26.95,
                sky_brightness: 23.0,
            },
            m_limit: 26.95,
            is_time_domain: true,
            seasonal_coverage: 1.0,
        };
        let result = compute_tde_rate(&survey, &lf, &BhmfModel::Shankar, &cosmo, 30000.0, 100, 42);

        // Paper: 14.2. Our Python gives ~14.6. Should be close.
        assert!(
            result.n_median > 5.0 && result.n_median < 50.0,
            "Roman deep Shankar N = {:.1}, expected ~14",
            result.n_median
        );
    }
}
