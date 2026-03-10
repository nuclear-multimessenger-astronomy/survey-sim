use std::collections::HashMap;


use crate::types::{Cosmology, SkyCoord, TransientInstance, TransientType};

use super::distributions::{self, *};
use super::PopulationGenerator;

/// Kilonova population generator.
pub struct KilonovaPopulation {
    /// Volumetric rate in Gpc^-3 yr^-1.
    pub rate: f64,
    /// Maximum redshift.
    pub z_max: f64,
    /// Peak absolute magnitude (default: -16.0).
    pub peak_abs_mag: f64,
    /// MJD range for explosion times.
    pub mjd_min: f64,
    pub mjd_max: f64,
    /// Cosmology.
    pub cosmology: Cosmology,
}

impl KilonovaPopulation {
    pub fn new(rate: f64, z_max: f64, peak_abs_mag: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }
}

impl PopulationGenerator for KilonovaPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            // Metzger KN model parameters: ejecta mass, velocity, opacity.
            let mut params = HashMap::new();
            params.insert("mej".to_string(), sample_log_uniform(0.001, 0.1, rng));
            params.insert("vej".to_string(), sample_gaussian_clamped(0.2, 0.05, 0.05, 0.5, rng));
            params.insert("kappa".to_string(), sample_log_uniform(0.5, 30.0, rng));

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: sample_gaussian_clamped(
                    self.peak_abs_mag,
                    1.0,
                    self.peak_abs_mag - 3.0,
                    self.peak_abs_mag + 3.0,
                    rng,
                ),
                transient_type: TransientType::Kilonova,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: sample_gaussian_clamped(0.1, 0.2, 0.0, 3.0, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::Kilonova
    }
}

/// Metzger 1-zone kilonova population with fixed physical parameters.
///
/// Only redshift, sky position, explosion time, and extinction are randomized.
/// The Metzger model computes per-band magnitudes via blackbody emission.
pub struct FixedMetzgerKilonovaPopulation {
    pub rate: f64,
    pub z_max: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub cosmology: Cosmology,
    pub mej: f64,
    pub vej: f64,
    pub kappa: f64,
}

impl FixedMetzgerKilonovaPopulation {
    pub fn new(
        rate: f64, z_max: f64, mjd_min: f64, mjd_max: f64,
        mej: f64, vej: f64, kappa: f64,
    ) -> Self {
        Self {
            rate, z_max, mjd_min, mjd_max,
            cosmology: Cosmology::default(),
            mej, vej, kappa,
        }
    }
}

impl PopulationGenerator for FixedMetzgerKilonovaPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();
            params.insert("mej".to_string(), self.mej);
            params.insert("vej".to_string(), self.vej);
            params.insert("kappa".to_string(), self.kappa);

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: 0.0, // Not used — Metzger computes physical magnitudes.
                transient_type: TransientType::Kilonova,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: sample_gaussian_clamped(0.1, 0.2, 0.0, 3.0, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::Kilonova
    }
}

/// Bu2026 two-component kilonova population generator (for fiesta surrogate).
pub struct Bu2026KilonovaPopulation {
    /// Volumetric rate in Gpc^-3 yr^-1.
    pub rate: f64,
    /// Maximum redshift.
    pub z_max: f64,
    /// MJD range for explosion times.
    pub mjd_min: f64,
    pub mjd_max: f64,
    /// Cosmology.
    pub cosmology: Cosmology,
}

impl Bu2026KilonovaPopulation {
    pub fn new(rate: f64, z_max: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            rate,
            z_max,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }
}

impl PopulationGenerator for Bu2026KilonovaPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        use rand::Rng;

        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            // Bu2026 training bounds (two-component ejecta + viewing angle).
            let mut params = HashMap::new();
            params.insert("log10_mej_dyn".to_string(), rng.random::<f64>() * (-1.3 - -4.0) + -4.0);
            params.insert("v_ej_dyn".to_string(), rng.random::<f64>() * (0.35 - 0.12) + 0.12);
            params.insert("Ye_dyn".to_string(), rng.random::<f64>() * (0.35 - 0.15) + 0.15);
            params.insert("log10_mej_wind".to_string(), rng.random::<f64>() * (-0.56 - -4.0) + -4.0);
            params.insert("v_ej_wind".to_string(), rng.random::<f64>() * (0.15 - 0.05) + 0.05);
            params.insert("Ye_wind".to_string(), rng.random::<f64>() * (0.4 - 0.2) + 0.2);
            // Uniform in cos(inclination): cos(i) ~ U(0, 1) => i = arccos(U(0,1)).
            params.insert(
                "inclination_EM".to_string(),
                rng.random::<f64>().acos(),
            );

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                // Bu2026 computes physical magnitudes; set placeholder.
                peak_abs_mag: 0.0,
                transient_type: TransientType::Kilonova,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: sample_gaussian_clamped(0.1, 0.2, 0.0, 3.0, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::Kilonova
    }
}

/// Bu2026 kilonova population with fixed physical parameters.
///
/// Only redshift, sky position, explosion time, and extinction are randomized.
/// If `vary_inclination` is true, inclination is drawn as uniform in cos(iota)
/// instead of using the fixed value.
pub struct FixedBu2026KilonovaPopulation {
    pub rate: f64,
    pub z_max: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub cosmology: Cosmology,
    // Fixed Bu2026 parameters.
    pub log10_mej_dyn: f64,
    pub v_ej_dyn: f64,
    pub ye_dyn: f64,
    pub log10_mej_wind: f64,
    pub v_ej_wind: f64,
    pub ye_wind: f64,
    pub inclination_em: f64,
    pub vary_inclination: bool,
}

impl FixedBu2026KilonovaPopulation {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rate: f64, z_max: f64, mjd_min: f64, mjd_max: f64,
        log10_mej_dyn: f64, v_ej_dyn: f64, ye_dyn: f64,
        log10_mej_wind: f64, v_ej_wind: f64, ye_wind: f64,
        inclination_em: f64,
    ) -> Self {
        Self {
            rate, z_max, mjd_min, mjd_max,
            cosmology: Cosmology::default(),
            log10_mej_dyn, v_ej_dyn, ye_dyn,
            log10_mej_wind, v_ej_wind, ye_wind,
            inclination_em,
            vary_inclination: false,
        }
    }
}

impl PopulationGenerator for FixedBu2026KilonovaPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        use rand::Rng;

        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            // If vary_inclination, draw uniform in cos(iota): iota = arccos(U(0,1)).
            let inclination = if self.vary_inclination {
                rng.random::<f64>().acos()
            } else {
                self.inclination_em
            };

            let mut params = HashMap::new();
            params.insert("log10_mej_dyn".to_string(), self.log10_mej_dyn);
            params.insert("v_ej_dyn".to_string(), self.v_ej_dyn);
            params.insert("Ye_dyn".to_string(), self.ye_dyn);
            params.insert("log10_mej_wind".to_string(), self.log10_mej_wind);
            params.insert("v_ej_wind".to_string(), self.v_ej_wind);
            params.insert("Ye_wind".to_string(), self.ye_wind);
            params.insert("inclination_EM".to_string(), inclination);

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: 0.0,
                transient_type: TransientType::Kilonova,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: sample_gaussian_clamped(0.1, 0.2, 0.0, 3.0, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::Kilonova
    }
}

/// Type Ia Supernova population generator.
pub struct SupernovaIaPopulation {
    pub rate: f64,
    pub z_max: f64,
    pub peak_abs_mag: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub cosmology: Cosmology,
}

impl SupernovaIaPopulation {
    pub fn new(rate: f64, z_max: f64, peak_abs_mag: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }
}

impl PopulationGenerator for SupernovaIaPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();
            // Stretch and color (SALT2-like).
            params.insert("x1".to_string(), sample_gaussian_clamped(0.0, 1.0, -3.0, 3.0, rng));
            params.insert("c".to_string(), sample_gaussian_clamped(0.0, 0.1, -0.3, 0.3, rng));

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: sample_gaussian_clamped(
                    self.peak_abs_mag,
                    0.15,
                    self.peak_abs_mag - 1.0,
                    self.peak_abs_mag + 1.0,
                    rng,
                ),
                transient_type: TransientType::SupernovaIa,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: sample_gaussian_clamped(0.2, 0.3, 0.0, 3.0, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::SupernovaIa
    }
}

/// Type II Supernova population generator.
pub struct SupernovaIIPopulation {
    pub rate: f64,
    pub z_max: f64,
    pub peak_abs_mag: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub cosmology: Cosmology,
}

impl SupernovaIIPopulation {
    pub fn new(rate: f64, z_max: f64, peak_abs_mag: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }
}

impl PopulationGenerator for SupernovaIIPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();
            params.insert("plateau_duration".to_string(), sample_gaussian_clamped(80.0, 20.0, 30.0, 150.0, rng));
            params.insert("plateau_slope".to_string(), sample_gaussian_clamped(0.01, 0.005, 0.0, 0.05, rng));

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: sample_gaussian_clamped(
                    self.peak_abs_mag,
                    0.8,
                    self.peak_abs_mag - 3.0,
                    self.peak_abs_mag + 3.0,
                    rng,
                ),
                transient_type: TransientType::SupernovaII,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: sample_gaussian_clamped(0.3, 0.5, 0.0, 5.0, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::SupernovaII
    }
}

/// Type Ibc Supernova population generator.
pub struct SupernovaIbcPopulation {
    pub rate: f64,
    pub z_max: f64,
    pub peak_abs_mag: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub cosmology: Cosmology,
}

impl SupernovaIbcPopulation {
    pub fn new(rate: f64, z_max: f64, peak_abs_mag: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }
}

impl PopulationGenerator for SupernovaIbcPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let params = HashMap::new();

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: sample_gaussian_clamped(
                    self.peak_abs_mag,
                    0.6,
                    self.peak_abs_mag - 2.0,
                    self.peak_abs_mag + 2.0,
                    rng,
                ),
                transient_type: TransientType::SupernovaIbc,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: sample_gaussian_clamped(0.2, 0.3, 0.0, 3.0, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::SupernovaIbc
    }
}

/// TDE population generator.
///
/// When `use_luminosity_function` is true, peak absolute magnitudes are drawn
/// from the Yao et al. (2023) broken power-law luminosity function and the
/// volumetric rate is computed by integrating the LF. This ensures identical
/// population parameters for any survey — only the survey depth determines
/// which TDEs are detected.
pub struct TdePopulation {
    pub rate: f64,
    pub z_max: f64,
    pub peak_abs_mag: f64,
    pub use_luminosity_function: bool,
    /// Yao+2023 broken power-law LF: ϕ(Lg) = N0 / [(Lg/Lbk)^γ1 + (Lg/Lbk)^γ2]
    pub lf_log_lbk: f64,
    pub lf_gamma1: f64,
    pub lf_gamma2: f64,
    pub lf_n0: f64,       // Mpc^-3 yr^-1 dex^-1
    pub lf_mag_min: f64,  // brightest (most negative M_g)
    pub lf_mag_max: f64,  // faintest (least negative M_g)
    /// Whether to apply Karmen+2025 rate evolution factors: F(z) × N_BH(z) × O(z).
    pub use_rate_evolution: bool,
    /// BHMF model for rate evolution (Illustris or Shankar).
    pub bhmf_model: crate::efficiency::tde::BhmfModel,
    /// Merger enhancement E factor (median of U[10,100] = 30).
    pub evolution_e_factor: f64,
    /// Density evolution slope α (median of U[1,2] = 1.5).
    pub evolution_density_alpha: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub cosmology: Cosmology,
}

/// Convert rest-frame g-band luminosity logLg (erg/s) to absolute g-band AB mag.
/// Relation: M_g = -2.5 * logLg + 88.6  (derived from νLν at ν_g = 6.3e14 Hz).
fn loglg_to_abs_mag(log_lg: f64) -> f64 {
    -2.5 * log_lg + 88.6
}

/// Evaluate the unnormalized broken power-law LF shape at logLg.
fn lf_shape(log_lg: f64, log_lbk: f64, gamma1: f64, gamma2: f64) -> f64 {
    let x = log_lg - log_lbk;
    1.0 / (10f64.powf(gamma1 * x) + 10f64.powf(gamma2 * x))
}

/// Integrate the LF over a magnitude range to get total volumetric rate.
fn integrate_lf(
    n0: f64, log_lbk: f64, gamma1: f64, gamma2: f64,
    mag_min: f64, mag_max: f64,
) -> f64 {
    // Convert mag bounds to logLg bounds.
    // mag_min (brightest) → largest logLg; mag_max (faintest) → smallest logLg
    let log_lg_min = (88.6 - mag_max) / 2.5; // faintest → smallest logLg
    let log_lg_max = (88.6 - mag_min) / 2.5; // brightest → largest logLg

    // Trapezoidal integration in logLg space.
    let n_steps = 1000;
    let d_log = (log_lg_max - log_lg_min) / n_steps as f64;
    let mut sum = 0.0;
    for i in 0..=n_steps {
        let log_lg = log_lg_min + i as f64 * d_log;
        let w = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
        sum += w * lf_shape(log_lg, log_lbk, gamma1, gamma2);
    }
    n0 * sum * d_log  // Mpc^-3 yr^-1
}

impl TdePopulation {
    /// Legacy constructor: fixed rate and Gaussian peak_abs_mag.
    pub fn new(rate: f64, z_max: f64, peak_abs_mag: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
            use_luminosity_function: false,
            lf_log_lbk: 43.13,
            lf_gamma1: 0.26,
            lf_gamma2: 2.58,
            lf_n0: 2.87e-7,
            lf_mag_min: -24.0,
            lf_mag_max: -15.0,
            use_rate_evolution: false,
            bhmf_model: crate::efficiency::tde::BhmfModel::Illustris,
            evolution_e_factor: 30.0,
            evolution_density_alpha: 1.5,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }

    /// LF-based constructor: rate and magnitudes from Yao+2023 broken power-law LF.
    pub fn from_luminosity_function(z_max: f64, mjd_min: f64, mjd_max: f64) -> Self {
        let log_lbk = 43.13;
        let gamma1 = 0.26;
        let gamma2 = 2.58;
        let n0 = 2.87e-7; // Mpc^-3 yr^-1 dex^-1
        let mag_min = -24.0;
        let mag_max = -15.0;

        // Integrate LF to get total volumetric rate in Gpc^-3 yr^-1.
        let rate_mpc3 = integrate_lf(n0, log_lbk, gamma1, gamma2, mag_min, mag_max);
        let rate_gpc3 = rate_mpc3 * 1e9;

        Self {
            rate: rate_gpc3,
            z_max,
            peak_abs_mag: -19.5, // not used when use_luminosity_function=true
            use_luminosity_function: true,
            lf_log_lbk: log_lbk,
            lf_gamma1: gamma1,
            lf_gamma2: gamma2,
            lf_n0: n0,
            lf_mag_min: mag_min,
            lf_mag_max: mag_max,
            use_rate_evolution: false,
            bhmf_model: crate::efficiency::tde::BhmfModel::Illustris,
            evolution_e_factor: 30.0,
            evolution_density_alpha: 1.5,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }

    /// LF-based constructor with Karmen+2025 rate evolution.
    ///
    /// Applies F(z) × N_BH(z) × O(z) weighting to the redshift distribution,
    /// accounting for galaxy evolution, BHMF decline, and dust obscuration.
    pub fn from_luminosity_function_evolved(
        z_max: f64,
        bhmf_model: crate::efficiency::tde::BhmfModel,
        mjd_min: f64,
        mjd_max: f64,
    ) -> Self {
        let mut pop = Self::from_luminosity_function(z_max, mjd_min, mjd_max);
        pop.use_rate_evolution = true;
        pop.bhmf_model = bhmf_model;
        pop
    }

    /// Draw a peak absolute magnitude from the broken power-law LF via rejection sampling.
    fn sample_mag_from_lf(&self, rng: &mut dyn rand::RngCore) -> f64 {
        use rand::Rng;
        // Find the envelope (maximum of the LF shape in mag range).
        // The LF peaks at the faint end (highest mag_max → lowest logLg) since gamma1 < 1.
        let log_lg_faint = (88.6 - self.lf_mag_max) / 2.5;
        let envelope = lf_shape(log_lg_faint, self.lf_log_lbk, self.lf_gamma1, self.lf_gamma2);

        loop {
            let mag: f64 = rng.random::<f64>() * (self.lf_mag_max - self.lf_mag_min) + self.lf_mag_min;
            let log_lg = (88.6 - mag) / 2.5;
            let prob = lf_shape(log_lg, self.lf_log_lbk, self.lf_gamma1, self.lf_gamma2) / envelope;
            if rng.random::<f64>() < prob {
                return mag;
            }
        }
    }
}

impl PopulationGenerator for TdePopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let mut instances = Vec::with_capacity(n);

        // Pre-compute rejection sampling envelope.
        let (use_evolved, envelope) = if self.use_rate_evolution {
            let env = distributions::max_dvdz_evolved(
                self.z_max,
                &self.cosmology,
                &self.bhmf_model,
                self.evolution_e_factor,
                self.evolution_density_alpha,
            );
            (true, env)
        } else {
            (false, distributions::max_dvdz(self.z_max, &self.cosmology))
        };

        for _ in 0..n {
            let z = if use_evolved {
                distributions::sample_redshift_evolved(
                    self.z_max,
                    &self.cosmology,
                    &self.bhmf_model,
                    self.evolution_e_factor,
                    self.evolution_density_alpha,
                    envelope,
                    rng,
                )
            } else {
                sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng)
            };
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();
            params.insert("m_bh".to_string(), sample_log_uniform(1e5, 1e8, rng));
            params.insert("m_star".to_string(), sample_gaussian_clamped(1.0, 0.5, 0.1, 10.0, rng));

            // TDE lightcurve timescale parameters (Yao et al. 2023 Table 4).
            // tau_rise: sigmoid rise timescale; t_1/2,rise ~ 1.1 × tau_rise.
            // Observed t_1/2,rise ~ 6–52 days (median ~20), so tau_rise ~ 5–47 (median ~18).
            let tau_rise = sample_gaussian_clamped(18.0, 10.0, 3.0, 60.0, rng);
            params.insert("log_tau_rise".to_string(), tau_rise.ln());

            // tau_fall: power-law decay timescale.  With alpha=5/3,
            // t_1/2,decline ~ 0.52 × tau_fall. Observed t_1/2,decline ~ 12–86 days
            // (median ~30), so tau_fall ~ 23–165 (median ~58).
            let tau_fall = sample_gaussian_clamped(58.0, 25.0, 10.0, 200.0, rng);
            params.insert("log_tau_fall".to_string(), tau_fall.ln());

            // alpha: power-law decay index. Theoretical TDE fallback = 5/3.
            let alpha = sample_gaussian_clamped(1.67, 0.3, 0.8, 3.0, rng);
            params.insert("alpha".to_string(), alpha);

            // log_a: ln(2) so that peak normalized flux ~ 1.0.
            params.insert("log_a".to_string(), 2.0_f64.ln());
            params.insert("b".to_string(), 0.0);

            let peak_mag = if self.use_luminosity_function {
                self.sample_mag_from_lf(rng)
            } else {
                sample_gaussian_clamped(
                    self.peak_abs_mag, 1.5,
                    self.peak_abs_mag - 5.0, self.peak_abs_mag + 5.0, rng,
                )
            };

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: peak_mag,
                transient_type: TransientType::Tde,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: sample_gaussian_clamped(0.1, 0.1, 0.0, 1.0, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::Tde
    }
}

/// Fast Blue Optical Transient (FBOT) population generator.
///
/// Parameters drawn from Ho et al. (2021, arXiv:2105.08811) Table 10:
/// 38 FBOTs from ZTF Phase I (March 2018 – October 2020).
///
/// Uses the Bazin model: F(t) = A × exp(-(t-t0)/tfall) / (1 + exp(-(t-t0)/trise)) + c
/// with timescales drawn from the observed sample distributions.
pub struct FbotPopulation {
    /// Volumetric rate in Gpc^-3 yr^-1.
    pub rate: f64,
    /// Maximum redshift.
    pub z_max: f64,
    /// Peak absolute magnitude (mean of Gaussian draw).
    pub peak_abs_mag: f64,
    /// MJD range for explosion times.
    pub mjd_min: f64,
    pub mjd_max: f64,
    /// Cosmology.
    pub cosmology: Cosmology,
}

impl FbotPopulation {
    pub fn new(rate: f64, z_max: f64, peak_abs_mag: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }
}

impl PopulationGenerator for FbotPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();

            // Bazin model params: [A, t0, tfall, trise, c]
            // A=1.0 (normalized), t0=0.0 (peak offset), c=0.0 (baseline).
            params.insert("A".to_string(), 1.0);
            params.insert("t0".to_string(), 0.0);
            params.insert("c".to_string(), 0.0);

            // Rise time: Ho+2021 Table 10 t_1/2,rise ~ 0.5–4.6 days (g-band).
            // Median ~2.5 days. Bazin trise is an e-folding time, roughly
            // trise ~ t_1/2,rise / ln(2) ≈ t_1/2,rise / 0.69.
            let t_half_rise = sample_gaussian_clamped(2.5, 1.2, 0.3, 6.0, rng);
            params.insert("trise".to_string(), t_half_rise / 0.693);

            // Fade time: Ho+2021 Table 10 t_1/2,fade ~ 1.5–8 days (g-band).
            // Median ~5.5 days. Bazin tfall ~ t_1/2,fade / ln(2).
            let t_half_fade = sample_gaussian_clamped(5.5, 2.0, 1.0, 15.0, rng);
            params.insert("tfall".to_string(), t_half_fade / 0.693);

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: sample_gaussian_clamped(
                    self.peak_abs_mag, 1.5,
                    self.peak_abs_mag - 4.0, self.peak_abs_mag + 4.0, rng,
                ),
                transient_type: TransientType::Fbot,
                model_params: params,
                mw_extinction_av: 0.02,
                host_extinction_av: sample_gaussian_clamped(0.2, 0.2, 0.0, 1.5, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::Fbot
    }
}

/// GRB Afterglow population generator.
pub struct AfterglowPopulation {
    pub rate: f64,
    pub z_max: f64,
    pub peak_abs_mag: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub cosmology: Cosmology,
}

impl AfterglowPopulation {
    pub fn new(rate: f64, z_max: f64, peak_abs_mag: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }
}

impl PopulationGenerator for AfterglowPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();
            params.insert("E_iso".to_string(), sample_log_uniform(1e50, 1e54, rng));
            params.insert("theta_obs".to_string(), sample_gaussian_clamped(0.1, 0.1, 0.0, 1.0, rng));
            params.insert("n_ism".to_string(), sample_log_uniform(1e-4, 1.0, rng));

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: sample_gaussian_clamped(
                    self.peak_abs_mag,
                    2.0,
                    self.peak_abs_mag - 5.0,
                    self.peak_abs_mag + 5.0,
                    rng,
                ),
                transient_type: TransientType::Afterglow,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: sample_gaussian_clamped(0.1, 0.2, 0.0, 2.0, rng),
            });
        }
        instances
    }

    fn volumetric_rate(&self) -> f64 {
        self.rate
    }

    fn transient_type(&self) -> TransientType {
        TransientType::Afterglow
    }
}
