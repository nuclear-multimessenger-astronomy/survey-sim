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
pub struct TdePopulation {
    pub rate: f64,
    pub z_max: f64,
    pub peak_abs_mag: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub cosmology: Cosmology,
}

impl TdePopulation {
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

impl PopulationGenerator for TdePopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();
            params.insert("m_bh".to_string(), sample_log_uniform(1e5, 1e8, rng));
            params.insert("m_star".to_string(), sample_gaussian_clamped(1.0, 0.5, 0.1, 10.0, rng));

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
