use std::collections::HashMap;
use std::sync::Arc;

use rand::Rng;
use serde::Deserialize;

use crate::types::{Cosmology, SkyCoord, TransientInstance, TransientType};

use super::distributions::{self, sample_explosion_time, sample_isotropic_sky, sample_redshift_volumetric};
use super::PopulationGenerator;

/// A single row from the GRB afterglow parameter catalog CSV.
#[derive(Debug, Clone, Deserialize)]
struct GrbRow {
    z: f64,
    #[serde(rename = "d_L")]
    d_l_cm: f64,
    #[serde(rename = "Eiso")]
    eiso: f64,
    #[serde(rename = "Gamma_0")]
    gamma_0: f64,
    thv: f64,
    logn0: f64,
    logepse: f64,
    #[serde(rename = "logepsB")]
    logepsb: f64,
    logthc: f64,
    p: f64,
    av: f64,
    p_rvs: f64,
    logepse_rvs: f64,
    #[serde(rename = "logepsB_rvs")]
    logepsb_rvs: f64,
    peak_mag: f64,
}

/// Shared catalog of pre-drawn GRB parameter sets loaded from CSV.
pub struct GrbCatalog {
    rows: Vec<GrbRow>,
}

impl GrbCatalog {
    /// Load the catalog from a CSV file.
    pub fn from_csv(path: &str) -> std::io::Result<Self> {
        let mut reader = csv::Reader::from_path(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut rows = Vec::new();
        for result in reader.deserialize() {
            let row: GrbRow =
                result.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            rows.push(row);
        }
        Ok(Self { rows })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Cm-to-Mpc conversion factor.
const MPC_CM: f64 = 3.09e24;

/// GRB afterglow population generator that samples from a pre-computed CSV catalog.
pub struct GrbPopulation {
    pub catalog: Arc<GrbCatalog>,
    pub rate: f64,
    pub z_max: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
}

impl GrbPopulation {
    pub fn new(catalog: Arc<GrbCatalog>, rate: f64, z_max: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            catalog,
            rate,
            z_max,
            mjd_min,
            mjd_max,
        }
    }
}

impl PopulationGenerator for GrbPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let nrows = self.catalog.rows.len();
        if nrows == 0 {
            return Vec::new();
        }

        let mut instances = Vec::with_capacity(n);
        for _ in 0..n {
            // Sample a random row with replacement.
            let idx: usize = rng.random_range(0..nrows);
            let row = &self.catalog.rows[idx];

            // Use the row's redshift and luminosity distance.
            let z = row.z;
            let d_l = row.d_l_cm / MPC_CM; // convert cm -> Mpc

            // Random sky position and explosion time.
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            // Build model parameters from the catalog row.
            let mut params = HashMap::new();
            params.insert("Eiso".to_string(), row.eiso);
            params.insert("Gamma_0".to_string(), row.gamma_0);
            params.insert("theta_v".to_string(), row.thv);
            params.insert("logthc".to_string(), row.logthc);
            params.insert("logn0".to_string(), row.logn0);
            params.insert("logepse".to_string(), row.logepse);
            params.insert("logepsB".to_string(), row.logepsb);
            params.insert("p".to_string(), row.p);
            params.insert("av".to_string(), row.av);
            params.insert("p_rvs".to_string(), row.p_rvs);
            params.insert("logepse_rvs".to_string(), row.logepse_rvs);
            params.insert("logepsB_rvs".to_string(), row.logepsb_rvs);

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: row.peak_mag, // informational; blastwave computes mags from physics
                transient_type: TransientType::Afterglow,
                model_params: params,
                mw_extinction_av: 0.02, // high galactic latitude assumption
                host_extinction_av: row.av,
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

/// On-axis GRB afterglow population: only catalog rows with θ_v ≤ θ_j.
///
/// Uses catalog redshifts and luminosity distances directly (these come from
/// the gamma-ray-detected GRB redshift distribution). Appropriate for predicting
/// detection rates of prompt-emission-triggered afterglows.
pub struct OnAxisGrbPopulation {
    pub catalog: Arc<GrbCatalog>,
    /// Indices into catalog.rows that are on-axis.
    on_axis_indices: Vec<usize>,
    pub rate: f64,
    pub z_max: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
}

impl OnAxisGrbPopulation {
    pub fn new(catalog: Arc<GrbCatalog>, rate: f64, z_max: f64, mjd_min: f64, mjd_max: f64) -> Self {
        let on_axis_indices: Vec<usize> = catalog
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| {
                let theta_j = 10.0_f64.powf(row.logthc);
                row.thv <= theta_j
            })
            .map(|(i, _)| i)
            .collect();
        Self {
            catalog,
            on_axis_indices,
            rate,
            z_max,
            mjd_min,
            mjd_max,
        }
    }

    pub fn n_on_axis(&self) -> usize {
        self.on_axis_indices.len()
    }
}

impl PopulationGenerator for OnAxisGrbPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        if self.on_axis_indices.is_empty() {
            return Vec::new();
        }
        let n_onaxis = self.on_axis_indices.len();
        let mut instances = Vec::with_capacity(n);

        for _ in 0..n {
            let pick: usize = rng.random_range(0..n_onaxis);
            let row = &self.catalog.rows[self.on_axis_indices[pick]];

            let z = row.z;
            let d_l = row.d_l_cm / MPC_CM;
            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();
            params.insert("Eiso".to_string(), row.eiso);
            params.insert("Gamma_0".to_string(), row.gamma_0);
            params.insert("theta_v".to_string(), row.thv);
            params.insert("logthc".to_string(), row.logthc);
            params.insert("logn0".to_string(), row.logn0);
            params.insert("logepse".to_string(), row.logepse);
            params.insert("logepsB".to_string(), row.logepsb);
            params.insert("p".to_string(), row.p);
            params.insert("av".to_string(), row.av);
            params.insert("p_rvs".to_string(), row.p_rvs);
            params.insert("logepse_rvs".to_string(), row.logepse_rvs);
            params.insert("logepsB_rvs".to_string(), row.logepsb_rvs);

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: row.peak_mag,
                transient_type: TransientType::Afterglow,
                model_params: params,
                mw_extinction_av: 0.02,
                host_extinction_av: row.av,
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

/// Off-axis GRB afterglow population: volumetric redshift + isotropic viewing angle.
///
/// Samples jet intrinsic properties (Eiso, Gamma_0, microphysics) from catalog rows,
/// but draws redshift from the volumetric rate (uniform in comoving volume) and
/// viewing angle isotropically (uniform in cos θ, constrained to θ_v > θ_j).
/// This produces the physically correct distribution of "orphan afterglows"
/// that are not associated with a detected gamma-ray burst.
pub struct OffAxisGrbPopulation {
    pub catalog: Arc<GrbCatalog>,
    pub rate: f64,
    pub z_max: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    pub cosmology: Cosmology,
}

impl OffAxisGrbPopulation {
    pub fn new(catalog: Arc<GrbCatalog>, rate: f64, z_max: f64, mjd_min: f64, mjd_max: f64) -> Self {
        Self {
            catalog,
            rate,
            z_max,
            mjd_min,
            mjd_max,
            cosmology: Cosmology::default(),
        }
    }
}

/// Sample a viewing angle isotropically (uniform in cos θ) with θ_v > θ_j.
fn sample_offaxis_viewing_angle(theta_j: f64, rng: &mut dyn rand::RngCore) -> f64 {
    // cos(θ) uniform in [cos(π/2), cos(θ_j)] → θ in [θ_j, π/2]
    // We cap at π/2 since afterglows viewed from behind the jet are symmetric.
    let cos_min = 0.0_f64; // cos(π/2)
    let cos_max = theta_j.cos();
    loop {
        let cos_theta = cos_min + rng.random::<f64>() * (cos_max - cos_min);
        let theta = cos_theta.acos();
        if theta > theta_j {
            return theta;
        }
    }
}

impl PopulationGenerator for OffAxisGrbPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        let nrows = self.catalog.rows.len();
        if nrows == 0 {
            return Vec::new();
        }

        let envelope = distributions::max_dvdz(self.z_max, &self.cosmology);
        let mut instances = Vec::with_capacity(n);

        for _ in 0..n {
            // Sample jet properties from a random catalog row.
            let idx: usize = rng.random_range(0..nrows);
            let row = &self.catalog.rows[idx];

            // Volumetric redshift (uniform in comoving volume).
            let z = sample_redshift_volumetric(self.z_max, &self.cosmology, envelope, rng);
            let d_l = self.cosmology.luminosity_distance(z);

            // Isotropic viewing angle, constrained to off-axis.
            let theta_j = 10.0_f64.powf(row.logthc);
            let theta_v = sample_offaxis_viewing_angle(theta_j, rng);

            let (ra, dec) = sample_isotropic_sky(rng);
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();
            params.insert("Eiso".to_string(), row.eiso);
            params.insert("Gamma_0".to_string(), row.gamma_0);
            params.insert("theta_v".to_string(), theta_v);
            params.insert("logthc".to_string(), row.logthc);
            params.insert("logn0".to_string(), row.logn0);
            params.insert("logepse".to_string(), row.logepse);
            params.insert("logepsB".to_string(), row.logepsb);
            params.insert("p".to_string(), row.p);
            params.insert("av".to_string(), row.av);
            params.insert("p_rvs".to_string(), row.p_rvs);
            params.insert("logepse_rvs".to_string(), row.logepse_rvs);
            params.insert("logepsB_rvs".to_string(), row.logepsb_rvs);

            instances.push(TransientInstance {
                coord: SkyCoord::new(ra, dec),
                z,
                d_l,
                t_exp,
                peak_abs_mag: -99.0, // not meaningful for off-axis; computed by blastwave
                transient_type: TransientType::Afterglow,
                model_params: params,
                mw_extinction_av: 0.02,
                host_extinction_av: row.av,
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_grb_catalog_parse_header() {
        // Verify we can at least construct a GrbRow from known field names.
        let csv_data = "\
z,d_L,Eiso,Gamma_0,thv,logn0,logepse,logepsB,logthc,p,av,p_rvs,logepse_rvs,logepsB_rvs,peak_mag
1.0,3.09e28,1e52,100.0,0.1,-1.0,-1.0,-2.0,-1.0,2.3,0.1,2.3,-1.0,-2.0,22.0\n";
        let mut reader = csv::Reader::from_reader(csv_data.as_bytes());
        let row: GrbRow = reader.deserialize().next().unwrap().unwrap();
        assert!((row.z - 1.0).abs() < 1e-10);
        assert!((row.eiso - 1e52).abs() / 1e52 < 1e-10);
    }

    #[test]
    fn test_grb_population_generates() {
        let csv_data = "\
z,d_L,Eiso,Gamma_0,thv,logn0,logepse,logepsB,logthc,p,av,p_rvs,logepse_rvs,logepsB_rvs,peak_mag
1.0,3.09e28,1e52,100.0,0.1,-1.0,-1.0,-2.0,-1.0,2.3,0.1,2.3,-1.0,-2.0,22.0
0.5,1.5e28,1e51,50.0,0.2,-2.0,-1.5,-3.0,-0.8,2.5,0.2,2.5,-1.5,-3.0,24.0\n";
        let mut reader = csv::Reader::from_reader(csv_data.as_bytes());
        let rows: Vec<GrbRow> = reader.deserialize().map(|r| r.unwrap()).collect();
        let catalog = Arc::new(GrbCatalog { rows });

        let pop = GrbPopulation::new(catalog, 1.0, 6.0, 60000.0, 60365.0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let instances = pop.generate(10, &mut rng);

        assert_eq!(instances.len(), 10);
        for inst in &instances {
            assert_eq!(inst.transient_type, TransientType::Afterglow);
            assert!(inst.t_exp >= 60000.0 && inst.t_exp <= 60365.0);
            assert!(inst.coord.ra >= 0.0 && inst.coord.ra <= 360.0);
            assert!(inst.model_params.contains_key("Eiso"));
            assert!(inst.model_params.contains_key("logepsB_rvs"));
        }
    }

    fn make_test_catalog() -> Arc<GrbCatalog> {
        // Row 1: on-axis (thv=0.05 < 10^(-1.0)=0.1)
        // Row 2: off-axis (thv=0.3 > 10^(-0.8)=0.158)
        let csv_data = "\
z,d_L,Eiso,Gamma_0,thv,logn0,logepse,logepsB,logthc,p,av,p_rvs,logepse_rvs,logepsB_rvs,peak_mag
1.0,3.09e28,1e52,100.0,0.05,-1.0,-1.0,-2.0,-1.0,2.3,0.1,2.3,-1.0,-2.0,22.0
2.0,6.0e28,1e53,200.0,0.3,-2.0,-1.5,-3.0,-0.8,2.5,0.2,2.5,-1.5,-3.0,24.0\n";
        let mut reader = csv::Reader::from_reader(csv_data.as_bytes());
        let rows: Vec<GrbRow> = reader.deserialize().map(|r| r.unwrap()).collect();
        Arc::new(GrbCatalog { rows })
    }

    #[test]
    fn test_on_axis_population_filters() {
        let catalog = make_test_catalog();
        let pop = OnAxisGrbPopulation::new(catalog, 1.3, 6.0, 60000.0, 60365.0);
        // Only row 0 is on-axis.
        assert_eq!(pop.n_on_axis(), 1);

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let instances = pop.generate(20, &mut rng);
        assert_eq!(instances.len(), 20);
        for inst in &instances {
            let thv = inst.model_params["theta_v"];
            let thc = 10.0_f64.powf(inst.model_params["logthc"]);
            assert!(thv <= thc, "on-axis population should only have thv <= thc");
        }
    }

    #[test]
    fn test_off_axis_population_viewing_angles() {
        let catalog = make_test_catalog();
        let pop = OffAxisGrbPopulation::new(catalog, 800.0, 1.0, 60000.0, 60365.0);

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let instances = pop.generate(100, &mut rng);
        assert_eq!(instances.len(), 100);
        for inst in &instances {
            let thv = inst.model_params["theta_v"];
            let thc = 10.0_f64.powf(inst.model_params["logthc"]);
            assert!(thv > thc, "off-axis population should only have thv > thc");
            assert!(inst.z <= 1.0, "redshift should be <= z_max");
            assert!(inst.z > 0.0, "redshift should be > 0");
        }
    }
}
