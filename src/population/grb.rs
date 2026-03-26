use std::collections::HashMap;
use std::sync::Arc;

use rand::Rng;
use serde::{Deserialize, Deserializer};

use crate::types::{Cosmology, SkyCoord, TransientInstance, TransientType};

use super::distributions::{self, sample_explosion_time, sample_isotropic_sky, sample_redshift_volumetric};
use super::PopulationGenerator;

/// Deserialize Python-style booleans (True/False) as well as standard true/false.
fn deserialize_python_bool<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "True" | "true" | "1" => Ok(true),
        "False" | "false" | "0" | "" => Ok(false),
        _ => Err(serde::de::Error::custom(format!("expected bool, got: {}", s))),
    }
}

/// Deserialize an f64 that may be empty (empty string → 0.0).
fn deserialize_optional_f64<'de, D>(deserializer: D) -> std::result::Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    if s.is_empty() || s == "nan" || s == "inf" {
        Ok(0.0)
    } else {
        s.parse::<f64>().map_err(serde::de::Error::custom)
    }
}

/// A single row from the GRB afterglow parameter catalog CSV.
#[derive(Debug, Clone, Deserialize)]
struct GrbRow {
    z: f64,
    #[serde(rename = "d_L")]
    d_l_cm: f64,
    #[serde(rename = "Eiso", deserialize_with = "deserialize_optional_f64")]
    eiso: f64,
    #[serde(rename = "Gamma_0", deserialize_with = "deserialize_optional_f64")]
    gamma_0: f64,
    #[serde(deserialize_with = "deserialize_optional_f64")]
    thv: f64,
    #[serde(deserialize_with = "deserialize_optional_f64")]
    logn0: f64,
    #[serde(deserialize_with = "deserialize_optional_f64")]
    logepse: f64,
    #[serde(rename = "logepsB", deserialize_with = "deserialize_optional_f64")]
    logepsb: f64,
    #[serde(deserialize_with = "deserialize_optional_f64")]
    logthc: f64,
    #[serde(deserialize_with = "deserialize_optional_f64")]
    p: f64,
    #[serde(deserialize_with = "deserialize_optional_f64")]
    av: f64,
    #[serde(deserialize_with = "deserialize_optional_f64")]
    p_rvs: f64,
    #[serde(deserialize_with = "deserialize_optional_f64")]
    logepse_rvs: f64,
    #[serde(rename = "logepsB_rvs", deserialize_with = "deserialize_optional_f64")]
    logepsb_rvs: f64,
    #[serde(deserialize_with = "deserialize_python_bool")]
    detectable: bool,
    #[serde(deserialize_with = "deserialize_optional_f64")]
    peak_mag: f64,
    /// Swift/BAT peak flux (erg/cm²/s, 15-150 keV).
    #[serde(rename = "Swift_flux", deserialize_with = "deserialize_optional_f64")]
    swift_flux: f64,
    /// Fermi/GBM peak flux (erg/cm²/s, 10-1000 keV).
    #[serde(rename = "Fermi_flux", deserialize_with = "deserialize_optional_f64")]
    fermi_flux: f64,
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

/// On-axis GRB afterglow population (Freeburn et al. catalog).
///
/// The catalog CSV contains GRBs already filtered to on-axis viewing
/// (θ_v ≤ max(θ_γ,j, 1/Γ_0)) per the gamma-ray jet criterion.
/// All catalog rows are on-axis. The `detectable` flag (peak R_c < 24.5 mag)
/// pre-filters to only simulate GRBs bright enough for optical detection,
/// matching Freeburn's methodology.
pub struct OnAxisGrbPopulation {
    pub catalog: Arc<GrbCatalog>,
    /// Indices into catalog.rows eligible for sampling (detectable=true).
    eligible_indices: Vec<usize>,
    /// Total number of on-axis GRBs in catalog (all rows).
    pub n_total: usize,
    /// Number of Swift/BAT-detectable GRBs (flux > 1.9e-8 erg/cm²/s).
    pub n_swift: usize,
    /// Number of Fermi/GBM-detectable GRBs (flux > 7.5e-8 erg/cm²/s).
    pub n_fermi: usize,
    pub rate: f64,
    pub z_max: f64,
    pub mjd_min: f64,
    pub mjd_max: f64,
    /// If set, place all transients at this (RA, Dec) instead of random sky.
    pub fixed_coord: Option<(f64, f64)>,
}

impl OnAxisGrbPopulation {
    pub fn new(catalog: Arc<GrbCatalog>, rate: f64, z_max: f64, mjd_min: f64, mjd_max: f64) -> Self {
        let n_total = catalog.rows.len();

        // Count instrument-detectable GRBs (Freeburn Table 2 thresholds).
        let n_swift = catalog.rows.iter()
            .filter(|r| r.swift_flux > 1.9e-8)
            .count();
        let n_fermi = catalog.rows.iter()
            .filter(|r| r.fermi_flux > 7.5e-8)
            .count();

        // Sample from detectable GRBs only (peak R_c < 24.5 mag).
        let eligible_indices: Vec<usize> = catalog
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.detectable)
            .map(|(i, _)| i)
            .collect();

        eprintln!("[pop] Catalog: {} total on-axis, {} detectable (peak<24.5), {} Swift/BAT, {} Fermi/GBM",
            n_total, eligible_indices.len(), n_swift, n_fermi);

        Self {
            catalog,
            eligible_indices,
            n_total,
            n_swift,
            n_fermi,
            rate,
            z_max,
            mjd_min,
            mjd_max,
            fixed_coord: None,
        }
    }

    pub fn with_fixed_coord(mut self, ra: f64, dec: f64) -> Self {
        self.fixed_coord = Some((ra, dec));
        self
    }

    pub fn n_eligible(&self) -> usize {
        self.eligible_indices.len()
    }
}

impl PopulationGenerator for OnAxisGrbPopulation {
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance> {
        if self.eligible_indices.is_empty() {
            return Vec::new();
        }
        let n_eligible = self.eligible_indices.len();
        let mut instances = Vec::with_capacity(n);

        for _ in 0..n {
            let pick: usize = rng.random_range(0..n_eligible);
            let row = &self.catalog.rows[self.eligible_indices[pick]];

            let z = row.z;
            let d_l = row.d_l_cm / MPC_CM;
            let (ra, dec) = if let Some((ra, dec)) = self.fixed_coord {
                (ra, dec)
            } else {
                sample_isotropic_sky(rng)
            };
            let t_exp = sample_explosion_time(self.mjd_min, self.mjd_max, rng);

            let mut params = HashMap::new();
            params.insert("Eiso".to_string(), row.eiso);
            params.insert("Gamma_0".to_string(), row.gamma_0);
            // Freeburn et al.: afterglow computed with θ_v=0 (on-axis).
            // The catalog thv is the viewing angle relative to the γ-ray jet,
            // not the afterglow jet. θ_c (afterglow) is decoupled from θ_γ,j.
            params.insert("theta_v".to_string(), 0.0);
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
z,d_L,Eiso,Gamma_0,thv,logn0,logepse,logepsB,logthc,p,av,p_rvs,logepse_rvs,logepsB_rvs,detectable,peak_mag,Swift_flux,Fermi_flux
1.0,3.09e28,1e52,100.0,0.1,-1.0,-1.0,-2.0,-1.0,2.3,0.1,2.3,-1.0,-2.0,True,22.0,1e-7,1e-7\n";
        let mut reader = csv::Reader::from_reader(csv_data.as_bytes());
        let row: GrbRow = reader.deserialize().next().unwrap().unwrap();
        assert!((row.z - 1.0).abs() < 1e-10);
        assert!((row.eiso - 1e52).abs() / 1e52 < 1e-10);
    }

    #[test]
    fn test_grb_population_generates() {
        let csv_data = "\
z,d_L,Eiso,Gamma_0,thv,logn0,logepse,logepsB,logthc,p,av,p_rvs,logepse_rvs,logepsB_rvs,detectable,peak_mag,Swift_flux,Fermi_flux
1.0,3.09e28,1e52,100.0,0.1,-1.0,-1.0,-2.0,-1.0,2.3,0.1,2.3,-1.0,-2.0,True,22.0,1e-7,1e-7
0.5,1.5e28,1e51,50.0,0.2,-2.0,-1.5,-3.0,-0.8,2.5,0.2,2.5,-1.5,-3.0,True,24.0,1e-8,1e-8\n";
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
z,d_L,Eiso,Gamma_0,thv,logn0,logepse,logepsB,logthc,p,av,p_rvs,logepse_rvs,logepsB_rvs,detectable,peak_mag,Swift_flux,Fermi_flux
1.0,3.09e28,1e52,100.0,0.05,-1.0,-1.0,-2.0,-1.0,2.3,0.1,2.3,-1.0,-2.0,True,22.0,1e-7,1e-7
2.0,6.0e28,1e53,200.0,0.3,-2.0,-1.5,-3.0,-0.8,2.5,0.2,2.5,-1.5,-3.0,True,24.0,1e-8,1e-8\n";
        let mut reader = csv::Reader::from_reader(csv_data.as_bytes());
        let rows: Vec<GrbRow> = reader.deserialize().map(|r| r.unwrap()).collect();
        Arc::new(GrbCatalog { rows })
    }

    #[test]
    fn test_on_axis_population_filters() {
        let catalog = make_test_catalog();
        let pop = OnAxisGrbPopulation::new(catalog, 1.3, 6.0, 60000.0, 60365.0);
        // Both rows have detectable=True, so both are eligible.
        assert_eq!(pop.n_eligible(), 2);

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let instances = pop.generate(20, &mut rng);
        assert_eq!(instances.len(), 20);
        for inst in &instances {
            // θ_v should always be 0 (Freeburn convention: on-axis for afterglow).
            let thv = inst.model_params["theta_v"];
            assert!((thv - 0.0).abs() < 1e-10, "on-axis population should set thv=0");
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
