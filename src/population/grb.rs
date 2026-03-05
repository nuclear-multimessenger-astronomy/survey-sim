use std::collections::HashMap;
use std::sync::Arc;

use rand::Rng;
use serde::Deserialize;

use crate::types::{SkyCoord, TransientInstance, TransientType};

use super::distributions::{sample_explosion_time, sample_gaussian_clamped, sample_isotropic_sky};
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
                mw_extinction_av: sample_gaussian_clamped(0.1, 0.1, 0.0, 2.0, rng),
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
}
