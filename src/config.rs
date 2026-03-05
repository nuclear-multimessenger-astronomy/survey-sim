use serde::{Deserialize, Serialize};

use crate::types::{Cosmology, TransientType};

/// Top-level simulation configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Cosmology parameters.
    #[serde(default)]
    pub cosmology: Cosmology,
    /// Number of transients to simulate per population.
    pub n_transients: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// HEALPix NSIDE for spatial indexing (must be power of 2).
    #[serde(default = "default_nside")]
    pub nside: u32,
    /// Optional path to an instrument YAML configuration file.
    #[serde(default)]
    pub instrument: Option<String>,
    /// Detection criteria.
    pub detection: DetectionConfig,
    /// Population configurations.
    pub populations: Vec<PopulationConfig>,
}

/// Detection criteria configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Minimum total detections above threshold.
    #[serde(default = "default_two")]
    pub min_detections: usize,
    /// Minimum number of bands with at least one detection.
    #[serde(default = "default_one")]
    pub min_bands: usize,
    /// Minimum detections required in a single band.
    #[serde(default = "default_one")]
    pub min_per_band: usize,
    /// Maximum timespan in days for required detections.
    #[serde(default = "default_max_timespan")]
    pub max_timespan_days: f64,
    /// SNR threshold (default 5.0, matching five_sigma_depth).
    #[serde(default = "default_snr")]
    pub snr_threshold: f64,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            min_detections: 2,
            min_bands: 1,
            min_per_band: 1,
            max_timespan_days: 30.0,
            snr_threshold: 5.0,
        }
    }
}

/// Population configuration for a single transient type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PopulationConfig {
    pub transient_type: TransientType,
    /// Volumetric rate in Gpc^-3 yr^-1.
    pub rate: f64,
    /// Maximum redshift.
    pub z_max: f64,
    /// Peak absolute magnitude.
    pub peak_abs_mag: f64,
    /// Additional parameters specific to the population.
    #[serde(default)]
    pub params: std::collections::HashMap<String, f64>,
}

fn default_nside() -> u32 {
    64
}

fn default_two() -> usize {
    2
}

fn default_one() -> usize {
    1
}

fn default_max_timespan() -> f64 {
    30.0
}

fn default_snr() -> f64 {
    5.0
}
