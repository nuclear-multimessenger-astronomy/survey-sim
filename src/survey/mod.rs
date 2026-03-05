pub mod argus;
pub mod rubin;
pub mod ztf;

use crate::instrument::InstrumentConfig;
use crate::spatial::SpatialIndex;
use crate::types::{Band, SkyCoord};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SurveyError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
    #[error("Parquet error: {0}")]
    Parquet(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, SurveyError>;

/// A single observation from a survey schedule.
#[derive(Clone, Debug)]
pub struct SurveyObservation {
    /// Unique observation ID.
    pub obs_id: u64,
    /// Sky pointing center.
    pub coord: SkyCoord,
    /// Modified Julian Date of observation start.
    pub mjd: f64,
    /// Photometric band.
    pub band: Band,
    /// Five-sigma limiting magnitude.
    pub five_sigma_depth: f64,
    /// Effective seeing FWHM in arcsec.
    pub seeing_fwhm: f64,
    /// Visit exposure time in seconds.
    pub exposure_time: f64,
    /// Airmass at observation.
    pub airmass: f64,
    /// Sky brightness in mag/arcsec^2.
    pub sky_brightness: f64,
    /// Night number (integer).
    pub night: i64,
}

/// Trait for loading survey observations from various formats.
pub trait SurveyLoader: Send + Sync {
    fn load(&self) -> Result<Vec<SurveyObservation>>;
    fn bands(&self) -> Vec<Band>;
    fn name(&self) -> &str;
    /// Return the instrument configuration for this survey, if known.
    fn instrument(&self) -> Option<InstrumentConfig> {
        None
    }
}

/// Spatially-indexed store of survey observations for efficient querying.
pub struct SurveyStore {
    observations: Vec<SurveyObservation>,
    spatial_index: SpatialIndex,
    /// MJD range of the survey.
    pub mjd_min: f64,
    pub mjd_max: f64,
    /// Duration of the survey in years.
    pub duration_years: f64,
    /// Unique bands in the survey.
    pub bands: Vec<Band>,
    /// Instrument configuration, if known.
    pub instrument: Option<InstrumentConfig>,
    /// Field-of-view radius in degrees for cone queries.
    /// Derived from instrument config or set manually.
    pub fov_radius_deg: f64,
}

impl SurveyStore {
    /// Build a SurveyStore from a list of observations.
    ///
    /// `nside` controls the HEALPix resolution for spatial indexing.
    pub fn new(observations: Vec<SurveyObservation>, nside: u32) -> Self {
        let mjd_min = observations
            .iter()
            .map(|o| o.mjd)
            .fold(f64::INFINITY, f64::min);
        let mjd_max = observations
            .iter()
            .map(|o| o.mjd)
            .fold(f64::NEG_INFINITY, f64::max);
        let duration_years = (mjd_max - mjd_min) / 365.25;

        // Collect unique bands.
        let mut band_set = std::collections::HashSet::new();
        for obs in &observations {
            band_set.insert(obs.band.0.clone());
        }
        let bands: Vec<Band> = band_set.into_iter().map(Band).collect();

        // Build spatial index.
        let coords: Vec<(f64, f64)> = observations
            .iter()
            .map(|o| (o.coord.ra, o.coord.dec))
            .collect();
        let spatial_index = SpatialIndex::new(&coords, nside);

        Self {
            observations,
            spatial_index,
            mjd_min,
            mjd_max,
            duration_years,
            bands,
            instrument: None,
            fov_radius_deg: 1.75, // default: Rubin-like FoV
        }
    }

    /// Attach an instrument configuration to this store.
    /// Automatically sets the FoV radius from the instrument's field of view.
    pub fn with_instrument(mut self, instrument: InstrumentConfig) -> Self {
        // Derive FoV radius from instrument's fov_deg2.
        let fov_deg2 = instrument.detector.fov_deg2;
        if fov_deg2 > 0.0 {
            // Circular approximation: A = pi * r^2.
            self.fov_radius_deg = (fov_deg2 / std::f64::consts::PI).sqrt();
        }
        self.instrument = Some(instrument);
        self
    }

    /// Set the field-of-view radius for spatial cone queries.
    pub fn with_fov_radius(mut self, radius_deg: f64) -> Self {
        self.fov_radius_deg = radius_deg;
        self
    }

    /// Query observations overlapping a sky position within an MJD range.
    ///
    /// Uses MOC cone coverage (`cdshealpix::cone_coverage_approx`) to find
    /// all observation pointing centers within the survey FoV radius of the
    /// given position. This correctly handles arbitrary FoV sizes at any NSIDE.
    ///
    /// Returns indices into the observations vector.
    pub fn query(&self, coord: &SkyCoord, mjd_min: f64, mjd_max: f64) -> Vec<usize> {
        // Use cone query with FoV radius for proper spatial coverage.
        let candidates =
            self.spatial_index
                .query_cone(coord.ra, coord.dec, self.fov_radius_deg);

        // Filter by MJD range.
        candidates
            .into_iter()
            .filter(|&idx| {
                let obs = &self.observations[idx];
                obs.mjd >= mjd_min && obs.mjd <= mjd_max
            })
            .collect()
    }

    /// Query observations overlapping a sky position within an MJD range,
    /// with an additional angular separation filter.
    pub fn query_with_radius(
        &self,
        coord: &SkyCoord,
        mjd_min: f64,
        mjd_max: f64,
        radius_deg: f64,
    ) -> Vec<usize> {
        let candidates = self.spatial_index.query(coord.ra, coord.dec);

        candidates
            .into_iter()
            .filter(|&idx| {
                let obs = &self.observations[idx];
                obs.mjd >= mjd_min
                    && obs.mjd <= mjd_max
                    && coord.separation(&obs.coord) <= radius_deg
            })
            .collect()
    }

    /// Get an observation by index.
    pub fn get(&self, idx: usize) -> &SurveyObservation {
        &self.observations[idx]
    }

    /// Total number of observations.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Number of unique HEALPix pixels with observations.
    pub fn n_pixels(&self) -> usize {
        self.spatial_index.n_pixels()
    }

    /// NSIDE of the spatial index.
    pub fn nside(&self) -> u32 {
        self.spatial_index.nside()
    }

    /// Get all observations (for iteration).
    pub fn observations(&self) -> &[SurveyObservation] {
        &self.observations
    }
}
