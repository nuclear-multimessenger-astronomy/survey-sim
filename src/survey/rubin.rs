use rusqlite::Connection;

use crate::instrument::InstrumentConfig;
use crate::types::{Band, SkyCoord};

use super::{Result, SurveyError, SurveyLoader, SurveyObservation};

/// Loads Rubin Observatory LSST observations from an OpSim SQLite database.
pub struct RubinLoader {
    db_path: String,
}

impl RubinLoader {
    pub fn new(db_path: &str) -> Self {
        Self {
            db_path: db_path.to_string(),
        }
    }
}

impl SurveyLoader for RubinLoader {
    fn load(&self) -> Result<Vec<SurveyObservation>> {
        let conn = Connection::open(&self.db_path)?;

        // Determine available table name — OpSim databases use either
        // "observations" or "SummaryAllProps".
        let table = detect_table(&conn)?;

        // Map columns based on schema.
        // Modern OpSim (v4+) schema uses these column names:
        let query = format!(
            "SELECT observationId, fieldRA, fieldDec, observationStartMJD, \
             filter, fiveSigmaDepth, seeingFwhmEff, visitExposureTime, \
             airmass, skyBrightness, night \
             FROM {table}"
        );

        let mut stmt = conn.prepare(&query)?;
        let observations = stmt
            .query_map([], |row| {
                let obs_id: i64 = row.get(0)?;
                let ra_deg: f64 = row.get(1)?;
                let dec_deg: f64 = row.get(2)?;
                let mjd: f64 = row.get(3)?;
                let band_str: String = row.get(4)?;
                let five_sigma_depth: f64 = row.get(5)?;
                let seeing_fwhm: f64 = row.get(6)?;
                let exposure_time: f64 = row.get(7)?;
                let airmass: f64 = row.get(8)?;
                let sky_brightness: f64 = row.get(9)?;
                let night: i64 = row.get(10)?;

                Ok(SurveyObservation {
                    obs_id: obs_id as u64,
                    coord: SkyCoord::new(ra_deg, dec_deg),
                    mjd,
                    band: Band::new(&band_str),
                    five_sigma_depth,
                    seeing_fwhm,
                    exposure_time,
                    airmass,
                    sky_brightness,
                    night,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        log::info!(
            "Loaded {} Rubin observations from {}",
            observations.len(),
            self.db_path
        );

        Ok(observations)
    }

    fn bands(&self) -> Vec<Band> {
        vec!["u", "g", "r", "i", "z", "y"]
            .into_iter()
            .map(Band::new)
            .collect()
    }

    fn name(&self) -> &str {
        "Rubin LSST"
    }

    fn instrument(&self) -> Option<InstrumentConfig> {
        Some(InstrumentConfig::rubin())
    }
}

/// Detect which table name the OpSim database uses.
fn detect_table(conn: &Connection) -> Result<String> {
    // Try modern schema first.
    let has_observations: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='observations'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|c| c > 0)?;

    if has_observations {
        return Ok("observations".to_string());
    }

    // Try legacy schema.
    let has_summary: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='SummaryAllProps'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|c| c > 0)?;

    if has_summary {
        return Ok("SummaryAllProps".to_string());
    }

    Err(SurveyError::InvalidData(
        "No recognized OpSim table found (expected 'observations' or 'SummaryAllProps')"
            .to_string(),
    ))
}
