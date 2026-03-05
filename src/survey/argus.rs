use std::fs::File;

use arrow::array::{AsArray, BooleanArray};
use arrow::datatypes::Float64Type;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::instrument::InstrumentConfig;
use crate::types::{Band, SkyCoord};

use super::{Result, SurveyError, SurveyLoader, SurveyObservation};

/// Loads Argus Array observations from parquet files.
///
/// Each parquet file typically corresponds to a single HEALPix pixel of the
/// simulated Argus schedule. Rows where `masked == true` are skipped.
pub struct ArgusLoader {
    parquet_paths: Vec<String>,
    band: String,
}

impl ArgusLoader {
    pub fn new(parquet_paths: Vec<String>, band: &str) -> Self {
        Self {
            parquet_paths,
            band: band.to_string(),
        }
    }
}

impl SurveyLoader for ArgusLoader {
    fn load(&self) -> Result<Vec<SurveyObservation>> {
        let band = Band::new(&self.band);
        let mut observations = Vec::new();
        let mut obs_id: u64 = 0;

        for path in &self.parquet_paths {
            let file = File::open(path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .map_err(|e| SurveyError::Parquet(e.to_string()))?;
            let reader = builder
                .build()
                .map_err(|e| SurveyError::Parquet(e.to_string()))?;

            for batch in reader {
                let batch = batch.map_err(|e| SurveyError::Parquet(e.to_string()))?;
                let n = batch.num_rows();

                let ra = batch
                    .column_by_name("ra")
                    .ok_or_else(|| SurveyError::InvalidData("Missing column 'ra'".into()))?
                    .as_primitive::<Float64Type>();
                let dec = batch
                    .column_by_name("dec")
                    .ok_or_else(|| SurveyError::InvalidData("Missing column 'dec'".into()))?
                    .as_primitive::<Float64Type>();
                let epoch = batch
                    .column_by_name("epoch")
                    .ok_or_else(|| SurveyError::InvalidData("Missing column 'epoch'".into()))?
                    .as_primitive::<Float64Type>();
                let limmag = batch
                    .column_by_name("limmag")
                    .ok_or_else(|| SurveyError::InvalidData("Missing column 'limmag'".into()))?
                    .as_primitive::<Float64Type>();
                let seeing = batch
                    .column_by_name("seeing")
                    .ok_or_else(|| SurveyError::InvalidData("Missing column 'seeing'".into()))?
                    .as_primitive::<Float64Type>();
                let exptime = batch
                    .column_by_name("exptime")
                    .ok_or_else(|| SurveyError::InvalidData("Missing column 'exptime'".into()))?
                    .as_primitive::<Float64Type>();
                let alt = batch
                    .column_by_name("alt")
                    .ok_or_else(|| SurveyError::InvalidData("Missing column 'alt'".into()))?
                    .as_primitive::<Float64Type>();
                let sky_brightness = batch
                    .column_by_name("sky_brightness")
                    .ok_or_else(|| {
                        SurveyError::InvalidData("Missing column 'sky_brightness'".into())
                    })?
                    .as_primitive::<Float64Type>();
                let masked = batch
                    .column_by_name("masked")
                    .ok_or_else(|| SurveyError::InvalidData("Missing column 'masked'".into()))?
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        SurveyError::InvalidData("Column 'masked' is not boolean".into())
                    })?;

                for i in 0..n {
                    // Skip masked observations.
                    if masked.value(i) {
                        continue;
                    }

                    let alt_deg = alt.value(i);
                    let alt_rad = alt_deg.to_radians();
                    let airmass = 1.0 / alt_rad.sin();
                    let mjd = epoch.value(i);

                    observations.push(SurveyObservation {
                        obs_id,
                        coord: SkyCoord::new(ra.value(i), dec.value(i)),
                        mjd,
                        band: band.clone(),
                        five_sigma_depth: limmag.value(i),
                        seeing_fwhm: seeing.value(i),
                        exposure_time: exptime.value(i),
                        airmass,
                        sky_brightness: sky_brightness.value(i),
                        night: mjd.floor() as i64,
                    });
                    obs_id += 1;
                }
            }
        }

        log::info!(
            "Loaded {} Argus observations from {} parquet file(s)",
            observations.len(),
            self.parquet_paths.len()
        );

        Ok(observations)
    }

    fn bands(&self) -> Vec<Band> {
        vec![Band::new(&self.band)]
    }

    fn name(&self) -> &str {
        "Argus Array"
    }

    fn instrument(&self) -> Option<InstrumentConfig> {
        Some(InstrumentConfig::argus())
    }
}
