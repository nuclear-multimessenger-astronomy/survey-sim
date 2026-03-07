use std::collections::HashMap;
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
///
/// When `stack_window_s` is set, consecutive exposures at the same sky position
/// within each time window are co-added. The stacked depth improvement uses
/// an interpolation table matching the Argus website science requirements,
/// which account for read-noise-dominated single frames transitioning to
/// sky-noise-limited stacks.
pub struct ArgusLoader {
    parquet_paths: Vec<String>,
    band: String,
    /// Stacking window in seconds. If `None`, no stacking (raw exposures).
    stack_window_s: Option<f64>,
}

impl ArgusLoader {
    pub fn new(parquet_paths: Vec<String>, band: &str) -> Self {
        Self {
            parquet_paths,
            band: band.to_string(),
            stack_window_s: None,
        }
    }

    pub fn with_stacking(mut self, window_s: f64) -> Self {
        self.stack_window_s = Some(window_s);
        self
    }
}

/// Intermediate struct for raw observations before stacking.
struct RawObs {
    ra: f64,
    dec: f64,
    mjd: f64,
    limmag: f64,
    seeing: f64,
    exptime: f64,
    airmass: f64,
    sky_brightness: f64,
    healpix: i64,
}

impl SurveyLoader for ArgusLoader {
    fn load(&self) -> Result<Vec<SurveyObservation>> {
        let band = Band::new(&self.band);

        // First pass: read all unmasked observations.
        let mut raw_obs = Vec::new();

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

                // Try to read healpix column (used for stacking grouping).
                let healpix_col = batch
                    .column_by_name("healpix")
                    .and_then(|c| c.as_any().downcast_ref::<arrow::array::Int64Array>().cloned());

                for i in 0..n {
                    if masked.value(i) {
                        continue;
                    }

                    let alt_deg = alt.value(i);
                    let alt_rad = alt_deg.to_radians();
                    let airmass = 1.0 / alt_rad.sin();

                    let hpx = healpix_col.as_ref().map_or(0, |c| c.value(i));

                    raw_obs.push(RawObs {
                        ra: ra.value(i),
                        dec: dec.value(i),
                        mjd: epoch.value(i),
                        limmag: limmag.value(i),
                        seeing: seeing.value(i),
                        exptime: exptime.value(i),
                        airmass,
                        sky_brightness: sky_brightness.value(i),
                        healpix: hpx,
                    });
                }
            }
        }

        let n_raw = raw_obs.len();

        let observations = if let Some(window_s) = self.stack_window_s {
            stack_observations(raw_obs, window_s, &band)
        } else {
            // No stacking: emit raw observations.
            raw_obs
                .into_iter()
                .enumerate()
                .map(|(i, o)| SurveyObservation {
                    obs_id: i as u64,
                    coord: SkyCoord::new(o.ra, o.dec),
                    mjd: o.mjd,
                    band: band.clone(),
                    five_sigma_depth: o.limmag,
                    seeing_fwhm: o.seeing,
                    exposure_time: o.exptime,
                    airmass: o.airmass,
                    sky_brightness: o.sky_brightness,
                    night: o.mjd.floor() as i64,
                })
                .collect()
        };

        log::info!(
            "Loaded {} Argus observations from {} parquet file(s) ({} raw, stacking={:?})",
            observations.len(),
            self.parquet_paths.len(),
            n_raw,
            self.stack_window_s,
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

/// Compute depth boost for stacking N Argus exposures.
///
/// Argus 1-second frames are read-noise dominated, so stacking improves
/// faster than the sky-limited `1.25 * log10(N)`. The depth boost is
/// interpolated from the Argus website science requirements (g-band, 5σ):
///
///   N=1 → 0.0, N=60 → 3.2, N=900 → 4.7, N=3600 → 5.5, N=28800 → 6.4
///
/// Uses log-linear interpolation between these anchor points.
fn argus_depth_boost(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    // (log10(N), depth_boost) from Argus website science requirements.
    // Base depth: 16.8 mag (1-sec, 5σ, science requirement).
    // Requirements: 1min=20.0, 15min=21.5, 1hr=22.3, 1night=23.2
    const TABLE: [(f64, f64); 5] = [
        (0.0, 0.0),       // N=1
        (1.778, 3.2),     // N=60 (1 min): 20.0 - 16.8
        (2.954, 4.7),     // N=900 (15 min): 21.5 - 16.8
        (3.556, 5.5),     // N=3600 (1 hr): 22.3 - 16.8
        (4.459, 6.4),     // N=28800 (1 night): 23.2 - 16.8
    ];

    let log_n = (n as f64).log10();

    // Clamp to table range.
    if log_n >= TABLE[TABLE.len() - 1].0 {
        return TABLE[TABLE.len() - 1].1;
    }

    // Find bracketing interval and interpolate.
    for i in 1..TABLE.len() {
        if log_n <= TABLE[i].0 {
            let frac = (log_n - TABLE[i - 1].0) / (TABLE[i].0 - TABLE[i - 1].0);
            return TABLE[i - 1].1 + frac * (TABLE[i].1 - TABLE[i - 1].1);
        }
    }

    TABLE[TABLE.len() - 1].1
}

/// Group observations by (healpix, time_bin) and co-add within each bin.
///
/// Depth improvement uses `argus_depth_boost(N)`, an interpolation table
/// matching the Argus website science requirements.
///
/// The stacked observation gets:
///   - MJD: midpoint of the window
///   - five_sigma_depth: median single-frame depth + argus_depth_boost(N)
///   - exposure_time: sum of individual exposure times
///   - seeing, airmass, sky_brightness: median of contributing exposures
fn stack_observations(
    mut raw: Vec<RawObs>,
    window_s: f64,
    band: &Band,
) -> Vec<SurveyObservation> {
    let window_days = window_s / 86400.0;

    // Sort by (healpix, mjd) for grouping.
    raw.sort_by(|a, b| {
        a.healpix
            .cmp(&b.healpix)
            .then(a.mjd.partial_cmp(&b.mjd).unwrap())
    });

    // Group into (healpix, time_bin) buckets.
    let mut groups: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, o) in raw.iter().enumerate() {
        let time_bin = (o.mjd / window_days).floor() as i64;
        groups.entry((o.healpix, time_bin)).or_default().push(i);
    }

    let mut observations = Vec::with_capacity(groups.len());
    let mut obs_id: u64 = 0;

    for (_, indices) in &groups {
        let n = indices.len();
        if n == 0 {
            continue;
        }

        // Compute stacked properties.
        let mjd_mid = {
            let mjd_min = indices.iter().map(|&i| raw[i].mjd).fold(f64::INFINITY, f64::min);
            let mjd_max = indices.iter().map(|&i| raw[i].mjd).fold(f64::NEG_INFINITY, f64::max);
            (mjd_min + mjd_max) / 2.0
        };

        let total_exptime: f64 = indices.iter().map(|&i| raw[i].exptime).sum();

        // Median single-frame depth (sort a copy).
        let mut depths: Vec<f64> = indices.iter().map(|&i| raw[i].limmag).collect();
        depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_depth = depths[depths.len() / 2];

        // Stacking depth boost from Argus website science requirements.
        let depth_boost = argus_depth_boost(n);
        let stacked_depth = median_depth + depth_boost;

        // Median seeing, airmass, sky_brightness.
        let median_f64 = |getter: fn(&RawObs) -> f64| -> f64 {
            let mut vals: Vec<f64> = indices.iter().map(|&i| getter(&raw[i])).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vals[vals.len() / 2]
        };

        let ra = raw[indices[0]].ra;
        let dec = raw[indices[0]].dec;

        observations.push(SurveyObservation {
            obs_id,
            coord: SkyCoord::new(ra, dec),
            mjd: mjd_mid,
            band: band.clone(),
            five_sigma_depth: stacked_depth,
            seeing_fwhm: median_f64(|o| o.seeing),
            exposure_time: total_exptime,
            airmass: median_f64(|o| o.airmass),
            sky_brightness: median_f64(|o| o.sky_brightness),
            night: mjd_mid.floor() as i64,
        });
        obs_id += 1;
    }

    observations
}
