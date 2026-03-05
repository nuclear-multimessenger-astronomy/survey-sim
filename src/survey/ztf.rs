use std::collections::HashMap;

use hdf5::H5Type;

use crate::instrument::InstrumentConfig;
use crate::types::{Band, SkyCoord};

use super::{Result, SurveyLoader, SurveyObservation};

/// Loads ZTF observations from a CSV observation log.
///
/// Expected columns: obsid, ra, dec, mjd, filter, maglim, seeing, exptime, airmass, skymag, night
pub struct ZtfLoader {
    csv_path: String,
}

impl ZtfLoader {
    pub fn new(csv_path: &str) -> Self {
        Self {
            csv_path: csv_path.to_string(),
        }
    }
}

impl SurveyLoader for ZtfLoader {
    fn load(&self) -> Result<Vec<SurveyObservation>> {
        let mut rdr = csv::Reader::from_path(&self.csv_path)?;
        let mut observations = Vec::new();

        for result in rdr.records() {
            let record = result?;
            let obs = SurveyObservation {
                obs_id: record[0].parse::<u64>().unwrap_or(0),
                coord: SkyCoord::new(
                    record[1].parse::<f64>().unwrap_or(0.0),
                    record[2].parse::<f64>().unwrap_or(0.0),
                ),
                mjd: record[3].parse::<f64>().unwrap_or(0.0),
                band: Band::new(&record[4]),
                five_sigma_depth: record[5].parse::<f64>().unwrap_or(0.0),
                seeing_fwhm: record[6].parse::<f64>().unwrap_or(0.0),
                exposure_time: record[7].parse::<f64>().unwrap_or(30.0),
                airmass: record[8].parse::<f64>().unwrap_or(1.0),
                sky_brightness: record[9].parse::<f64>().unwrap_or(21.0),
                night: record[10].parse::<i64>().unwrap_or(0),
            };
            observations.push(obs);
        }

        log::info!(
            "Loaded {} ZTF observations from {}",
            observations.len(),
            self.csv_path
        );

        Ok(observations)
    }

    fn bands(&self) -> Vec<Band> {
        vec!["g", "r", "i"].into_iter().map(Band::new).collect()
    }

    fn name(&self) -> &str {
        "ZTF"
    }

    fn instrument(&self) -> Option<InstrumentConfig> {
        Some(InstrumentConfig::ztf())
    }
}

// ---------------------------------------------------------------------------
// ZTF HDF5 loader (IRSA TAP per-year files)
// ---------------------------------------------------------------------------

/// Raw CCD-level row matching the HDF5 compound dtype.
#[derive(H5Type, Clone, Debug)]
#[repr(C)]
struct ZtfCcdObs {
    obsjd: f64,
    field: i16,
    rcid: i8,
    ra: f64,
    dec: f64,
    programid: i8,
    expid: i32,
    fid: i8,
    maglimit: f32,
    seeing: f32,
    airmass: f32,
    exptime: f32,
}

/// Loads ZTF observations from IRSA HDF5 files (one or more per-year files).
/// Aggregates CCD-level rows to per-exposure observations using median statistics.
pub struct ZtfHdf5Loader {
    h5_paths: Vec<String>,
}

impl ZtfHdf5Loader {
    pub fn new(paths: &[&str]) -> Self {
        Self {
            h5_paths: paths.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Aggregate CCD-level rows into per-exposure `SurveyObservation`s.
    fn aggregate(rows: Vec<ZtfCcdObs>) -> Vec<SurveyObservation> {
        // Group by expid.
        let mut groups: HashMap<i32, Vec<ZtfCcdObs>> = HashMap::new();
        for row in rows {
            groups.entry(row.expid).or_default().push(row);
        }

        let mut observations = Vec::with_capacity(groups.len());
        for (expid, ccds) in groups {
            let first = &ccds[0];
            let obsjd = first.obsjd;
            let fid = first.fid;
            let exptime = first.exptime;

            // Median helper: sort and pick middle value.
            let median_f64 = |vals: &mut Vec<f64>| -> f64 {
                vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let n = vals.len();
                if n % 2 == 0 {
                    (vals[n / 2 - 1] + vals[n / 2]) / 2.0
                } else {
                    vals[n / 2]
                }
            };

            let mut ras: Vec<f64> = ccds.iter().map(|c| c.ra).collect();
            let mut decs: Vec<f64> = ccds.iter().map(|c| c.dec).collect();
            let mut maglimits: Vec<f64> = ccds.iter().map(|c| c.maglimit as f64).collect();
            let mut seeings: Vec<f64> = ccds.iter().map(|c| c.seeing as f64).collect();
            let mut airmasses: Vec<f64> = ccds.iter().map(|c| c.airmass as f64).collect();

            let ra = median_f64(&mut ras);
            let dec = median_f64(&mut decs);
            let maglimit = median_f64(&mut maglimits);
            let seeing = median_f64(&mut seeings);
            let airmass = median_f64(&mut airmasses);

            let band_name = match fid {
                1 => "g",
                2 => "r",
                3 => "i",
                _ => "unknown",
            };

            let mjd = obsjd - 2_400_000.5;
            let night = (obsjd - 0.5).floor() as i64 - 2_400_000;

            observations.push(SurveyObservation {
                obs_id: expid as u64,
                coord: SkyCoord::new(ra, dec),
                mjd,
                band: Band::new(band_name),
                five_sigma_depth: maglimit,
                seeing_fwhm: seeing,
                exposure_time: exptime as f64,
                airmass,
                sky_brightness: 0.0, // not available in HDF5 metadata
                night,
            });
        }

        observations
    }
}

impl SurveyLoader for ZtfHdf5Loader {
    fn load(&self) -> Result<Vec<SurveyObservation>> {
        let mut all_rows: Vec<ZtfCcdObs> = Vec::new();

        for path in &self.h5_paths {
            let file = hdf5::File::open(path)?;
            let dataset = file.dataset("observations")?;
            let rows = dataset.read_1d::<ZtfCcdObs>()?;
            log::info!("Read {} CCD rows from {}", rows.len(), path);
            all_rows.extend(rows.to_vec());
        }

        let observations = Self::aggregate(all_rows);
        log::info!(
            "Aggregated to {} unique exposures from {} HDF5 file(s)",
            observations.len(),
            self.h5_paths.len()
        );

        Ok(observations)
    }

    fn bands(&self) -> Vec<Band> {
        vec!["g", "r", "i"].into_iter().map(Band::new).collect()
    }

    fn name(&self) -> &str {
        "ZTF-HDF5"
    }

    fn instrument(&self) -> Option<InstrumentConfig> {
        Some(InstrumentConfig::ztf())
    }
}

// ---------------------------------------------------------------------------
// ZTF Boom HDF5 loader (monthly per-CCD files from ztf_boom pipeline)
// ---------------------------------------------------------------------------

/// Raw CCD-level row matching the boom HDF5 compound dtype.
#[derive(H5Type, Clone, Debug)]
#[repr(C)]
struct ZtfBoomObs {
    obsjd: f64,
    field: i16,
    rcid: i8,
    ra: f64,
    dec: f64,
    nalertpackets: i16,
    programid: i8,
    expid: i32,
    fid: i8,
    scimaglim: f32,
    diffmaglim: f32,
    sciinpseeing: f32,
    difffwhm: f32,
    exptime: i16,
}

/// Loads ZTF observations from boom-pipeline monthly HDF5 files.
/// Aggregates CCD-level rows to per-exposure observations using median statistics.
pub struct ZtfBoomLoader {
    h5_paths: Vec<String>,
}

impl ZtfBoomLoader {
    pub fn new(paths: &[&str]) -> Self {
        Self {
            h5_paths: paths.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn aggregate(rows: Vec<ZtfBoomObs>) -> Vec<SurveyObservation> {
        // Group by (field, obsjd, fid) since expid is not populated in boom files.
        let mut groups: HashMap<(i16, i64, i8), Vec<ZtfBoomObs>> = HashMap::new();
        for row in rows {
            // Quantize obsjd to avoid floating-point grouping issues.
            // ZTF exposures are 30s apart minimum, so rounding to nearest second is safe.
            let obsjd_key = (row.obsjd * 86400.0).round() as i64;
            groups.entry((row.field, obsjd_key, row.fid)).or_default().push(row);
        }

        let mut obs_id_counter = 0u64;
        let mut observations = Vec::with_capacity(groups.len());
        for (_key, ccds) in groups {
            let first = &ccds[0];
            let obsjd = first.obsjd;
            let fid = first.fid;
            let exptime = first.exptime;

            let median_f64 = |vals: &mut Vec<f64>| -> f64 {
                vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let n = vals.len();
                if n % 2 == 0 {
                    (vals[n / 2 - 1] + vals[n / 2]) / 2.0
                } else {
                    vals[n / 2]
                }
            };

            let mut ras: Vec<f64> = ccds.iter().map(|c| c.ra).collect();
            let mut decs: Vec<f64> = ccds.iter().map(|c| c.dec).collect();
            let mut diffmaglims: Vec<f64> = ccds.iter().map(|c| c.diffmaglim as f64).collect();
            let mut seeings: Vec<f64> = ccds.iter().map(|c| c.sciinpseeing as f64).collect();

            let ra = median_f64(&mut ras);
            let dec = median_f64(&mut decs);
            let maglimit = median_f64(&mut diffmaglims);
            let seeing = median_f64(&mut seeings);

            let band_name = match fid {
                1 => "g",
                2 => "r",
                3 => "i",
                _ => "unknown",
            };

            let mjd = obsjd - 2_400_000.5;
            let night = (obsjd - 0.5).floor() as i64 - 2_400_000;

            observations.push(SurveyObservation {
                obs_id: obs_id_counter,
                coord: SkyCoord::new(ra, dec),
                mjd,
                band: Band::new(band_name),
                five_sigma_depth: maglimit,
                seeing_fwhm: seeing,
                exposure_time: exptime as f64,
                airmass: 1.0,
                sky_brightness: 0.0,
                night,
            });
            obs_id_counter += 1;
        }

        observations
    }
}

impl SurveyLoader for ZtfBoomLoader {
    fn load(&self) -> Result<Vec<SurveyObservation>> {
        let mut all_rows: Vec<ZtfBoomObs> = Vec::new();

        for path in &self.h5_paths {
            let file = hdf5::File::open(path)?;
            let dataset = file.dataset("observations")?;
            let rows = dataset.read_1d::<ZtfBoomObs>()?;
            log::info!("Read {} CCD rows from {}", rows.len(), path);
            all_rows.extend(rows.to_vec());
        }

        let observations = Self::aggregate(all_rows);
        log::info!(
            "Aggregated to {} unique exposures from {} boom HDF5 file(s)",
            observations.len(),
            self.h5_paths.len()
        );

        Ok(observations)
    }

    fn bands(&self) -> Vec<Band> {
        vec!["g", "r", "i"].into_iter().map(Band::new).collect()
    }

    fn name(&self) -> &str {
        "ZTF-Boom"
    }

    fn instrument(&self) -> Option<InstrumentConfig> {
        Some(InstrumentConfig::ztf())
    }
}
