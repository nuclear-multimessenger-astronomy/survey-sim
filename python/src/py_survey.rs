use pyo3::prelude::*;
use std::collections::HashMap;

use survey_sim::instrument::InstrumentConfig;
use survey_sim::survey::argus::ArgusLoader;
use survey_sim::survey::rubin::RubinLoader;
use survey_sim::survey::too::{
    self, TooTrigger,
};
use survey_sim::survey::ztf::{ZtfLoader, ZtfHdf5Loader, ZtfBoomLoader};
use survey_sim::survey::{SurveyLoader, SurveyStore};
use survey_sim::types::SkyCoord;

/// Python wrapper for SurveyStore.
#[pyclass]
pub struct PySurveyStore {
    pub(crate) inner: SurveyStore,
}

#[pymethods]
impl PySurveyStore {
    /// Load a Rubin OpSim SQLite database.
    #[staticmethod]
    #[pyo3(signature = (db_path, nside=64))]
    fn from_rubin(db_path: &str, nside: u32) -> PyResult<Self> {
        let loader = RubinLoader::new(db_path);
        let instrument = loader.instrument();
        let obs = loader
            .load()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut store = SurveyStore::new(obs, nside);
        if let Some(inst) = instrument {
            store = store.with_instrument(inst);
        }
        Ok(Self { inner: store })
    }

    /// Load a ZTF CSV observation log.
    #[staticmethod]
    #[pyo3(signature = (csv_path, nside=64))]
    fn from_ztf(csv_path: &str, nside: u32) -> PyResult<Self> {
        let loader = ZtfLoader::new(csv_path);
        let instrument = loader.instrument();
        let obs = loader
            .load()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut store = SurveyStore::new(obs, nside);
        if let Some(inst) = instrument {
            store = store.with_instrument(inst);
        }
        Ok(Self { inner: store })
    }

    /// Load ZTF observations from IRSA HDF5 files.
    #[staticmethod]
    #[pyo3(signature = (h5_paths, nside=64))]
    fn from_ztf_hdf5(h5_paths: Vec<String>, nside: u32) -> PyResult<Self> {
        let path_refs: Vec<&str> = h5_paths.iter().map(|s| s.as_str()).collect();
        let loader = ZtfHdf5Loader::new(&path_refs);
        let instrument = loader.instrument();
        let obs = loader
            .load()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut store = SurveyStore::new(obs, nside);
        if let Some(inst) = instrument {
            store = store.with_instrument(inst);
        }
        Ok(Self { inner: store })
    }

    /// Load ZTF observations from boom-pipeline monthly HDF5 files.
    #[staticmethod]
    #[pyo3(signature = (h5_paths, nside=64))]
    fn from_ztf_boom(h5_paths: Vec<String>, nside: u32) -> PyResult<Self> {
        let path_refs: Vec<&str> = h5_paths.iter().map(|s| s.as_str()).collect();
        let loader = ZtfBoomLoader::new(&path_refs);
        let instrument = loader.instrument();
        let obs = loader
            .load()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut store = SurveyStore::new(obs, nside);
        if let Some(inst) = instrument {
            store = store.with_instrument(inst);
        }
        Ok(Self { inner: store })
    }

    /// Load Argus Array observations from parquet files.
    ///
    /// If `stack_window_s` is provided, consecutive exposures within each
    /// time window (in seconds) are co-added. The stacked depth improves
    /// by 1.25 * log10(N) magnitudes, where N is the number of exposures
    /// in each bin.
    #[staticmethod]
    #[pyo3(signature = (parquet_paths, band="g", nside=64, stack_window_s=None))]
    fn from_argus(
        parquet_paths: Vec<String>,
        band: &str,
        nside: u32,
        stack_window_s: Option<f64>,
    ) -> PyResult<Self> {
        let mut loader = ArgusLoader::new(parquet_paths, band);
        if let Some(window) = stack_window_s {
            loader = loader.with_stacking(window);
        }
        let instrument = loader.instrument();
        let obs = loader
            .load()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let mut store = SurveyStore::new(obs, nside);
        if let Some(inst) = instrument {
            store = store.with_instrument(inst);
        }
        Ok(Self { inner: store })
    }

    /// Load Argus observations from a CSV pointing database (redback format).
    ///
    /// Expected columns: expMJD, _ra, _dec, filter, fiveSigmaDepth
    /// This matches the format used by Freeburn et al.'s SimulateAfterglows.py.
    #[staticmethod]
    #[pyo3(signature = (csv_path, nside=64))]
    fn from_argus_csv(csv_path: &str, nside: u32) -> PyResult<Self> {
        use std::io::BufRead;
        let file = std::fs::File::open(csv_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let reader = std::io::BufReader::new(file);
        let mut obs_vec = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            if i == 0 {
                continue; // skip header
            }
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 5 {
                continue;
            }
            let mjd: f64 = fields[0].parse().unwrap_or(0.0);
            let ra: f64 = fields[1].parse().unwrap_or(0.0);
            let dec: f64 = fields[2].parse().unwrap_or(0.0);
            let band_str = fields[3].trim();
            let depth: f64 = fields[4].parse().unwrap_or(0.0);

            // Map redback filter names to survey-sim bands.
            let band = match band_str {
                "sdssg" | "desg" | "lsstg" | "ztfg" | "argus_g" => "g",
                "sdssr" | "desr" | "lsstr" | "ztfr" | "argus_r" => "r",
                "sdssi" | "desi" | "lssti" | "ztfi" => "i",
                other => other,
            };

            obs_vec.push(survey_sim::survey::SurveyObservation {
                obs_id: (i - 1) as u64,
                coord: survey_sim::types::SkyCoord::new(ra, dec),
                mjd,
                band: survey_sim::types::Band::new(band),
                five_sigma_depth: depth,
                seeing_fwhm: 1.0,
                exposure_time: 1.0,
                airmass: 1.0,
                sky_brightness: 21.0,
                night: mjd.floor() as i64,
            });
        }

        eprintln!("[survey] Loaded {} observations from CSV: {}", obs_vec.len(), csv_path);
        let store = SurveyStore::new(obs_vec, nside);
        Ok(Self { inner: store })
    }

    /// Load observations from a Python list of dicts.
    ///
    /// Each dict must have: ra, dec, mjd, band, five_sigma_depth, seeing_fwhm,
    /// exposure_time, airmass, sky_brightness, night
    #[staticmethod]
    #[pyo3(signature = (observations, nside=64))]
    fn from_python(observations: Vec<Bound<'_, pyo3::types::PyDict>>, nside: u32) -> PyResult<Self> {
        let mut obs_vec = Vec::with_capacity(observations.len());
        for (i, d) in observations.iter().enumerate() {
            let get_f64 = |key: &str| -> PyResult<f64> {
                d.get_item(key)?
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                            "Missing key '{}' in observation {}",
                            key, i
                        ))
                    })?
                    .extract()
            };
            let get_str = |key: &str| -> PyResult<String> {
                d.get_item(key)?
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                            "Missing key '{}' in observation {}",
                            key, i
                        ))
                    })?
                    .extract()
            };

            obs_vec.push(survey_sim::survey::SurveyObservation {
                obs_id: i as u64,
                coord: survey_sim::types::SkyCoord::new(get_f64("ra")?, get_f64("dec")?),
                mjd: get_f64("mjd")?,
                band: survey_sim::types::Band::new(&get_str("band")?),
                five_sigma_depth: get_f64("five_sigma_depth")?,
                seeing_fwhm: get_f64("seeing_fwhm").unwrap_or(1.0),
                exposure_time: get_f64("exposure_time").unwrap_or(30.0),
                airmass: get_f64("airmass").unwrap_or(1.0),
                sky_brightness: get_f64("sky_brightness").unwrap_or(21.0),
                night: get_f64("night").unwrap_or(0.0) as i64,
            });
        }
        Ok(Self {
            inner: SurveyStore::new(obs_vec, nside),
        })
    }

    /// Number of observations in the store.
    #[getter]
    fn n_observations(&self) -> usize {
        self.inner.len()
    }

    /// MJD range of the survey.
    #[getter]
    fn mjd_range(&self) -> (f64, f64) {
        (self.inner.mjd_min, self.inner.mjd_max)
    }

    /// Duration of the survey in years.
    #[getter]
    fn duration_years(&self) -> f64 {
        self.inner.duration_years
    }

    /// Band names in the survey.
    #[getter]
    fn bands(&self) -> Vec<String> {
        self.inner.bands.iter().map(|b| b.0.clone()).collect()
    }

    /// Instrument configuration, if available.
    #[getter]
    fn instrument(&self) -> Option<PyInstrument> {
        self.inner
            .instrument
            .as_ref()
            .map(|i| PyInstrument { inner: i.clone() })
    }

    /// Add ToO follow-up observations for a trigger event.
    ///
    /// Args:
    ///     strategy_name: Name of the built-in ToO strategy
    ///         ("rubin_gold", "rubin_silver", "ztf", "ultrasat", "uvex")
    ///     ra: Trigger right ascension in degrees
    ///     dec: Trigger declination in degrees
    ///     trigger_mjd: MJD of the trigger event
    ///     localization_area_deg2: 90% credible localization area in sq deg
    ///     distance_mpc: Distance estimate in Mpc (optional)
    #[pyo3(signature = (strategy_name, ra, dec, trigger_mjd, localization_area_deg2, distance_mpc=None))]
    fn add_too(
        &mut self,
        strategy_name: &str,
        ra: f64,
        dec: f64,
        trigger_mjd: f64,
        localization_area_deg2: f64,
        distance_mpc: Option<f64>,
    ) -> PyResult<usize> {
        let strategy = too::builtin_strategy(strategy_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown ToO strategy: '{}'. Available: rubin_gold, rubin_silver, ztf, ultrasat, uvex",
                strategy_name
            ))
        })?;
        let trigger = TooTrigger {
            coord: SkyCoord::new(ra, dec),
            trigger_mjd,
            localization_area_deg2,
            distance_mpc,
        };
        let start_id = self.inner.len() as u64;
        let obs = strategy.generate_observations(&trigger, start_id);
        let n_added = obs.len();
        self.inner.add_observations(obs);
        Ok(n_added)
    }

    /// Create a SurveyStore from ToO observations only (no base survey).
    ///
    /// Args:
    ///     strategy_name: Name of the built-in ToO strategy
    ///     ra: Trigger right ascension in degrees
    ///     dec: Trigger declination in degrees
    ///     trigger_mjd: MJD of the trigger event
    ///     localization_area_deg2: 90% credible localization area in sq deg
    ///     distance_mpc: Distance estimate in Mpc (optional)
    ///     nside: HEALPix NSIDE for spatial indexing (default 64)
    #[staticmethod]
    #[pyo3(signature = (strategy_name, ra, dec, trigger_mjd, localization_area_deg2, distance_mpc=None, nside=64))]
    fn from_too(
        strategy_name: &str,
        ra: f64,
        dec: f64,
        trigger_mjd: f64,
        localization_area_deg2: f64,
        distance_mpc: Option<f64>,
        nside: u32,
    ) -> PyResult<Self> {
        let strategy = too::builtin_strategy(strategy_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown ToO strategy: '{}'. Available: rubin_gold, rubin_silver, ztf, ultrasat, uvex",
                strategy_name
            ))
        })?;
        let trigger = TooTrigger {
            coord: SkyCoord::new(ra, dec),
            trigger_mjd,
            localization_area_deg2,
            distance_mpc,
        };
        let obs = strategy.generate_observations(&trigger, 0);
        let instrument = strategy.instrument();
        let store = SurveyStore::from_too(obs, nside, instrument);
        Ok(Self { inner: store })
    }
}

/// Python wrapper for InstrumentConfig.
#[pyclass(name = "Instrument")]
pub struct PyInstrument {
    inner: InstrumentConfig,
}

#[pymethods]
impl PyInstrument {
    /// Load an instrument configuration from a YAML file.
    #[staticmethod]
    fn from_yaml(path: &str) -> PyResult<Self> {
        let config = InstrumentConfig::from_yaml(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner: config })
    }

    /// Built-in Rubin LSST instrument.
    #[staticmethod]
    fn rubin() -> Self {
        Self {
            inner: InstrumentConfig::rubin(),
        }
    }

    /// Built-in ZTF instrument.
    #[staticmethod]
    fn ztf() -> Self {
        Self {
            inner: InstrumentConfig::ztf(),
        }
    }

    /// Built-in ULTRASAT instrument.
    #[staticmethod]
    fn ultrasat() -> Self {
        Self {
            inner: InstrumentConfig::ultrasat(),
        }
    }

    /// Built-in UVEX instrument.
    #[staticmethod]
    fn uvex() -> Self {
        Self {
            inner: InstrumentConfig::uvex(),
        }
    }

    /// Built-in Argus Array instrument.
    #[staticmethod]
    fn argus() -> Self {
        Self {
            inner: InstrumentConfig::argus(),
        }
    }

    /// Instrument name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Instrument description.
    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    /// Band names defined by this instrument.
    #[getter]
    fn bands(&self) -> Vec<String> {
        self.inner.bands.keys().cloned().collect()
    }

    /// Field of view in square degrees.
    #[getter]
    fn fov_deg2(&self) -> f64 {
        self.inner.detector.fov_deg2
    }

    /// Telescope aperture in meters.
    #[getter]
    fn aperture_m(&self) -> f64 {
        self.inner.telescope.aperture_m
    }

    /// Default exposure time in seconds.
    #[getter]
    fn default_exposure_s(&self) -> f64 {
        self.inner.observing.default_exposure_s
    }

    /// Extinction coefficients (A_band / A_V).
    #[getter]
    fn extinction_coefficients(&self) -> HashMap<String, f64> {
        self.inner.extinction_coefficients.clone()
    }

    /// Get the extinction ratio for a specific band.
    fn extinction_ratio(&self, band: &str) -> f64 {
        self.inner.extinction_ratio(band)
    }
}

/// Python wrapper for GW observing scenario events.
#[pyclass(name = "GwEvent")]
pub struct PyGwEvent {
    pub(crate) inner: survey_sim::survey::observing_scenario::GwEvent,
}

#[pymethods]
impl PyGwEvent {
    #[getter]
    fn simulation_id(&self) -> u64 { self.inner.simulation_id }
    #[getter]
    fn ra(&self) -> f64 { self.inner.ra }
    #[getter]
    fn dec(&self) -> f64 { self.inner.dec }
    #[getter]
    fn distance_mpc(&self) -> f64 { self.inner.distance_mpc }
    #[getter]
    fn mass1(&self) -> f64 { self.inner.mass1 }
    #[getter]
    fn mass2(&self) -> f64 { self.inner.mass2 }
    #[getter]
    fn inclination(&self) -> f64 { self.inner.inclination }
    #[getter]
    fn snr(&self) -> f64 { self.inner.snr }
    #[getter]
    fn area_90(&self) -> f64 { self.inner.area_90 }
    #[getter]
    fn area_50(&self) -> f64 { self.inner.area_50 }
    #[getter]
    fn dist_mean(&self) -> f64 { self.inner.dist_mean }
    #[getter]
    fn dist_std(&self) -> f64 { self.inner.dist_std }
    #[getter]
    fn ifos(&self) -> &str { &self.inner.ifos }
    #[getter]
    fn is_bns(&self) -> bool { self.inner.is_bns() }
    #[getter]
    fn is_nsbh(&self) -> bool { self.inner.is_nsbh() }
    #[getter]
    fn is_bbh(&self) -> bool { self.inner.is_bbh() }
    #[getter]
    fn chirp_mass(&self) -> f64 { self.inner.chirp_mass() }

    /// Construct a GwEvent from individual values (used by HDF5 loader).
    #[staticmethod]
    #[pyo3(signature = (simulation_id, ra, dec, distance_mpc, mass1, mass2, inclination, snr, far, area_90, area_50, dist_mean, dist_std))]
    fn _from_values(
        simulation_id: u64, ra: f64, dec: f64, distance_mpc: f64,
        mass1: f64, mass2: f64, inclination: f64,
        snr: f64, far: f64,
        area_90: f64, area_50: f64, dist_mean: f64, dist_std: f64,
    ) -> Self {
        use survey_sim::survey::observing_scenario::GwEvent;
        PyGwEvent {
            inner: GwEvent {
                simulation_id,
                coinc_event_id: simulation_id,
                longitude: ra.to_radians(),
                latitude: dec.to_radians(),
                ra, dec, distance_mpc, mass1, mass2,
                spin1z: 0.0, spin2z: 0.0,
                inclination, snr, far,
                area_90, area_50, dist_mean, dist_std,
                ifos: String::new(),
            },
        }
    }

    fn __repr__(&self) -> String {
        let kind = if self.inner.is_bns() { "BNS" }
            else if self.inner.is_nsbh() { "NSBH" }
            else { "BBH" };
        format!(
            "GwEvent(id={}, {}, d={:.0}Mpc, area90={:.0}deg², snr={:.1})",
            self.inner.simulation_id, kind,
            self.inner.distance_mpc, self.inner.area_90, self.inner.snr,
        )
    }
}

/// Load GW events from an observing scenario run directory.
///
/// Args:
///     run_dir: Path to run directory (e.g. "runs/O5a/bgp/")
///
/// Returns:
///     List of GwEvent objects with truth parameters and skymap statistics.
#[pyfunction]
fn load_gw_events(run_dir: &str) -> PyResult<Vec<PyGwEvent>> {
    let events = survey_sim::survey::observing_scenario::load_observing_scenario(run_dir)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
    Ok(events.into_iter().map(|e| PyGwEvent { inner: e }).collect())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySurveyStore>()?;
    m.add_class::<PyInstrument>()?;
    m.add_class::<PyGwEvent>()?;
    m.add_function(wrap_pyfunction!(load_gw_events, m)?)?;
    Ok(())
}
