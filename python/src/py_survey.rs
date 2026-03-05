use pyo3::prelude::*;
use std::collections::HashMap;

use survey_sim::instrument::InstrumentConfig;
use survey_sim::survey::argus::ArgusLoader;
use survey_sim::survey::rubin::RubinLoader;
use survey_sim::survey::ztf::{ZtfLoader, ZtfHdf5Loader, ZtfBoomLoader};
use survey_sim::survey::{SurveyLoader, SurveyStore};

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
    #[staticmethod]
    #[pyo3(signature = (parquet_paths, band="g", nside=64))]
    fn from_argus(parquet_paths: Vec<String>, band: &str, nside: u32) -> PyResult<Self> {
        let loader = ArgusLoader::new(parquet_paths, band);
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySurveyStore>()?;
    m.add_class::<PyInstrument>()?;
    Ok(())
}
