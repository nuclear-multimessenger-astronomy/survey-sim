use pyo3::prelude::*;
use pyo3::types::PyDict;

use survey_sim::efficiency::tde::{
    compute_tde_rate, BhmfModel, TdeLuminosityFunction, TdeRateSurvey,
};
use survey_sim::instrument::{BandConfig, InstrumentConfig};
use survey_sim::lightcurve::kcorrection;
use survey_sim::types::Cosmology;

/// Python wrapper for TDE rate forecast.
#[pyclass]
#[pyo3(name = "TdeRateForecast")]
pub struct PyTdeRateForecast {
    temperature_k: f64,
    n_mc: usize,
    seed: u64,
}

#[pymethods]
impl PyTdeRateForecast {
    #[new]
    #[pyo3(signature = (temperature_k=30000.0, n_mc=200, seed=42))]
    fn new(temperature_k: f64, n_mc: usize, seed: u64) -> Self {
        Self {
            temperature_k,
            n_mc,
            seed,
        }
    }

    /// Compute TDE rate for a built-in survey configuration.
    ///
    /// survey_name: one of "rubin", "rubin_ddf", "roman_wide", "roman_deep"
    /// bhmf_model: "illustris" or "shankar"
    #[pyo3(signature = (survey_name, bhmf_model="illustris"))]
    fn compute_rate<'py>(
        &self,
        py: Python<'py>,
        survey_name: &str,
        bhmf_model: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let survey = builtin_survey(survey_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown survey '{}'. Use: rubin, rubin_ddf, roman_wide, roman_deep",
                survey_name
            ))
        })?;
        let bhmf = parse_bhmf(bhmf_model)?;
        let cosmo = Cosmology::default();
        let lf = TdeLuminosityFunction::default();

        let result =
            compute_tde_rate(&survey, &lf, &bhmf, &cosmo, self.temperature_k, self.n_mc, self.seed);

        let dict = PyDict::new(py);
        dict.set_item("N_median", result.n_median)?;
        dict.set_item("N_16", result.n_16)?;
        dict.set_item("N_84", result.n_84)?;
        dict.set_item("z_median", result.z_median)?;
        dict.set_item("z_mean", result.z_mean)?;
        dict.set_item("z_max", result.z_max)?;
        dict.set_item("survey", survey.name.clone())?;
        dict.set_item("bhmf", bhmf_model)?;
        Ok(dict)
    }

    /// Compute TDE rate with custom survey parameters.
    #[pyo3(signature = (name, area_deg2, depth, central_wavelength_nm, width_nm,
                         bhmf_model="illustris", is_time_domain=true, seasonal_coverage=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn compute_rate_custom<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        area_deg2: f64,
        depth: f64,
        central_wavelength_nm: f64,
        width_nm: f64,
        bhmf_model: &str,
        is_time_domain: bool,
        seasonal_coverage: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let survey = TdeRateSurvey {
            name: name.to_string(),
            area_deg2,
            best_filter: BandConfig {
                central_wavelength_nm,
                width_nm,
                typical_seeing_arcsec: 1.0,
                single_visit_depth: depth,
                sky_brightness: 22.0,
            },
            m_limit: depth,
            is_time_domain,
            seasonal_coverage,
        };
        let bhmf = parse_bhmf(bhmf_model)?;
        let cosmo = Cosmology::default();
        let lf = TdeLuminosityFunction::default();

        let result =
            compute_tde_rate(&survey, &lf, &bhmf, &cosmo, self.temperature_k, self.n_mc, self.seed);

        let dict = PyDict::new(py);
        dict.set_item("N_median", result.n_median)?;
        dict.set_item("N_16", result.n_16)?;
        dict.set_item("N_84", result.n_84)?;
        dict.set_item("z_median", result.z_median)?;
        dict.set_item("z_mean", result.z_mean)?;
        dict.set_item("z_max", result.z_max)?;
        dict.set_item("survey", name)?;
        dict.set_item("bhmf", bhmf_model)?;
        Ok(dict)
    }
}

/// Broadband K-correction for a blackbody SED through a filter.
///
/// Returns K(z) such that m_obs = M + mu + K.
#[pyfunction]
#[pyo3(signature = (z, temperature_k, central_wavelength_nm, width_nm))]
fn blackbody_k_correction(
    z: f64,
    temperature_k: f64,
    central_wavelength_nm: f64,
    width_nm: f64,
) -> f64 {
    let band = BandConfig {
        central_wavelength_nm,
        width_nm,
        typical_seeing_arcsec: 1.0,
        single_visit_depth: 25.0,
        sky_brightness: 22.0,
    };
    kcorrection::k_correction_blackbody(temperature_k, &band, z)
}

/// K-correction for a blackbody through a named band of a built-in instrument.
///
/// instrument: "rubin", "ztf", "roman", "ultrasat", "uvex", "argus"
#[pyfunction]
#[pyo3(signature = (z, temperature_k, instrument, band_name))]
fn blackbody_k_correction_instrument(
    z: f64,
    temperature_k: f64,
    instrument: &str,
    band_name: &str,
) -> PyResult<f64> {
    let inst = builtin_instrument(instrument).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown instrument '{}'", instrument
        ))
    })?;
    kcorrection::k_correction_blackbody_named(temperature_k, &inst, band_name, z).ok_or_else(
        || {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Band '{}' not found in instrument '{}'",
                band_name, instrument
            ))
        },
    )
}

fn parse_bhmf(name: &str) -> PyResult<BhmfModel> {
    match name {
        "illustris" => Ok(BhmfModel::Illustris),
        "shankar" => Ok(BhmfModel::Shankar),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown BHMF model '{}'. Use 'illustris' or 'shankar'",
            name
        ))),
    }
}

fn builtin_instrument(name: &str) -> Option<InstrumentConfig> {
    match name {
        "rubin" => Some(InstrumentConfig::rubin()),
        "ztf" => Some(InstrumentConfig::ztf()),
        "roman" => Some(InstrumentConfig::roman()),
        "ultrasat" => Some(InstrumentConfig::ultrasat()),
        "uvex" => Some(InstrumentConfig::uvex()),
        "argus" => Some(InstrumentConfig::argus()),
        _ => None,
    }
}

fn builtin_survey(name: &str) -> Option<TdeRateSurvey> {
    match name {
        "rubin" => {
            let inst = InstrumentConfig::rubin();
            let band = inst.bands.get("g")?.clone();
            Some(TdeRateSurvey {
                name: "Rubin (LSST)".to_string(),
                area_deg2: 18000.0,
                best_filter: band.clone(),
                m_limit: band.single_visit_depth,
                is_time_domain: true,
                seasonal_coverage: 1.0,
            })
        }
        "rubin_ddf" => {
            let inst = InstrumentConfig::rubin();
            let band = inst.bands.get("g")?.clone();
            Some(TdeRateSurvey {
                name: "Rubin (LSST deep drilling)".to_string(),
                area_deg2: 50.0,
                best_filter: BandConfig {
                    single_visit_depth: 26.0,
                    ..band
                },
                m_limit: 26.0,
                is_time_domain: true,
                seasonal_coverage: 0.5,
            })
        }
        "roman_wide" => {
            let inst = InstrumentConfig::roman();
            let band = inst.bands.get("F062")?.clone();
            Some(TdeRateSurvey {
                name: "Roman HLTDS (wide tier)".to_string(),
                area_deg2: 19.0,
                best_filter: band.clone(),
                m_limit: band.single_visit_depth,
                is_time_domain: true,
                seasonal_coverage: 1.0,
            })
        }
        "roman_deep" => {
            let inst = InstrumentConfig::roman();
            let band = inst.bands.get("F062")?.clone();
            Some(TdeRateSurvey {
                name: "Roman HLTDS (deep tier)".to_string(),
                area_deg2: 6.0,
                best_filter: BandConfig {
                    single_visit_depth: 26.95,
                    ..band
                },
                m_limit: 26.95,
                is_time_domain: true,
                seasonal_coverage: 1.0,
            })
        }
        _ => None,
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTdeRateForecast>()?;
    m.add_function(wrap_pyfunction!(blackbody_k_correction, m)?)?;
    m.add_function(wrap_pyfunction!(blackbody_k_correction_instrument, m)?)?;
    Ok(())
}
