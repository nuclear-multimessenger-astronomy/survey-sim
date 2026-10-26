use pyo3::prelude::*;
use pyo3::types::PyDict;

use survey_sim::efficiency::imposters::{
    ImposterPopulation, KilonovaImposterAnalysis, KilonovaSignal,
};
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


/// Python wrapper for Kilonova Imposter and Detection Efficiency Analysis.
#[pyclass]
#[pyo3(name = "KilonovaImposterAnalysis")]
#[derive(Clone)]
pub struct PyKilonovaImposterAnalysis {
    pub(crate) inner: KilonovaImposterAnalysis,
}

#[pymethods]
impl PyKilonovaImposterAnalysis {
    #[staticmethod]
    #[pyo3(signature = (
        signal_name,
        volumetric_rate,
        expected_detections,
        effective_vt,
        overall_efficiency,
        imposters,
        followup_budget=None
    ))]
    pub fn from_counts(
        signal_name: &str,
        volumetric_rate: f64,
        expected_detections: f64,
        effective_vt: f64,
        overall_efficiency: f64,
        imposters: Vec<(String, f64, Option<f64>)>,
        followup_budget: Option<f64>,
    ) -> Self {
        let kn = KilonovaSignal::new(
            signal_name,
            volumetric_rate,
            expected_detections,
            effective_vt,
            overall_efficiency,
        );

        let imp_pops: Vec<ImposterPopulation> = imposters
            .into_iter()
            .map(|(name, count, sys_err)| {
                ImposterPopulation::from_count(&name, count, sys_err.unwrap_or(0.20))
            })
            .collect();

        let mut analysis = KilonovaImposterAnalysis::new(kn, imp_pops);
        if let Some(budget) = followup_budget {
            analysis = analysis.with_followup_budget(budget);
        }

        Self { inner: analysis }
    }

    #[staticmethod]
    #[pyo3(signature = (result, signal_type="Kilonova", default_sys_err=0.20, followup_budget=None))]
    pub fn from_simulation_result(
        result: &crate::py_pipeline::PySimulationResult,
        signal_type: &str,
        default_sys_err: f64,
        followup_budget: Option<f64>,
    ) -> PyResult<Self> {
        result.analyze_imposters(signal_type, default_sys_err, followup_budget)
    }

    #[staticmethod]
    #[pyo3(signature = (rate_summaries, signal_type="Kilonova", default_sys_err=0.20, followup_budget=None))]
    pub fn from_rate_summaries(
        rate_summaries: Vec<crate::py_pipeline::PyRateSummary>,
        signal_type: &str,
        default_sys_err: f64,
        followup_budget: Option<f64>,
    ) -> PyResult<Self> {
        let signal_summary = rate_summaries
            .iter()
            .find(|rs| rs.transient_type == signal_type)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Signal transient type '{}' not found in rate summaries",
                    signal_type
                ))
            })?;

        let imposter_summaries: Vec<crate::py_pipeline::PyRateSummary> = rate_summaries
            .iter()
            .filter(|rs| rs.transient_type != signal_type)
            .cloned()
            .collect();

        signal_summary.analyze_imposters(None, Some(imposter_summaries), default_sys_err, followup_budget)
    }

    #[getter]
    pub fn signal_name(&self) -> String {
        self.inner.kilonova.name.clone()
    }

    #[getter]
    pub fn signal_detections(&self) -> f64 {
        self.inner.metrics.signal_detections
    }

    #[getter]
    pub fn total_imposters(&self) -> f64 {
        self.inner.metrics.total_imposters
    }

    #[getter]
    pub fn total_candidates(&self) -> f64 {
        self.inner.metrics.total_detections
    }

    #[getter]
    pub fn imposter_to_signal_ratio(&self) -> f64 {
        self.inner.metrics.imposter_to_signal_ratio
    }

    #[getter]
    pub fn purity(&self) -> f64 {
        self.inner.metrics.purity
    }

    #[getter]
    pub fn false_discovery_rate(&self) -> f64 {
        self.inner.metrics.false_discovery_rate
    }

    #[getter]
    pub fn pure_efficiency(&self) -> f64 {
        self.inner.metrics.pure_efficiency
    }

    #[getter]
    pub fn raw_efficiency(&self) -> f64 {
        self.inner.kilonova.overall_efficiency
    }

    #[getter]
    pub fn f1_score(&self) -> f64 {
        self.inner.metrics.f1_score
    }

    pub fn f_beta(&self, beta: f64) -> f64 {
        self.inner.metrics.f_beta(beta)
    }

    #[getter]
    pub fn significance_fom(&self) -> f64 {
        self.inner.metrics.significance_fom
    }

    #[getter]
    pub fn naive_recovered_rate(&self) -> f64 {
        self.inner.rate_recovery.naive_recovered_rate
    }

    #[getter]
    pub fn rate_bias_absolute(&self) -> f64 {
        self.inner.rate_recovery.rate_bias_absolute
    }

    #[getter]
    pub fn rate_bias_fractional(&self) -> f64 {
        self.inner.rate_recovery.rate_bias_fractional
    }

    #[getter]
    pub fn naive_poisson_1sig_interval(&self) -> (f64, f64) {
        self.inner.rate_recovery.naive_poisson_1sig_interval
    }

    #[getter]
    pub fn decontaminated_recovered_rate(&self) -> f64 {
        self.inner.rate_recovery.decontaminated_recovered_rate
    }

    #[getter]
    pub fn decontaminated_1sig_interval(&self) -> (f64, f64) {
        self.inner.rate_recovery.decontaminated_1sig_interval
    }

    #[getter]
    pub fn decontaminated_2sig_interval(&self) -> (f64, f64) {
        self.inner.rate_recovery.decontaminated_2sig_interval
    }

    #[getter]
    pub fn variance_inflation_factor(&self) -> f64 {
        self.inner.rate_recovery.variance_inflation_factor
    }

    #[getter]
    pub fn uncertainty_inflation_factor(&self) -> f64 {
        self.inner.rate_recovery.uncertainty_inflation_factor
    }

    #[getter]
    pub fn consistent_1sig(&self) -> bool {
        self.inner.rate_recovery.consistent_1sig
    }

    #[getter]
    pub fn consistent_2sig(&self) -> bool {
        self.inner.rate_recovery.consistent_2sig
    }

    #[getter]
    pub fn imposter_breakdown(&self) -> Vec<(String, f64)> {
        self.inner.metrics.imposter_breakdown.clone()
    }

    pub fn with_followup_budget(&self, budget: f64) -> Self {
        Self {
            inner: self.inner.clone().with_followup_budget(budget),
        }
    }

    #[getter]
    pub fn followup_metrics<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if let Some(ref fb) = self.inner.followup_budget {
            let dict = PyDict::new(py);
            dict.set_item("budget", fb.budget)?;
            dict.set_item("total_candidates", fb.total_candidates)?;
            dict.set_item("allocation_fraction", fb.allocation_fraction)?;
            dict.set_item("expected_confirmed_kn", fb.expected_confirmed_kn)?;
            dict.set_item("confirmed_efficiency", fb.confirmed_efficiency)?;
            dict.set_item("wasted_imposter_followups", fb.wasted_imposter_followups)?;
            dict.set_item("is_budget_exceeded", fb.is_budget_exceeded)?;
            Ok(Some(dict))
        } else {
            Ok(None)
        }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("signal_name", self.inner.kilonova.name.clone())?;
        dict.set_item("signal_rate", self.inner.kilonova.volumetric_rate)?;
        dict.set_item("effective_vt", self.inner.kilonova.effective_vt)?;
        dict.set_item("signal_detections", self.inner.metrics.signal_detections)?;
        dict.set_item("total_imposters", self.inner.metrics.total_imposters)?;
        dict.set_item("total_candidates", self.inner.metrics.total_detections)?;
        dict.set_item("imposter_to_signal_ratio", self.inner.metrics.imposter_to_signal_ratio)?;
        dict.set_item("purity", self.inner.metrics.purity)?;
        dict.set_item("false_discovery_rate", self.inner.metrics.false_discovery_rate)?;
        dict.set_item("raw_efficiency", self.inner.kilonova.overall_efficiency)?;
        dict.set_item("pure_efficiency", self.inner.metrics.pure_efficiency)?;
        dict.set_item("f1_score", self.inner.metrics.f1_score)?;
        dict.set_item("significance_fom", self.inner.metrics.significance_fom)?;
        dict.set_item("naive_recovered_rate", self.inner.rate_recovery.naive_recovered_rate)?;
        dict.set_item("rate_bias_absolute", self.inner.rate_recovery.rate_bias_absolute)?;
        dict.set_item("rate_bias_fractional", self.inner.rate_recovery.rate_bias_fractional)?;
        dict.set_item("decontaminated_rate", self.inner.rate_recovery.decontaminated_recovered_rate)?;
        dict.set_item("variance_inflation_factor", self.inner.rate_recovery.variance_inflation_factor)?;
        dict.set_item("uncertainty_inflation_factor", self.inner.rate_recovery.uncertainty_inflation_factor)?;
        dict.set_item("consistent_2sig", self.inner.rate_recovery.consistent_2sig)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTdeRateForecast>()?;
    m.add_class::<PyKilonovaImposterAnalysis>()?;
    m.add_function(wrap_pyfunction!(blackbody_k_correction, m)?)?;
    m.add_function(wrap_pyfunction!(blackbody_k_correction_instrument, m)?)?;
    Ok(())
}
