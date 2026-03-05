use std::collections::HashMap;

use pyo3::prelude::*;

use survey_sim::lightcurve::blastwave_model::BlastwaveModel;
use survey_sim::lightcurve::python_model::python_result_to_evaluation;
use survey_sim::lightcurve::{LightcurveEvaluation, LightcurveModel, LightcurveError};
use survey_sim::lightcurve::parametric::ParametricModel;
use survey_sim::types::{Band, TransientInstance};

/// Python wrapper for ParametricModel (MetzgerKN, etc.).
#[pyclass]
#[pyo3(name = "MetzgerKNModel")]
pub struct PyMetzgerKNModel {
    pub(crate) peak_abs_mag: f64,
}

#[pymethods]
impl PyMetzgerKNModel {
    #[new]
    #[pyo3(signature = (peak_abs_mag=-16.0))]
    fn new(peak_abs_mag: f64) -> Self {
        Self { peak_abs_mag }
    }
}

impl PyMetzgerKNModel {
    pub fn to_model(&self) -> ParametricModel {
        use lightcurve_fitting::SviModelName;
        ParametricModel::new().with_model(SviModelName::MetzgerKN)
    }
}

/// Python callback lightcurve model wrapping a Python object with .predict().
///
/// Designed for fiestaEM SurrogateModel integration.
pub struct PythonCallbackModel {
    callback: PyObject,
}

impl PythonCallbackModel {
    pub fn new(callback: PyObject) -> Self {
        Self { callback }
    }
}

impl LightcurveModel for PythonCallbackModel {
    fn evaluate(
        &self,
        instance: &TransientInstance,
        _times_mjd: &[f64],
        _bands: &[Band],
    ) -> survey_sim::lightcurve::Result<LightcurveEvaluation> {
        Python::with_gil(|py| {
            // Build params dict from instance.model_params.
            let params_dict = pyo3::types::PyDict::new(py);
            for (key, val) in &instance.model_params {
                params_dict
                    .set_item(key, val)
                    .map_err(|e| LightcurveError::PythonError(e.to_string()))?;
            }

            // Add standard parameters.
            params_dict
                .set_item("luminosity_distance", instance.d_l)
                .map_err(|e| LightcurveError::PythonError(e.to_string()))?;
            params_dict
                .set_item("redshift", instance.z)
                .map_err(|e| LightcurveError::PythonError(e.to_string()))?;

            // Pass observation context so adapters can interpolate to obs times.
            let obs_times: Vec<f64> = _times_mjd.to_vec();
            let obs_bands: Vec<String> = _bands.iter().map(|b| b.to_string()).collect();
            params_dict
                .set_item("_obs_times_mjd", obs_times)
                .map_err(|e| LightcurveError::PythonError(e.to_string()))?;
            params_dict
                .set_item("_obs_bands", obs_bands)
                .map_err(|e| LightcurveError::PythonError(e.to_string()))?;
            params_dict
                .set_item("_t_exp", instance.t_exp)
                .map_err(|e| LightcurveError::PythonError(e.to_string()))?;

            // Call model.predict(params).
            let result = self
                .callback
                .call_method1(py, "predict", (params_dict,))
                .map_err(|e| LightcurveError::PythonError(e.to_string()))?;

            // Unpack (times, {band: mags}).
            let tuple = result
                .downcast_bound::<pyo3::types::PyTuple>(py)
                .map_err(|e| LightcurveError::PythonError(format!("Expected tuple: {}", e)))?;

            let py_times: Vec<f64> = tuple
                .get_item(0)
                .map_err(|e| LightcurveError::PythonError(e.to_string()))?
                .extract()
                .map_err(|e| LightcurveError::PythonError(e.to_string()))?;

            let py_mags_dict = tuple
                .get_item(1)
                .map_err(|e| LightcurveError::PythonError(e.to_string()))?;
            let py_mags_dict = py_mags_dict
                .downcast::<pyo3::types::PyDict>()
                .map_err(|e| LightcurveError::PythonError(format!("Expected dict: {}", e)))?;

            let mut mags_per_band = HashMap::new();
            for (key, val) in py_mags_dict.iter() {
                let band_name: String = key
                    .extract()
                    .map_err(|e| LightcurveError::PythonError(e.to_string()))?;
                let mags: Vec<f64> = val
                    .extract()
                    .map_err(|e| LightcurveError::PythonError(e.to_string()))?;
                mags_per_band.insert(band_name, mags);
            }

            Ok(python_result_to_evaluation(py_times, mags_per_band))
        })
    }

    fn requires_gil(&self) -> bool {
        true
    }
}

/// Python wrapper for BlastwaveModel (GRB afterglow).
#[pyclass]
#[pyo3(name = "BlastwaveModel")]
pub struct PyBlastwaveModel {
    pub(crate) radiation_model: String,
    pub(crate) band_frequencies: HashMap<String, f64>,
}

#[pymethods]
impl PyBlastwaveModel {
    #[new]
    #[pyo3(signature = (radiation_model="sync_ssa_smooth", band_frequencies=None))]
    fn new(radiation_model: &str, band_frequencies: Option<HashMap<String, f64>>) -> Self {
        let freqs = band_frequencies.unwrap_or_else(|| {
            let mut m = HashMap::new();
            m.insert("g".to_string(), 6.3e14);
            m
        });
        Self {
            radiation_model: radiation_model.to_string(),
            band_frequencies: freqs,
        }
    }
}

impl PyBlastwaveModel {
    pub fn to_model(&self) -> BlastwaveModel {
        BlastwaveModel::new(&self.radiation_model, self.band_frequencies.clone())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMetzgerKNModel>()?;
    m.add_class::<PyBlastwaveModel>()?;
    Ok(())
}
