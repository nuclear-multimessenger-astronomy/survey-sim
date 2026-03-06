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
/// If the Python object has a `batch_predict` method, batch evaluation
/// via GPU (e.g., JAX vmap) is automatically enabled.
pub struct PythonCallbackModel {
    callback: PyObject,
    has_batch: bool,
    has_columnar: bool,
}

impl PythonCallbackModel {
    pub fn new(callback: PyObject) -> Self {
        let (has_batch, has_columnar) = Python::with_gil(|py| {
            let batch = callback.getattr(py, "batch_predict").is_ok();
            let columnar = callback.getattr(py, "batch_evaluate_arrays").is_ok();
            (batch, columnar)
        });
        Self { callback, has_batch, has_columnar }
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
            params_dict
                .set_item("peak_abs_mag", instance.peak_abs_mag)
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

    fn supports_batch(&self) -> bool {
        self.has_batch
    }

    fn batch_evaluate(
        &self,
        instances: &[&TransientInstance],
        times_mjd: &[&[f64]],
        bands: &[&[Band]],
    ) -> Vec<survey_sim::lightcurve::Result<LightcurveEvaluation>> {
        if !self.has_batch {
            return instances
                .iter()
                .zip(times_mjd.iter())
                .zip(bands.iter())
                .map(|((inst, t), b)| self.evaluate(inst, t, b))
                .collect();
        }

        // Fast columnar path: pass arrays instead of 66K dicts.
        if self.has_columnar {
            return self.batch_evaluate_columnar(instances, times_mjd, bands);
        }

        // Legacy dict-based path.
        self.batch_evaluate_dicts(instances, times_mjd, bands)
    }
}

impl PythonCallbackModel {
    /// Fast columnar path: pass numpy arrays instead of per-transient dicts.
    fn batch_evaluate_columnar(
        &self,
        instances: &[&TransientInstance],
        times_mjd: &[&[f64]],
        bands: &[&[Band]],
    ) -> Vec<survey_sim::lightcurve::Result<LightcurveEvaluation>> {
        Python::with_gil(|py| {
            let n = instances.len();

            // Build columnar arrays.
            let redshifts: Vec<f64> = instances.iter().map(|i| i.z).collect();
            let d_ls: Vec<f64> = instances.iter().map(|i| i.d_l).collect();
            let peak_mags: Vec<f64> = instances.iter().map(|i| i.peak_abs_mag).collect();
            let t_exps: Vec<f64> = instances.iter().map(|i| i.t_exp).collect();

            // Collect model param names (from first instance).
            let param_names: Vec<String> = if n > 0 {
                instances[0].model_params.keys().cloned().collect()
            } else {
                vec![]
            };
            let param_arrays: Vec<Vec<f64>> = param_names
                .iter()
                .map(|name| {
                    instances
                        .iter()
                        .map(|inst| inst.model_params.get(name).copied().unwrap_or(0.0))
                        .collect()
                })
                .collect();

            // Flatten obs times and bands.
            let obs_counts: Vec<i64> = times_mjd.iter().map(|t| t.len() as i64).collect();
            let obs_times_flat: Vec<f64> = times_mjd.iter().flat_map(|t| t.iter().copied()).collect();
            let obs_bands_flat: Vec<String> = bands
                .iter()
                .flat_map(|b| b.iter().map(|band| band.to_string()))
                .collect();

            // Convert to Python objects.
            let py_param_names = pyo3::types::PyList::new(py, &param_names)
                .expect("Failed to create param names list");
            let py_param_arrays = pyo3::types::PyList::new(
                py,
                param_arrays.iter().map(|a| pyo3::types::PyList::new(py, a).unwrap()),
            )
            .expect("Failed to create param arrays list");

            let result = match self.callback.call_method(
                py,
                "batch_evaluate_arrays",
                (
                    redshifts,
                    d_ls,
                    peak_mags,
                    py_param_names,
                    py_param_arrays,
                    obs_times_flat,
                    obs_bands_flat,
                    obs_counts,
                    t_exps,
                ),
                None,
            ) {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("batch_evaluate_arrays failed: {}", e);
                    return (0..n)
                        .map(|_| Err(LightcurveError::PythonError(msg.clone())))
                        .collect();
                }
            };

            // Unpack flat return: (obs_times_flat, {band: flat_mags}, obs_counts)
            let tuple = match result.downcast_bound::<pyo3::types::PyTuple>(py) {
                Ok(t) => t,
                Err(e) => {
                    let msg = format!("Expected tuple: {}", e);
                    return (0..n)
                        .map(|_| Err(LightcurveError::PythonError(msg.clone())))
                        .collect();
                }
            };

            let all_times: Vec<f64> = tuple
                .get_item(0)
                .and_then(|v| v.extract())
                .map_err(|e| LightcurveError::PythonError(e.to_string()))
                .unwrap_or_default();
            let mags_dict = match tuple.get_item(1).and_then(|v| {
                v.downcast::<pyo3::types::PyDict>().map(|d| d.clone()).map_err(|e| e.into())
            }) {
                Ok(d) => d,
                Err(e) => {
                    let msg = format!("Expected mags dict: {}", e);
                    return (0..n)
                        .map(|_| Err(LightcurveError::PythonError(msg.clone())))
                        .collect();
                }
            };
            let ret_counts: Vec<i64> = tuple
                .get_item(2)
                .and_then(|v| v.extract())
                .map_err(|e| LightcurveError::PythonError(e.to_string()))
                .unwrap_or_default();

            // Extract flat magnitude arrays per band.
            let mut flat_mags: HashMap<String, Vec<f64>> = HashMap::new();
            for (key, val) in mags_dict.iter() {
                let band_name: String = key
                    .extract()
                    .map_err(|e| LightcurveError::PythonError(e.to_string()))
                    .unwrap_or_default();
                let mags: Vec<f64> = val
                    .extract()
                    .map_err(|e| LightcurveError::PythonError(e.to_string()))
                    .unwrap_or_default();
                flat_mags.insert(band_name, mags);
            }

            // Split flat arrays into per-transient evaluations.
            let mut offset = 0usize;
            (0..n)
                .map(|i| {
                    let count = ret_counts[i] as usize;
                    let times_slice = all_times[offset..offset + count].to_vec();
                    let mut mags_per_band = HashMap::new();
                    for (band, flat) in &flat_mags {
                        mags_per_band.insert(band.clone(), flat[offset..offset + count].to_vec());
                    }
                    offset += count;
                    Ok(python_result_to_evaluation(times_slice, mags_per_band))
                })
                .collect()
        })
    }

    /// Legacy dict-based batch evaluation path.
    fn batch_evaluate_dicts(
        &self,
        instances: &[&TransientInstance],
        times_mjd: &[&[f64]],
        bands: &[&[Band]],
    ) -> Vec<survey_sim::lightcurve::Result<LightcurveEvaluation>> {
        Python::with_gil(|py| {
            let params_list = pyo3::types::PyList::empty(py);
            for (i, inst) in instances.iter().enumerate() {
                let params_dict = pyo3::types::PyDict::new(py);
                for (key, val) in &inst.model_params {
                    let _ = params_dict.set_item(key, val);
                }
                let _ = params_dict.set_item("luminosity_distance", inst.d_l);
                let _ = params_dict.set_item("redshift", inst.z);
                let _ = params_dict.set_item("peak_abs_mag", inst.peak_abs_mag);
                let obs_times: Vec<f64> = times_mjd[i].to_vec();
                let obs_bands: Vec<String> = bands[i].iter().map(|b| b.to_string()).collect();
                let _ = params_dict.set_item("_obs_times_mjd", obs_times);
                let _ = params_dict.set_item("_obs_bands", obs_bands);
                let _ = params_dict.set_item("_t_exp", inst.t_exp);
                let _ = params_list.append(params_dict);
            }

            let result = match self
                .callback
                .call_method1(py, "batch_predict", (params_list,))
            {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("batch_predict failed: {}", e);
                    return (0..instances.len())
                        .map(|_| Err(LightcurveError::PythonError(msg.clone())))
                        .collect();
                }
            };

            let py_list = match result.downcast_bound::<pyo3::types::PyList>(py) {
                Ok(l) => l,
                Err(e) => {
                    let msg = format!("Expected list: {}", e);
                    return (0..instances.len())
                        .map(|_| Err(LightcurveError::PythonError(msg.clone())))
                        .collect();
                }
            };

            py_list
                .iter()
                .map(|item| {
                    let tuple = item
                        .downcast::<pyo3::types::PyTuple>()
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
                .collect()
        })
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
