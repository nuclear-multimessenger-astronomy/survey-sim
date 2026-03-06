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

#[pymethods]
impl PyBlastwaveModel {
    /// Evaluate the blastwave model for a single GRB at given observation times.
    ///
    /// Parameters
    /// ----------
    /// params : dict
    ///     GRB physical parameters. Required keys:
    ///     ``Eiso``, ``Gamma_0``, ``theta_v``, ``logthc``, ``logn0``,
    ///     ``logepse``, ``logepsB``, ``p``, ``av``,
    ///     ``p_rvs``, ``logepse_rvs``, ``logepsB_rvs``.
    /// z : float
    ///     Source redshift.
    /// d_l_mpc : float
    ///     Luminosity distance in Mpc.
    /// times_s : list[float]
    ///     Observer-frame times in seconds since explosion.
    /// band : str
    ///     Band name (must be in ``band_frequencies``).
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Flux density in mJy at each time.
    #[pyo3(signature = (params, z, d_l_mpc, times_s, band))]
    fn evaluate_flux(
        &self,
        params: HashMap<String, f64>,
        z: f64,
        d_l_mpc: f64,
        times_s: Vec<f64>,
        band: String,
    ) -> PyResult<Vec<f64>> {
        use blastwave::afterglow::eats::EATS;
        use blastwave::afterglow::forward_grid::ForwardGrid;
        use blastwave::afterglow::models::{Dict, get_radiation_model};
        use blastwave::constants::{C_SPEED, MASS_P, MPC};
        use blastwave::hydro::sim_box::SimBox;

        let freq = *self.band_frequencies.get(&band).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Band '{}' not in band_frequencies", band),
            )
        })?;

        let get = |key: &str| -> PyResult<f64> {
            params.get(key).copied().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Missing parameter: {}", key),
                )
            })
        };

        let eiso = get("Eiso")?;
        let gamma_0 = get("Gamma_0")?;
        let theta_v = get("theta_v")?;
        let logthc = get("logthc")?;
        let logn0 = get("logn0")?;
        let logepse = get("logepse")?;
        let logepsb = get("logepsB")?;
        let p = get("p")?;
        let p_rs = get("p_rvs")?;
        let logepse_rs = get("logepse_rvs")?;
        let logepsb_rs = get("logepsB_rvs")?;

        let theta_c = 10.0_f64.powf(logthc);
        let n0 = 10.0_f64.powf(logn0);
        let eps_e = 10.0_f64.powf(logepse);
        let eps_b = 10.0_f64.powf(logepsb);
        let eps_e_rs = 10.0_f64.powf(logepse_rs);
        let eps_b_rs = 10.0_f64.powf(logepsb_rs);

        let d_cm = d_l_mpc * MPC;

        // Filter valid times (positive only).
        let valid_times: Vec<(usize, f64)> = times_s
            .iter()
            .enumerate()
            .filter(|(_, &t)| t > 0.0)
            .map(|(i, &t)| (i, t))
            .collect();

        if valid_times.is_empty() {
            return Ok(vec![0.0; times_s.len()]);
        }

        let tmax = valid_times.iter().map(|(_, t)| *t).fold(0.0_f64, f64::max) * 2.0;
        let tmax = tmax.max(1e5).min(1e8);

        // Build jet config and solve.
        let config = survey_sim::lightcurve::blastwave_model::build_jet_config(
            eiso, gamma_0, theta_c, n0, p, eps_e, eps_b, p_rs, eps_e_rs, eps_b_rs, tmax,
        );
        let mut sim_box = SimBox::new(&config);
        sim_box.solve_pde();

        let theta_data = sim_box.get_theta();
        let t_data = &sim_box.ts;
        let y_data = &sim_box.ys;
        let rs_data = sim_box.ys_rs.as_ref();
        let eats = EATS::new(theta_data, t_data);
        let tool = sim_box.tool();

        let radiation_model = get_radiation_model(&self.radiation_model).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown radiation model: {}", self.radiation_model),
            )
        })?;

        let mut ag_params = Dict::new();
        ag_params.insert("eps_e".into(), eps_e);
        ag_params.insert("eps_b".into(), eps_b);
        ag_params.insert("p".into(), p);

        let mut rs_ag_params = Dict::new();
        rs_ag_params.insert("eps_e".into(), eps_e_rs);
        rs_ag_params.insert("eps_b".into(), eps_b_rs);
        rs_ag_params.insert("p".into(), p_rs);

        let nu_z = freq * (1.0 + z);
        let flux_factor = (1.0 + z) / (4.0 * std::f64::consts::PI * d_cm * d_cm);

        // Rest-frame times for batch evaluation.
        let t_rest: Vec<f64> = valid_times.iter().map(|(_, t)| t / (1.0 + z)).collect();

        let fs_grid = ForwardGrid::precompute(
            nu_z, theta_v, y_data, t_data, theta_data,
            &eats, tool, &ag_params, radiation_model,
        );
        let fs_lum = fs_grid.luminosity_batch(&t_rest);

        let rs_lum = if let Some(rs) = rs_data {
            let rs_grid = ForwardGrid::precompute_reverse(
                nu_z, theta_v, y_data, rs, t_data, theta_data,
                &eats, tool, &rs_ag_params, radiation_model,
            );
            rs_grid.luminosity_batch(&t_rest)
        } else {
            vec![0.0; t_rest.len()]
        };

        // Build output array.
        let mut flux_mjy = vec![0.0_f64; times_s.len()];
        for (q_idx, &(orig_idx, _)) in valid_times.iter().enumerate() {
            let lum = fs_lum[q_idx] + rs_lum[q_idx];
            let f_nu = lum * flux_factor;
            flux_mjy[orig_idx] = f_nu / 1e-26;
        }

        Ok(flux_mjy)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMetzgerKNModel>()?;
    m.add_class::<PyBlastwaveModel>()?;
    Ok(())
}
