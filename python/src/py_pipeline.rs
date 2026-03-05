use std::collections::HashMap;

use pyo3::prelude::*;

use survey_sim::detection::DetectionCriteria;

use crate::py_detection::PyDetectionCriteria;
use crate::py_lightcurve::{PyBlastwaveModel, PyMetzgerKNModel, PythonCallbackModel};
use crate::py_population::*;
use crate::py_survey::PySurveyStore;

/// Python wrapper for SimulationPipeline.
#[pyclass]
#[pyo3(name = "SimulationPipeline")]
pub struct PySimulationPipeline {
    survey: Py<PySurveyStore>,
    populations: Vec<PyObject>,
    models: HashMap<String, PyObject>,
    detection: Py<PyDetectionCriteria>,
    n_transients: usize,
    seed: u64,
}

#[pymethods]
impl PySimulationPipeline {
    #[new]
    #[pyo3(signature = (survey, populations, models, detection, n_transients=100000, seed=42))]
    fn new(
        survey: Py<PySurveyStore>,
        populations: Vec<PyObject>,
        models: HashMap<String, PyObject>,
        detection: Py<PyDetectionCriteria>,
        n_transients: usize,
        seed: u64,
    ) -> Self {
        Self {
            survey,
            populations,
            models,
            detection,
            n_transients,
            seed,
        }
    }

    /// Run the simulation pipeline.
    fn run(&self, py: Python<'_>) -> PyResult<PySimulationResult> {
        let survey_ref = self.survey.borrow(py);
        let det_ref = self.detection.borrow(py);

        // We need to build the Rust pipeline from the Python objects.
        // First, reconstruct the survey store reference.
        // Since SimulationPipeline takes ownership, we need to reload observations.
        // For now, we'll build a simpler approach: re-extract from the Python objects.

        // Build detection criteria.
        let criteria = det_ref.inner.clone();

        // Build pipeline (we need the survey's inner data).
        // This requires the survey to be borrowed for the duration of the run.
        let mjd_min = survey_ref.inner.mjd_min;
        let mjd_max = survey_ref.inner.mjd_max;

        // Build populations.
        let mut rust_populations: Vec<Box<dyn survey_sim::population::PopulationGenerator>> =
            Vec::new();

        for pop_obj in &self.populations {
            if let Ok(kn) = pop_obj.extract::<PyRef<PyKilonovaPopulation>>(py) {
                rust_populations.push(Box::new(kn.to_generator(mjd_min, mjd_max)));
            } else if let Ok(bu) = pop_obj.extract::<PyRef<PyBu2026KilonovaPopulation>>(py) {
                rust_populations.push(Box::new(bu.to_generator(mjd_min, mjd_max)));
            } else if let Ok(fbu) = pop_obj.extract::<PyRef<PyFixedBu2026KilonovaPopulation>>(py) {
                rust_populations.push(Box::new(fbu.to_generator(mjd_min, mjd_max)));
            } else if let Ok(snia) = pop_obj.extract::<PyRef<PySupernovaIaPopulation>>(py) {
                rust_populations.push(Box::new(snia.to_generator(mjd_min, mjd_max)));
            } else if let Ok(snii) = pop_obj.extract::<PyRef<PySupernovaIIPopulation>>(py) {
                rust_populations.push(Box::new(snii.to_generator(mjd_min, mjd_max)));
            } else if let Ok(tde) = pop_obj.extract::<PyRef<PyTdePopulation>>(py) {
                rust_populations.push(Box::new(tde.to_generator(mjd_min, mjd_max)));
            } else if let Ok(grb) = pop_obj.extract::<PyRef<PyGrbPopulation>>(py) {
                rust_populations.push(Box::new(grb.to_generator(mjd_min, mjd_max)?));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Unsupported population type",
                ));
            }
        }

        // Build models.
        let mut rust_models: HashMap<String, Box<dyn survey_sim::lightcurve::LightcurveModel>> =
            HashMap::new();

        for (name, model_obj) in &self.models {
            if let Ok(kn_model) = model_obj.extract::<PyRef<PyMetzgerKNModel>>(py) {
                rust_models.insert(name.clone(), Box::new(kn_model.to_model()));
            } else if let Ok(bw_model) = model_obj.extract::<PyRef<PyBlastwaveModel>>(py) {
                rust_models.insert(name.clone(), Box::new(bw_model.to_model()));
            } else {
                // Assume it's a Python callback model (e.g., fiestaEM).
                rust_models.insert(
                    name.clone(),
                    Box::new(PythonCallbackModel::new(model_obj.clone_ref(py))),
                );
            }
        }

        // We need to construct the pipeline, but it takes ownership of SurveyStore.
        // For the Python binding, we'll reconstruct by loading observations again,
        // or we can work around this by using the survey reference directly.
        // For simplicity, let's reconstruct the observations from the borrow.
        drop(survey_ref);
        drop(det_ref);

        // Unfortunately, SimulationPipeline takes ownership of SurveyStore.
        // For the Python API, we need to reconstruct. Let's modify the approach:
        // We'll run the pipeline phases manually using the borrowed survey.

        let survey_ref = self.survey.borrow(py);
        let survey = &survey_ref.inner;

        // Run simulation manually (simplified version that borrows survey).
        let result = py.allow_threads(|| {
            run_pipeline_borrowed(
                survey,
                &rust_populations,
                &rust_models,
                &criteria,
                self.n_transients,
                self.seed,
            )
        });

        Ok(PySimulationResult {
            n_simulated: result.n_simulated,
            n_detected: result.n_detected,
            rate_summaries: result
                .rate_summaries
                .iter()
                .map(|rs| PyRateSummary {
                    transient_type: rs.transient_type.clone(),
                    volumetric_rate: rs.volumetric_rate,
                    detections_per_year: rs.detections_per_year,
                    detections_total: rs.detections_total,
                    overall_efficiency: rs.overall_efficiency,
                })
                .collect(),
        })
    }
}

/// Run the pipeline with a borrowed SurveyStore (avoids ownership issues).
fn run_pipeline_borrowed(
    survey: &survey_sim::survey::SurveyStore,
    populations: &[Box<dyn survey_sim::population::PopulationGenerator>],
    models: &HashMap<String, Box<dyn survey_sim::lightcurve::LightcurveModel>>,
    criteria: &DetectionCriteria,
    n_transients: usize,
    seed: u64,
) -> PipelineResult {
    use rayon::prelude::*;
    use rand::SeedableRng;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let _cosmo = survey_sim::types::Cosmology::default();

    let mut total_simulated = 0usize;
    let mut total_detected = 0usize;
    let mut rate_summaries = Vec::new();

    for pop in populations {
        let instances = pop.generate(n_transients, &mut rng);
        let type_name = pop.transient_type().to_string();

        let model = match models.get(&type_name) {
            Some(m) => m,
            None => continue,
        };

        let time_window = 100.0;

        // Phase 1: Spatial matching (parallel).
        let matched: Vec<(usize, Vec<usize>)> = instances
            .par_iter()
            .enumerate()
            .map(|(i, inst)| {
                let mjd_min = inst.t_exp;
                let mjd_max = inst.t_exp + time_window * (1.0 + inst.z);
                let obs_indices = survey.query(&inst.coord, mjd_min, mjd_max);
                (i, obs_indices)
            })
            .filter(|(_, obs)| !obs.is_empty())
            .collect();

        // Phase 2: Lightcurve evaluation.
        let evaluations: Vec<(usize, Vec<usize>, survey_sim::lightcurve::LightcurveEvaluation)> =
            if model.supports_batch() {
                // GPU-batched path: collect all instances, evaluate in one call.
                let batch_instances: Vec<&survey_sim::types::TransientInstance> =
                    matched.iter().map(|(idx, _)| &instances[*idx]).collect();
                let batch_times: Vec<Vec<f64>> = matched
                    .iter()
                    .map(|(_, obs_indices)| obs_indices.iter().map(|&oi| survey.get(oi).mjd).collect())
                    .collect();
                let batch_bands: Vec<Vec<survey_sim::types::Band>> = matched
                    .iter()
                    .map(|(_, obs_indices)| obs_indices.iter().map(|&oi| survey.get(oi).band.clone()).collect())
                    .collect();
                let times_refs: Vec<&[f64]> = batch_times.iter().map(|t| t.as_slice()).collect();
                let bands_refs: Vec<&[survey_sim::types::Band]> = batch_bands.iter().map(|b| b.as_slice()).collect();

                let results = model.batch_evaluate(&batch_instances, &times_refs, &bands_refs);

                matched
                    .iter()
                    .zip(results.into_iter())
                    .filter_map(|((idx, obs_indices), result)| {
                        result.ok().map(|eval| (*idx, obs_indices.clone(), eval))
                    })
                    .collect()
            } else if model.requires_gil() {
                // Sequential GIL path: one transient at a time.
                matched
                    .iter()
                    .filter_map(|(idx, obs_indices)| {
                        let inst = &instances[*idx];
                        let times: Vec<f64> =
                            obs_indices.iter().map(|&oi| survey.get(oi).mjd).collect();
                        let bands: Vec<_> =
                            obs_indices.iter().map(|&oi| survey.get(oi).band.clone()).collect();
                        model.evaluate(inst, &times, &bands).ok().map(|eval| (*idx, obs_indices.clone(), eval))
                    })
                    .collect()
            } else {
                // Rayon-parallel path for pure Rust models.
                matched
                    .par_iter()
                    .filter_map(|(idx, obs_indices)| {
                        let inst = &instances[*idx];
                        let times: Vec<f64> =
                            obs_indices.iter().map(|&oi| survey.get(oi).mjd).collect();
                        let bands: Vec<_> =
                            obs_indices.iter().map(|&oi| survey.get(oi).band.clone()).collect();
                        model.evaluate(inst, &times, &bands).ok().map(|eval| (*idx, obs_indices.clone(), eval))
                    })
                    .collect()
            };

        // Phase 3: Detection (parallel).
        let detection_results: Vec<(usize, survey_sim::detection::DetectionResult)> = evaluations
            .par_iter()
            .map(|(idx, obs_indices, eval)| {
                let obs_refs: Vec<&survey_sim::survey::SurveyObservation> =
                    obs_indices.iter().map(|&oi| survey.get(oi)).collect();
                let result = survey_sim::detection::evaluate_detection(eval, &obs_refs, criteria);
                (*idx, result)
            })
            .collect();

        let n_detected = detection_results.iter().filter(|(_, d)| d.detected).count();
        let overall_eff = n_detected as f64 / n_transients.max(1) as f64;

        let actual_z_max = instances.iter().map(|i| i.z).fold(0.0f64, f64::max);

        rate_summaries.push(survey_sim::efficiency::rates::RateSummary {
            transient_type: type_name,
            volumetric_rate: pop.volumetric_rate(),
            detections_per_year: 0.0, // Simplified for now.
            detections_total: 0.0,
            overall_efficiency: overall_eff,
            survey_omega_sr: 0.0,
            z_max: actual_z_max,
            recovery: None,
        });

        total_simulated += n_transients;
        total_detected += n_detected;
    }

    PipelineResult {
        n_simulated: total_simulated,
        n_detected: total_detected,
        rate_summaries,
    }
}

struct PipelineResult {
    n_simulated: usize,
    n_detected: usize,
    rate_summaries: Vec<survey_sim::efficiency::rates::RateSummary>,
}

/// Python result from the simulation pipeline.
#[pyclass]
#[pyo3(name = "SimulationResult")]
pub struct PySimulationResult {
    #[pyo3(get)]
    pub n_simulated: usize,
    #[pyo3(get)]
    pub n_detected: usize,
    #[pyo3(get)]
    pub rate_summaries: Vec<PyRateSummary>,
}

#[pymethods]
impl PySimulationResult {
    fn __repr__(&self) -> String {
        format!(
            "SimulationResult(n_simulated={}, n_detected={}, efficiency={:.4})",
            self.n_simulated,
            self.n_detected,
            self.n_detected as f64 / self.n_simulated.max(1) as f64,
        )
    }
}

/// Python rate summary.
#[pyclass]
#[pyo3(name = "RateSummary")]
#[derive(Clone)]
pub struct PyRateSummary {
    #[pyo3(get)]
    pub transient_type: String,
    #[pyo3(get)]
    pub volumetric_rate: f64,
    #[pyo3(get)]
    pub detections_per_year: f64,
    #[pyo3(get)]
    pub detections_total: f64,
    #[pyo3(get)]
    pub overall_efficiency: f64,
}

#[pymethods]
impl PyRateSummary {
    fn __repr__(&self) -> String {
        format!(
            "RateSummary(type={}, rate={:.1} Gpc^-3/yr, eff={:.4}, det/yr={:.1})",
            self.transient_type,
            self.volumetric_rate,
            self.overall_efficiency,
            self.detections_per_year,
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulationPipeline>()?;
    m.add_class::<PySimulationResult>()?;
    m.add_class::<PyRateSummary>()?;
    Ok(())
}
