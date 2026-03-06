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
            } else if let Ok(fm) = pop_obj.extract::<PyRef<PyFixedMetzgerKilonovaPopulation>>(py) {
                rust_populations.push(Box::new(fm.to_generator(mjd_min, mjd_max)));
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
    use std::time::Instant;

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
        let t_phase1 = Instant::now();

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

        eprintln!("[pipeline] Phase 1 spatial match: {:.1}s, {} matched of {} generated",
            t_phase1.elapsed().as_secs_f64(), matched.len(), instances.len());
        let t_phase2 = Instant::now();

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

        eprintln!("[pipeline] Phase 2 lightcurve eval: {:.1}s, {} evaluations",
            t_phase2.elapsed().as_secs_f64(), evaluations.len());
        let t_phase3 = Instant::now();

        // Phase 3: Detection (parallel).
        let gal_lat_cut = criteria.min_galactic_lat;
        let spec_k = criteria.spectroscopic_completeness_k;
        let spec_m0 = criteria.spectroscopic_completeness_m0;
        let detection_results: Vec<(usize, survey_sim::detection::DetectionResult)> = evaluations
            .par_iter()
            .map(|(idx, obs_indices, eval)| {
                let obs_refs: Vec<&survey_sim::survey::SurveyObservation> =
                    obs_indices.iter().map(|&oi| survey.get(oi)).collect();
                let mut result = survey_sim::detection::evaluate_detection(eval, &obs_refs, criteria);
                // Apply galactic latitude cut on the transient sky position.
                if gal_lat_cut > 0.0 && result.detected {
                    let b = instances[*idx].coord.galactic_lat().abs();
                    if b < gal_lat_cut {
                        result.detected = false;
                    }
                }
                // Apply spectroscopic completeness (magnitude-dependent classification probability).
                if spec_k > 0.0 && result.detected {
                    if let Some(peak) = result.peak_mag {
                        let p = 1.0 / (1.0 + (spec_k * (peak - spec_m0)).exp());
                        // Deterministic pseudo-random from transient index + seed.
                        // Use a simple hash: fract(idx * phi) where phi = golden ratio.
                        let hash = ((*idx as f64 + 0.5) * std::f64::consts::FRAC_1_PI * 7.31).fract();
                        if hash > p {
                            result.detected = false;
                        }
                    }
                }
                (*idx, result)
            })
            .collect();

        eprintln!("[pipeline] Phase 3 detection: {:.1}s", t_phase3.elapsed().as_secs_f64());

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

/// Python wrapper for ToO simulation results.
#[pyclass]
#[pyo3(name = "TooSimulationResult")]
pub struct PyTooSimulationResult {
    #[pyo3(get)]
    pub strategy_name: String,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_detected: usize,
    #[pyo3(get)]
    pub efficiency: f64,
    /// Per-event detection flags (parallel to input events).
    #[pyo3(get)]
    pub detected: Vec<bool>,
    /// Per-event distances in Mpc.
    #[pyo3(get)]
    pub distances: Vec<f64>,
    /// Per-event area_90 in sq deg.
    #[pyo3(get)]
    pub areas_90: Vec<f64>,
    /// Per-event number of detections.
    #[pyo3(get)]
    pub n_detections_per_event: Vec<usize>,
}

#[pymethods]
impl PyTooSimulationResult {
    fn __repr__(&self) -> String {
        format!(
            "TooSimulationResult(strategy='{}', events={}, detected={}, efficiency={:.3})",
            self.strategy_name, self.n_events, self.n_detected, self.efficiency,
        )
    }
}

/// Run a targeted ToO simulation for GW events.
///
/// For each BNS/NSBH event, places a kilonova at the event's true position
/// and evaluates detection with the given ToO strategy.
///
/// Args:
///     events: List of GwEvent objects from load_gw_events().
///     strategy: Strategy name ("rubin_gold", "rubin_silver", "ztf", "ultrasat", "uvex").
///     population: A population generator for KN parameters.
///     model: A lightcurve model (e.g., MetzgerKNModel).
///     detection: Detection criteria.
///     trigger_mjd: MJD to use as the explosion/trigger time (default: 60000.0).
///     include_bbh: Whether to include BBH events (default: false).
#[pyfunction]
#[pyo3(signature = (events, strategy, population, model, detection, trigger_mjd=60000.0, include_bbh=false))]
pub fn run_too_simulation(
    py: Python<'_>,
    events: Vec<PyRef<crate::py_survey::PyGwEvent>>,
    strategy: &str,
    population: PyObject,
    model: PyObject,
    detection: PyRef<PyDetectionCriteria>,
    trigger_mjd: f64,
    include_bbh: bool,
) -> PyResult<PyTooSimulationResult> {
    // Convert GwEvents.
    let gw_events: Vec<survey_sim::survey::observing_scenario::GwEvent> =
        events.iter().map(|e| e.inner.clone()).collect();

    // Get the strategy.
    let too_strategy = survey_sim::survey::too::builtin_strategy(strategy)
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown ToO strategy '{}'. Available: rubin_gold, rubin_silver, ztf, ultrasat, uvex",
                strategy
            ))
        })?;

    // Build Rust population.
    let mjd_min = trigger_mjd - 1.0;
    let mjd_max = trigger_mjd + 365.25;
    let rust_pop: Box<dyn survey_sim::population::PopulationGenerator> =
        if let Ok(kn) = population.extract::<PyRef<PyKilonovaPopulation>>(py) {
            Box::new(kn.to_generator(mjd_min, mjd_max))
        } else if let Ok(fm) = population.extract::<PyRef<PyFixedMetzgerKilonovaPopulation>>(py) {
            Box::new(fm.to_generator(mjd_min, mjd_max))
        } else if let Ok(bu) = population.extract::<PyRef<PyBu2026KilonovaPopulation>>(py) {
            Box::new(bu.to_generator(mjd_min, mjd_max))
        } else if let Ok(fbu) = population.extract::<PyRef<PyFixedBu2026KilonovaPopulation>>(py) {
            Box::new(fbu.to_generator(mjd_min, mjd_max))
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported population type for ToO simulation",
            ));
        };

    // Build Rust model.
    let rust_model: Box<dyn survey_sim::lightcurve::LightcurveModel> =
        if let Ok(kn_model) = model.extract::<PyRef<PyMetzgerKNModel>>(py) {
            Box::new(kn_model.to_model())
        } else {
            Box::new(PythonCallbackModel::new(model.clone_ref(py)))
        };

    let criteria = detection.inner.clone();

    // Run the simulation (release GIL for Rust models).
    let result = if rust_model.requires_gil() {
        survey_sim::pipeline::too::run_too_simulation(
            &gw_events,
            too_strategy.as_ref(),
            rust_pop.as_ref(),
            rust_model.as_ref(),
            &criteria,
            trigger_mjd,
            include_bbh,
        )
    } else {
        py.allow_threads(|| {
            survey_sim::pipeline::too::run_too_simulation(
                &gw_events,
                too_strategy.as_ref(),
                rust_pop.as_ref(),
                rust_model.as_ref(),
                &criteria,
                trigger_mjd,
                include_bbh,
            )
        })
    };

    Ok(PyTooSimulationResult {
        strategy_name: result.strategy_name,
        n_events: result.n_events,
        n_detected: result.n_detected,
        efficiency: result.efficiency,
        detected: result.event_results.iter().map(|r| r.detection.detected).collect(),
        distances: result.event_results.iter().map(|r| r.event.distance_mpc).collect(),
        areas_90: result.event_results.iter().map(|r| r.event.area_90).collect(),
        n_detections_per_event: result.event_results.iter().map(|r| r.detection.n_detections).collect(),
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulationPipeline>()?;
    m.add_class::<PySimulationResult>()?;
    m.add_class::<PyRateSummary>()?;
    m.add_class::<PyTooSimulationResult>()?;
    m.add_function(wrap_pyfunction!(run_too_simulation, m)?)?;
    Ok(())
}
