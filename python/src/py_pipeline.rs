use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use survey_sim::detection::DetectionCriteria;

use crate::py_detection::PyDetectionCriteria;
use crate::py_lightcurve::{PyBlastwaveModel, PyMetzgerKNModel, PyParametricModel, PythonCallbackModel};
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
    /// If set, apply flux-averaging stacking between lightcurve eval and detection.
    /// Multiple windows produce independent stacked measurements that are all
    /// combined for detection (e.g., [900, 3600] for 15-min + 1-hour stacks).
    stack_windows_s: Option<Vec<f64>>,
    /// Max rest-frame days after explosion to query for observations (default 100).
    time_window_days: f64,
}

#[pymethods]
impl PySimulationPipeline {
    #[new]
    #[pyo3(signature = (survey, populations, models, detection, n_transients=100000, seed=42, stack_windows_s=None, time_window_days=100.0))]
    fn new(
        survey: Py<PySurveyStore>,
        populations: Vec<PyObject>,
        models: HashMap<String, PyObject>,
        detection: Py<PyDetectionCriteria>,
        n_transients: usize,
        seed: u64,
        stack_windows_s: Option<Vec<f64>>,
        time_window_days: f64,
    ) -> Self {
        Self {
            survey,
            populations,
            models,
            detection,
            n_transients,
            seed,
            stack_windows_s,
            time_window_days,
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
            } else if let Ok(fbot) = pop_obj.extract::<PyRef<PyFbotPopulation>>(py) {
                rust_populations.push(Box::new(fbot.to_generator(mjd_min, mjd_max)));
            } else if let Ok(grb) = pop_obj.extract::<PyRef<PyGrbPopulation>>(py) {
                rust_populations.push(Box::new(grb.to_generator(mjd_min, mjd_max)?));
            } else if let Ok(grb_on) = pop_obj.extract::<PyRef<PyOnAxisGrbPopulation>>(py) {
                rust_populations.push(Box::new(grb_on.to_generator(mjd_min, mjd_max)?));
            } else if let Ok(grb_off) = pop_obj.extract::<PyRef<PyOffAxisGrbPopulation>>(py) {
                rust_populations.push(Box::new(grb_off.to_generator(mjd_min, mjd_max)?));
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
            } else if let Ok(param_model) = model_obj.extract::<PyRef<PyParametricModel>>(py) {
                rust_models.insert(name.clone(), Box::new(param_model.to_model()));
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
        let stack_windows_s = self.stack_windows_s.clone();
        let time_window_days = self.time_window_days;
        let result = py.allow_threads(|| {
            run_pipeline_borrowed(
                survey,
                &rust_populations,
                &rust_models,
                &criteria,
                self.n_transients,
                self.seed,
                stack_windows_s,
                time_window_days,
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
            detected_sources: result
                .detected_sources
                .into_iter()
                .map(|ds| PyDetectedSource { data: ds })
                .collect(),
        })
    }
}

/// Run the pipeline with a borrowed SurveyStore (avoids ownership issues).
/// Build realistic photometry from a lightcurve evaluation and survey observations.
///
/// For each observation where the transient is brighter than the 5-sigma depth,
/// compute a magnitude error from the SNR and add Gaussian noise.
/// Build detection photometry and non-detection upper limits from a lightcurve evaluation.
fn build_photometry_from_eval(
    eval: &survey_sim::lightcurve::LightcurveEvaluation,
    obs: Vec<&survey_sim::survey::SurveyObservation>,
    inst: &survey_sim::types::TransientInstance,
) -> (Vec<(f64, f64, f64, String)>, Vec<(f64, f64, String)>) {
    let mut photometry = Vec::new();
    let mut non_detections = Vec::new();
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(
        (inst.t_exp * 1000.0) as u64 ^ (inst.z * 1e6) as u64
    );

    for (i, observation) in obs.iter().enumerate() {
        if i >= eval.times_mjd.len() {
            break;
        }
        let band_name = observation.band.0.as_str();
        let depth = observation.five_sigma_depth;

        // Pre-explosion observations are always non-detections
        if observation.mjd < inst.t_exp {
            non_detections.push((observation.mjd, depth, band_name.to_string()));
            continue;
        }

        // Get model apparent magnitude for this observation
        let model_mag = match eval.apparent_mags.get(band_name) {
            Some(mags) if i < mags.len() => mags[i],
            _ => continue,
        };

        if !model_mag.is_finite() || model_mag > 90.0 {
            // No valid model prediction — record as non-detection
            non_detections.push((observation.mjd, depth, band_name.to_string()));
            continue;
        }

        // SNR = 5 * 10^(0.4 * (depth - mag))
        let snr = 5.0 * 10f64.powf(0.4 * (depth - model_mag));

        if snr >= 5.0 {
            // Detection: add noise and record
            let mag_err = 1.0857362 / snr;
            let u1: f64 = rng.random::<f64>().max(1e-10);
            let u2: f64 = rng.random::<f64>();
            let gauss: f64 = (-2.0f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let observed_mag = model_mag + mag_err * gauss;
            photometry.push((observation.mjd, observed_mag, mag_err, band_name.to_string()));
        } else {
            // Non-detection: record the depth as upper limit
            non_detections.push((observation.mjd, depth, band_name.to_string()));
        }
    }

    (photometry, non_detections)
}

fn run_pipeline_borrowed(
    survey: &survey_sim::survey::SurveyStore,
    populations: &[Box<dyn survey_sim::population::PopulationGenerator>],
    models: &HashMap<String, Box<dyn survey_sim::lightcurve::LightcurveModel>>,
    criteria: &DetectionCriteria,
    n_transients: usize,
    seed: u64,
    stack_windows_s: Option<Vec<f64>>,
    time_window_days: f64,
) -> PipelineResult {
    use rayon::prelude::*;
    use rand::SeedableRng;
    use std::time::Instant;

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let _cosmo = survey_sim::types::Cosmology::default();

    let mut total_simulated = 0usize;
    let mut total_detected = 0usize;
    let mut rate_summaries = Vec::new();
    let mut all_detected_sources: Vec<DetectedSourceData> = Vec::new();

    for pop in populations {
        let instances = pop.generate(n_transients, &mut rng);
        let type_name = pop.transient_type().to_string();

        let model = match models.get(&type_name) {
            Some(m) => m,
            None => continue,
        };

        let time_window = time_window_days;
        let t_phase1 = Instant::now();

        // Phase 1: Spatial matching (parallel).
        let matched: Vec<(usize, Vec<usize>)> = instances
            .par_iter()
            .enumerate()
            .map(|(i, inst)| {
                // Look back 30 days before explosion for non-detection upper limits
                let mjd_min = inst.t_exp - 30.0;
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

        // Phase 2.5: Flux-averaging stacking (if configured).
        // Converts mags to flux, averages within time bins per band,
        // converts back to mag, and boosts depth by 2.5*log10(sqrt(N)).
        // Multiple windows produce independent measurements that are combined.
        let stacked: Option<Vec<(usize, Vec<survey_sim::survey::SurveyObservation>, survey_sim::lightcurve::LightcurveEvaluation)>>;

        if let Some(ref windows) = stack_windows_s {
            let t_stack = Instant::now();
            let windows_days: Vec<f64> = windows.iter().map(|w| w / 86400.0).collect();

            stacked = Some(evaluations
                .par_iter()
                .map(|(idx, obs_indices, eval)| {
                    let raw_obs: Vec<&survey_sim::survey::SurveyObservation> =
                        obs_indices.iter().map(|&oi| survey.get(oi)).collect();
                    let (combined_obs, combined_eval) =
                        stack_multi_window(eval, &raw_obs, &windows_days);
                    (*idx, combined_obs, combined_eval)
                })
                .collect());

            let window_labels: Vec<String> = windows.iter().map(|w| format!("{:.0}s", w)).collect();
            eprintln!("[pipeline] Phase 2.5 stacking: {:.1}s (windows=[{}])",
                t_stack.elapsed().as_secs_f64(), window_labels.join(", "));
        } else {
            stacked = None;
        }

        let t_phase3 = Instant::now();

        // Phase 3: Detection (parallel).
        let gal_lat_cut = criteria.min_galactic_lat;
        let spec_k = criteria.spectroscopic_completeness_k;
        let spec_m0 = criteria.spectroscopic_completeness_m0;

        let detection_results: Vec<(usize, survey_sim::detection::DetectionResult)> =
            if let Some(ref stacked_data) = stacked {
                // Use stacked observations and evaluations.
                stacked_data
                    .par_iter()
                    .map(|(idx, stacked_obs, stacked_eval)| {
                        let obs_refs: Vec<&survey_sim::survey::SurveyObservation> =
                            stacked_obs.iter().collect();
                        let mut result = survey_sim::detection::evaluate_detection(stacked_eval, &obs_refs, criteria);
                        if gal_lat_cut > 0.0 && result.detected {
                            let b = instances[*idx].coord.galactic_lat().abs();
                            if b < gal_lat_cut {
                                result.detected = false;
                            }
                        }
                        if spec_k > 0.0 && result.detected {
                            if let Some(peak) = result.peak_mag {
                                let p = 1.0 / (1.0 + (spec_k * (peak - spec_m0)).exp());
                                let hash = ((*idx as f64 + 0.5) * std::f64::consts::FRAC_1_PI * 7.31).fract();
                                if hash > p {
                                    result.detected = false;
                                }
                            }
                        }
                        (*idx, result)
                    })
                    .collect()
            } else {
                // No stacking: use raw observations from survey.
                evaluations
                    .par_iter()
                    .map(|(idx, obs_indices, eval)| {
                        let obs_refs: Vec<&survey_sim::survey::SurveyObservation> =
                            obs_indices.iter().map(|&oi| survey.get(oi)).collect();
                        let mut result = survey_sim::detection::evaluate_detection(eval, &obs_refs, criteria);
                        if gal_lat_cut > 0.0 && result.detected {
                            let b = instances[*idx].coord.galactic_lat().abs();
                            if b < gal_lat_cut {
                                result.detected = false;
                            }
                        }
                        if spec_k > 0.0 && result.detected {
                            if let Some(peak) = result.peak_mag {
                                let p = 1.0 / (1.0 + (spec_k * (peak - spec_m0)).exp());
                                let hash = ((*idx as f64 + 0.5) * std::f64::consts::FRAC_1_PI * 7.31).fract();
                                if hash > p {
                                    result.detected = false;
                                }
                            }
                        }
                        (*idx, result)
                    })
                    .collect()
            };

        eprintln!("[pipeline] Phase 3 detection: {:.1}s", t_phase3.elapsed().as_secs_f64());

        let n_detected = detection_results.iter().filter(|(_, d)| d.detected).count();
        let overall_eff = n_detected as f64 / n_transients.max(1) as f64;

        let actual_z_max = instances.iter().map(|i| i.z).fold(0.0f64, f64::max);

        rate_summaries.push(survey_sim::efficiency::rates::RateSummary {
            transient_type: type_name.clone(),
            volumetric_rate: pop.volumetric_rate(),
            detections_per_year: 0.0, // Simplified for now.
            detections_total: 0.0,
            overall_efficiency: overall_eff,
            survey_omega_sr: 0.0,
            z_max: actual_z_max,
            recovery: None,
        });

        // Phase 4: Build photometry for detected sources.
        // For each detected transient, pair model magnitudes with observation
        // depths to produce realistic (mjd, mag, mag_err, band) tuples.
        let detected_idx: Vec<usize> = detection_results
            .iter()
            .filter(|(_, d)| d.detected)
            .map(|(idx, _)| *idx)
            .collect();

        // Build lookup from instance index to evaluation data.
        let eval_map: HashMap<usize, &(usize, Vec<usize>, survey_sim::lightcurve::LightcurveEvaluation)> =
            evaluations.iter().map(|e| (e.0, e)).collect();
        // Also check stacked data if available.
        let stacked_map: Option<HashMap<usize, &(usize, Vec<survey_sim::survey::SurveyObservation>, survey_sim::lightcurve::LightcurveEvaluation)>> =
            stacked.as_ref().map(|s| s.iter().map(|e| (e.0, e)).collect());

        for &idx in &detected_idx {
            let inst = &instances[idx];

            // Get evaluation + observations (prefer stacked if available).
            let (photometry, non_detections) = if let Some(ref sm) = stacked_map {
                if let Some(&&ref entry) = sm.get(&idx) {
                    let (_, ref obs_vec, ref eval) = entry;
                    build_photometry_from_eval(eval, obs_vec.iter().collect(), inst)
                } else {
                    continue;
                }
            } else if let Some(&&ref entry) = eval_map.get(&idx) {
                let (_, ref obs_indices, ref eval) = entry;
                let obs_refs: Vec<&survey_sim::survey::SurveyObservation> =
                    obs_indices.iter().map(|&oi| survey.get(oi)).collect();
                build_photometry_from_eval(eval, obs_refs, inst)
            } else {
                continue;
            };

            if !photometry.is_empty() {
                all_detected_sources.push(DetectedSourceData {
                    true_params: inst.model_params.clone(),
                    z: inst.z,
                    peak_abs_mag: inst.peak_abs_mag,
                    t_exp: inst.t_exp,
                    transient_type: type_name.to_string(),
                    photometry,
                    non_detections,
                });
            }
        }

        total_simulated += n_transients;
        total_detected += n_detected;
    }

    PipelineResult {
        n_simulated: total_simulated,
        n_detected: total_detected,
        rate_summaries,
        detected_sources: all_detected_sources,
    }
}

/// Apply stacking at multiple time windows and combine all stacked observations.
///
/// Each window produces independent measurements. For example, with [15min, 1hr]:
/// - 15-min stacks give a set of observations with moderate depth boost
/// - 1-hr stacks give fewer observations with larger depth boost
/// Both sets are combined as independent measurements for detection.
fn stack_multi_window(
    eval: &survey_sim::lightcurve::LightcurveEvaluation,
    raw_obs: &[&survey_sim::survey::SurveyObservation],
    windows_days: &[f64],
) -> (Vec<survey_sim::survey::SurveyObservation>, survey_sim::lightcurve::LightcurveEvaluation) {
    use survey_sim::survey::SurveyObservation;
    use survey_sim::lightcurve::LightcurveEvaluation;

    if windows_days.is_empty() {
        return (
            raw_obs.iter().map(|o| (*o).clone()).collect(),
            eval.clone(),
        );
    }

    if windows_days.len() == 1 {
        return stack_flux_average(eval, raw_obs, windows_days[0]);
    }

    // Stack at each window independently, then combine all results.
    let mut all_obs: Vec<SurveyObservation> = Vec::new();
    let mut all_mags: HashMap<String, Vec<f64>> = HashMap::new();
    let mut all_times: Vec<f64> = Vec::new();

    for &window in windows_days {
        let (stacked_obs, stacked_eval) = stack_flux_average(eval, raw_obs, window);

        let offset = all_obs.len();

        for obs in &stacked_obs {
            let mut obs_copy = obs.clone();
            obs_copy.obs_id = (offset + all_obs.len() - offset) as u64 + all_obs.len() as u64;
            all_obs.push(obs.clone());
        }

        all_times.extend_from_slice(&stacked_eval.times_mjd);

        for (band, mags) in &stacked_eval.apparent_mags {
            all_mags.entry(band.clone()).or_default().extend_from_slice(mags);
        }
    }

    // Sort everything by MJD.
    let mut sort_idx: Vec<usize> = (0..all_obs.len()).collect();
    sort_idx.sort_by(|&a, &b| all_obs[a].mjd.partial_cmp(&all_obs[b].mjd).unwrap());

    let sorted_obs: Vec<SurveyObservation> = sort_idx.iter().map(|&i| {
        let mut o = all_obs[i].clone();
        o.obs_id = sort_idx.iter().position(|&j| j == i).unwrap_or(0) as u64;
        o
    }).collect();
    let sorted_times: Vec<f64> = sort_idx.iter().map(|&i| all_times[i]).collect();

    let mut sorted_mags: HashMap<String, Vec<f64>> = HashMap::new();
    for (band, mags) in &all_mags {
        sorted_mags.insert(
            band.clone(),
            sort_idx.iter().map(|&i| mags[i]).collect(),
        );
    }

    (
        sorted_obs,
        LightcurveEvaluation {
            apparent_mags: sorted_mags,
            times_mjd: sorted_times,
        },
    )
}

/// Stack observations by flux-averaging within time bins per band.
///
/// Replicates the stacking logic from the Argus pipeline:
/// 1. Group observations by band, then sequentially bin within each band
///    (a new bin starts when an observation exceeds `window_days` from the bin start)
/// 2. Average flux density within each bin (convert mag→flux, average, flux→mag)
/// 3. Boost depth: new_depth = median(depth) + 2.5 * log10(sqrt(n_obs))
fn stack_flux_average(
    eval: &survey_sim::lightcurve::LightcurveEvaluation,
    raw_obs: &[&survey_sim::survey::SurveyObservation],
    window_days: f64,
) -> (Vec<survey_sim::survey::SurveyObservation>, survey_sim::lightcurve::LightcurveEvaluation) {
    use survey_sim::survey::SurveyObservation;
    use survey_sim::lightcurve::LightcurveEvaluation;
    use survey_sim::types::{Band, SkyCoord};

    let n = raw_obs.len();
    if n < 2 {
        // Nothing to stack.
        return (
            raw_obs.iter().map(|o| (*o).clone()).collect(),
            eval.clone(),
        );
    }

    // Group observation indices by band.
    let mut band_groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, obs) in raw_obs.iter().enumerate() {
        band_groups.entry(obs.band.0.clone()).or_default().push(i);
    }

    let mut stacked_obs: Vec<SurveyObservation> = Vec::new();
    let mut stacked_mags: HashMap<String, Vec<f64>> = HashMap::new();
    let mut stacked_times: Vec<f64> = Vec::new();

    for (band_name, indices) in &band_groups {
        // Sort by MJD within this band.
        let mut sorted_idx = indices.clone();
        sorted_idx.sort_by(|&a, &b| raw_obs[a].mjd.partial_cmp(&raw_obs[b].mjd).unwrap());

        // Sequential binning (matching colleague's approach).
        let mut bins: Vec<Vec<usize>> = Vec::new();
        let mut current_bin: Vec<usize> = vec![sorted_idx[0]];
        let mut bin_start = raw_obs[sorted_idx[0]].mjd;

        for &idx in &sorted_idx[1..] {
            if raw_obs[idx].mjd - bin_start < window_days {
                current_bin.push(idx);
            } else {
                bins.push(std::mem::take(&mut current_bin));
                current_bin.push(idx);
                bin_start = raw_obs[idx].mjd;
            }
        }
        bins.push(current_bin);

        // Get the magnitude array for this band.
        let band_mags = match eval.apparent_mags.get(band_name) {
            Some(m) => m,
            None => continue,
        };

        let band_stacked_mags = stacked_mags.entry(band_name.clone()).or_default();

        for bin in &bins {
            let n_bin = bin.len();
            if n_bin == 0 {
                continue;
            }

            // Convert magnitudes to flux density and average.
            let mut flux_sum = 0.0f64;
            let mut mjd_sum = 0.0f64;
            let mut depths: Vec<f64> = Vec::with_capacity(n_bin);
            let mut valid_count = 0usize;

            for &idx in bin {
                if idx < band_mags.len() {
                    let mag = band_mags[idx];
                    if mag < 90.0 {
                        // flux = 10^(-0.4 * mag) — constant factor cancels in average→mag conversion.
                        flux_sum += 10.0_f64.powf(-0.4 * mag);
                        valid_count += 1;
                    }
                }
                mjd_sum += raw_obs[idx].mjd;
                depths.push(raw_obs[idx].five_sigma_depth);
            }

            let mean_mjd = mjd_sum / n_bin as f64;

            // Average magnitude from averaged flux.
            let avg_mag = if valid_count > 0 {
                let avg_flux = flux_sum / valid_count as f64;
                -2.5 * avg_flux.log10()
            } else {
                99.0
            };

            // Depth boost: median depth + 2.5 * log10(sqrt(n)) = median + 1.25 * log10(n).
            depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_depth = depths[depths.len() / 2];
            let stacked_depth = median_depth + 1.25 * (n_bin as f64).log10();

            // Use first observation in bin as template for other fields.
            let template = raw_obs[bin[0]];

            stacked_obs.push(SurveyObservation {
                obs_id: stacked_obs.len() as u64,
                coord: SkyCoord::new(template.coord.ra, template.coord.dec),
                mjd: mean_mjd,
                band: Band::new(band_name),
                five_sigma_depth: stacked_depth,
                seeing_fwhm: template.seeing_fwhm,
                exposure_time: template.exposure_time * n_bin as f64,
                airmass: template.airmass,
                sky_brightness: template.sky_brightness,
                night: mean_mjd.floor() as i64,
            });

            stacked_times.push(mean_mjd);
            band_stacked_mags.push(avg_mag);
        }
    }

    // Sort stacked observations by MJD and align magnitude arrays.
    // First, create an index mapping.
    let mut sort_indices: Vec<usize> = (0..stacked_obs.len()).collect();
    sort_indices.sort_by(|&a, &b| stacked_obs[a].mjd.partial_cmp(&stacked_obs[b].mjd).unwrap());

    let sorted_obs: Vec<SurveyObservation> = sort_indices.iter().map(|&i| stacked_obs[i].clone()).collect();

    // Rebuild magnitude arrays: each band gets a vec of length = total stacked obs,
    // with 99.0 for obs in other bands.
    let total = sorted_obs.len();
    let mut final_mags: HashMap<String, Vec<f64>> = HashMap::new();

    // Build a mapping: for each stacked observation, what band is it and what's its
    // position in the per-band stacked_mags.
    // We need to track which per-band index each stacked obs came from.
    // Since we built stacked_obs by iterating bands then bins, we know the structure.
    // Let's rebuild more carefully.

    // Track (band, per-band-index) for each stacked obs.
    let mut obs_band_idx: Vec<(String, usize)> = Vec::new();
    for (band_name, indices) in &band_groups {
        let count = stacked_mags.get(band_name).map_or(0, |v| v.len());
        for i in 0..count {
            obs_band_idx.push((band_name.clone(), i));
        }
    }

    // Initialize all bands with 99.0.
    for band_name in stacked_mags.keys() {
        final_mags.insert(band_name.clone(), vec![99.0; total]);
    }

    // Fill in actual values.
    for (new_pos, &orig_pos) in sort_indices.iter().enumerate() {
        let (ref band, band_idx) = obs_band_idx[orig_pos];
        if let Some(mags) = stacked_mags.get(band) {
            if let Some(final_band) = final_mags.get_mut(band) {
                final_band[new_pos] = mags[band_idx];
            }
        }
    }

    let final_times: Vec<f64> = sorted_obs.iter().map(|o| o.mjd).collect();

    (
        sorted_obs,
        LightcurveEvaluation {
            apparent_mags: final_mags,
            times_mjd: final_times,
        },
    )
}

struct PipelineResult {
    n_simulated: usize,
    n_detected: usize,
    rate_summaries: Vec<survey_sim::efficiency::rates::RateSummary>,
    detected_sources: Vec<DetectedSourceData>,
}

/// Per-detected-source photometry and ground truth.
struct DetectedSourceData {
    /// True model parameters.
    true_params: HashMap<String, f64>,
    /// True redshift.
    z: f64,
    /// True peak absolute magnitude.
    peak_abs_mag: f64,
    /// True explosion MJD.
    t_exp: f64,
    /// Transient type name.
    transient_type: String,
    /// Observed photometry: (mjd, apparent_mag, mag_err, band_name).
    photometry: Vec<(f64, f64, f64, String)>,
    /// Non-detections (upper limits): (mjd, depth_mag, band_name).
    non_detections: Vec<(f64, f64, String)>,
}

/// Python-visible detected source with photometry and truth.
#[pyclass]
#[pyo3(name = "DetectedSource")]
pub struct PyDetectedSource {
    data: DetectedSourceData,
}

#[pymethods]
impl PyDetectedSource {
    /// True model parameters as a dict.
    #[getter]
    fn true_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.data.true_params {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    #[getter]
    fn z(&self) -> f64 { self.data.z }

    #[getter]
    fn peak_abs_mag(&self) -> f64 { self.data.peak_abs_mag }

    #[getter]
    fn t_exp(&self) -> f64 { self.data.t_exp }

    #[getter]
    fn transient_type(&self) -> &str { &self.data.transient_type }

    /// Number of photometric observations.
    #[getter]
    fn n_obs(&self) -> usize { self.data.photometry.len() }

    /// Photometry as (times, mags, mag_errs, bands) tuple of lists.
    fn photometry<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let times: Vec<f64> = self.data.photometry.iter().map(|p| p.0).collect();
        let mags: Vec<f64> = self.data.photometry.iter().map(|p| p.1).collect();
        let errs: Vec<f64> = self.data.photometry.iter().map(|p| p.2).collect();
        let bands: Vec<&str> = self.data.photometry.iter().map(|p| p.3.as_str()).collect();
        Ok((times, mags, errs, bands).into_pyobject(py)?.into())
    }

    /// Non-detections (upper limits) as (times, depths, bands) tuple of lists.
    fn non_detections<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let times: Vec<f64> = self.data.non_detections.iter().map(|p| p.0).collect();
        let depths: Vec<f64> = self.data.non_detections.iter().map(|p| p.1).collect();
        let bands: Vec<&str> = self.data.non_detections.iter().map(|p| p.2.as_str()).collect();
        Ok((times, depths, bands).into_pyobject(py)?.into())
    }

    /// Number of non-detection upper limits.
    #[getter]
    fn n_non_detections(&self) -> usize { self.data.non_detections.len() }

    fn __repr__(&self) -> String {
        format!(
            "DetectedSource(type={}, z={:.3}, n_obs={}, bands={})",
            self.data.transient_type, self.data.z, self.data.photometry.len(),
            {
                let mut bs: Vec<&str> = self.data.photometry.iter().map(|p| p.3.as_str()).collect();
                bs.sort();
                bs.dedup();
                bs.join(",")
            }
        )
    }
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
    pub detected_sources: Vec<PyDetectedSource>,
}

#[pymethods]
impl PySimulationResult {
    /// Number of detected sources with photometry.
    #[getter]
    fn n_sources(&self) -> usize {
        self.detected_sources.len()
    }

    /// Get a detected source by index.
    fn get_source(&self, index: usize) -> PyResult<PyDetectedSource> {
        self.detected_sources.get(index)
            .map(|s| PyDetectedSource { data: DetectedSourceData {
                true_params: s.data.true_params.clone(),
                z: s.data.z,
                peak_abs_mag: s.data.peak_abs_mag,
                t_exp: s.data.t_exp,
                transient_type: s.data.transient_type.clone(),
                photometry: s.data.photometry.clone(),
                non_detections: s.data.non_detections.clone(),
            }})
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("source index out of range"))
    }

    /// Get all detected sources as a list.
    fn sources(&self) -> Vec<PyDetectedSource> {
        self.detected_sources.iter().map(|s| PyDetectedSource {
            data: DetectedSourceData {
                true_params: s.data.true_params.clone(),
                z: s.data.z,
                peak_abs_mag: s.data.peak_abs_mag,
                t_exp: s.data.t_exp,
                transient_type: s.data.transient_type.clone(),
                photometry: s.data.photometry.clone(),
                non_detections: s.data.non_detections.clone(),
            }
        }).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SimulationResult(n_simulated={}, n_detected={}, efficiency={:.4}, sources={})",
            self.n_simulated,
            self.n_detected,
            self.n_detected as f64 / self.n_simulated.max(1) as f64,
            self.detected_sources.len(),
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
    m.add_class::<PyDetectedSource>()?;
    m.add_class::<PyRateSummary>()?;
    m.add_class::<PyTooSimulationResult>()?;
    m.add_function(wrap_pyfunction!(run_too_simulation, m)?)?;
    Ok(())
}
