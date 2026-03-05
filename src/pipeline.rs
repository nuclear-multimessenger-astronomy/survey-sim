use std::collections::HashMap;

use rayon::prelude::*;

use crate::detection::{evaluate_detection, DetectionCriteria, DetectionResult};
use crate::efficiency::rates::{compute_rate, estimate_survey_omega, recover_rate, RateSummary};
use crate::efficiency::{EfficiencyGrid, GridAxis};
use crate::lightcurve::{LightcurveEvaluation, LightcurveModel};
use crate::population::PopulationGenerator;
use crate::survey::{SurveyObservation, SurveyStore};
use crate::types::{Cosmology, TransientInstance};

/// End-to-end simulation pipeline.
pub struct SimulationPipeline {
    /// The loaded survey schedule.
    pub survey: SurveyStore,
    /// Transient populations to simulate.
    pub populations: Vec<Box<dyn PopulationGenerator>>,
    /// Lightcurve models keyed by transient type name.
    pub models: HashMap<String, Box<dyn LightcurveModel>>,
    /// Detection criteria.
    pub criteria: DetectionCriteria,
    /// Number of transients to simulate per population.
    pub n_transients: usize,
    /// Random seed.
    pub seed: u64,
    /// Cosmology.
    pub cosmology: Cosmology,
    /// Number of redshift bins for the efficiency grid.
    pub n_z_bins: usize,
}

/// Result of a simulation run.
pub struct SimulationResult {
    /// Rate summaries per population.
    pub rate_summaries: Vec<RateSummary>,
    /// Efficiency grid (redshift only for simplicity; extensible).
    pub efficiency_grid: EfficiencyGrid,
    /// All detection results (for detailed analysis).
    pub detection_results: Vec<(TransientInstance, DetectionResult)>,
    /// Total transients simulated.
    pub n_simulated: usize,
    /// Total transients detected.
    pub n_detected: usize,
}

impl std::fmt::Display for SimulationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Simulation Result ===")?;
        writeln!(f, "Simulated: {}", self.n_simulated)?;
        writeln!(
            f,
            "Detected: {} ({:.2}%)",
            self.n_detected,
            100.0 * self.n_detected as f64 / self.n_simulated.max(1) as f64
        )?;
        writeln!(f)?;
        for rs in &self.rate_summaries {
            write!(f, "{}", rs)?;
        }
        Ok(())
    }
}

impl SimulationPipeline {
    pub fn new(
        survey: SurveyStore,
        criteria: DetectionCriteria,
        n_transients: usize,
        seed: u64,
    ) -> Self {
        Self {
            survey,
            populations: Vec::new(),
            models: HashMap::new(),
            criteria,
            n_transients,
            seed,
            cosmology: Cosmology::default(),
            n_z_bins: 30,
        }
    }

    pub fn add_population(&mut self, pop: Box<dyn PopulationGenerator>) {
        self.populations.push(pop);
    }

    pub fn add_model(&mut self, name: &str, model: Box<dyn LightcurveModel>) {
        self.models.insert(name.to_string(), model);
    }

    /// Run the full simulation pipeline.
    ///
    /// Three-phase execution:
    /// 1. Spatial-temporal matching (parallel)
    /// 2. Lightcurve evaluation (sequential for Python models, parallel otherwise)
    /// 3. Detection evaluation (parallel)
    pub fn run(&self) -> SimulationResult {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(self.seed);

        let mut all_instances = Vec::new();
        let mut all_detection_results = Vec::new();
        let mut rate_summaries = Vec::new();

        // Determine z_max from populations for efficiency grid.
        let _z_max = self
            .populations
            .iter()
            .map(|_p| {
                // We need to get z_max from somewhere. For now, use the max z
                // from generated instances.
                0.5 // default; will be updated after generation
            })
            .fold(0.0f64, f64::max);

        // Build efficiency grid with redshift axis.
        let mut grid = EfficiencyGrid::new(vec![GridAxis::uniform("z", 0.0, 1.0, self.n_z_bins)]);

        for pop in &self.populations {
            log::info!(
                "Generating {} {} transients...",
                self.n_transients,
                pop.transient_type()
            );

            // Phase 0: Generate population.
            let instances = pop.generate(self.n_transients, &mut rng);

            let actual_z_max = instances
                .iter()
                .map(|i| i.z)
                .fold(0.0f64, f64::max);

            // Rebuild grid with actual z range if needed.
            if actual_z_max > 0.0 {
                grid = EfficiencyGrid::new(vec![GridAxis::uniform(
                    "z",
                    0.0,
                    actual_z_max * 1.1,
                    self.n_z_bins,
                )]);
            }

            // Determine which model to use.
            let type_name = pop.transient_type().to_string();
            let model = self.models.get(&type_name);

            if model.is_none() {
                log::warn!("No model configured for type '{}', skipping.", type_name);
                continue;
            }
            let model = model.unwrap();

            let uses_python = model.requires_gil();

            // Phase 1: Spatial-temporal matching (parallel).
            log::info!("Phase 1: Spatial-temporal matching...");
            let time_window = 100.0; // days after explosion to check

            let matched: Vec<(usize, Vec<usize>)> = instances
                .par_iter()
                .enumerate()
                .map(|(i, inst)| {
                    let mjd_min = inst.t_exp;
                    let mjd_max = inst.t_exp + time_window * (1.0 + inst.z);
                    let obs_indices = self.survey.query(&inst.coord, mjd_min, mjd_max);
                    (i, obs_indices)
                })
                .filter(|(_, obs)| !obs.is_empty())
                .collect();

            log::info!(
                "  {} / {} transients have overlapping observations",
                matched.len(),
                instances.len()
            );

            // Phase 2: Lightcurve evaluation.
            log::info!("Phase 2: Lightcurve evaluation...");
            let evaluations: Vec<(usize, Vec<usize>, LightcurveEvaluation)> = if uses_python {
                // Sequential for Python models (GIL constraint).
                matched
                    .iter()
                    .filter_map(|(inst_idx, obs_indices)| {
                        let inst = &instances[*inst_idx];
                        let times: Vec<f64> = obs_indices
                            .iter()
                            .map(|&oi| self.survey.get(oi).mjd)
                            .collect();
                        let bands: Vec<_> = obs_indices
                            .iter()
                            .map(|&oi| self.survey.get(oi).band.clone())
                            .collect();

                        match model.evaluate(inst, &times, &bands) {
                            Ok(eval) => Some((*inst_idx, obs_indices.clone(), eval)),
                            Err(e) => {
                                log::debug!("Lightcurve eval failed for instance {}: {}", inst_idx, e);
                                None
                            }
                        }
                    })
                    .collect()
            } else {
                // Parallel for Rust-native models.
                matched
                    .par_iter()
                    .filter_map(|(inst_idx, obs_indices)| {
                        let inst = &instances[*inst_idx];
                        let times: Vec<f64> = obs_indices
                            .iter()
                            .map(|&oi| self.survey.get(oi).mjd)
                            .collect();
                        let bands: Vec<_> = obs_indices
                            .iter()
                            .map(|&oi| self.survey.get(oi).band.clone())
                            .collect();

                        match model.evaluate(inst, &times, &bands) {
                            Ok(eval) => Some((*inst_idx, obs_indices.clone(), eval)),
                            Err(e) => {
                                log::debug!("Lightcurve eval failed for instance {}: {}", inst_idx, e);
                                None
                            }
                        }
                    })
                    .collect()
            };

            // Phase 3: Detection evaluation (parallel).
            log::info!("Phase 3: Detection evaluation...");
            let criteria = &self.criteria;
            let detection_results: Vec<(usize, DetectionResult)> = evaluations
                .par_iter()
                .map(|(inst_idx, obs_indices, eval)| {
                    let obs_refs: Vec<&SurveyObservation> = obs_indices
                        .iter()
                        .map(|&oi| self.survey.get(oi))
                        .collect();
                    let result = evaluate_detection(eval, &obs_refs, criteria);
                    (*inst_idx, result)
                })
                .collect();

            // Accumulate into efficiency grid.
            let mut n_detected = 0usize;
            for (inst_idx, det) in &detection_results {
                let inst = &instances[*inst_idx];
                let mut vals = HashMap::new();
                vals.insert("z".to_string(), inst.z);
                grid.record(&vals, det.detected);
                if det.detected {
                    n_detected += 1;
                }
            }

            // Also record unmatched transients as non-detections.
            let matched_set: std::collections::HashSet<usize> =
                matched.iter().map(|(i, _)| *i).collect();
            for i in 0..instances.len() {
                if !matched_set.contains(&i) {
                    let inst = &instances[i];
                    let mut vals = HashMap::new();
                    vals.insert("z".to_string(), inst.z);
                    grid.record(&vals, false);
                }
            }

            // Compute rate.
            // Use Omega=4π because eff(z) was measured from isotropic (full-sky)
            // injection, so it already incorporates the sky coverage fraction.
            let eff_vs_z = grid.marginalize_over("z").unwrap_or_default();
            let survey_omega = estimate_survey_omega(
                self.survey.n_pixels(),
                self.survey.nside(),
            );
            let omega_full_sky = 4.0 * std::f64::consts::PI;
            let rate_per_year = compute_rate(
                &eff_vs_z,
                pop.volumetric_rate(),
                omega_full_sky,
                &self.cosmology,
            );

            let overall_eff = if self.n_transients > 0 {
                n_detected as f64 / self.n_transients as f64
            } else {
                0.0
            };

            // Compute rate recovery (inverse problem).
            // The recovery denominator uses compute_rate which integrates
            //   ∫ eff(z) × dV/dz / (1+z) dz
            // The MC numerator must estimate the same integral. Since the MC
            // samples z from p(z) ∝ dV/dz, we reweight each detection by
            // 1/(1+z) to match the time-dilation factor in compute_rate:
            //   N_obs = R_vol × T × V_full/N_sim × Σ_{detected} 1/(1+z_i)
            let v_total_gpc3 = self.cosmology.comoving_volume(actual_z_max); // full sky
            let sum_inv_1pz: f64 = detection_results
                .iter()
                .filter(|(_, det)| det.detected)
                .map(|(idx, _)| 1.0 / (1.0 + instances[*idx].z))
                .sum();
            let n_observed = pop.volumetric_rate()
                * self.survey.duration_years
                * v_total_gpc3
                / self.n_transients as f64
                * sum_inv_1pz;
            let recovery = recover_rate(
                &eff_vs_z,
                n_observed,
                self.survey.duration_years,
                omega_full_sky,
                pop.volumetric_rate(),
                &self.cosmology,
            );

            rate_summaries.push(RateSummary {
                transient_type: type_name,
                volumetric_rate: pop.volumetric_rate(),
                detections_per_year: rate_per_year,
                detections_total: rate_per_year * self.survey.duration_years,
                overall_efficiency: overall_eff,
                survey_omega_sr: survey_omega,
                z_max: actual_z_max,
                recovery: Some(recovery),
            });

            // Collect results.
            for (inst_idx, det) in detection_results {
                all_detection_results.push((instances[inst_idx].clone(), det));
            }
            all_instances.extend(instances);
        }

        let n_simulated = all_instances.len();
        let n_detected = all_detection_results
            .iter()
            .filter(|(_, d)| d.detected)
            .count();

        log::info!("Pipeline complete: {}/{} detected", n_detected, n_simulated);

        SimulationResult {
            rate_summaries,
            efficiency_grid: grid,
            detection_results: all_detection_results,
            n_simulated,
            n_detected,
        }
    }
}
