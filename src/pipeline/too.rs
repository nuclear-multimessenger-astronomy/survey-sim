//! Targeted ToO simulation pipeline.
//!
//! For each GW event, places a kilonova at the event's true sky position and
//! distance, generates synthetic ToO observations from a strategy, and evaluates
//! detection. This solves the fundamental mismatch between isotropic injection
//! (which misses small FoV instruments) and targeted follow-up.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::detection::{evaluate_detection, DetectionCriteria, DetectionResult};
use crate::lightcurve::LightcurveModel;
use crate::population::PopulationGenerator;
use crate::survey::observing_scenario::GwEvent;
use crate::survey::too::{TooStrategy, TooTrigger};
use crate::survey::SurveyStore;
use crate::types::{Cosmology, TransientInstance};

/// Result of a ToO simulation for a single GW event.
#[derive(Clone, Debug)]
pub struct TooEventResult {
    /// The GW event.
    pub event: GwEvent,
    /// The generated transient instance.
    pub instance: TransientInstance,
    /// Detection result.
    pub detection: DetectionResult,
    /// Number of ToO observations generated.
    pub n_observations: usize,
}

/// Aggregate result of a ToO simulation campaign.
pub struct TooSimulationResult {
    /// Per-event results.
    pub event_results: Vec<TooEventResult>,
    /// Strategy name.
    pub strategy_name: String,
    /// Total events evaluated.
    pub n_events: usize,
    /// Total detected.
    pub n_detected: usize,
    /// Detection efficiency.
    pub efficiency: f64,
}

impl std::fmt::Display for TooSimulationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== ToO Simulation: {} ===", self.strategy_name)?;
        writeln!(f, "Events evaluated: {}", self.n_events)?;
        writeln!(
            f,
            "Detected: {} ({:.1}%)",
            self.n_detected,
            self.efficiency * 100.0
        )?;

        // Breakdown by type.
        let bns: Vec<_> = self.event_results.iter().filter(|r| r.event.is_bns()).collect();
        let nsbh: Vec<_> = self.event_results.iter().filter(|r| r.event.is_nsbh()).collect();
        let bns_det = bns.iter().filter(|r| r.detection.detected).count();
        let nsbh_det = nsbh.iter().filter(|r| r.detection.detected).count();
        if !bns.is_empty() {
            writeln!(
                f,
                "  BNS: {}/{} ({:.1}%)",
                bns_det,
                bns.len(),
                100.0 * bns_det as f64 / bns.len() as f64
            )?;
        }
        if !nsbh.is_empty() {
            writeln!(
                f,
                "  NSBH: {}/{} ({:.1}%)",
                nsbh_det,
                nsbh.len(),
                100.0 * nsbh_det as f64 / nsbh.len() as f64
            )?;
        }

        // Distance stats for detected events.
        let det_dists: Vec<f64> = self
            .event_results
            .iter()
            .filter(|r| r.detection.detected)
            .map(|r| r.event.distance_mpc)
            .collect();
        if !det_dists.is_empty() {
            let mean_d = det_dists.iter().sum::<f64>() / det_dists.len() as f64;
            let max_d = det_dists.iter().cloned().fold(0.0f64, f64::max);
            writeln!(
                f,
                "  Detected distance: mean={:.0} Mpc, max={:.0} Mpc",
                mean_d, max_d
            )?;
        }

        Ok(())
    }
}

/// Run a targeted ToO simulation for a list of GW events.
///
/// For each event:
/// 1. Generate a KN at the event's true sky position and distance using the
///    provided population generator (to get realistic model parameters).
/// 2. Override the KN's position, distance, and explosion time to match the event.
/// 3. Generate ToO observations from the strategy.
/// 4. Evaluate detection.
///
/// Events are filtered by type (BNS/NSBH only by default).
pub fn run_too_simulation(
    events: &[GwEvent],
    strategy: &dyn TooStrategy,
    population: &dyn PopulationGenerator,
    model: &dyn LightcurveModel,
    criteria: &DetectionCriteria,
    trigger_mjd: f64,
    include_bbh: bool,
) -> TooSimulationResult {
    use rand::SeedableRng;

    let strategy_name = strategy.name().to_string();
    let cosmology = Cosmology::default();

    // Filter to EM-relevant events (BNS + NSBH, optionally BBH).
    let relevant_events: Vec<&GwEvent> = events
        .iter()
        .filter(|e| e.is_bns() || e.is_nsbh() || (include_bbh && e.is_bbh()))
        .collect();

    log::info!(
        "ToO simulation: {} events ({} relevant) with strategy '{}'",
        events.len(),
        relevant_events.len(),
        strategy_name,
    );

    let uses_python = model.requires_gil();

    // Process events — parallel for Rust models, sequential for Python.
    let process_event = |event: &&GwEvent, idx: usize| -> Option<TooEventResult> {
        // Generate one KN instance to get realistic model parameters.
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42 + idx as u64);
        let mut instances = population.generate(1, &mut rng);
        if instances.is_empty() {
            return None;
        }
        let mut instance = instances.remove(0);

        // Override position, distance, and timing to match the GW event.
        instance.coord = crate::types::SkyCoord::new(event.ra, event.dec);
        instance.d_l = event.distance_mpc;
        instance.z = cosmology.redshift_from_distance(event.distance_mpc);
        instance.t_exp = trigger_mjd; // explosion at trigger time

        // Store inclination from the GW event if available.
        instance
            .model_params
            .insert("inclination".to_string(), event.inclination);

        // Generate ToO observations for this event.
        let trigger = TooTrigger {
            coord: instance.coord,
            trigger_mjd,
            localization_area_deg2: event.area_90,
            distance_mpc: Some(event.distance_mpc),
        };
        let observations = strategy.generate_observations(&trigger, 0);
        let n_observations = observations.len();

        if observations.is_empty() {
            return None;
        }

        // Build a temporary survey store.
        let store = SurveyStore::new(observations, 64);

        // Phase 1: Spatial matching.
        let time_window = 100.0;
        let mjd_min = instance.t_exp;
        let mjd_max = instance.t_exp + time_window * (1.0 + instance.z);
        let obs_indices = store.query(&instance.coord, mjd_min, mjd_max);

        if obs_indices.is_empty() {
            return Some(TooEventResult {
                event: (*event).clone(),
                instance,
                detection: DetectionResult {
                    detected: false,
                    n_detections: 0,
                    n_detections_primary: 0,
                    n_bands_detected: 0,
                    first_detection_mjd: None,
                    last_detection_mjd: None,
                    detections_per_band: HashMap::new(),
                    best_rise_rate: None,
                    best_fade_rate: None,
                    is_fast_transient: false,
                    peak_mjd: None,
                    peak_mag: None,
                    n_pre_peak: 0,
                    n_post_peak: 0,
                    phase_min_days: None,
                    phase_max_days: None,
                },
                n_observations,
            });
        }

        // Phase 2: Lightcurve evaluation.
        let times: Vec<f64> = obs_indices.iter().map(|&oi| store.get(oi).mjd).collect();
        let bands: Vec<_> = obs_indices
            .iter()
            .map(|&oi| store.get(oi).band.clone())
            .collect();

        let eval = match model.evaluate(&instance, &times, &bands) {
            Ok(e) => e,
            Err(_) => return None,
        };

        // Phase 3: Detection evaluation.
        let obs_refs: Vec<_> = obs_indices.iter().map(|&oi| store.get(oi)).collect();
        let detection = evaluate_detection(&eval, &obs_refs, criteria);

        Some(TooEventResult {
            event: (*event).clone(),
            instance,
            detection,
            n_observations,
        })
    };

    let event_results: Vec<TooEventResult> = if uses_python {
        relevant_events
            .iter()
            .enumerate()
            .filter_map(|(i, e)| process_event(e, i))
            .collect()
    } else {
        relevant_events
            .par_iter()
            .enumerate()
            .filter_map(|(i, e)| process_event(e, i))
            .collect()
    };

    let n_events = event_results.len();
    let n_detected = event_results.iter().filter(|r| r.detection.detected).count();
    let efficiency = if n_events > 0 {
        n_detected as f64 / n_events as f64
    } else {
        0.0
    };

    log::info!(
        "ToO simulation complete: {}/{} detected ({:.1}%)",
        n_detected,
        n_events,
        efficiency * 100.0,
    );

    TooSimulationResult {
        event_results,
        strategy_name,
        n_events,
        n_detected,
        efficiency,
    }
}
