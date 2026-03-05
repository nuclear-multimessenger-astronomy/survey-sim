use std::collections::HashMap;
use std::time::Instant;

use rayon::prelude::*;

use survey_sim::detection::{evaluate_detection, DetectionCriteria};
use survey_sim::efficiency::{EfficiencyGrid, GridAxis};
use survey_sim::instrument::InstrumentConfig;
use survey_sim::lightcurve::parametric::ParametricModel;
use survey_sim::lightcurve::LightcurveModel;
use survey_sim::population::generator::KilonovaPopulation;
use survey_sim::population::PopulationGenerator;
use survey_sim::survey::rubin::RubinLoader;
use survey_sim::survey::{SurveyLoader, SurveyStore};
use survey_sim::types::SkyCoord;

/// Profile each pipeline phase against Rubin Year 1.
#[test]
#[ignore]
fn profile_pipeline_phases() {
    let t0 = Instant::now();

    // --- Load survey ---
    let db_path = "/fred/oz480/mcoughli/simulations/TESS-Rubin/baseline_v5.1.1_10yrs.db";
    let loader = RubinLoader::new(db_path);
    let mut observations = loader.load().expect("Failed to load OpSim database");
    let mjd_min = observations.iter().map(|o| o.mjd).fold(f64::INFINITY, f64::min);
    let mjd_year1_max = mjd_min + 365.25;
    observations.retain(|obs| obs.mjd <= mjd_year1_max);
    println!("[{:.2}s] Loaded {} Year 1 observations", t0.elapsed().as_secs_f64(), observations.len());

    let t1 = Instant::now();
    let survey = SurveyStore::new(observations, 64).with_instrument(InstrumentConfig::rubin());
    println!("[{:.2}s] Built SurveyStore + spatial index (nside=64, fov_radius={:.2}°)",
        t1.elapsed().as_secs_f64(), survey.fov_radius_deg);

    // --- Generate population ---
    let t2 = Instant::now();
    let pop = KilonovaPopulation::new(45.0, 0.3, -16.0, mjd_min, mjd_year1_max);
    let n_transients = 5000; // smaller for profiling
    let mut rng = <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(42);
    let instances = pop.generate(n_transients, &mut rng);
    println!("[{:.2}s] Generated {} transients (z_max={:.2})",
        t2.elapsed().as_secs_f64(), instances.len(),
        instances.iter().map(|i| i.z).fold(0.0f64, f64::max));

    // --- Phase 1: Spatial-temporal matching ---
    let t3 = Instant::now();
    let time_window = 100.0;
    let matched: Vec<(usize, Vec<usize>)> = instances
        .par_iter()
        .enumerate()
        .map(|(i, inst)| {
            let mjd_lo = inst.t_exp;
            let mjd_hi = inst.t_exp + time_window * (1.0 + inst.z);
            let obs_indices = survey.query(&inst.coord, mjd_lo, mjd_hi);
            (i, obs_indices)
        })
        .filter(|(_, obs)| !obs.is_empty())
        .collect();
    let phase1_time = t3.elapsed().as_secs_f64();
    let total_matched_obs: usize = matched.iter().map(|(_, obs)| obs.len()).sum();
    println!("[{:.2}s] Phase 1 (spatial match): {} / {} transients matched, {} total obs pairs",
        phase1_time, matched.len(), n_transients, total_matched_obs);
    if !matched.is_empty() {
        let avg_obs = total_matched_obs as f64 / matched.len() as f64;
        println!("  avg {:.1} obs/transient, {:.1} us/query",
            avg_obs, phase1_time * 1e6 / n_transients as f64);
    }

    // --- Phase 2: Lightcurve evaluation ---
    let model = ParametricModel::new().with_model(lightcurve_fitting::SviModelName::MetzgerKN);
    let t4 = Instant::now();
    let evaluations: Vec<_> = matched
        .par_iter()
        .filter_map(|(inst_idx, obs_indices)| {
            let inst = &instances[*inst_idx];
            let times: Vec<f64> = obs_indices.iter().map(|&oi| survey.get(oi).mjd).collect();
            let bands: Vec<_> = obs_indices.iter().map(|&oi| survey.get(oi).band.clone()).collect();
            match model.evaluate(inst, &times, &bands) {
                Ok(eval) => Some((*inst_idx, obs_indices.clone(), eval)),
                Err(_) => None,
            }
        })
        .collect();
    let phase2_time = t4.elapsed().as_secs_f64();
    println!("[{:.2}s] Phase 2 (lightcurve eval): {} evaluations, {:.1} us/eval",
        phase2_time, evaluations.len(),
        if evaluations.is_empty() { 0.0 } else { phase2_time * 1e6 / evaluations.len() as f64 });

    // --- Phase 3: Detection evaluation ---
    let criteria = DetectionCriteria {
        min_detections: 2,
        min_bands: 1,
        min_per_band: 2,
        max_timespan_days: 30.0,
        snr_threshold: 5.0,
        require_fast_transient: true,
        min_rise_rate: 1.0,
        min_fade_rate: 0.3,
        ..Default::default()
    };
    let t5 = Instant::now();
    let detection_results: Vec<_> = evaluations
        .par_iter()
        .map(|(inst_idx, obs_indices, eval)| {
            let obs_refs: Vec<_> = obs_indices.iter().map(|&oi| survey.get(oi)).collect();
            let result = evaluate_detection(eval, &obs_refs, &criteria);
            (*inst_idx, result)
        })
        .collect();
    let phase3_time = t5.elapsed().as_secs_f64();
    let n_detected = detection_results.iter().filter(|(_, d)| d.detected).count();
    let n_fast = detection_results.iter().filter(|(_, d)| d.is_fast_transient).count();
    println!("[{:.2}s] Phase 3 (detection eval): {} detected, {} fast transients, {:.1} us/eval",
        phase3_time, n_detected, n_fast,
        if detection_results.is_empty() { 0.0 } else { phase3_time * 1e6 / detection_results.len() as f64 });

    // --- Summary ---
    let total = t0.elapsed().as_secs_f64();
    println!("\n=== PROFILE SUMMARY ({} transients) ===", n_transients);
    println!("  DB load:           {:.2}s", t1.elapsed().as_secs_f64());
    println!("  Phase 1 (spatial): {:.2}s ({:.1}%)", phase1_time, 100.0 * phase1_time / total);
    println!("  Phase 2 (LC eval): {:.2}s ({:.1}%)", phase2_time, 100.0 * phase2_time / total);
    println!("  Phase 3 (detect):  {:.2}s ({:.1}%)", phase3_time, 100.0 * phase3_time / total);
    println!("  Total:             {:.2}s", total);
    println!("  Detected: {} / {} ({:.2}%)", n_detected, n_transients,
        100.0 * n_detected as f64 / n_transients as f64);
}
