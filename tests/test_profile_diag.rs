use std::time::Instant;

use rayon::prelude::*;

use survey_sim::detection::{evaluate_detection, DetectionCriteria};
use survey_sim::instrument::InstrumentConfig;
use survey_sim::lightcurve::parametric::ParametricModel;
use survey_sim::lightcurve::LightcurveModel;
use survey_sim::population::generator::KilonovaPopulation;
use survey_sim::population::PopulationGenerator;
use survey_sim::survey::rubin::RubinLoader;
use survey_sim::survey::{SurveyLoader, SurveyStore};

/// Diagnose why 0 detections with Rubin Year 1.
#[test]
#[ignore]
fn diagnose_zero_detections() {
    let db_path = "/fred/oz480/mcoughli/simulations/TESS-Rubin/baseline_v5.1.1_10yrs.db";
    let loader = RubinLoader::new(db_path);
    let mut observations = loader.load().unwrap();
    let mjd_min = observations.iter().map(|o| o.mjd).fold(f64::INFINITY, f64::min);
    observations.retain(|obs| obs.mjd <= mjd_min + 365.25);
    let survey = SurveyStore::new(observations, 64).with_instrument(InstrumentConfig::rubin());

    let pop = KilonovaPopulation::new(45.0, 0.3, -16.0, mjd_min, mjd_min + 365.25);
    let mut rng = <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(42);
    let instances = pop.generate(500, &mut rng);
    let model = ParametricModel::new().with_model(lightcurve_fitting::SviModelName::MetzgerKN);

    // First: test WITHOUT fast transient requirement
    let criteria_basic = DetectionCriteria {
        min_detections: 2,
        min_bands: 1,
        min_per_band: 1,
        max_timespan_days: 30.0,
        snr_threshold: 5.0,
        ..Default::default()
    };

    let criteria_fast = DetectionCriteria {
        min_per_band: 2,
        require_fast_transient: true,
        ..Default::default()
    };

    let time_window = 100.0;
    let mut n_matched = 0;
    let mut n_basic_det = 0;
    let mut n_fast_det = 0;
    let mut sample_mags: Vec<(f64, f64, String)> = Vec::new(); // (z, brightest_mag, band)

    for (i, inst) in instances.iter().enumerate() {
        let mjd_lo = inst.t_exp;
        let mjd_hi = inst.t_exp + time_window * (1.0 + inst.z);
        let obs_indices = survey.query(&inst.coord, mjd_lo, mjd_hi);
        if obs_indices.is_empty() {
            continue;
        }
        n_matched += 1;

        let times: Vec<f64> = obs_indices.iter().map(|&oi| survey.get(oi).mjd).collect();
        let bands: Vec<_> = obs_indices.iter().map(|&oi| survey.get(oi).band.clone()).collect();

        let eval = match model.evaluate(inst, &times, &bands) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Find brightest magnitude across all bands
        let mut brightest = 99.0;
        let mut brightest_band = String::new();
        for (band, mags) in &eval.apparent_mags {
            for &m in mags {
                if m < brightest {
                    brightest = m;
                    brightest_band = band.clone();
                }
            }
        }

        // Find deepest observation
        let deepest = obs_indices.iter().map(|&oi| survey.get(oi).five_sigma_depth).fold(0.0f64, f64::max);

        if i < 10 && n_matched <= 10 {
            println!("  Transient {}: z={:.3}, peak_abs_mag={:.1}, brightest_app={:.2} ({}), deepest_obs={:.2}, {} obs",
                i, inst.z, inst.peak_abs_mag, brightest, brightest_band, deepest, obs_indices.len());
        }

        // Collect sample for stats
        sample_mags.push((inst.z, brightest, brightest_band.clone()));

        let obs_refs: Vec<_> = obs_indices.iter().map(|&oi| survey.get(oi)).collect();
        let result_basic = evaluate_detection(&eval, &obs_refs, &criteria_basic);
        let result_fast = evaluate_detection(&eval, &obs_refs, &criteria_fast);

        if result_basic.detected {
            n_basic_det += 1;
            if result_basic.best_rise_rate.is_some() || result_basic.best_fade_rate.is_some() {
                println!("  HAS RATE: transient {}, z={:.3}, M={:.1}, rise={:?}, fade={:?}, fast={}",
                    i, inst.z, inst.peak_abs_mag,
                    result_basic.best_rise_rate, result_basic.best_fade_rate, result_basic.is_fast_transient);
            }
        }
        if result_fast.detected {
            n_fast_det += 1;
        }

        if i < 20 && result_basic.detected {
            println!("    -> BASIC detected: {} detections, {} bands, rise={:?}, fade={:?}, fast={}",
                result_basic.n_detections, result_basic.n_bands_detected,
                result_basic.best_rise_rate, result_basic.best_fade_rate, result_basic.is_fast_transient);
            // Show per-band detection breakdown and detected mags
            for (band, count) in &result_basic.detections_per_band {
                // Show the detected magnitudes for this band
                let det_mags: Vec<(f64, f64)> = obs_indices.iter()
                    .filter_map(|&oi| {
                        let obs = survey.get(oi);
                        if obs.band.0 == *band {
                            eval.apparent_mags.get(band).and_then(|mags_vec| {
                                let idx = obs_indices.iter().position(|&x| x == oi)?;
                                let m = *mags_vec.get(idx)?;
                                if m < obs.five_sigma_depth { Some((obs.mjd, m)) } else { None }
                            })
                        } else { None }
                    })
                    .collect();
                println!("      {}-band: {} det, mags: {:?}", band, count, det_mags);
            }
            // Also show fast transient result
            println!("    -> FAST: detected={}, rise={:?}, fade={:?}, fast={}",
                result_fast.detected, result_fast.best_rise_rate, result_fast.best_fade_rate, result_fast.is_fast_transient);
        }
    }

    // Stats on brightest mags
    if !sample_mags.is_empty() {
        let brightest_overall = sample_mags.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
        let median_idx = sample_mags.len() / 2;
        let mut sorted: Vec<f64> = sample_mags.iter().map(|s| s.1).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("\nMagnitude stats ({} matched transients):", sample_mags.len());
        println!("  Brightest: {:.2}", brightest_overall);
        println!("  Median:    {:.2}", sorted[median_idx]);
        println!("  Faintest:  {:.2}", sorted.last().unwrap());
    }

    // Count multi-night detections
    let mut n_multi_night = 0;
    let mut night_counts: Vec<usize> = Vec::new();
    for (i, inst) in instances.iter().enumerate() {
        let mjd_lo = inst.t_exp;
        let mjd_hi = inst.t_exp + time_window * (1.0 + inst.z);
        let obs_indices = survey.query(&inst.coord, mjd_lo, mjd_hi);
        if obs_indices.is_empty() { continue; }

        let times: Vec<f64> = obs_indices.iter().map(|&oi| survey.get(oi).mjd).collect();
        let bands: Vec<_> = obs_indices.iter().map(|&oi| survey.get(oi).band.clone()).collect();
        let eval = match model.evaluate(inst, &times, &bands) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let obs_refs: Vec<_> = obs_indices.iter().map(|&oi| survey.get(oi)).collect();
        let result = evaluate_detection(&eval, &obs_refs, &criteria_basic);
        if !result.detected { continue; }

        // Count unique nights with detections
        let mut det_nights: Vec<f64> = Vec::new();
        for (j, obs) in obs_refs.iter().enumerate() {
            let band_name = &obs.band.0;
            if let Some(mags) = eval.apparent_mags.get(band_name) {
                if j < mags.len() && mags[j] < obs.five_sigma_depth {
                    det_nights.push(obs.mjd);
                }
            }
        }
        det_nights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        det_nights.dedup_by(|a, b| (*a - *b).abs() < 0.5);
        night_counts.push(det_nights.len());
        if det_nights.len() >= 2 {
            n_multi_night += 1;
            if n_multi_night <= 5 {
                println!("  Multi-night: transient {}, z={:.3}, M={:.1}, {} unique nights, nights: {:?}",
                    i, inst.z, inst.peak_abs_mag, det_nights.len(),
                    det_nights.iter().map(|&d| format!("{:.2}", d)).collect::<Vec<_>>());
            }
        }
    }
    night_counts.sort();
    println!("\nUnique detection nights distribution (of {} basic detections):", n_basic_det);
    for n in 1..=5 {
        let count = night_counts.iter().filter(|&&c| c == n).count();
        if count > 0 { println!("  {} night(s): {} transients", n, count); }
    }
    let max_nights = night_counts.last().copied().unwrap_or(0);
    if max_nights > 5 {
        let count = night_counts.iter().filter(|&&c| c > 5).count();
        println!("  >5 nights: {} transients (max={})", count, max_nights);
    }

    println!("\n=== DETECTION SUMMARY ({} transients) ===", instances.len());
    println!("  Spatially matched:   {}", n_matched);
    println!("  Basic detection:     {}", n_basic_det);
    println!("  Multi-night det:     {}", n_multi_night);
    println!("  Fast transient det:  {}", n_fast_det);
}
