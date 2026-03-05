//! Reproduce the ZTFReST kilonova rate analysis from Andreoni & Coughlin 2021
//! (arXiv:2104.06352).
//!
//! Methodology:
//! 1. Load ZTF 2018–2020 observations from IRSA HDF5 files
//! 2. Inject synthetic GW170817-like kilonovae into the survey footprint
//! 3. Apply ZTFReST detection criteria (2 det, one 5σ one 3σ, >3hr sep, fade >0.3 mag/day)
//! 4. Measure efficiency vs redshift
//! 5. Compute 90% CL Poisson upper limit on kilonova rate (zero detections)
//!
//! Expected result: R_KN < ~900 Gpc^-3 yr^-1 (paper reports R < 900 combined)

use std::time::Instant;

use survey_sim::detection::{evaluate_detection, DetectionCriteria};
use survey_sim::efficiency::rates::{compute_rate_upper_limit, estimate_survey_omega};
use survey_sim::efficiency::{EfficiencyGrid, GridAxis};
use survey_sim::instrument::InstrumentConfig;
use survey_sim::lightcurve::parametric::ParametricModel;
use survey_sim::lightcurve::LightcurveModel;
use survey_sim::population::generator::KilonovaPopulation;
use survey_sim::population::PopulationGenerator;
use survey_sim::survey::ztf::ZtfHdf5Loader;
use survey_sim::survey::{SurveyLoader, SurveyStore};
use survey_sim::types::Cosmology;

/// Quick sanity check with 10K injections (runs in ~1 min).
#[test]
#[ignore]
fn test_ztfrest_sanity_10k() {
    run_ztfrest_analysis(10_000);
}

/// Full ZTFReST-style analysis with 100K injections.
#[test]
#[ignore]
fn test_ztfrest_kilonova_rate() {
    run_ztfrest_analysis(100_000);
}

fn run_ztfrest_analysis(n_inject: usize) {
    env_logger::try_init().ok();
    let t_total = Instant::now();

    // ==== Phase 0: Load ZTF observations ====
    println!("\n[Phase 0] Loading ZTF observations...");
    let t0 = Instant::now();
    let loader = ZtfHdf5Loader::new(&[
        "/fred/oz480/mcoughli/simulations/ztf_data/ztf_2018.h5",
        "/fred/oz480/mcoughli/simulations/ztf_data/ztf_2019.h5",
        "/fred/oz480/mcoughli/simulations/ztf_data/ztf_2020.h5",
    ]);
    let observations = loader.load().expect("Failed to load ZTF HDF5 files");
    let n_obs = observations.len();

    let mjd_min = observations
        .iter()
        .map(|o| o.mjd)
        .fold(f64::INFINITY, f64::min);
    let mjd_max = observations
        .iter()
        .map(|o| o.mjd)
        .fold(f64::NEG_INFINITY, f64::max);
    let duration_days = mjd_max - mjd_min;
    let duration_years = duration_days / 365.25;
    println!(
        "[Phase 0] Loaded {} exposures in {:.1}s",
        n_obs,
        t0.elapsed().as_secs_f64()
    );
    println!(
        "  MJD range: {:.1} – {:.1} ({:.2} years, {:.0} days)",
        mjd_min, mjd_max, duration_years, duration_days
    );

    // Build spatially-indexed survey store with ZTF FoV.
    println!("[Phase 0] Building spatial index...");
    let t0 = Instant::now();
    let survey = SurveyStore::new(observations, 64).with_instrument(InstrumentConfig::ztf());
    println!(
        "[Phase 0] SurveyStore ready in {:.1}s: {} obs, FoV radius {:.2} deg, {} bands",
        t0.elapsed().as_secs_f64(),
        survey.len(),
        survey.fov_radius_deg,
        survey.bands.len(),
    );

    // ==== Detection criteria (ZTFReST) ====
    let criteria = DetectionCriteria::ztfrest();
    println!(
        "\nDetection criteria (ZTFReST):\n  min_det={}, SNR primary={:.0}σ secondary={:.0}σ\n  min_time_sep={:.1}hr, max_timespan={:.0}d\n  fast_transient: fade>{:.1} mag/day",
        criteria.min_detections,
        criteria.snr_threshold,
        criteria.snr_threshold_secondary,
        criteria.min_time_separation_hours,
        criteria.max_timespan_days,
        criteria.min_fade_rate,
    );

    // ==== Phase 1: Generate kilonova population ====
    println!("\n[Phase 1] Generating {} kilonovae...", n_inject);
    let t0 = Instant::now();
    let z_max = 0.1;
    let bns_rate = 320.0; // GWTC-3: 320 Gpc^-3 yr^-1

    let pop = KilonovaPopulation::new(bns_rate, z_max, -16.0, mjd_min, mjd_max);
    let mut rng = <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(42);
    let instances = pop.generate(n_inject, &mut rng);
    println!(
        "[Phase 1] Generated {} kilonovae in {:.2}s (z_max={:.3}, M_peak=-16, R={} Gpc^-3 yr^-1)",
        instances.len(),
        t0.elapsed().as_secs_f64(),
        z_max,
        bns_rate,
    );

    // ==== Phase 2: Inject & detect ====
    println!("\n[Phase 2] Running injection-recovery (match → lightcurve → detect)...");
    let t0 = Instant::now();
    let model = ParametricModel::new();
    let n_z_bins = 20;
    let mut grid = EfficiencyGrid::new(vec![GridAxis::uniform("z", 0.0, z_max * 1.1, n_z_bins)]);
    let time_window = 30.0; // days post-explosion

    // Also run with relaxed criteria to diagnose bottlenecks.
    let criteria_basic = DetectionCriteria {
        min_detections: 2,
        min_bands: 1,
        ..Default::default()
    };
    let criteria_no_fast = DetectionCriteria {
        min_time_separation_hours: 3.0,
        snr_threshold_secondary: 3.0,
        max_timespan_days: 14.0,
        ..Default::default()
    };

    let mut n_matched = 0usize;
    let mut n_evaluated = 0usize;
    let mut n_detected = 0usize;
    let mut n_detected_basic = 0usize;
    let mut n_detected_no_fast = 0usize;
    let mut n_any_bright = 0usize; // any obs brighter than depth
    let report_interval = (n_inject / 5).max(1000);

    for (i, inst) in instances.iter().enumerate() {
        let mjd_lo = inst.t_exp;
        let mjd_hi = inst.t_exp + time_window * (1.0 + inst.z);

        // Spatial-temporal match.
        let obs_indices = survey.query(&inst.coord, mjd_lo, mjd_hi);
        if obs_indices.is_empty() {
            let mut vals = std::collections::HashMap::new();
            vals.insert("z".to_string(), inst.z);
            grid.record(&vals, false);
            continue;
        }
        n_matched += 1;

        // Lightcurve evaluation.
        let times: Vec<f64> = obs_indices.iter().map(|&oi| survey.get(oi).mjd).collect();
        let bands: Vec<_> = obs_indices
            .iter()
            .map(|&oi| survey.get(oi).band.clone())
            .collect();

        let eval = match model.evaluate(inst, &times, &bands) {
            Ok(e) => e,
            Err(_) => {
                let mut vals = std::collections::HashMap::new();
                vals.insert("z".to_string(), inst.z);
                grid.record(&vals, false);
                continue;
            }
        };
        n_evaluated += 1;

        // Detection evaluation with all criteria sets.
        let obs_refs: Vec<_> = obs_indices.iter().map(|&oi| survey.get(oi)).collect();
        let result = evaluate_detection(&eval, &obs_refs, &criteria);
        let result_basic = evaluate_detection(&eval, &obs_refs, &criteria_basic);
        let result_no_fast = evaluate_detection(&eval, &obs_refs, &criteria_no_fast);

        // Check if any observation is brighter than depth.
        let mut any_bright = false;
        for (j, obs) in obs_refs.iter().enumerate() {
            if let Some(mags) = eval.apparent_mags.get(&obs.band.0) {
                if j < mags.len() && mags[j] < obs.five_sigma_depth {
                    any_bright = true;
                    break;
                }
            }
        }
        if any_bright {
            n_any_bright += 1;
        }

        let mut vals = std::collections::HashMap::new();
        vals.insert("z".to_string(), inst.z);
        grid.record(&vals, result.detected);

        if result_basic.detected {
            n_detected_basic += 1;
        }
        if result_no_fast.detected {
            n_detected_no_fast += 1;
            // Show fade rate info for first few.
            if n_detected_no_fast <= 10 {
                println!(
                    "  [no-fast #{}] z={:.4}, M={:.1}, {} obs, n_det={}, rise={:?}, fade={:?}, is_fast={}",
                    n_detected_no_fast, inst.z, inst.peak_abs_mag,
                    obs_indices.len(), result_no_fast.n_detections,
                    result_no_fast.best_rise_rate, result_no_fast.best_fade_rate,
                    result_no_fast.is_fast_transient,
                );
            }
        }
        if result.detected {
            n_detected += 1;
            if n_detected <= 10 {
                println!(
                    "  Detection #{}: z={:.4}, M={:.1}, {} obs, n_det={} ({}@5σ), fade={:?} mag/day",
                    n_detected,
                    inst.z,
                    inst.peak_abs_mag,
                    obs_indices.len(),
                    result.n_detections,
                    result.n_detections_primary,
                    result.best_fade_rate,
                );
            }
        }

        // Diagnostic: print details for first few low-z matched transients.
        if inst.z < 0.015 && n_matched <= 20 {
            let mut brightest = 99.0f64;
            let mut brightest_band = String::new();
            let mut brightest_depth = 0.0f64;
            for (j, obs) in obs_refs.iter().enumerate() {
                if let Some(mags) = eval.apparent_mags.get(&obs.band.0) {
                    if j < mags.len() && mags[j] < brightest {
                        brightest = mags[j];
                        brightest_band = obs.band.0.clone();
                        brightest_depth = obs.five_sigma_depth;
                    }
                }
            }
            println!(
                "  [diag] z={:.4}, M={:.1}, brightest_app={:.2} ({}-band, depth={:.1}), {} obs, basic_det={}, no_fast_det={}, ztfrest_det={}",
                inst.z, inst.peak_abs_mag, brightest, brightest_band, brightest_depth,
                obs_indices.len(), result_basic.detected, result_no_fast.detected, result.detected,
            );
        }

        if (i + 1) % report_interval == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            let eta = (n_inject - i - 1) as f64 / rate;
            println!(
                "  [{}/{}] matched={} eval={} det={} ({:.1}/s, ETA {:.0}s)",
                i + 1,
                n_inject,
                n_matched,
                n_evaluated,
                n_detected,
                rate,
                eta,
            );
        }
    }

    let phase2_time = t0.elapsed().as_secs_f64();
    let overall_eff = n_detected as f64 / n_inject as f64;
    println!(
        "[Phase 2] Complete in {:.1}s",
        phase2_time,
    );
    println!(
        "\n=== INJECTION-RECOVERY RESULTS ===\n  Injected:   {}\n  Matched:    {} ({:.1}%)\n  Evaluated:  {} ({:.1}%)\n  Any bright: {} ({:.2}%)\n  Basic det:  {} ({:.2}%)\n  No-fast det:{} ({:.2}%)\n  ZTFReST det:{} ({:.2}%)",
        n_inject,
        n_matched,
        100.0 * n_matched as f64 / n_inject as f64,
        n_evaluated,
        100.0 * n_evaluated as f64 / n_inject as f64,
        n_any_bright,
        100.0 * n_any_bright as f64 / n_inject as f64,
        n_detected_basic,
        100.0 * n_detected_basic as f64 / n_inject as f64,
        n_detected_no_fast,
        100.0 * n_detected_no_fast as f64 / n_inject as f64,
        n_detected,
        overall_eff * 100.0,
    );

    // ==== Efficiency vs redshift ====
    let eff_vs_z = grid.marginalize_over("z").unwrap();
    println!("\nEfficiency vs redshift:");
    println!("  {:>8}  {:>10}", "z", "eff");
    for &(z, eff) in &eff_vs_z {
        let bar = "#".repeat((eff * 40.0) as usize);
        println!("  {:>8.4}  {:>10.4}  {}", z, eff, bar);
    }

    // ==== Survey solid angle ====
    let cosmo = Cosmology::default();
    let n_pixels = survey.n_pixels();
    let nside = survey.nside();
    let survey_omega = estimate_survey_omega(n_pixels, nside);
    let sky_fraction = survey_omega / (4.0 * std::f64::consts::PI);
    println!(
        "\nSurvey: {} unique pixels (nside={}), Omega={:.4} sr ({:.1}% of sky), T={:.2} yr",
        n_pixels, nside,
        survey_omega,
        sky_fraction * 100.0,
        duration_years,
    );

    // ==== Rate upper limit (zero confirmed KN detections) ====
    // Use Omega=4π because eff(z) was measured from isotropic (full-sky) injection,
    // so it already incorporates the sky coverage fraction. Using survey_omega here
    // would double-count the sky fraction.
    let omega_full_sky = 4.0 * std::f64::consts::PI;
    let ul_90 = compute_rate_upper_limit(&eff_vs_z, 0, duration_years, omega_full_sky, 0.90, &cosmo);
    let ul_95 = compute_rate_upper_limit(&eff_vs_z, 0, duration_years, omega_full_sky, 0.95, &cosmo);

    println!("\n{}", ul_90);
    println!("{}", ul_95);

    // ==== Sanity checks ====
    assert!(n_obs > 300_000, "Expected >300K ZTF exposures for 3 years");
    assert!(duration_years > 2.0, "Expected >2yr survey duration");
    assert!(n_matched > 0, "Expected some transients to match survey");
    if ul_95.rate_upper.is_finite() {
        assert!(
            ul_95.rate_upper > 0.0 && ul_95.rate_upper < 1e7,
            "Rate upper limit {:.1} out of expected range",
            ul_95.rate_upper
        );
    } else {
        println!("WARNING: VT_eff=0 → infinite rate upper limit (zero efficiency).");
        println!("  This indicates a problem with detection efficiency — see diagnostics above.");
    }

    println!("\n=== ZTFReST Analysis Complete ({:.1}s total) ===", t_total.elapsed().as_secs_f64());
    println!("Paper: R_KN < 900 Gpc^-3 yr^-1 (95% CL combined, 2018-2021)");
    println!(
        "Ours:  R_KN < {:.0} Gpc^-3 yr^-1 (90% CL), < {:.0} Gpc^-3 yr^-1 (95% CL)",
        ul_90.rate_upper, ul_95.rate_upper,
    );
}
