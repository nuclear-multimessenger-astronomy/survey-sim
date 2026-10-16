use std::collections::HashMap;

use survey_sim::detection::{evaluate_detection, DetectionCriteria, DetectionResult};
use survey_sim::lightcurve::LightcurveEvaluation;
use survey_sim::survey::SurveyObservation;
use survey_sim::types::{Band, SkyCoord};

fn make_obs(mjd: f64, band: &str, depth: f64) -> SurveyObservation {
    SurveyObservation {
        obs_id: 0,
        coord: SkyCoord::new(0.0, 0.0),
        mjd,
        band: Band::new(band),
        five_sigma_depth: depth,
        seeing_fwhm: 1.0,
        exposure_time: 30.0,
        airmass: 1.0,
        sky_brightness: 21.0,
        night: 0,
    }
}

#[test]
fn test_multi_band_detection() {
    let obs = vec![
        make_obs(60000.0, "g", 25.0),
        make_obs(60001.0, "r", 24.5),
        make_obs(60002.0, "g", 25.0),
    ];
    let obs_refs: Vec<&SurveyObservation> = obs.iter().collect();

    let mut mags = HashMap::new();
    mags.insert("g".to_string(), vec![23.0, 99.0, 23.5]);
    mags.insert("r".to_string(), vec![99.0, 22.0, 99.0]);

    let eval = LightcurveEvaluation {
        apparent_mags: mags,
        times_mjd: vec![60000.0, 60001.0, 60002.0],
    };

    let criteria = DetectionCriteria {
        min_detections: 2,
        min_bands: 2,
        min_per_band: 1,
        max_timespan_days: 30.0,
        snr_threshold: 5.0,
        ..Default::default()
    };

    let result = evaluate_detection(&eval, &obs_refs, &criteria);
    assert!(result.detected);
    assert_eq!(result.n_detections, 3);
    assert_eq!(result.n_bands_detected, 2);
}

#[test]
fn test_insufficient_bands() {
    // Only g-band detections, but require 2 bands.
    let obs = vec![
        make_obs(60000.0, "g", 25.0),
        make_obs(60001.0, "g", 25.0),
    ];
    let obs_refs: Vec<&SurveyObservation> = obs.iter().collect();

    let mut mags = HashMap::new();
    mags.insert("g".to_string(), vec![23.0, 23.5]);

    let eval = LightcurveEvaluation {
        apparent_mags: mags,
        times_mjd: vec![60000.0, 60001.0],
    };

    let criteria = DetectionCriteria {
        min_detections: 2,
        min_bands: 2, // requires 2 bands
        ..Default::default()
    };

    let result = evaluate_detection(&eval, &obs_refs, &criteria);
    assert!(!result.detected);
}

/// Verify fade rate measurement with a realistic kilonova lightcurve at ZTF cadence.
///
/// Simulates: KN peaks at mag 19.0, fades 1 mag/day in g-band.
/// ZTF observes once per night in g (depth 20.8). Should detect for ~2 nights.
/// Expected fade rate: ~1.0 mag/day.
#[test]
fn test_kilonova_fade_rate_ztf_cadence() {
    // ZTF-like: nightly cadence, alternating g/r.
    // KN peaks at day 0, fades at ~1 mag/day in both bands.
    let depth_g = 20.8;
    let depth_r = 20.6;

    // Night 0: g=19.0 (peak), Night 0.02: r=19.2
    // Night 1: g=20.0, Night 1.02: r=20.2
    // Night 2: g=21.0 (below depth), Night 2.02: r=21.2 (below depth)
    let obs = vec![
        make_obs(60000.0, "g", depth_g),
        make_obs(60000.02, "r", depth_r),
        make_obs(60001.0, "g", depth_g),
        make_obs(60001.02, "r", depth_r),
        make_obs(60002.0, "g", depth_g),
        make_obs(60002.02, "r", depth_r),
    ];
    let obs_refs: Vec<&SurveyObservation> = obs.iter().collect();

    let mut mags = HashMap::new();
    // All observations get magnitude arrays indexed by total obs index.
    // Fading at 1 mag/day.
    mags.insert(
        "g".to_string(),
        vec![19.0, 99.0, 20.0, 99.0, 21.0, 99.0],
    );
    mags.insert(
        "r".to_string(),
        vec![99.0, 19.2, 99.0, 20.2, 99.0, 21.2],
    );

    let eval = LightcurveEvaluation {
        apparent_mags: mags,
        times_mjd: vec![60000.0, 60000.02, 60001.0, 60001.02, 60002.0, 60002.02],
    };

    // Basic criteria (no fast transient requirement).
    let criteria_basic = DetectionCriteria::default();
    let result = evaluate_detection(&eval, &obs_refs, &criteria_basic);

    println!("KN fade test (basic): detected={}, n_det={}, rise={:?}, fade={:?}, is_fast={}",
        result.detected, result.n_detections, result.best_rise_rate, result.best_fade_rate, result.is_fast_transient);
    println!("  per_band: {:?}", result.detections_per_band);

    // Should detect 4 observations (g@19, g@20, r@19.2, r@20.2).
    assert_eq!(result.n_detections, 4, "Expected 4 detections");
    assert!(result.detected);

    // Fade rate should be ~1 mag/day.
    let fade = result.best_fade_rate.expect("Expected a fade rate");
    println!("  fade_rate = {:.4} mag/day", fade);
    assert!(
        fade > 0.5 && fade < 2.0,
        "Fade rate {:.4} should be ~1.0 mag/day",
        fade
    );

    // Now with ZTFReST criteria.
    let criteria_ztfrest = DetectionCriteria::ztfrest();
    let result_ztfrest = evaluate_detection(&eval, &obs_refs, &criteria_ztfrest);
    println!("KN fade test (ZTFReST): detected={}, n_det={} ({}@5σ), fade={:?}, is_fast={}",
        result_ztfrest.detected, result_ztfrest.n_detections, result_ztfrest.n_detections_primary,
        result_ztfrest.best_fade_rate, result_ztfrest.is_fast_transient);
    assert!(result_ztfrest.is_fast_transient, "Should qualify as fast transient");
    assert!(result_ztfrest.detected, "Should be detected with ZTFReST criteria");
}

/// Check that the fade rate is correct for a single-band, 3-point fade.
#[test]
fn test_fade_rate_single_band_3pt() {
    // 3 observations in g-band, 1 day apart, fading 0.5 mag/day.
    let obs = vec![
        make_obs(60000.0, "g", 25.0),
        make_obs(60001.0, "g", 25.0),
        make_obs(60002.0, "g", 25.0),
    ];
    let obs_refs: Vec<&SurveyObservation> = obs.iter().collect();

    let mut mags = HashMap::new();
    // Monotonically fading: 20.0, 20.5, 21.0.
    mags.insert("g".to_string(), vec![20.0, 20.5, 21.0]);

    let eval = LightcurveEvaluation {
        apparent_mags: mags,
        times_mjd: vec![60000.0, 60001.0, 60002.0],
    };

    let criteria = DetectionCriteria {
        require_fast_transient: true,
        min_fade_rate: 0.3,
        ..Default::default()
    };

    let result = evaluate_detection(&eval, &obs_refs, &criteria);
    println!("3pt fade test: fade={:?}, rise={:?}, is_fast={}", result.best_fade_rate, result.best_rise_rate, result.is_fast_transient);

    let fade = result.best_fade_rate.expect("Expected fade rate");
    assert!(
        (fade - 0.5).abs() < 0.1,
        "Fade rate {:.4} should be ~0.5 mag/day",
        fade
    );
    assert!(result.is_fast_transient);
    assert!(result.detected);
}

#[test]
fn test_snr_calculation_and_error() {
    use survey_sim::detection::{mag_error_from_snr, snr_from_depth};

    // At 5-sigma depth, SNR is exactly 5.0
    let snr_at_depth = snr_from_depth(20.0, 20.0);
    assert!((snr_at_depth - 5.0).abs() < 1e-10);

    let err_at_5sigma = mag_error_from_snr(5.0);
    assert!((err_at_5sigma - (1.0857362 / 5.0)).abs() < 1e-10);

    // 2.5 magnitudes brighter -> 10x flux -> 10x SNR
    let snr_bright = snr_from_depth(17.5, 20.0);
    assert!((snr_bright - 50.0).abs() < 1e-9);

    // 2.5 magnitudes fainter -> 0.1x flux -> 0.1x SNR
    let snr_faint = snr_from_depth(22.5, 20.0);
    assert!((snr_faint - 0.5).abs() < 1e-9);
}

#[test]
fn test_custom_snr_thresholds() {
    let obs = vec![
        make_obs(60000.0, "g", 20.0), // depth 20.0
        make_obs(60001.0, "g", 20.0),
    ];
    let obs_refs: Vec<&SurveyObservation> = obs.iter().collect();

    let mut mags = HashMap::new();
    // Mag 20.0 has SNR = 5.0
    mags.insert("g".to_string(), vec![20.0, 20.0]);

    let eval = LightcurveEvaluation {
        apparent_mags: mags,
        times_mjd: vec![60000.0, 60001.0],
    };

    // Stricter SNR threshold: snr_threshold = 10.0 (requires mag <= 19.247)
    let criteria_strict = DetectionCriteria {
        min_detections: 2,
        min_bands: 1,
        min_per_band: 1,
        snr_threshold: 10.0,
        min_detections_primary: 1,
        ..Default::default()
    };
    let res_strict = evaluate_detection(&eval, &obs_refs, &criteria_strict);
    // Observations have SNR=5.0, which is below 10.0 -> not detected
    assert!(!res_strict.detected);
    assert_eq!(res_strict.n_detections_primary, 0);

    // Looser SNR threshold: snr_threshold = 3.0, snr_threshold_secondary = 3.0
    // Mag 20.5 has SNR = 5.0 * 10^(0.4 * (20.0 - 20.5)) = 5.0 * 10^(-0.2) ≈ 3.15
    let mut faint_mags = HashMap::new();
    faint_mags.insert("g".to_string(), vec![20.5, 20.5]);
    let eval_faint = LightcurveEvaluation {
        apparent_mags: faint_mags,
        times_mjd: vec![60000.0, 60001.0],
    };

    let criteria_loose = DetectionCriteria {
        min_detections: 2,
        min_bands: 1,
        min_per_band: 1,
        snr_threshold: 3.0,
        snr_threshold_secondary: 3.0,
        min_detections_primary: 1,
        ..Default::default()
    };
    let res_loose = evaluate_detection(&eval_faint, &obs_refs, &criteria_loose);
    assert!(res_loose.detected);
    assert_eq!(res_loose.n_detections, 2);
    assert_eq!(res_loose.n_detections_primary, 2);
}
