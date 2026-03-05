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
