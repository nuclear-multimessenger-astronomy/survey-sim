use std::collections::HashMap;

use survey_sim::detection::DetectionCriteria;
use survey_sim::lightcurve::parametric::ParametricModel;
use survey_sim::pipeline::SimulationPipeline;
use survey_sim::population::generator::KilonovaPopulation;
use survey_sim::survey::{SurveyObservation, SurveyStore};
use survey_sim::types::{Band, SkyCoord};

/// Create a small synthetic survey for testing.
fn make_synthetic_survey() -> SurveyStore {
    let mut observations = Vec::new();
    let mut obs_id = 0u64;

    // Create observations covering a grid of the sky over 1 year.
    for night in 0..365 {
        let mjd = 60000.0 + night as f64;
        for ra_block in 0..12 {
            let ra = ra_block as f64 * 30.0;
            for dec_block in 0..6 {
                let dec = -60.0 + dec_block as f64 * 20.0;
                for band in &["g", "r", "i"] {
                    observations.push(SurveyObservation {
                        obs_id,
                        coord: SkyCoord::new(ra, dec),
                        mjd,
                        band: Band::new(band),
                        five_sigma_depth: 24.0,
                        seeing_fwhm: 1.0,
                        exposure_time: 30.0,
                        airmass: 1.2,
                        sky_brightness: 21.0,
                        night: night as i64,
                    });
                    obs_id += 1;
                }
            }
        }
    }

    SurveyStore::new(observations, 4)
}

#[test]
fn test_pipeline_smoke() {
    let survey = make_synthetic_survey();
    println!(
        "Synthetic survey: {} obs, {:.1} years",
        survey.len(),
        survey.duration_years
    );

    let mut pipeline = SimulationPipeline::new(
        survey,
        DetectionCriteria {
            min_detections: 2,
            min_bands: 1,
            min_per_band: 1,
            max_timespan_days: 30.0,
            snr_threshold: 5.0,
            ..Default::default()
        },
        1000, // small for smoke test
        42,
    );

    let kn_pop = KilonovaPopulation::new(1000.0, 0.1, -16.0, 60000.0, 60365.0);
    pipeline.add_population(Box::new(kn_pop));
    pipeline.add_model("Kilonova", Box::new(ParametricModel::new().with_model(
        lightcurve_fitting::SviModelName::MetzgerKN,
    )));

    let result = pipeline.run();
    println!("{}", result);

    assert_eq!(result.n_simulated, 1000);
    // At least some should be detected with our very dense survey.
    println!(
        "Detected: {} / {} ({:.1}%)",
        result.n_detected,
        result.n_simulated,
        100.0 * result.n_detected as f64 / result.n_simulated as f64
    );
    // With z_max=0.1 and depth=24, we expect a reasonable detection rate.
    assert!(result.n_detected > 0, "Expected at least some detections");
}
