use survey_sim::detection::DetectionCriteria;
use survey_sim::lightcurve::parametric::ParametricModel;
use survey_sim::pipeline::SimulationPipeline;
use survey_sim::population::generator::KilonovaPopulation;
use survey_sim::survey::{SurveyObservation, SurveyStore};
use survey_sim::types::{Band, SkyCoord};

/// Create a synthetic survey matching the test_pipeline pattern.
fn make_synthetic_survey() -> SurveyStore {
    let mut observations = Vec::new();
    let mut obs_id = 0u64;

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

fn make_pipeline(rate: f64, z_max: f64, peak_abs_mag: f64, n_transients: usize) -> SimulationPipeline {
    let survey = make_synthetic_survey();
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
        n_transients,
        42,
    );

    let kn_pop = KilonovaPopulation::new(rate, z_max, peak_abs_mag, 60000.0, 60365.0);
    pipeline.add_population(Box::new(kn_pop));
    pipeline.add_model(
        "Kilonova",
        Box::new(ParametricModel::new().with_model(lightcurve_fitting::SviModelName::MetzgerKN)),
    );

    pipeline
}

#[test]
fn test_bns_kilonova_rate_recovery() {
    let rate_bns = 45.0; // Gpc^-3 yr^-1 (LIGO BNS)
    let pipeline = make_pipeline(rate_bns, 0.3, -16.0, 50_000);
    let result = pipeline.run();

    println!("{}", result);

    assert!(!result.rate_summaries.is_empty());
    let summary = &result.rate_summaries[0];
    let recovery = summary.recovery.as_ref().expect("recovery should be computed");

    println!("BNS Rate Recovery:");
    println!("  Input rate:     {:.2} Gpc^-3 yr^-1", recovery.input_rate);
    println!("  Recovered rate: {:.2} Gpc^-3 yr^-1", recovery.recovered_rate);
    println!("  Effective VT:   {:.4} Gpc^3 yr", recovery.effective_vt);
    println!("  N_expected:     {:.1}", recovery.n_expected_detections);
    println!(
        "  2sig interval:  [{:.2}, {:.2}]",
        recovery.poisson_lower_2sig, recovery.poisson_upper_2sig
    );
    println!("  2sig consistent: {}", recovery.consistent_2sig);

    assert!(
        recovery.consistent_2sig,
        "BNS: recovered rate {:.2} not within 2sig of input {:.2} (interval: [{:.2}, {:.2}])",
        recovery.recovered_rate,
        recovery.input_rate,
        recovery.poisson_lower_2sig,
        recovery.poisson_upper_2sig,
    );
}

#[test]
fn test_nsbh_kilonova_rate_recovery() {
    let rate_nsbh = 25.0; // Gpc^-3 yr^-1 (LIGO NSBH)
    let pipeline = make_pipeline(rate_nsbh, 0.3, -15.0, 50_000);
    let result = pipeline.run();

    println!("{}", result);

    assert!(!result.rate_summaries.is_empty());
    let summary = &result.rate_summaries[0];
    let recovery = summary.recovery.as_ref().expect("recovery should be computed");

    println!("NSBH Rate Recovery:");
    println!("  Input rate:     {:.2} Gpc^-3 yr^-1", recovery.input_rate);
    println!("  Recovered rate: {:.2} Gpc^-3 yr^-1", recovery.recovered_rate);
    println!("  Effective VT:   {:.4} Gpc^3 yr", recovery.effective_vt);
    println!("  N_expected:     {:.1}", recovery.n_expected_detections);
    println!(
        "  2sig interval:  [{:.2}, {:.2}]",
        recovery.poisson_lower_2sig, recovery.poisson_upper_2sig
    );
    println!("  2sig consistent: {}", recovery.consistent_2sig);

    assert!(
        recovery.consistent_2sig,
        "NSBH: recovered rate {:.2} not within 2sig of input {:.2} (interval: [{:.2}, {:.2}])",
        recovery.recovered_rate,
        recovery.input_rate,
        recovery.poisson_lower_2sig,
        recovery.poisson_upper_2sig,
    );
}
