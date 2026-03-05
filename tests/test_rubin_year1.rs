use survey_sim::detection::DetectionCriteria;
use survey_sim::instrument::InstrumentConfig;
use survey_sim::lightcurve::parametric::ParametricModel;
use survey_sim::pipeline::SimulationPipeline;
use survey_sim::population::generator::KilonovaPopulation;
use survey_sim::survey::rubin::RubinLoader;
use survey_sim::survey::{SurveyLoader, SurveyStore};

/// Load Rubin OpSim Year 1 (first 365 days of baseline_v5.1.1).
fn load_rubin_year1() -> SurveyStore {
    let db_path = "/fred/oz480/mcoughli/simulations/TESS-Rubin/baseline_v5.1.1_10yrs.db";
    let loader = RubinLoader::new(db_path);
    let mut observations = loader.load().expect("Failed to load OpSim database");

    // Filter to Year 1: first 365.25 days.
    let mjd_min = observations
        .iter()
        .map(|o| o.mjd)
        .fold(f64::INFINITY, f64::min);
    let mjd_year1_max = mjd_min + 365.25;
    observations.retain(|obs| obs.mjd <= mjd_year1_max);

    println!(
        "Rubin Year 1: {} observations, MJD {:.1} to {:.1}",
        observations.len(),
        mjd_min,
        mjd_year1_max
    );

    // Use nside=64 with MOC cone queries — FoV radius set from instrument config.
    SurveyStore::new(observations, 64).with_instrument(InstrumentConfig::rubin())
}

/// BNS kilonova rate recovery against Rubin Year 1 with fast transient criterion.
///
/// Injects KN at LIGO BNS rate (45 Gpc^-3 yr^-1), requires fast transient
/// (rise > 1 mag/day or fade > 0.3 mag/day), and recovers the rate.
#[test]
#[ignore] // Requires OpSim database on disk.
fn test_rubin_year1_bns_rate_recovery() {
    let survey = load_rubin_year1();
    let mjd_min = survey.mjd_min;
    let mjd_max = survey.mjd_max;
    println!(
        "Survey: {} obs, {:.2} yr, {} bands",
        survey.len(),
        survey.duration_years,
        survey.bands.len(),
    );

    let rate_bns = 45.0; // Gpc^-3 yr^-1 (LIGO O4 BNS median)

    let mut pipeline = SimulationPipeline::new(
        survey,
        DetectionCriteria {
            min_detections: 2,
            min_bands: 1,
            min_per_band: 2,
            max_timespan_days: 30.0,
            snr_threshold: 5.0,
            require_fast_transient: true,
            min_rise_rate: 1.0,   // > 1 mag/day brightening
            min_fade_rate: 0.3,   // > 0.3 mag/day fading
            ..Default::default()
        },
        50_000,
        42,
    );

    let kn_pop = KilonovaPopulation::new(rate_bns, 0.3, -16.0, mjd_min, mjd_max);
    pipeline.add_population(Box::new(kn_pop));
    pipeline.add_model(
        "Kilonova",
        Box::new(
            ParametricModel::new().with_model(lightcurve_fitting::SviModelName::MetzgerKN),
        ),
    );

    let result = pipeline.run();
    println!("{}", result);

    assert!(!result.rate_summaries.is_empty());
    let summary = &result.rate_summaries[0];
    let recovery = summary
        .recovery
        .as_ref()
        .expect("recovery should be computed");

    println!("=== Rubin Year 1 — BNS KN Rate Recovery (fast transient) ===");
    println!("  Input rate:     {:.2} Gpc^-3 yr^-1", recovery.input_rate);
    println!(
        "  Recovered rate: {:.2} Gpc^-3 yr^-1",
        recovery.recovered_rate
    );
    println!("  Effective VT:   {:.6} Gpc^3 yr", recovery.effective_vt);
    println!("  N_expected:     {:.1}", recovery.n_expected_detections);
    println!(
        "  2sig interval:  [{:.2}, {:.2}]",
        recovery.poisson_lower_2sig, recovery.poisson_upper_2sig
    );
    println!("  1sig consistent: {}", recovery.consistent_1sig);
    println!("  2sig consistent: {}", recovery.consistent_2sig);
    println!("  Overall eff:    {:.4}", summary.overall_efficiency);
    println!(
        "  Detected:       {} / {} ({:.2}%)",
        result.n_detected,
        result.n_simulated,
        100.0 * result.n_detected as f64 / result.n_simulated.max(1) as f64,
    );

    assert!(
        recovery.consistent_2sig,
        "BNS rate recovery failed: recovered {:.2} not within 2sig of input {:.2}",
        recovery.recovered_rate,
        recovery.input_rate,
    );
}

/// NSBH kilonova rate recovery against Rubin Year 1 with fast transient criterion.
///
/// Dimmer peak (M=-15) and lower rate (25 Gpc^-3 yr^-1).
#[test]
#[ignore] // Requires OpSim database on disk.
fn test_rubin_year1_nsbh_rate_recovery() {
    let survey = load_rubin_year1();
    let mjd_min = survey.mjd_min;
    let mjd_max = survey.mjd_max;
    println!(
        "Survey: {} obs, {:.2} yr, {} bands",
        survey.len(),
        survey.duration_years,
        survey.bands.len(),
    );

    let rate_nsbh = 25.0; // Gpc^-3 yr^-1 (LIGO NSBH)

    let mut pipeline = SimulationPipeline::new(
        survey,
        DetectionCriteria {
            min_detections: 2,
            min_bands: 1,
            min_per_band: 2,
            max_timespan_days: 30.0,
            snr_threshold: 5.0,
            require_fast_transient: true,
            min_rise_rate: 1.0,
            min_fade_rate: 0.3,
            ..Default::default()
        },
        50_000,
        42,
    );

    let kn_pop = KilonovaPopulation::new(rate_nsbh, 0.3, -15.0, mjd_min, mjd_max);
    pipeline.add_population(Box::new(kn_pop));
    pipeline.add_model(
        "Kilonova",
        Box::new(
            ParametricModel::new().with_model(lightcurve_fitting::SviModelName::MetzgerKN),
        ),
    );

    let result = pipeline.run();
    println!("{}", result);

    assert!(!result.rate_summaries.is_empty());
    let summary = &result.rate_summaries[0];
    let recovery = summary
        .recovery
        .as_ref()
        .expect("recovery should be computed");

    println!("=== Rubin Year 1 — NSBH KN Rate Recovery (fast transient) ===");
    println!("  Input rate:     {:.2} Gpc^-3 yr^-1", recovery.input_rate);
    println!(
        "  Recovered rate: {:.2} Gpc^-3 yr^-1",
        recovery.recovered_rate
    );
    println!("  Effective VT:   {:.6} Gpc^3 yr", recovery.effective_vt);
    println!("  N_expected:     {:.1}", recovery.n_expected_detections);
    println!(
        "  2sig interval:  [{:.2}, {:.2}]",
        recovery.poisson_lower_2sig, recovery.poisson_upper_2sig
    );
    println!("  1sig consistent: {}", recovery.consistent_1sig);
    println!("  2sig consistent: {}", recovery.consistent_2sig);
    println!("  Overall eff:    {:.4}", summary.overall_efficiency);
    println!(
        "  Detected:       {} / {} ({:.2}%)",
        result.n_detected,
        result.n_simulated,
        100.0 * result.n_detected as f64 / result.n_simulated.max(1) as f64,
    );

    assert!(
        recovery.consistent_2sig,
        "NSBH rate recovery failed: recovered {:.2} not within 2sig of input {:.2}",
        recovery.recovered_rate,
        recovery.input_rate,
    );
}
