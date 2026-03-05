use survey_sim::cadence::ReturnTimeAnalysis;
use survey_sim::instrument::InstrumentConfig;
use survey_sim::survey::rubin::RubinLoader;
use survey_sim::survey::{SurveyLoader, SurveyStore};

/// Load Rubin OpSim Year 1 (first 365 days of baseline_v5.1.1).
fn load_rubin_year1() -> SurveyStore {
    let db_path = "/fred/oz480/mcoughli/simulations/TESS-Rubin/baseline_v5.1.1_10yrs.db";
    let loader = RubinLoader::new(db_path);
    let mut observations = loader.load().expect("Failed to load OpSim database");

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

    SurveyStore::new(observations, 64).with_instrument(InstrumentConfig::rubin())
}

#[test]
#[ignore] // Requires OpSim database on disk
fn test_rubin_year1_cadence() {
    let survey = load_rubin_year1();
    let analysis = ReturnTimeAnalysis::analyze(&survey, 1000);
    println!("{}", analysis);

    // Sanity checks per band.
    // Median can be very short (~minutes) due to overlapping field revisits within a night.
    // The 75th percentile better reflects inter-night cadence (~3-7 days for Rubin).
    for stats in &analysis.band_stats {
        assert!(
            stats.n_gaps > 0,
            "{}: expected some gaps",
            stats.band,
        );
        assert!(
            stats.percentiles[3] > 1.0,  // 75th percentile > 1 day
            "{}: 75th percentile too short ({:.2} d)",
            stats.band,
            stats.percentiles[3],
        );
        assert!(
            stats.percentiles[3] < 30.0, // 75th percentile < 30 days
            "{}: 75th percentile too long ({:.2} d)",
            stats.band,
            stats.percentiles[3],
        );
    }

    // Any-filter revisit should be faster than single-band.
    assert!(
        analysis.all_bands_stats.median_days < 5.0,
        "any-filter median should be < 5 days, got {:.2}",
        analysis.all_bands_stats.median_days,
    );
}
