use std::collections::HashSet;

use survey_sim::survey::rubin::RubinLoader;
use survey_sim::survey::ztf::ZtfHdf5Loader;
use survey_sim::survey::{SurveyLoader, SurveyStore};

/// Test loading the real Rubin OpSim database.
/// This test requires the file to be present on disk.
#[test]
#[ignore] // Run with --ignored flag when the .db file is available.
fn test_load_rubin_opsim() {
    let db_path = "/fred/oz480/mcoughli/simulations/TESS-Rubin/baseline_v5.1.1_10yrs.db";

    let loader = RubinLoader::new(db_path);
    let bands = loader.bands();
    assert_eq!(bands.len(), 6);

    let observations = loader.load().expect("Failed to load OpSim database");

    // The baseline v5.1.1 10yr simulation should have ~2M observations.
    println!("Loaded {} observations", observations.len());
    assert!(
        observations.len() > 1_000_000,
        "Expected >1M observations, got {}",
        observations.len()
    );

    // Build the survey store.
    let store = SurveyStore::new(observations, 64);
    println!(
        "Survey: {} observations, MJD {:.1} to {:.1}, {:.1} years, {} bands",
        store.len(),
        store.mjd_min,
        store.mjd_max,
        store.duration_years,
        store.bands.len()
    );

    // Verify we have 6 bands.
    assert!(store.bands.len() >= 6, "Expected >= 6 bands");

    // Verify MJD range is reasonable (10 years ~ 3652 days).
    let duration_days = store.mjd_max - store.mjd_min;
    assert!(
        duration_days > 3000.0 && duration_days < 4000.0,
        "Expected ~10yr duration, got {:.0} days",
        duration_days
    );
}

/// Test loading ZTF 2018 observations from IRSA HDF5 file.
/// Requires the HDF5 file to be present on disk.
#[test]
#[ignore]
fn test_ztf_2018_hdf5() {
    let loader = ZtfHdf5Loader::new(&["/fred/oz480/mcoughli/simulations/ztf_data/ztf_2018.h5"]);
    let obs = loader.load().unwrap();

    // 2018 has ~146K unique exposures.
    assert!(
        obs.len() > 100_000,
        "Expected >100K exposures, got {}",
        obs.len()
    );

    // Check bands.
    let bands: HashSet<_> = obs.iter().map(|o| o.band.0.clone()).collect();
    assert!(bands.contains("g") && bands.contains("r"));

    // MJD sanity (2018 = MJD ~58178–58483).
    assert!(obs.iter().all(|o| o.mjd > 58000.0 && o.mjd < 59000.0));

    println!("ZTF 2018: {} exposures, bands: {:?}", obs.len(), bands);
}
