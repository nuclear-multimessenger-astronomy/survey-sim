use survey_sim::instrument::{InstrumentConfig, Location};

#[test]
fn test_rubin_builtin_loads() {
    let rubin = InstrumentConfig::rubin();
    assert_eq!(rubin.name, "Rubin LSST");
    assert_eq!(rubin.bands.len(), 6);
    assert!(rubin.bands.contains_key("u"));
    assert!(rubin.bands.contains_key("g"));
    assert!(rubin.bands.contains_key("r"));
    assert!(rubin.bands.contains_key("i"));
    assert!(rubin.bands.contains_key("z"));
    assert!(rubin.bands.contains_key("y"));
}

#[test]
fn test_ztf_builtin_loads() {
    let ztf = InstrumentConfig::ztf();
    assert_eq!(ztf.name, "ZTF");
    assert_eq!(ztf.bands.len(), 3);
    assert!(ztf.bands.contains_key("g"));
    assert!(ztf.bands.contains_key("r"));
    assert!(ztf.bands.contains_key("i"));
}

#[test]
fn test_rubin_yaml_loads() {
    let rubin = InstrumentConfig::from_yaml("instruments/rubin.yaml").unwrap();
    assert_eq!(rubin.name, "Rubin LSST");
    assert_eq!(rubin.bands.len(), 6);
    assert!((rubin.telescope.aperture_m - 6.423).abs() < 1e-3);
    assert!((rubin.detector.fov_deg2 - 9.6).abs() < 1e-3);

    match &rubin.telescope.location {
        Location::Ground {
            latitude_deg,
            longitude_deg,
            altitude_m,
        } => {
            assert!((*latitude_deg - (-30.2446)).abs() < 1e-4);
            assert!((*longitude_deg - (-70.7494)).abs() < 1e-4);
            assert!((*altitude_m - 2663.0).abs() < 1.0);
        }
        Location::Space => panic!("Rubin should be ground-based"),
    }
}

#[test]
fn test_ztf_yaml_loads() {
    let ztf = InstrumentConfig::from_yaml("instruments/ztf.yaml").unwrap();
    assert_eq!(ztf.name, "ZTF");
    assert_eq!(ztf.bands.len(), 3);
    assert!((ztf.detector.fov_deg2 - 47.0).abs() < 1e-3);
}

#[test]
fn test_argus_builtin_loads() {
    let argus = InstrumentConfig::argus();
    assert_eq!(argus.name, "Argus Array");
    assert_eq!(argus.bands.len(), 2);
    assert!(argus.bands.contains_key("b"));
    assert!(argus.bands.contains_key("r"));
    assert!((argus.detector.fov_deg2 - 8000.0).abs() < 1e-3);
    assert!((argus.telescope.aperture_m - 8.0).abs() < 1e-3);
}

#[test]
fn test_argus_yaml_loads() {
    let argus = InstrumentConfig::from_yaml("instruments/argus.yaml").unwrap();
    assert_eq!(argus.name, "Argus Array");
    assert_eq!(argus.bands.len(), 2);
    assert!((argus.detector.fov_deg2 - 8000.0).abs() < 1e-3);
    assert!((argus.observing.default_exposure_s - 60.0).abs() < 1e-3);
    assert!((argus.observing.readout_s - 0.001).abs() < 1e-6);

    match &argus.telescope.location {
        Location::Ground {
            latitude_deg,
            longitude_deg,
            ..
        } => {
            assert!((*latitude_deg - 35.9132).abs() < 1e-4);
            assert!((*longitude_deg - (-79.0558)).abs() < 1e-4);
        }
        Location::Space => panic!("Argus should be ground-based"),
    }
}

#[test]
fn test_yaml_roundtrip() {
    let original = InstrumentConfig::rubin();
    let yaml = serde_yaml::to_string(&original).unwrap();
    let parsed = InstrumentConfig::from_yaml_str(&yaml).unwrap();

    assert_eq!(parsed.name, original.name);
    assert_eq!(parsed.bands.len(), original.bands.len());
    assert!((parsed.telescope.aperture_m - original.telescope.aperture_m).abs() < 1e-10);
    assert!((parsed.detector.fov_deg2 - original.detector.fov_deg2).abs() < 1e-10);
}

#[test]
fn test_extinction_from_instrument() {
    let rubin = InstrumentConfig::rubin();
    assert!((rubin.extinction_ratio("u") - 1.56).abs() < 1e-10);
    assert!((rubin.extinction_ratio("g") - 1.31).abs() < 1e-10);
    assert!((rubin.extinction_ratio("r") - 1.0).abs() < 1e-10);
    assert!((rubin.extinction_ratio("y") - 0.47).abs() < 1e-10);
    // Fallback for band not in config
    assert!((rubin.extinction_ratio("J") - 0.29).abs() < 1e-10);
}

#[test]
fn test_extinction_with_instrument() {
    use survey_sim::lightcurve::cosmology::extinction_in_band_with_instrument;

    let rubin = InstrumentConfig::rubin();
    let a_v = 1.0;

    // With instrument
    let a_u = extinction_in_band_with_instrument(a_v, "u", Some(&rubin));
    assert!((a_u - 1.56).abs() < 1e-10);

    // Without instrument (None) — same fallback
    let a_u_none = extinction_in_band_with_instrument(a_v, "u", None);
    assert!((a_u_none - 1.56).abs() < 1e-10);
}

#[test]
fn test_survey_store_with_instrument() {
    use survey_sim::survey::SurveyStore;

    let store = SurveyStore::new(vec![], 64);
    assert!(store.instrument.is_none());

    let store = store.with_instrument(InstrumentConfig::rubin());
    assert!(store.instrument.is_some());
    assert_eq!(store.instrument.as_ref().unwrap().name, "Rubin LSST");
}

#[test]
fn test_loader_returns_instrument() {
    use survey_sim::survey::SurveyLoader;
    use survey_sim::survey::rubin::RubinLoader;
    use survey_sim::survey::ztf::ZtfLoader;

    let rubin = RubinLoader::new("/nonexistent/path.db");
    let inst = rubin.instrument();
    assert!(inst.is_some());
    assert_eq!(inst.unwrap().name, "Rubin LSST");

    let ztf = ZtfLoader::new("/nonexistent/path.csv");
    let inst = ztf.instrument();
    assert!(inst.is_some());
    assert_eq!(inst.unwrap().name, "ZTF");
}
