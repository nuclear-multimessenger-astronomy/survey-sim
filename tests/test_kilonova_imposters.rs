use survey_sim::efficiency::imposters::{
    analyze_simulation_imposters, ImposterPopulation, KilonovaImposterAnalysis, KilonovaSignal,
};
use survey_sim::detection::DetectionCriteria;
use survey_sim::lightcurve::parametric::ParametricModel;
use survey_sim::pipeline::SimulationPipeline;
use survey_sim::population::generator::{FbotPopulation, KilonovaPopulation};
use survey_sim::survey::{SurveyObservation, SurveyStore};
use survey_sim::types::{Band, Cosmology, SkyCoord};

fn make_synthetic_survey() -> SurveyStore {
    let mut observations = Vec::new();
    let mut obs_id = 0u64;

    for night in 0..180 {
        let mjd = 60000.0 + night as f64 * 0.5; // twice-nightly / 12-hr cadence
        for ra_block in 0..6 {
            let ra = ra_block as f64 * 60.0;
            for dec_block in 0..3 {
                let dec = -30.0 + dec_block as f64 * 30.0;
                for band in &["g", "r"] {
                    observations.push(SurveyObservation {
                        obs_id,
                        coord: SkyCoord::new(ra, dec),
                        mjd,
                        band: Band::new(band),
                        five_sigma_depth: 21.5,
                        seeing_fwhm: 1.5,
                        exposure_time: 30.0,
                        airmass: 1.2,
                        sky_brightness: 21.0,
                        night: (night / 2) as i64,
                    });
                    obs_id += 1;
                }
            }
        }
    }

    SurveyStore::new(observations, 4)
}

#[test]
fn test_analytical_imposter_analysis_bns_and_shock_cooling() {
    let cosmo = Cosmology::default();
    let duration_yr = 3.0;
    let omega_sr = 2.0 * std::f64::consts::PI; // half-sky survey

    // Kilonova signal: 45 Gpc^-3 yr^-1 (LVK BNS rate)
    let kn_eff_vs_z: Vec<(f64, f64)> = (0..=10)
        .map(|i| {
            let z = i as f64 * 0.02;
            let eff = (1.0 - z / 0.15).clamp(0.0, 1.0);
            (z, eff * 0.50)
        })
        .collect();

    let kn_signal = KilonovaSignal::from_efficiency(
        "BNS Kilonova",
        45.0,
        &kn_eff_vs_z,
        duration_yr,
        omega_sr,
        &cosmo,
    );

    // Imposter 1: Type IIb supernova shock cooling
    let iib_imposter = ImposterPopulation::type_iib_shock_cooling(duration_yr, omega_sr, &cosmo);

    // Imposter 2: Fast Blue Optical Transients (FBOTs)
    let fbot_imposter = ImposterPopulation::fbot(duration_yr, omega_sr, &cosmo);

    let analysis = KilonovaImposterAnalysis::new(
        kn_signal,
        vec![iib_imposter, fbot_imposter],
    ).with_followup_budget(20.0);

    println!("{}", analysis);

    // Verify properties
    assert!(analysis.metrics.signal_detections > 0.0);
    assert!(analysis.metrics.total_imposters > 0.0);
    assert!(analysis.metrics.purity > 0.0 && analysis.metrics.purity < 1.0);
    assert!(analysis.metrics.false_discovery_rate > 0.0 && analysis.metrics.false_discovery_rate < 1.0);
    assert!((analysis.metrics.purity + analysis.metrics.false_discovery_rate - 1.0).abs() < 1e-10);

    // Effective pure efficiency must be strictly less than raw efficiency
    assert!(analysis.metrics.pure_efficiency < analysis.kilonova.overall_efficiency);

    // Naive rate must exceed true rate due to imposter contamination
    assert!(analysis.rate_recovery.naive_recovered_rate > analysis.kilonova.volumetric_rate);
    assert!(analysis.rate_recovery.rate_bias_absolute > 0.0);

    // Decontaminated rate should recover the true rate
    assert!(
        (analysis.rate_recovery.decontaminated_recovered_rate - analysis.kilonova.volumetric_rate).abs() < 1e-6
    );

    // Variance inflation factor must be > 1.0 (imposter background adds variance)
    assert!(analysis.rate_recovery.variance_inflation_factor > 1.0);
    assert!(analysis.rate_recovery.uncertainty_inflation_factor > 1.0);
    assert!(analysis.rate_recovery.consistent_2sig);

    // Follow-up budget checks
    let fb = analysis.followup_budget.as_ref().unwrap();
    assert_eq!(fb.budget, 20.0);
    assert!(fb.expected_confirmed_kn <= analysis.metrics.signal_detections);
}

#[test]
fn test_pipeline_simulation_with_kilonova_and_imposters() {
    let survey = make_synthetic_survey();
    let criteria = DetectionCriteria {
        min_detections: 2,
        min_bands: 1,
        min_per_band: 1,
        max_timespan_days: 14.0,
        snr_threshold: 5.0,
        ..Default::default()
    };

    let n_transients = 1000;
    let mut pipeline = SimulationPipeline::new(survey, criteria, n_transients, 12345);

    // Add Kilonova population
    let kn_pop = KilonovaPopulation::new(500.0, 0.15, -16.0, 60000.0, 60080.0);
    pipeline.add_population(Box::new(kn_pop));
    pipeline.add_model(
        "Kilonova",
        Box::new(ParametricModel::new().with_model(lightcurve_fitting::SviModelName::MetzgerKN)),
    );

    // Add FBOT imposter population
    let fbot_pop = FbotPopulation::new(2000.0, 0.20, -17.0, 60000.0, 60080.0);
    pipeline.add_population(Box::new(fbot_pop));
    pipeline.add_model(
        "FBOT",
        Box::new(ParametricModel::new().with_model(lightcurve_fitting::SviModelName::Bazin)),
    );

    let result = pipeline.run();
    println!("{}", result);

    assert_eq!(result.rate_summaries.len(), 2);

    // Run imposter analysis across simulated populations
    let analysis = analyze_simulation_imposters(&result, "Kilonova", &["FBOT"], 0.25)
        .expect("Should find Kilonova and FBOT summaries");

    println!("{}", analysis);

    assert_eq!(analysis.kilonova.name, "Kilonova");
    assert_eq!(analysis.imposters.len(), 1);
    assert_eq!(analysis.imposters[0].name, "FBOT");

    assert!(analysis.metrics.total_detections >= analysis.metrics.signal_detections);
    assert!(analysis.rate_recovery.consistent_2sig);

    // Also test direct from_simulation_result and from_rate_summaries
    let direct_analysis = KilonovaImposterAnalysis::from_simulation_result(&result, "Kilonova", 0.25)
        .expect("Should construct from simulation result");
    assert_eq!(direct_analysis.kilonova.name, "Kilonova");
    assert_eq!(direct_analysis.imposters.len(), 1);
    assert!((direct_analysis.metrics.purity - analysis.metrics.purity).abs() < 1e-10);

    let summary_analysis = KilonovaImposterAnalysis::from_rate_summaries(&result.rate_summaries, "Kilonova", 0.25)
        .expect("Should construct from rate summaries");
    assert!((summary_analysis.metrics.f1_score - analysis.metrics.f1_score).abs() < 1e-10);
}

#[test]
fn test_all_standard_imposter_presets() {
    let cosmo = Cosmology::default();
    let duration_yr = 3.0;
    let omega_sr = 2.0 * std::f64::consts::PI;

    let p1 = ImposterPopulation::type_iib_shock_cooling(duration_yr, omega_sr, &cosmo);
    let p2 = ImposterPopulation::fbot(duration_yr, omega_sr, &cosmo);
    let p3 = ImposterPopulation::shocked_cocoon(duration_yr, omega_sr, &cosmo);
    let p4 = ImposterPopulation::sn_iax(duration_yr, omega_sr, &cosmo);
    let p5 = ImposterPopulation::csm_interaction(duration_yr, omega_sr, &cosmo);
    let p6 = ImposterPopulation::orphan_afterglow(duration_yr, omega_sr, &cosmo);

    assert_eq!(p1.name, "Type IIb SN (Shock Cooling)");
    assert_eq!(p2.name, "Fast Blue Optical Transient (FBOT)");
    assert_eq!(p3.name, "Relativistic Shocked Cocoon");
    assert_eq!(p4.name, "Type Iax Supernova");
    assert_eq!(p5.name, "CSM Interaction Transient");
    assert_eq!(p6.name, "Orphan GRB Afterglow");

    let all_imposters = vec![p1, p2, p3, p4, p5, p6];
    for imp in &all_imposters {
        println!("{}: rate={}, vt={:?}, expected={}", imp.name, imp.rate, imp.effective_vt, imp.expected_detections);
        assert!(imp.expected_detections > 0.0);
        assert!(imp.effective_vt.unwrap_or(0.0) > 0.0);
        assert!(imp.rate > 0.0);
        assert!(imp.rate_uncertainty_frac > 0.0);
    }

    let kn_signal = KilonovaSignal::new("BNS Kilonova", 45.0, 18.5, 0.411, 0.0035);
    let analysis = KilonovaImposterAnalysis::new(kn_signal, all_imposters)
        .with_followup_budget(25.0);

    assert_eq!(analysis.imposters.len(), 6);
    assert!(analysis.metrics.total_imposters > 25.0);
    assert!(analysis.metrics.purity > 0.15 && analysis.metrics.purity < 0.50);
    assert!(analysis.rate_recovery.variance_inflation_factor > 1.0);
    assert!(analysis.rate_recovery.consistent_2sig);
}
