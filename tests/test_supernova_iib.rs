use rand::SeedableRng;

use survey_sim::population::generator::SupernovaIIbPopulation;
use survey_sim::population::PopulationGenerator;
use survey_sim::types::TransientType;

#[test]
fn test_supernova_iib_population_generation() {
    let pop = SupernovaIIbPopulation::new(1200.0, 0.15, -16.5, 60000.0, 61000.0);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let instances = pop.generate(500, &mut rng);
    assert_eq!(instances.len(), 500);

    // Verify transient type
    assert!(instances
        .iter()
        .all(|i| i.transient_type == TransientType::SupernovaIIb));

    // Redshifts within [0, 0.15]
    assert!(instances.iter().all(|i| i.z >= 0.0 && i.z <= 0.15));

    // Explosion times in range
    assert!(instances
        .iter()
        .all(|i| i.t_exp >= 60000.0 && i.t_exp <= 61000.0));

    // Luminosity distances positive
    assert!(instances.iter().all(|i| i.d_l > 0.0));

    // Verify physical shock-cooling parameters exist and are in reasonable physical bounds
    for inst in &instances {
        let menv = inst.model_params.get("log10_Menv").expect("log10_Menv must be present");
        assert!(*menv >= -2.0 && *menv <= -0.5, "Menv out of physical range: {}", menv);

        let renv = inst.model_params.get("log10_Renv").expect("log10_Renv must be present");
        assert!(*renv >= 1.4 && *renv <= 2.8, "Renv out of physical range: {}", renv);

        let ee = inst.model_params.get("log10_Ee").expect("log10_Ee must be present");
        assert!(*ee >= 50.5 && *ee <= 51.5, "Ee out of physical range: {}", ee);
    }

    assert_eq!(pop.volumetric_rate(), 1200.0);
    assert_eq!(pop.transient_type(), TransientType::SupernovaIIb);
}
