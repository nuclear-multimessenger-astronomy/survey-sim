use rand::SeedableRng;
use survey_sim::population::generator::ShockedCocoonPopulation;
use survey_sim::population::PopulationGenerator;
use survey_sim::types::TransientType;

#[test]
fn test_shocked_cocoon_population_generation() {
    let pop = ShockedCocoonPopulation::new(30.0, 0.30, -16.0, 60000.0, 61000.0);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let instances = pop.generate(300, &mut rng);
    assert_eq!(instances.len(), 300);

    assert!(instances
        .iter()
        .all(|i| i.transient_type == TransientType::ShockedCocoon));

    assert!(instances.iter().all(|i| i.z >= 0.0 && i.z <= 0.30));
    assert!(instances.iter().all(|i| i.d_l > 0.0));

    for inst in &instances {
        let mej = inst.model_params.get("log10_mej").expect("log10_mej");
        assert!(*mej >= -3.5 && *mej <= -0.5);

        let vej = inst.model_params.get("log10_vej").expect("log10_vej");
        assert!(*vej >= -1.1 && *vej <= -0.3);

        let eta = inst.model_params.get("eta").expect("eta");
        assert!(*eta >= 1.0 && *eta <= 3.0);
    }

    assert_eq!(pop.volumetric_rate(), 30.0);
    assert_eq!(pop.transient_type(), TransientType::ShockedCocoon);
}
