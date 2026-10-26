use rand::SeedableRng;

use survey_sim::population::generator::SupernovaIaxPopulation;
use survey_sim::population::PopulationGenerator;
use survey_sim::types::TransientType;

#[test]
fn test_supernova_iax_population_generation() {
    let pop = SupernovaIaxPopulation::new(3000.0, 0.20, -15.0, 60000.0, 61000.0);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let instances = pop.generate(500, &mut rng);
    assert_eq!(instances.len(), 500);

    // Verify transient type
    assert!(instances
        .iter()
        .all(|i| i.transient_type == TransientType::SupernovaIax));

    // Redshifts within [0, 0.20]
    assert!(instances.iter().all(|i| i.z >= 0.0 && i.z <= 0.20));

    // Explosion times in range
    assert!(instances
        .iter()
        .all(|i| i.t_exp >= 60000.0 && i.t_exp <= 61000.0));

    // Luminosity distances positive
    assert!(instances.iter().all(|i| i.d_l > 0.0));

    // Verify physical Iax parameters exist and are in expected physical bounds
    for inst in &instances {
        let m_ni = inst.model_params.get("m_ni").expect("m_ni must be present");
        assert!(*m_ni >= 0.005 && *m_ni <= 0.08, "m_ni out of physical range: {}", m_ni);

        let v_ej = inst.model_params.get("v_ej").expect("v_ej must be present");
        assert!(*v_ej >= 2000.0 && *v_ej <= 6000.0, "v_ej out of physical range: {}", v_ej);

        let m_ej = inst.model_params.get("m_ej").expect("m_ej must be present");
        assert!(*m_ej >= 0.10 && *m_ej <= 0.50, "m_ej out of physical range: {}", m_ej);

        let t_rise = inst.model_params.get("t_rise").expect("t_rise must be present");
        assert!(*t_rise >= 6.0 && *t_rise <= 14.0, "t_rise out of physical range: {}", t_rise);

        let t_fall = inst.model_params.get("t_fall").expect("t_fall must be present");
        assert!(*t_fall >= 12.0 && *t_fall <= 25.0, "t_fall out of physical range: {}", t_fall);

        // Peak absolute magnitude within clamped bounds [-18.0, -13.0]
        assert!(inst.peak_abs_mag >= -18.0 && inst.peak_abs_mag <= -13.0);
    }

    assert_eq!(pop.volumetric_rate(), 3000.0);
    assert_eq!(pop.transient_type(), TransientType::SupernovaIax);
}
