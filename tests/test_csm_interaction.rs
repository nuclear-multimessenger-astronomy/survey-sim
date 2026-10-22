use rand::SeedableRng;

use survey_sim::population::generator::CsmInteractionPopulation;
use survey_sim::population::PopulationGenerator;
use survey_sim::types::TransientType;

#[test]
fn test_csm_interaction_population_generation() {
    let pop = CsmInteractionPopulation::new(250.0, 0.25, -17.5, 60000.0, 61000.0);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let instances = pop.generate(500, &mut rng);
    assert_eq!(instances.len(), 500);

    // Verify transient type
    assert!(instances
        .iter()
        .all(|i| i.transient_type == TransientType::CsmInteraction));

    // Redshifts within [0, 0.25]
    assert!(instances.iter().all(|i| i.z >= 0.0 && i.z <= 0.25));

    // Explosion times in range
    assert!(instances
        .iter()
        .all(|i| i.t_exp >= 60000.0 && i.t_exp <= 61000.0));

    // Luminosity distances positive
    assert!(instances.iter().all(|i| i.d_l > 0.0));

    // Verify physical CSM parameters exist and are in reasonable physical bounds
    for inst in &instances {
        let mej = inst.model_params.get("log10_mej").expect("log10_mej must be present");
        assert!(*mej >= -0.3 && *mej <= 0.7, "log10_mej out of range: {}", mej);

        let mcsm = inst.model_params.get("log10_csm_mass").expect("log10_csm_mass must be present");
        assert!(*mcsm >= -1.5 && *mcsm <= 0.2, "log10_csm_mass out of range: {}", mcsm);

        let vej = inst.model_params.get("log10_vej").expect("log10_vej must be present");
        assert!(*vej >= 3.7 && *vej <= 4.2, "log10_vej out of range: {}", vej);

        let eta = inst.model_params.get("eta").expect("eta must be present");
        assert!(*eta >= 0.5 && *eta <= 2.5, "eta out of range: {}", eta);

        let rho = inst.model_params.get("log10_rho").expect("log10_rho must be present");
        assert!(*rho >= -14.5 && *rho <= -11.5, "log10_rho out of range: {}", rho);

        let r0 = inst.model_params.get("log10_r0").expect("log10_r0 must be present");
        assert!(*r0 >= 0.5 && *r0 <= 1.8, "log10_r0 out of range: {}", r0);

        let kappa = inst.model_params.get("log10_kappa").expect("log10_kappa must be present");
        assert!(*kappa >= -0.8 && *kappa <= -0.2, "log10_kappa out of range: {}", kappa);

        // Clamped peak absolute magnitude
        assert!(inst.peak_abs_mag >= -20.0 && inst.peak_abs_mag <= -15.0);
    }

    assert_eq!(pop.volumetric_rate(), 250.0);
    assert_eq!(pop.transient_type(), TransientType::CsmInteraction);
}
