use rand::SeedableRng;

use survey_sim::population::generator::KilonovaPopulation;
use survey_sim::population::PopulationGenerator;
use survey_sim::types::TransientType;

#[test]
fn test_kilonova_population_basic() {
    let pop = KilonovaPopulation::new(1000.0, 0.3, -16.0, 60000.0, 63652.0);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let instances = pop.generate(1000, &mut rng);
    assert_eq!(instances.len(), 1000);

    // All instances should have correct type.
    assert!(instances
        .iter()
        .all(|i| i.transient_type == TransientType::Kilonova));

    // All redshifts should be in [0, 0.3].
    assert!(instances.iter().all(|i| i.z >= 0.0 && i.z <= 0.3));

    // All explosion times in [60000, 63652].
    assert!(instances
        .iter()
        .all(|i| i.t_exp >= 60000.0 && i.t_exp <= 63652.0));

    // Luminosity distances should be positive.
    assert!(instances.iter().all(|i| i.d_l > 0.0));
}

#[test]
fn test_redshift_distribution_skewed_high() {
    // dV/dz increases with z, so mean redshift should be > z_max/2.
    let pop = KilonovaPopulation::new(1000.0, 0.3, -16.0, 60000.0, 63652.0);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let instances = pop.generate(10_000, &mut rng);
    let mean_z: f64 = instances.iter().map(|i| i.z).sum::<f64>() / instances.len() as f64;
    assert!(
        mean_z > 0.15,
        "Mean z should be > 0.15 (volumetric weighting), got {:.3}",
        mean_z
    );
}

#[test]
fn test_volumetric_rate() {
    let pop = KilonovaPopulation::new(1000.0, 0.3, -16.0, 60000.0, 63652.0);
    assert!((pop.volumetric_rate() - 1000.0).abs() < 1e-10);
}
