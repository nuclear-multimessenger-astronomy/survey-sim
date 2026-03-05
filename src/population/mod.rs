pub mod distributions;
pub mod generator;
pub mod grb;

use crate::types::{TransientInstance, TransientType};

/// Trait for generating transient populations.
pub trait PopulationGenerator: Send + Sync {
    /// Generate n transient instances using the provided RNG.
    fn generate(&self, n: usize, rng: &mut dyn rand::RngCore) -> Vec<TransientInstance>;

    /// Volumetric rate in Gpc^-3 yr^-1.
    fn volumetric_rate(&self) -> f64;

    /// Transient type for this population.
    fn transient_type(&self) -> TransientType;
}

/// A named population with its generator.
pub struct TransientPopulation {
    pub name: String,
    pub generator: Box<dyn PopulationGenerator>,
}

impl TransientPopulation {
    pub fn new(name: &str, generator: Box<dyn PopulationGenerator>) -> Self {
        Self {
            name: name.to_string(),
            generator,
        }
    }
}
