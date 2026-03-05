pub mod blastwave_model;
pub mod cosmology;
pub mod parametric;
pub mod python_model;

use std::collections::HashMap;

use thiserror::Error;

use crate::types::{Band, TransientInstance};

#[derive(Error, Debug)]
pub enum LightcurveError {
    #[error("Model evaluation failed: {0}")]
    EvaluationFailed(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Python callback error: {0}")]
    PythonError(String),
}

pub type Result<T> = std::result::Result<T, LightcurveError>;

/// Result of evaluating a lightcurve model at specific times and bands.
#[derive(Clone, Debug)]
pub struct LightcurveEvaluation {
    /// Apparent magnitudes per band. Key = band name, Value = magnitudes at each time.
    pub apparent_mags: HashMap<String, Vec<f64>>,
    /// Observation times (MJD) corresponding to the magnitude arrays.
    pub times_mjd: Vec<f64>,
}

/// Trait for lightcurve model evaluation.
pub trait LightcurveModel: Send + Sync {
    /// Evaluate the lightcurve model for a transient instance at given times and bands.
    ///
    /// Returns apparent magnitudes at the observer frame.
    fn evaluate(
        &self,
        instance: &TransientInstance,
        times_mjd: &[f64],
        bands: &[Band],
    ) -> Result<LightcurveEvaluation>;

    /// Whether this model requires Python GIL (i.e., cannot be parallelized with rayon).
    fn requires_gil(&self) -> bool {
        false
    }

    /// Whether this model supports batch evaluation (e.g., GPU-vectorized via JAX vmap).
    fn supports_batch(&self) -> bool {
        false
    }

    /// Evaluate multiple transients in a single batch call.
    ///
    /// Default implementation falls back to sequential `evaluate()` calls.
    /// Models that support GPU batching (e.g., fiestaEM with JAX vmap) should
    /// override this for dramatically better throughput.
    fn batch_evaluate(
        &self,
        instances: &[&TransientInstance],
        times_mjd: &[&[f64]],
        bands: &[&[Band]],
    ) -> Vec<Result<LightcurveEvaluation>> {
        instances
            .iter()
            .zip(times_mjd.iter())
            .zip(bands.iter())
            .map(|((inst, t), b)| self.evaluate(inst, t, b))
            .collect()
    }
}
