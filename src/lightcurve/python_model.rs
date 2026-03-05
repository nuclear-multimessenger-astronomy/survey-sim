/// Python callback wrapper for lightcurve models (e.g., fiestaEM).
///
/// This module is only functional when compiled with PyO3 support (the python crate).
/// The core library provides the trait and a stub; the actual PyO3 implementation
/// lives in `python/src/py_lightcurve.rs`.
use std::collections::HashMap;

use crate::types::{Band, TransientInstance};

use super::{LightcurveEvaluation, LightcurveModel, Result};

/// A lightcurve model backed by a Python callable.
///
/// The Python object must implement a `predict(params: dict) -> (times, {band: mags})` method.
/// This is designed to wrap fiestaEM SurrogateModel.predict().
///
/// Because it requires the GIL, it cannot be parallelized with rayon.
/// The pipeline handles this via three-phase execution.
pub struct PythonModelStub;

impl LightcurveModel for PythonModelStub {
    fn evaluate(
        &self,
        _instance: &TransientInstance,
        _times_mjd: &[f64],
        _bands: &[Band],
    ) -> Result<LightcurveEvaluation> {
        Err(super::LightcurveError::PythonError(
            "PythonModel requires PyO3 bindings. Use survey_sim Python package.".to_string(),
        ))
    }

    fn requires_gil(&self) -> bool {
        true
    }
}

/// Helper to convert a Python predict() result into a LightcurveEvaluation.
///
/// This is used by the PyO3 binding code in the python crate.
pub fn python_result_to_evaluation(
    times: Vec<f64>,
    mags_per_band: HashMap<String, Vec<f64>>,
) -> LightcurveEvaluation {
    LightcurveEvaluation {
        apparent_mags: mags_per_band,
        times_mjd: times,
    }
}
