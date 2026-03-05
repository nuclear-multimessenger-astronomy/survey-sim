use std::collections::HashMap;

use pyo3::prelude::*;

use survey_sim::detection::DetectionCriteria;

/// Python wrapper for DetectionCriteria.
#[pyclass]
#[pyo3(name = "DetectionCriteria")]
pub struct PyDetectionCriteria {
    pub(crate) inner: DetectionCriteria,
}

#[pymethods]
impl PyDetectionCriteria {
    #[new]
    #[pyo3(signature = (
        min_detections=2,
        min_bands=1,
        min_per_band=1,
        max_timespan_days=30.0,
        snr_threshold=5.0,
        snr_threshold_secondary=None,
        min_detections_primary=1,
        min_time_separation_hours=0.0,
        require_fast_transient=false,
        min_rise_rate=1.0,
        min_fade_rate=0.3,
    ))]
    fn new(
        min_detections: usize,
        min_bands: usize,
        min_per_band: usize,
        max_timespan_days: f64,
        snr_threshold: f64,
        snr_threshold_secondary: Option<f64>,
        min_detections_primary: usize,
        min_time_separation_hours: f64,
        require_fast_transient: bool,
        min_rise_rate: f64,
        min_fade_rate: f64,
    ) -> Self {
        Self {
            inner: DetectionCriteria {
                min_detections,
                min_bands,
                min_per_band,
                max_timespan_days,
                snr_threshold,
                snr_threshold_secondary: snr_threshold_secondary.unwrap_or(snr_threshold),
                min_detections_primary,
                min_time_separation_hours,
                require_fast_transient,
                min_rise_rate,
                min_fade_rate,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectionCriteria(min_detections={}, min_bands={}, min_per_band={}, max_timespan_days={}, snr_threshold={})",
            self.inner.min_detections,
            self.inner.min_bands,
            self.inner.min_per_band,
            self.inner.max_timespan_days,
            self.inner.snr_threshold,
        )
    }
}

/// Python wrapper for DetectionResult.
#[pyclass]
#[pyo3(name = "DetectionResult")]
pub struct PyDetectionResult {
    #[pyo3(get)]
    pub detected: bool,
    #[pyo3(get)]
    pub n_detections: usize,
    #[pyo3(get)]
    pub n_bands_detected: usize,
    #[pyo3(get)]
    pub first_detection_mjd: Option<f64>,
    #[pyo3(get)]
    pub last_detection_mjd: Option<f64>,
    #[pyo3(get)]
    pub detections_per_band: HashMap<String, usize>,
}

#[pymethods]
impl PyDetectionResult {
    fn __repr__(&self) -> String {
        format!(
            "DetectionResult(detected={}, n_detections={}, n_bands={})",
            self.detected, self.n_detections, self.n_bands_detected
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDetectionCriteria>()?;
    m.add_class::<PyDetectionResult>()?;
    Ok(())
}
