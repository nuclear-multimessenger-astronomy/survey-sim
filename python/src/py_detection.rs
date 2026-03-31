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
        min_pre_peak_detections=0,
        min_post_peak_detections=0,
        min_phase_range_days=0.0,
        min_galactic_lat=0.0,
        spectroscopic_completeness_k=0.0,
        spectroscopic_completeness_m0=19.46,
        early_detection_fast_days=0.0,
        rate_dedup_window_days=None,
        stack_windows_s=None,
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
        min_pre_peak_detections: usize,
        min_post_peak_detections: usize,
        min_phase_range_days: f64,
        min_galactic_lat: f64,
        spectroscopic_completeness_k: f64,
        spectroscopic_completeness_m0: f64,
        early_detection_fast_days: f64,
        rate_dedup_window_days: Option<f64>,
        stack_windows_s: Option<Vec<f64>>,
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
                min_pre_peak_detections,
                min_post_peak_detections,
                min_phase_range_days,
                min_galactic_lat,
                spectroscopic_completeness_k,
                spectroscopic_completeness_m0,
                early_detection_fast_days,
                rate_dedup_window_days: rate_dedup_window_days.unwrap_or(2.0 / 24.0),
                stack_windows_s: stack_windows_s.unwrap_or_default(),
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
    pub n_detections_primary: usize,
    #[pyo3(get)]
    pub n_bands_detected: usize,
    #[pyo3(get)]
    pub first_detection_mjd: Option<f64>,
    #[pyo3(get)]
    pub last_detection_mjd: Option<f64>,
    #[pyo3(get)]
    pub detections_per_band: HashMap<String, usize>,
    #[pyo3(get)]
    pub peak_mjd: Option<f64>,
    #[pyo3(get)]
    pub peak_mag: Option<f64>,
    #[pyo3(get)]
    pub n_pre_peak: usize,
    #[pyo3(get)]
    pub n_post_peak: usize,
    #[pyo3(get)]
    pub phase_min_days: Option<f64>,
    #[pyo3(get)]
    pub phase_max_days: Option<f64>,
}

#[pymethods]
impl PyDetectionResult {
    fn __repr__(&self) -> String {
        format!(
            "DetectionResult(detected={}, n_det={}, n_bands={}, peak_mag={}, pre/post={}/{})",
            self.detected, self.n_detections, self.n_bands_detected,
            self.peak_mag.map_or("None".to_string(), |m| format!("{:.1}", m)),
            self.n_pre_peak, self.n_post_peak,
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDetectionCriteria>()?;
    m.add_class::<PyDetectionResult>()?;
    Ok(())
}
