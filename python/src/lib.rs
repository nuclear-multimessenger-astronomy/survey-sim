use pyo3::prelude::*;

mod py_survey;
mod py_population;
mod py_lightcurve;
mod py_detection;
mod py_pipeline;
mod py_skymap;

#[pymodule]
fn survey_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_survey::register(m)?;
    py_population::register(m)?;
    py_lightcurve::register(m)?;
    py_detection::register(m)?;
    py_pipeline::register(m)?;
    py_skymap::register(m)?;
    Ok(())
}
