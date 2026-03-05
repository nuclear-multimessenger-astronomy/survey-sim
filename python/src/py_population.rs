use std::sync::Arc;

use pyo3::prelude::*;

use survey_sim::population::generator::*;
use survey_sim::population::grb::{GrbCatalog, GrbPopulation};

/// Python wrapper for KilonovaPopulation.
#[pyclass]
#[pyo3(name = "KilonovaPopulation")]
pub struct PyKilonovaPopulation {
    pub(crate) rate: f64,
    pub(crate) z_max: f64,
    pub(crate) peak_abs_mag: f64,
}

#[pymethods]
impl PyKilonovaPopulation {
    #[new]
    #[pyo3(signature = (rate=1000.0, z_max=0.3, peak_abs_mag=-16.0))]
    fn new(rate: f64, z_max: f64, peak_abs_mag: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
        }
    }
}

impl PyKilonovaPopulation {
    pub fn to_generator(&self, mjd_min: f64, mjd_max: f64) -> KilonovaPopulation {
        KilonovaPopulation::new(self.rate, self.z_max, self.peak_abs_mag, mjd_min, mjd_max)
    }
}

/// Python wrapper for Bu2026KilonovaPopulation.
#[pyclass]
#[pyo3(name = "Bu2026KilonovaPopulation")]
pub struct PyBu2026KilonovaPopulation {
    pub(crate) rate: f64,
    pub(crate) z_max: f64,
}

#[pymethods]
impl PyBu2026KilonovaPopulation {
    #[new]
    #[pyo3(signature = (rate=1000.0, z_max=0.3))]
    fn new(rate: f64, z_max: f64) -> Self {
        Self { rate, z_max }
    }
}

impl PyBu2026KilonovaPopulation {
    pub fn to_generator(&self, mjd_min: f64, mjd_max: f64) -> Bu2026KilonovaPopulation {
        Bu2026KilonovaPopulation::new(self.rate, self.z_max, mjd_min, mjd_max)
    }
}

/// Python wrapper for FixedBu2026KilonovaPopulation.
#[pyclass]
#[pyo3(name = "FixedBu2026KilonovaPopulation")]
pub struct PyFixedBu2026KilonovaPopulation {
    pub(crate) rate: f64,
    pub(crate) z_max: f64,
    pub(crate) log10_mej_dyn: f64,
    pub(crate) v_ej_dyn: f64,
    pub(crate) ye_dyn: f64,
    pub(crate) log10_mej_wind: f64,
    pub(crate) v_ej_wind: f64,
    pub(crate) ye_wind: f64,
    pub(crate) inclination_em: f64,
    pub(crate) vary_inclination: bool,
}

#[pymethods]
impl PyFixedBu2026KilonovaPopulation {
    #[new]
    #[pyo3(signature = (
        log10_mej_dyn, v_ej_dyn, ye_dyn,
        log10_mej_wind, v_ej_wind, ye_wind,
        inclination_em=0.0,
        rate=1000.0, z_max=0.3,
        vary_inclination=false,
    ))]
    fn new(
        log10_mej_dyn: f64, v_ej_dyn: f64, ye_dyn: f64,
        log10_mej_wind: f64, v_ej_wind: f64, ye_wind: f64,
        inclination_em: f64,
        rate: f64, z_max: f64,
        vary_inclination: bool,
    ) -> Self {
        Self {
            rate, z_max,
            log10_mej_dyn, v_ej_dyn, ye_dyn,
            log10_mej_wind, v_ej_wind, ye_wind,
            inclination_em,
            vary_inclination,
        }
    }
}

impl PyFixedBu2026KilonovaPopulation {
    pub fn to_generator(&self, mjd_min: f64, mjd_max: f64) -> FixedBu2026KilonovaPopulation {
        let mut pop = FixedBu2026KilonovaPopulation::new(
            self.rate, self.z_max, mjd_min, mjd_max,
            self.log10_mej_dyn, self.v_ej_dyn, self.ye_dyn,
            self.log10_mej_wind, self.v_ej_wind, self.ye_wind,
            self.inclination_em,
        );
        pop.vary_inclination = self.vary_inclination;
        pop
    }
}

/// Python wrapper for SupernovaIaPopulation.
#[pyclass]
#[pyo3(name = "SupernovaIaPopulation")]
pub struct PySupernovaIaPopulation {
    pub(crate) rate: f64,
    pub(crate) z_max: f64,
    pub(crate) peak_abs_mag: f64,
}

#[pymethods]
impl PySupernovaIaPopulation {
    #[new]
    #[pyo3(signature = (rate=30000.0, z_max=1.0, peak_abs_mag=-19.3))]
    fn new(rate: f64, z_max: f64, peak_abs_mag: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
        }
    }
}

impl PySupernovaIaPopulation {
    pub fn to_generator(&self, mjd_min: f64, mjd_max: f64) -> SupernovaIaPopulation {
        SupernovaIaPopulation::new(self.rate, self.z_max, self.peak_abs_mag, mjd_min, mjd_max)
    }
}

/// Python wrapper for SupernovaIIPopulation.
#[pyclass]
#[pyo3(name = "SupernovaIIPopulation")]
pub struct PySupernovaIIPopulation {
    pub(crate) rate: f64,
    pub(crate) z_max: f64,
    pub(crate) peak_abs_mag: f64,
}

#[pymethods]
impl PySupernovaIIPopulation {
    #[new]
    #[pyo3(signature = (rate=70000.0, z_max=0.5, peak_abs_mag=-17.0))]
    fn new(rate: f64, z_max: f64, peak_abs_mag: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
        }
    }
}

impl PySupernovaIIPopulation {
    pub fn to_generator(&self, mjd_min: f64, mjd_max: f64) -> SupernovaIIPopulation {
        SupernovaIIPopulation::new(self.rate, self.z_max, self.peak_abs_mag, mjd_min, mjd_max)
    }
}

/// Python wrapper for TdePopulation.
#[pyclass]
#[pyo3(name = "TdePopulation")]
pub struct PyTdePopulation {
    pub(crate) rate: f64,
    pub(crate) z_max: f64,
    pub(crate) peak_abs_mag: f64,
}

#[pymethods]
impl PyTdePopulation {
    #[new]
    #[pyo3(signature = (rate=100.0, z_max=0.5, peak_abs_mag=-20.0))]
    fn new(rate: f64, z_max: f64, peak_abs_mag: f64) -> Self {
        Self {
            rate,
            z_max,
            peak_abs_mag,
        }
    }
}

impl PyTdePopulation {
    pub fn to_generator(&self, mjd_min: f64, mjd_max: f64) -> TdePopulation {
        TdePopulation::new(self.rate, self.z_max, self.peak_abs_mag, mjd_min, mjd_max)
    }
}

/// Python wrapper for GrbPopulation (blastwave afterglow from CSV catalog).
#[pyclass]
#[pyo3(name = "GrbPopulation")]
pub struct PyGrbPopulation {
    pub(crate) csv_path: String,
    pub(crate) rate: f64,
    pub(crate) z_max: f64,
}

#[pymethods]
impl PyGrbPopulation {
    #[new]
    #[pyo3(signature = (csv_path, rate=1.0, z_max=6.0))]
    fn new(csv_path: &str, rate: f64, z_max: f64) -> Self {
        Self {
            csv_path: csv_path.to_string(),
            rate,
            z_max,
        }
    }
}

impl PyGrbPopulation {
    pub fn to_generator(&self, mjd_min: f64, mjd_max: f64) -> PyResult<GrbPopulation> {
        let catalog = GrbCatalog::from_csv(&self.csv_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(GrbPopulation::new(
            Arc::new(catalog),
            self.rate,
            self.z_max,
            mjd_min,
            mjd_max,
        ))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKilonovaPopulation>()?;
    m.add_class::<PyBu2026KilonovaPopulation>()?;
    m.add_class::<PyFixedBu2026KilonovaPopulation>()?;
    m.add_class::<PySupernovaIaPopulation>()?;
    m.add_class::<PySupernovaIIPopulation>()?;
    m.add_class::<PyTdePopulation>()?;
    m.add_class::<PyGrbPopulation>()?;
    Ok(())
}
