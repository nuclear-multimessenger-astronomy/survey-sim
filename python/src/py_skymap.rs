use pyo3::prelude::*;
use rand::SeedableRng;
use survey_sim::skymap::Skymap;

/// Python wrapper for a rasterized HEALPix skymap.
#[pyclass(name = "Skymap")]
pub struct PySkymap {
    inner: Skymap,
}

#[pymethods]
impl PySkymap {
    /// Load a rasterized skymap from HDF5.
    ///
    /// The HDF5 file should contain:
    ///   - Attribute `nside` (u32)
    ///   - Dataset `PROB` (f64 array)
    ///   - Optional datasets `DISTMU`, `DISTSIGMA`, `DISTNORM`
    ///
    /// Use `ligo.skymap.bayestar.rasterize()` to convert multi-order FITS.
    #[staticmethod]
    fn from_hdf5(path: &str) -> PyResult<Self> {
        let skymap = Skymap::from_hdf5(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner: skymap })
    }

    /// Construct from arrays (e.g. after rasterizing with ligo.skymap).
    ///
    /// Args:
    ///     nside: HEALPix NSIDE (must be a power of 2)
    ///     prob: Probability array (nested ordering, length 12*nside²)
    ///     distmu: Per-pixel distance mean in Mpc (optional)
    ///     distsigma: Per-pixel distance std in Mpc (optional)
    ///     distnorm: Per-pixel distance normalization (optional)
    #[staticmethod]
    #[pyo3(signature = (nside, prob, distmu=None, distsigma=None, distnorm=None))]
    fn from_arrays(
        nside: u32,
        prob: Vec<f64>,
        distmu: Option<Vec<f64>>,
        distsigma: Option<Vec<f64>>,
        distnorm: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let skymap = Skymap::from_arrays(nside, prob, distmu, distsigma, distnorm)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner: skymap })
    }

    /// HEALPix NSIDE parameter.
    #[getter]
    fn nside(&self) -> u32 {
        self.inner.nside
    }

    /// Number of pixels.
    #[getter]
    fn npix(&self) -> usize {
        self.inner.npix()
    }

    /// Whether this skymap has 3D distance information.
    #[getter]
    fn has_distance(&self) -> bool {
        self.inner.has_distance()
    }

    /// 90% credible area in square degrees.
    #[getter]
    fn area_90(&self) -> f64 {
        self.inner.area_90()
    }

    /// Get probability at a sky position (ra, dec in degrees).
    fn prob_at(&self, ra: f64, dec: f64) -> f64 {
        self.inner.prob_at(ra, dec)
    }

    /// Compute 2D probability coverage from rectangular observations.
    ///
    /// Args:
    ///     obs_ra: RA of observation centers (degrees)
    ///     obs_dec: Dec of observation centers (degrees)
    ///     hw_ra: Half-width in RA direction (degrees)
    ///     hw_dec: Half-width in Dec direction (degrees)
    ///
    /// Returns:
    ///     CoverageResult with prob_2d, area_deg2, n_pixels, covered mask
    #[pyo3(signature = (obs_ra, obs_dec, hw_ra, hw_dec))]
    fn coverage_2d(
        &self,
        obs_ra: Vec<f64>,
        obs_dec: Vec<f64>,
        hw_ra: f64,
        hw_dec: f64,
    ) -> PyResult<PyCoverageResult> {
        if obs_ra.len() != obs_dec.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "obs_ra and obs_dec must have the same length",
            ));
        }
        let result = self.inner.coverage_2d(&obs_ra, &obs_dec, hw_ra, hw_dec);
        Ok(PyCoverageResult { inner: result })
    }
}

/// Result of a 2D coverage computation.
#[pyclass(name = "CoverageResult")]
pub struct PyCoverageResult {
    inner: survey_sim::skymap::CoverageResult,
}

#[pymethods]
impl PyCoverageResult {
    /// Integrated 2D probability covered.
    #[getter]
    fn prob_2d(&self) -> f64 {
        self.inner.prob_2d
    }

    /// Sky area covered in square degrees.
    #[getter]
    fn area_deg2(&self) -> f64 {
        self.inner.area_deg2
    }

    /// Number of HEALPix pixels covered.
    #[getter]
    fn n_pixels(&self) -> usize {
        self.inner.n_pixels
    }

    /// Per-pixel coverage mask (list of bools, length = skymap npix).
    #[getter]
    fn covered(&self) -> Vec<bool> {
        self.inner.covered.clone()
    }

    /// Compute 3D distance-weighted probability for the covered pixels.
    ///
    /// Args:
    ///     skymap: The Skymap object (for distance posteriors)
    ///     d_max_mpc: Maximum detectable distance in Mpc
    ///     n_samples: Number of Monte Carlo samples per pixel (default 2000)
    ///     seed: Random seed (default 42)
    #[pyo3(signature = (skymap, d_max_mpc, n_samples=2000, seed=42))]
    fn coverage_3d(
        &self,
        skymap: &PySkymap,
        d_max_mpc: f64,
        n_samples: usize,
        seed: u64,
    ) -> f64 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        skymap
            .inner
            .coverage_3d(&self.inner.covered, d_max_mpc, n_samples, &mut rng)
    }

    /// Compute 3D probability with per-pixel detection horizons.
    ///
    /// Like coverage_3d, but each pixel has its own d_max (e.g. from
    /// a time-dependent kilonova model). Pixels with d_max <= 0 are skipped.
    ///
    /// Args:
    ///     skymap: The Skymap object (for distance posteriors)
    ///     d_max_per_pixel: Per-pixel max distance in Mpc (length = npix)
    ///     n_samples: Number of Monte Carlo samples per pixel (default 2000)
    ///     seed: Random seed (default 42)
    #[pyo3(signature = (skymap, d_max_per_pixel, n_samples=2000, seed=42))]
    fn coverage_3d_variable(
        &self,
        skymap: &PySkymap,
        d_max_per_pixel: Vec<f64>,
        n_samples: usize,
        seed: u64,
    ) -> PyResult<f64> {
        if d_max_per_pixel.len() != skymap.inner.npix() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "d_max_per_pixel length {} != skymap npix {}",
                    d_max_per_pixel.len(), skymap.inner.npix()
                ),
            ));
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Ok(skymap.inner.coverage_3d_variable(
            &self.inner.covered,
            &d_max_per_pixel,
            n_samples,
            &mut rng,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "CoverageResult(prob_2d={:.4}, area_deg2={:.0}, n_pixels={})",
            self.inner.prob_2d, self.inner.area_deg2, self.inner.n_pixels
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySkymap>()?;
    m.add_class::<PyCoverageResult>()?;
    Ok(())
}
