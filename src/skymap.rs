//! Gravitational-wave skymap loading and coverage computation.
//!
//! Supports loading rasterized HEALPix skymaps (fixed NSIDE, nested ordering)
//! from HDF5 files containing PROB, DISTMU, DISTSIGMA, DISTNORM arrays.
//!
//! Use `ligo.skymap.bayestar.rasterize()` in Python to convert multi-order
//! FITS skymaps to this format.

use cdshealpix::nested;
use hdf5;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use thiserror::Error;

/// Errors from skymap operations.
#[derive(Error, Debug)]
pub enum SkymapError {
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
    #[error("Invalid skymap: {0}")]
    Invalid(String),
}

/// A rasterized HEALPix skymap with optional 3D distance information.
///
/// The probability array is in nested HEALPix ordering at the stored NSIDE.
/// Distance parameters (DISTMU, DISTSIGMA, DISTNORM) follow the
/// `ligo.skymap` convention where the per-pixel distance posterior is
/// proportional to `d^2 * N(d; distmu, distsigma) * distnorm`.
pub struct Skymap {
    /// NSIDE parameter (power of 2).
    pub nside: u32,
    /// HEALPix depth (= log2(nside)).
    pub depth: u8,
    /// Per-pixel probability (sums to 1.0). Nested ordering.
    pub prob: Vec<f64>,
    /// Per-pixel distance mean (Mpc). `None` if 2D-only skymap.
    pub distmu: Option<Vec<f64>>,
    /// Per-pixel distance std (Mpc).
    pub distsigma: Option<Vec<f64>>,
    /// Per-pixel distance normalization.
    pub distnorm: Option<Vec<f64>>,
}

impl Skymap {
    /// Load a rasterized skymap from HDF5.
    ///
    /// Expected format:
    /// - Attribute `nside` (u32)
    /// - Dataset `PROB` (f64 array, length = 12 * nside^2)
    /// - Optional datasets `DISTMU`, `DISTSIGMA`, `DISTNORM`
    pub fn from_hdf5(path: &str) -> Result<Self, SkymapError> {
        let file = hdf5::File::open(path)?;

        let nside: u32 = file.attr("nside")?.read_scalar()?;
        if !nside.is_power_of_two() || nside == 0 {
            return Err(SkymapError::Invalid(format!(
                "NSIDE must be a power of 2, got {}",
                nside
            )));
        }
        let depth = nside.trailing_zeros() as u8;
        let npix = 12 * (nside as usize) * (nside as usize);

        let prob: Vec<f64> = file.dataset("PROB")?.read_raw()?;
        if prob.len() != npix {
            return Err(SkymapError::Invalid(format!(
                "PROB length {} != expected {} for NSIDE={}",
                prob.len(),
                npix,
                nside
            )));
        }

        let distmu = file.dataset("DISTMU").ok().map(|d| d.read_raw::<f64>().ok()).flatten();
        let distsigma = file.dataset("DISTSIGMA").ok().map(|d| d.read_raw::<f64>().ok()).flatten();
        let distnorm = file.dataset("DISTNORM").ok().map(|d| d.read_raw::<f64>().ok()).flatten();

        Ok(Self {
            nside,
            depth,
            prob,
            distmu,
            distsigma,
            distnorm,
        })
    }

    /// Construct a skymap from arrays (e.g. from Python after rasterization).
    pub fn from_arrays(
        nside: u32,
        prob: Vec<f64>,
        distmu: Option<Vec<f64>>,
        distsigma: Option<Vec<f64>>,
        distnorm: Option<Vec<f64>>,
    ) -> Result<Self, SkymapError> {
        if !nside.is_power_of_two() || nside == 0 {
            return Err(SkymapError::Invalid(format!(
                "NSIDE must be a power of 2, got {}",
                nside
            )));
        }
        let depth = nside.trailing_zeros() as u8;
        let npix = 12 * (nside as usize) * (nside as usize);
        if prob.len() != npix {
            return Err(SkymapError::Invalid(format!(
                "PROB length {} != expected {} for NSIDE={}",
                prob.len(), npix, nside
            )));
        }
        if let Some(ref v) = distmu {
            if v.len() != npix {
                return Err(SkymapError::Invalid("DISTMU length mismatch".into()));
            }
        }
        if let Some(ref v) = distsigma {
            if v.len() != npix {
                return Err(SkymapError::Invalid("DISTSIGMA length mismatch".into()));
            }
        }
        if let Some(ref v) = distnorm {
            if v.len() != npix {
                return Err(SkymapError::Invalid("DISTNORM length mismatch".into()));
            }
        }
        Ok(Self { nside, depth, prob, distmu, distsigma, distnorm })
    }

    /// Number of pixels in the skymap.
    pub fn npix(&self) -> usize {
        self.prob.len()
    }

    /// Whether this skymap has 3D distance information.
    pub fn has_distance(&self) -> bool {
        self.distmu.is_some() && self.distsigma.is_some()
    }

    /// Get the probability at a given (ra, dec) in degrees.
    pub fn prob_at(&self, ra_deg: f64, dec_deg: f64) -> f64 {
        let ipix = nested::hash(self.depth, ra_deg.to_radians(), dec_deg.to_radians()) as usize;
        self.prob.get(ipix).copied().unwrap_or(0.0)
    }

    /// Get the pixel index for a given (ra, dec) in degrees.
    pub fn pixel_at(&self, ra_deg: f64, dec_deg: f64) -> usize {
        nested::hash(self.depth, ra_deg.to_radians(), dec_deg.to_radians()) as usize
    }

    /// Compute the 90% credible area in square degrees.
    pub fn area_90(&self) -> f64 {
        let mut sorted_prob = self.prob.clone();
        sorted_prob.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut cum = 0.0;
        let mut n = 0;
        for p in &sorted_prob {
            cum += p;
            n += 1;
            if cum >= 0.9 {
                break;
            }
        }
        let pixel_area_sr = 4.0 * std::f64::consts::PI / (self.npix() as f64);
        let pixel_area_deg2 = pixel_area_sr * (180.0 / std::f64::consts::PI).powi(2);
        n as f64 * pixel_area_deg2
    }

    /// Compute 2D probability covered by a set of HEALPix pixel indices.
    ///
    /// The pixel indices must be in nested ordering at the skymap's NSIDE.
    pub fn prob_in_pixels(&self, pixels: &[usize]) -> f64 {
        pixels.iter().filter_map(|&i| self.prob.get(i)).sum()
    }

    /// Compute 2D probability covered by observations at given (ra, dec) positions
    /// with rectangular footprints of half-width (hw_ra_deg, hw_dec_deg).
    ///
    /// This is the main coverage computation. For each observation, it finds
    /// all HEALPix pixels within the rectangular footprint and sums their
    /// probability (avoiding double-counting).
    pub fn coverage_2d(
        &self,
        obs_ra: &[f64],
        obs_dec: &[f64],
        hw_ra_deg: f64,
        hw_dec_deg: f64,
    ) -> CoverageResult {
        let mut covered = vec![false; self.npix()];
        let radius_deg = (hw_ra_deg * hw_ra_deg + hw_dec_deg * hw_dec_deg).sqrt();

        for (&ra, &dec) in obs_ra.iter().zip(obs_dec.iter()) {
            // Use cone query to find candidate pixels, then filter to rectangle
            let cone = nested::cone_coverage_approx(
                self.depth,
                ra.to_radians(),
                dec.to_radians().clamp(
                    -std::f64::consts::FRAC_PI_2 + 1e-10,
                    std::f64::consts::FRAC_PI_2 - 1e-10,
                ),
                radius_deg.to_radians(),
            );

            let cos_dec = dec.to_radians().cos().max(1e-10);

            for ipix in cone.flat_iter() {
                let (lon, lat) = nested::center(self.depth, ipix);
                let pix_ra = lon.to_degrees();
                let pix_dec = lat.to_degrees();

                // Check rectangular footprint (accounting for cos(dec) in RA)
                let dra = ((pix_ra - ra + 180.0).rem_euclid(360.0) - 180.0).abs();
                let ddec = (pix_dec - dec).abs();

                if dra * cos_dec <= hw_ra_deg && ddec <= hw_dec_deg {
                    covered[ipix as usize] = true;
                }
            }
        }

        let prob_covered: f64 = covered
            .iter()
            .enumerate()
            .filter(|(_, &c)| c)
            .map(|(i, _)| self.prob[i])
            .sum();

        let pixel_area_sr = 4.0 * std::f64::consts::PI / (self.npix() as f64);
        let pixel_area_deg2 = pixel_area_sr * (180.0 / std::f64::consts::PI).powi(2);
        let area_deg2 = covered.iter().filter(|&&c| c).count() as f64 * pixel_area_deg2;

        CoverageResult {
            prob_2d: prob_covered,
            area_deg2,
            n_pixels: covered.iter().filter(|&&c| c).count(),
            covered,
        }
    }

    /// Compute 3D distance-weighted probability with per-pixel detection horizons.
    ///
    /// Like `coverage_3d`, but each pixel has its own maximum detectable distance.
    /// This supports time-dependent kilonova models where the detection horizon
    /// depends on when each pixel was observed.
    ///
    /// `d_max_per_pixel` must have length == `self.npix()`. Pixels with
    /// `d_max_per_pixel[i] <= 0` are skipped.
    pub fn coverage_3d_variable(
        &self,
        covered: &[bool],
        d_max_per_pixel: &[f64],
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> f64 {
        let distmu = match &self.distmu {
            Some(d) => d,
            None => return 0.0,
        };
        let distsigma = match &self.distsigma {
            Some(d) => d,
            None => return 0.0,
        };

        let mut prob_3d = 0.0;

        for (i, &is_covered) in covered.iter().enumerate() {
            if !is_covered || self.prob[i] < 1e-10 {
                continue;
            }
            let d_max = d_max_per_pixel[i];
            if d_max <= 0.0 {
                continue;
            }

            let mu = distmu[i];
            let sigma = distsigma[i];
            if sigma <= 0.0 || mu.is_nan() || mu.is_infinite() || mu <= 0.0 {
                continue;
            }

            let normal = match Normal::new(mu, sigma) {
                Ok(n) => n,
                Err(_) => continue,
            };

            let mut n_detectable = 0usize;
            for _ in 0..n_samples {
                let d: f64 = normal.sample(rng).abs();
                if d < d_max {
                    n_detectable += 1;
                }
            }

            prob_3d += self.prob[i] * (n_detectable as f64 / n_samples as f64);
        }

        prob_3d
    }

    /// Combined 2D + 3D coverage from rectangular observations with per-observation
    /// detection horizons.
    ///
    /// Each observation has its own `d_max` (e.g. from a time-dependent kilonova
    /// model evaluated at that observation's time and depth). For pixels covered
    /// by multiple observations, the best (highest) `d_max` is kept — this
    /// corresponds to the observation with the deepest effective sensitivity.
    ///
    /// Returns a `CoverageResult3D` with 2D coverage, the per-pixel best `d_max`,
    /// and the integrated 3D probability.
    pub fn coverage_2d_3d(
        &self,
        obs_ra: &[f64],
        obs_dec: &[f64],
        hw_ra_deg: f64,
        hw_dec_deg: f64,
        obs_d_max: &[f64],
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> CoverageResult3D {
        assert_eq!(obs_ra.len(), obs_dec.len());
        assert_eq!(obs_ra.len(), obs_d_max.len());

        let npix = self.npix();
        let mut best_d_max = vec![0.0_f64; npix];
        let mut covered = vec![false; npix];
        let radius_deg = (hw_ra_deg * hw_ra_deg + hw_dec_deg * hw_dec_deg).sqrt();

        // Phase 1: find covered pixels, keep best d_max per pixel
        for (k, (&ra, &dec)) in obs_ra.iter().zip(obs_dec.iter()).enumerate() {
            let d_max_k = obs_d_max[k];

            let cone = nested::cone_coverage_approx(
                self.depth,
                ra.to_radians(),
                dec.to_radians().clamp(
                    -std::f64::consts::FRAC_PI_2 + 1e-10,
                    std::f64::consts::FRAC_PI_2 - 1e-10,
                ),
                radius_deg.to_radians(),
            );

            let cos_dec = dec.to_radians().cos().max(1e-10);

            for ipix in cone.flat_iter() {
                let (lon, lat) = nested::center(self.depth, ipix);
                let pix_ra = lon.to_degrees();
                let pix_dec = lat.to_degrees();

                let dra = ((pix_ra - ra + 180.0).rem_euclid(360.0) - 180.0).abs();
                let ddec = (pix_dec - dec).abs();

                if dra * cos_dec <= hw_ra_deg && ddec <= hw_dec_deg {
                    let idx = ipix as usize;
                    covered[idx] = true;
                    if d_max_k > best_d_max[idx] {
                        best_d_max[idx] = d_max_k;
                    }
                }
            }
        }

        // 2D statistics
        let prob_2d: f64 = covered.iter().enumerate()
            .filter(|(_, &c)| c)
            .map(|(i, _)| self.prob[i])
            .sum();
        let pixel_area_sr = 4.0 * std::f64::consts::PI / (npix as f64);
        let pixel_area_deg2 = pixel_area_sr * (180.0 / std::f64::consts::PI).powi(2);
        let n_covered = covered.iter().filter(|&&c| c).count();
        let area_deg2 = n_covered as f64 * pixel_area_deg2;

        // Phase 2: 3D MC integration using best d_max per pixel
        let prob_3d = self.coverage_3d_variable(&covered, &best_d_max, n_samples, rng);

        CoverageResult3D {
            prob_2d,
            prob_3d,
            area_deg2,
            n_pixels: n_covered,
            covered,
            best_d_max,
        }
    }

    /// Compute 3D distance-weighted probability for observed pixels.
    ///
    /// For each observed pixel, samples from the distance posterior and computes
    /// the fraction of samples within the detection horizon (d < d_max_mpc).
    ///
    /// Returns the sum over observed pixels of: prob[i] * P(d < d_max | pixel i).
    pub fn coverage_3d(
        &self,
        covered: &[bool],
        d_max_mpc: f64,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> f64 {
        let distmu = match &self.distmu {
            Some(d) => d,
            None => return 0.0,
        };
        let distsigma = match &self.distsigma {
            Some(d) => d,
            None => return 0.0,
        };

        let mut prob_3d = 0.0;

        for (i, &is_covered) in covered.iter().enumerate() {
            if !is_covered || self.prob[i] < 1e-10 {
                continue;
            }

            let mu = distmu[i];
            let sigma = distsigma[i];

            if sigma <= 0.0 || mu.is_nan() || mu.is_infinite() || mu <= 0.0 {
                continue;
            }

            // Sample from the distance posterior
            let normal = match Normal::new(mu, sigma) {
                Ok(n) => n,
                Err(_) => continue,
            };

            let mut n_detectable = 0usize;
            for _ in 0..n_samples {
                let d: f64 = normal.sample(rng).abs();
                if d < d_max_mpc {
                    n_detectable += 1;
                }
            }

            prob_3d += self.prob[i] * (n_detectable as f64 / n_samples as f64);
        }

        prob_3d
    }
}

/// Result of a combined 2D+3D coverage computation with per-observation horizons.
pub struct CoverageResult3D {
    /// Integrated 2D probability covered.
    pub prob_2d: f64,
    /// Integrated 3D probability (using best per-pixel d_max).
    pub prob_3d: f64,
    /// Sky area covered in square degrees.
    pub area_deg2: f64,
    /// Number of HEALPix pixels covered.
    pub n_pixels: usize,
    /// Per-pixel coverage mask.
    pub covered: Vec<bool>,
    /// Per-pixel best detection horizon (Mpc). 0 if not covered.
    pub best_d_max: Vec<f64>,
}

/// Result of a 2D coverage computation.
pub struct CoverageResult {
    /// Integrated 2D probability covered.
    pub prob_2d: f64,
    /// Sky area covered in square degrees.
    pub area_deg2: f64,
    /// Number of HEALPix pixels covered.
    pub n_pixels: usize,
    /// Per-pixel coverage mask (for use with `coverage_3d`).
    pub covered: Vec<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skymap_prob_at() {
        // Create a simple skymap with uniform probability
        let nside = 8u32;
        let npix = 12 * (nside as usize) * (nside as usize);
        let prob = vec![1.0 / npix as f64; npix];
        let skymap = Skymap {
            nside,
            depth: nside.trailing_zeros() as u8,
            prob,
            distmu: None,
            distsigma: None,
            distnorm: None,
        };

        // Any position should have the same probability
        let p = skymap.prob_at(45.0, 30.0);
        assert!((p - 1.0 / npix as f64).abs() < 1e-15);
    }

    #[test]
    fn test_area_90() {
        // Uniform skymap: 90% area should be ~90% of full sky
        let nside = 8u32;
        let npix = 12 * (nside as usize) * (nside as usize);
        let prob = vec![1.0 / npix as f64; npix];
        let skymap = Skymap {
            nside,
            depth: nside.trailing_zeros() as u8,
            prob,
            distmu: None,
            distsigma: None,
            distnorm: None,
        };

        let area = skymap.area_90();
        let full_sky = 4.0 * 180.0 * 180.0 / std::f64::consts::PI;
        // Should be close to 90% of full sky
        assert!((area / full_sky - 0.9).abs() < 0.05);
    }

    #[test]
    fn test_coverage_2d_point_source() {
        // Concentrate all probability in one pixel, check coverage
        // Use NSIDE=32 (depth=5), pixel size ~1.83°
        let nside = 32u32;
        let npix = 12 * (nside as usize) * (nside as usize);
        let mut prob = vec![0.0; npix];

        // Put all probability at (180°, 0°)
        let depth = nside.trailing_zeros() as u8;
        let target_pix = nested::hash(depth, std::f64::consts::PI, 0.0) as usize;
        prob[target_pix] = 1.0;

        let skymap = Skymap {
            nside,
            depth,
            prob,
            distmu: None,
            distsigma: None,
            distnorm: None,
        };

        // Observation with ZTF-like footprint (3.5° half-width, larger than pixel)
        let result = skymap.coverage_2d(&[180.0], &[0.0], 3.5, 3.5);
        assert!(
            result.prob_2d > 0.99,
            "Should cover >99% of probability, got {}",
            result.prob_2d
        );

        // Observation far away should cover nothing
        let result_far = skymap.coverage_2d(&[0.0], &[80.0], 3.5, 3.5);
        assert!(
            result_far.prob_2d < 0.01,
            "Should cover <1% of probability, got {}",
            result_far.prob_2d
        );
    }
}
