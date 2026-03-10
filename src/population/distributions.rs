use rand::Rng;

use crate::efficiency::tde::{self, BhmfModel};
use crate::types::Cosmology;

/// Compute the maximum dV/dz over [0, z_max] for rejection sampling envelope.
pub fn max_dvdz(z_max: f64, cosmo: &Cosmology) -> f64 {
    let n_probe = 100;
    let mut max_val: f64 = 0.0;
    for i in 0..=n_probe {
        let z = z_max * i as f64 / n_probe as f64;
        let dvdz = cosmo.dv_dz(z);
        if dvdz > max_val {
            max_val = dvdz;
        }
    }
    max_val
}

/// Sample a redshift from the volumetric rate distribution dN/dz ~ dV/dz.
///
/// Uses rejection sampling with dV/dz as the weight.
/// `envelope` is the precomputed maximum of dV/dz over [0, z_max].
pub fn sample_redshift_volumetric(
    z_max: f64,
    cosmo: &Cosmology,
    envelope: f64,
    rng: &mut (impl Rng + ?Sized),
) -> f64 {
    loop {
        let z: f64 = rng.random::<f64>() * z_max;
        let dvdz = cosmo.dv_dz(z);
        let accept_prob = dvdz / envelope;
        if rng.random::<f64>() < accept_prob {
            return z;
        }
    }
}

/// Compute the maximum of dV/dz × W(z) over [0, z_max] for evolved rejection sampling.
///
/// W(z) = F(z, E, α) × N_BH(z) × O(z) / (1+z)
/// The (1+z) factor converts observer-frame to source-frame rate.
pub fn max_dvdz_evolved(
    z_max: f64,
    cosmo: &Cosmology,
    bhmf: &BhmfModel,
    e_factor: f64,
    density_alpha: f64,
) -> f64 {
    let n_probe = 200;
    let mut max_val: f64 = 0.0;
    for i in 0..=n_probe {
        let z = z_max * i as f64 / n_probe as f64;
        let dvdz = cosmo.dv_dz(z);
        let w = tde::galaxy_effects(z, e_factor, density_alpha)
            * bhmf.evolution(z)
            * tde::dust_obscuration(z)
            / (1.0 + z);
        let val = dvdz * w;
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

/// Sample a redshift from the evolved TDE rate distribution.
///
/// dN/dz ∝ dV/dz × F(z) × N_BH(z) × O(z) / (1+z)
/// This accounts for galaxy evolution, BHMF decline, and dust obscuration.
pub fn sample_redshift_evolved(
    z_max: f64,
    cosmo: &Cosmology,
    bhmf: &BhmfModel,
    e_factor: f64,
    density_alpha: f64,
    envelope: f64,
    rng: &mut (impl Rng + ?Sized),
) -> f64 {
    loop {
        let z: f64 = rng.random::<f64>() * z_max;
        let dvdz = cosmo.dv_dz(z);
        let w = tde::galaxy_effects(z, e_factor, density_alpha)
            * bhmf.evolution(z)
            * tde::dust_obscuration(z)
            / (1.0 + z);
        let accept_prob = dvdz * w / envelope;
        if rng.random::<f64>() < accept_prob {
            return z;
        }
    }
}

/// Sample an isotropic sky position (uniform on the sphere).
pub fn sample_isotropic_sky(rng: &mut (impl Rng + ?Sized)) -> (f64, f64) {
    let ra = rng.random::<f64>() * 360.0;
    // Uniform in sin(dec) for isotropic distribution.
    let sin_dec = rng.random::<f64>() * 2.0 - 1.0;
    let dec = sin_dec.asin().to_degrees();
    (ra, dec)
}

/// Sample an explosion time uniformly within an MJD range.
pub fn sample_explosion_time(mjd_min: f64, mjd_max: f64, rng: &mut (impl Rng + ?Sized)) -> f64 {
    mjd_min + rng.random::<f64>() * (mjd_max - mjd_min)
}

/// Sample from a log-uniform distribution in [lo, hi].
pub fn sample_log_uniform(lo: f64, hi: f64, rng: &mut (impl Rng + ?Sized)) -> f64 {
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    (log_lo + rng.random::<f64>() * (log_hi - log_lo)).exp()
}

/// Sample from a Gaussian distribution clamped to [lo, hi].
pub fn sample_gaussian_clamped(
    mean: f64,
    std: f64,
    lo: f64,
    hi: f64,
    rng: &mut (impl Rng + ?Sized),
) -> f64 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(mean, std).unwrap();
    loop {
        let x = normal.sample(rng);
        if x >= lo && x <= hi {
            return x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_volumetric_redshift_distribution() {
        let cosmo = Cosmology::default();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let n = 10_000;
        let z_max = 0.3;

        let envelope = max_dvdz(z_max, &cosmo);
        let samples: Vec<f64> = (0..n)
            .map(|_| sample_redshift_volumetric(z_max, &cosmo, envelope, &mut rng))
            .collect();

        // All samples should be in [0, z_max].
        assert!(samples.iter().all(|&z| z >= 0.0 && z <= z_max));

        // Mean should be skewed toward higher z (dV/dz increases with z).
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        assert!(mean > z_max * 0.4, "Mean redshift should be skewed high");
    }

    #[test]
    fn test_isotropic_sky() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let n = 10_000;
        let samples: Vec<(f64, f64)> =
            (0..n).map(|_| sample_isotropic_sky(&mut rng)).collect();

        // RA in [0, 360], Dec in [-90, 90].
        assert!(samples.iter().all(|&(ra, dec)| ra >= 0.0
            && ra <= 360.0
            && dec >= -90.0
            && dec <= 90.0));

        // Mean dec should be ~0 for isotropic.
        let mean_dec: f64 = samples.iter().map(|&(_, dec)| dec).sum::<f64>() / n as f64;
        assert!(mean_dec.abs() < 3.0);
    }
}
