use crate::types::Cosmology;

/// Speed of light in km/s.
const C_KMS: f64 = 299792.458;

impl Cosmology {
    /// Hubble parameter H(z) in km/s/Mpc for flat LCDM.
    pub fn hubble(&self, z: f64) -> f64 {
        let h0 = self.h * 100.0; // km/s/Mpc
        let z1 = 1.0 + z;
        h0 * (self.omega_m * z1.powi(3) + self.omega_lambda).sqrt()
    }

    /// Comoving distance in Mpc via numerical integration (Simpson's rule).
    pub fn comoving_distance(&self, z: f64) -> f64 {
        if z <= 0.0 {
            return 0.0;
        }
        let n = 1000;
        let dz = z / n as f64;
        let h0 = self.h * 100.0;

        let integrand = |zp: f64| -> f64 {
            let z1 = 1.0 + zp;
            1.0 / (self.omega_m * z1.powi(3) + self.omega_lambda).sqrt()
        };

        // Simpson's rule.
        let mut sum = integrand(0.0) + integrand(z);
        for i in 1..n {
            let zp = i as f64 * dz;
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * integrand(zp);
        }
        sum * dz / 3.0 * C_KMS / h0
    }

    /// Luminosity distance in Mpc.
    pub fn luminosity_distance(&self, z: f64) -> f64 {
        self.comoving_distance(z) * (1.0 + z)
    }

    /// Distance modulus mu = 5 * log10(d_L / 10pc).
    pub fn distance_modulus(&self, z: f64) -> f64 {
        let d_l_mpc = self.luminosity_distance(z);
        let d_l_pc = d_l_mpc * 1e6;
        5.0 * (d_l_pc / 10.0).log10()
    }

    /// Differential comoving volume element dV/dz in Gpc^3/sr.
    pub fn dv_dz(&self, z: f64) -> f64 {
        let d_c = self.comoving_distance(z); // Mpc
        let h_z = self.hubble(z); // km/s/Mpc
        let d_c_gpc = d_c / 1000.0;
        // dV/dz/dOmega = (c/H(z)) * d_c^2
        // c/H(z) is in Mpc, so convert to Gpc by dividing by 1000.
        let c_over_h_gpc = C_KMS / h_z / 1000.0;
        c_over_h_gpc * d_c_gpc * d_c_gpc
    }

    /// Invert luminosity_distance to find redshift from distance (bisection).
    pub fn redshift_from_distance(&self, d_l_mpc: f64) -> f64 {
        if d_l_mpc <= 0.0 {
            return 0.0;
        }
        let mut z_lo = 0.0;
        let mut z_hi = 20.0;
        for _ in 0..100 {
            let z_mid = (z_lo + z_hi) / 2.0;
            if self.luminosity_distance(z_mid) < d_l_mpc {
                z_lo = z_mid;
            } else {
                z_hi = z_mid;
            }
        }
        (z_lo + z_hi) / 2.0
    }

    /// Comoving volume out to redshift z in Gpc^3 (full sky).
    pub fn comoving_volume(&self, z: f64) -> f64 {
        let d_c = self.comoving_distance(z) / 1000.0; // Gpc
        4.0 / 3.0 * std::f64::consts::PI * d_c.powi(3)
    }
}

/// Bolometric K-correction approximation: -2.5 * log10(1+z).
pub fn k_correction_bolometric(z: f64) -> f64 {
    -2.5 * (1.0 + z).log10()
}

/// Milky Way extinction in a specific band given A_V.
/// Uses Cardelli et al. (1989) R_V=3.1 ratios for common bands.
/// If an `InstrumentConfig` is provided, its extinction coefficients take precedence.
pub fn extinction_in_band(a_v: f64, band: &str) -> f64 {
    extinction_in_band_with_instrument(a_v, band, None)
}

/// Milky Way extinction in a specific band, optionally using instrument-defined coefficients.
pub fn extinction_in_band_with_instrument(
    a_v: f64,
    band: &str,
    instrument: Option<&crate::instrument::InstrumentConfig>,
) -> f64 {
    if let Some(inst) = instrument {
        return a_v * inst.extinction_ratio(band);
    }
    let ratio = match band {
        // UV (Cardelli+1989 R_V=3.1 extrapolation)
        "FUV" => 8.4,
        "NUV" => 8.0,
        // Optical/NIR
        "u" | "U" => 1.56,
        "g" | "B" | "bessellb" => 1.31,
        "r" | "V" | "bessellv" => 1.0,
        "i" | "R" | "bessellr" => 0.75,
        "z" => 0.55,
        "y" | "Y" => 0.47,
        "J" => 0.29,
        "H" => 0.18,
        "K" | "Ks" => 0.11,
        _ => 1.0, // default to V-band ratio
    };
    a_v * ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luminosity_distance_z0() {
        let cosmo = Cosmology::default();
        let d_l = cosmo.luminosity_distance(0.0);
        assert!((d_l).abs() < 1e-10);
    }

    #[test]
    fn test_luminosity_distance_low_z() {
        let cosmo = Cosmology::default();
        // At low z, d_L ~ c*z/H0 (Hubble law).
        let z = 0.01;
        let d_l = cosmo.luminosity_distance(z);
        let h0 = cosmo.h * 100.0;
        let d_hubble = C_KMS * z / h0;
        // Should agree to ~1% at z=0.01.
        assert!((d_l - d_hubble).abs() / d_hubble < 0.01);
    }

    #[test]
    fn test_distance_modulus_known() {
        let cosmo = Cosmology::default();
        // At z=0.1, d_L ~ 460 Mpc -> mu ~ 38.3
        let mu = cosmo.distance_modulus(0.1);
        assert!(mu > 38.0 && mu < 39.0);
    }

    #[test]
    fn test_k_correction() {
        assert!((k_correction_bolometric(0.0)).abs() < 1e-10);
        // At z=1, k = -2.5*log10(2) ~ -0.753
        let k = k_correction_bolometric(1.0);
        assert!((k - (-0.7526)).abs() < 0.01);
    }

    #[test]
    fn test_comoving_volume() {
        let cosmo = Cosmology::default();
        let v = cosmo.comoving_volume(0.3);
        // Should be ~10 Gpc^3 at z=0.3.
        assert!(v > 5.0 && v < 20.0);
    }

    #[test]
    fn test_dv_dz_consistency() {
        // Verify dV/dz integrates to V(z) = (4/3)π d_c³.
        // ∫₀^z dV/dz' dΩ dz' = 4π ∫₀^z dV/dz'/dΩ dz' should equal V(z).
        let cosmo = Cosmology::default();
        let z_max = 0.3;
        let n = 1000;
        let dz = z_max / n as f64;

        // Trapezoidal integration of 4π × dV/dz/dΩ.
        let mut integral = 0.0;
        for i in 0..n {
            let z0 = i as f64 * dz;
            let z1 = (i + 1) as f64 * dz;
            integral += 0.5 * (cosmo.dv_dz(z0) + cosmo.dv_dz(z1)) * dz;
        }
        let v_integrated = integral * 4.0 * std::f64::consts::PI;
        let v_analytic = cosmo.comoving_volume(z_max);

        let rel_err = (v_integrated - v_analytic).abs() / v_analytic;
        assert!(
            rel_err < 0.01,
            "dV/dz integral ({:.4}) vs comoving_volume ({:.4}): {:.2}% error",
            v_integrated, v_analytic, rel_err * 100.0
        );
    }
}
