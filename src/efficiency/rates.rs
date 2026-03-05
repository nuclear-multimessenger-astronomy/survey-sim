use crate::types::Cosmology;

/// Result of recovering a volumetric rate from observed detections.
#[derive(Clone, Debug)]
pub struct RateRecovery {
    /// Injected volumetric rate in Gpc^-3 yr^-1.
    pub input_rate: f64,
    /// Recovered volumetric rate in Gpc^-3 yr^-1.
    pub recovered_rate: f64,
    /// Expected number of real detections over the survey.
    pub n_expected_detections: f64,
    /// Effective volume-time: Omega * T * integral(eff(z) dV/dz dz) in Gpc^3 yr.
    pub effective_vt: f64,
    /// 1-sigma lower bound on recovered rate.
    pub poisson_lower_1sig: f64,
    /// 1-sigma upper bound on recovered rate.
    pub poisson_upper_1sig: f64,
    /// 2-sigma lower bound on recovered rate.
    pub poisson_lower_2sig: f64,
    /// 2-sigma upper bound on recovered rate.
    pub poisson_upper_2sig: f64,
    /// Whether input rate is within 1-sigma interval.
    pub consistent_1sig: bool,
    /// Whether input rate is within 2-sigma interval.
    pub consistent_2sig: bool,
}

impl std::fmt::Display for RateRecovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lower = self.recovered_rate - self.poisson_lower_2sig;
        let upper = self.poisson_upper_2sig - self.recovered_rate;
        writeln!(
            f,
            "  Recovered: {:.2} +{:.2}/-{:.2} Gpc^-3 yr^-1 (input: {:.2}, 2sig consistent: {})",
            self.recovered_rate,
            upper,
            lower,
            self.input_rate,
            if self.consistent_2sig { "yes" } else { "no" },
        )
    }
}

/// Recover the volumetric rate from observed detections by inverting the rate integral.
///
/// `R_recovered = N_det / (T * Omega * integral[eff(z) * dV/dz dz])`
///
/// Poisson confidence intervals use Gaussian approximation (sigma = R / sqrt(N)).
/// For low N, uses Gehrels (1986) upper limits.
pub fn recover_rate(
    efficiency_vs_z: &[(f64, f64)],
    n_expected_detections: f64,
    survey_duration_years: f64,
    survey_omega_sr: f64,
    input_rate: f64,
    cosmo: &Cosmology,
) -> RateRecovery {
    // Compute effective VT by calling compute_rate with unit volumetric rate.
    // compute_rate(eff, 1.0, Omega, cosmo) gives detections/yr per unit rate.
    // Multiply by T to get VT in Gpc^3 yr.
    let rate_per_unit = compute_rate(efficiency_vs_z, 1.0, survey_omega_sr, cosmo);
    let effective_vt = rate_per_unit * survey_duration_years;

    let recovered_rate = if effective_vt > 0.0 {
        n_expected_detections / effective_vt
    } else {
        0.0
    };

    // Poisson uncertainty: sigma_N = sqrt(N), so sigma_R = R / sqrt(N).
    // For low N, use Gehrels (1986): upper = N + 1 + sqrt(N + 0.75).
    let (lower_1sig, upper_1sig, lower_2sig, upper_2sig) = if n_expected_detections > 0.0 {
        let n = n_expected_detections;
        if n >= 20.0 {
            // Gaussian approximation.
            let sigma_r = recovered_rate / n.sqrt();
            (
                (recovered_rate - sigma_r).max(0.0),
                recovered_rate + sigma_r,
                (recovered_rate - 2.0 * sigma_r).max(0.0),
                recovered_rate + 2.0 * sigma_r,
            )
        } else {
            // Gehrels (1986) for low-N Poisson intervals.
            let n_upper_1sig = n + 1.0 + (n + 0.75).sqrt();
            let n_lower_1sig = if n > 0.0 {
                n * (1.0 - 1.0 / (9.0 * n) - 1.0 / (3.0 * n.sqrt())).powi(3).max(0.0)
            } else {
                0.0
            };
            let n_upper_2sig = n + 2.0 + 2.0 * (n + 1.0).sqrt();
            let n_lower_2sig = if n > 1.0 {
                n * (1.0 - 1.0 / (9.0 * n) - 2.0 / (3.0 * n.sqrt())).powi(3).max(0.0)
            } else {
                0.0
            };
            (
                n_lower_1sig / effective_vt,
                n_upper_1sig / effective_vt,
                n_lower_2sig / effective_vt,
                n_upper_2sig / effective_vt,
            )
        }
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let consistent_1sig = input_rate >= lower_1sig && input_rate <= upper_1sig;
    let consistent_2sig = input_rate >= lower_2sig && input_rate <= upper_2sig;

    RateRecovery {
        input_rate,
        recovered_rate,
        n_expected_detections,
        effective_vt,
        poisson_lower_1sig: lower_1sig,
        poisson_upper_1sig: upper_1sig,
        poisson_lower_2sig: lower_2sig,
        poisson_upper_2sig: upper_2sig,
        consistent_1sig,
        consistent_2sig,
    }
}

/// Result of computing a rate upper limit from zero (or few) detections.
#[derive(Clone, Debug)]
pub struct RateUpperLimit {
    /// Observed number of detections.
    pub n_observed: u64,
    /// Confidence level (e.g., 0.90 for 90% CL).
    pub confidence_level: f64,
    /// Poisson upper limit on count (e.g., 2.303 for 90% CL with 0 obs).
    pub n_upper: f64,
    /// Effective volume-time in Gpc^3 yr.
    pub effective_vt: f64,
    /// Upper limit on volumetric rate in Gpc^-3 yr^-1.
    pub rate_upper: f64,
    /// Survey duration in years.
    pub survey_duration_years: f64,
    /// Survey solid angle in steradians.
    pub survey_omega_sr: f64,
}

impl std::fmt::Display for RateUpperLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Rate Upper Limit ({:.0}% CL):",
            self.confidence_level * 100.0
        )?;
        writeln!(f, "  N_observed: {}", self.n_observed)?;
        writeln!(f, "  N_upper (Poisson): {:.3}", self.n_upper)?;
        writeln!(f, "  Effective VT: {:.6} Gpc^3 yr", self.effective_vt)?;
        writeln!(
            f,
            "  R_upper: {:.1} Gpc^-3 yr^-1",
            self.rate_upper
        )
    }
}

/// Compute the Poisson upper limit on count for a given observed count and CL.
///
/// For n_observed = 0, the exact result is -ln(1 - CL).
/// For n_observed > 0, uses the Gehrels (1986) approximation.
fn poisson_upper_limit(n_observed: u64, confidence_level: f64) -> f64 {
    if n_observed == 0 {
        // Exact: P(k=0 | lambda) = exp(-lambda) <= 1 - CL
        // => lambda >= -ln(1 - CL)
        -(1.0 - confidence_level).ln()
    } else {
        // Gehrels (1986) Table 1 approximation for upper limits.
        // For 90% CL (1.28σ) and small n:
        let n = n_observed as f64;
        let s = if confidence_level > 0.95 {
            2.0 // ~2σ for 95% CL
        } else if confidence_level > 0.85 {
            1.282 // ~1.28σ for 90% CL
        } else {
            1.0 // ~1σ for 68% CL
        };
        n + s * n.sqrt() + 1.0
    }
}

/// Compute a rate upper limit from an efficiency curve, observed count, and survey parameters.
///
/// R_upper = N_upper / (T × Omega × ∫ eff(z) × dV/dz dz)
pub fn compute_rate_upper_limit(
    efficiency_vs_z: &[(f64, f64)],
    n_observed: u64,
    survey_duration_years: f64,
    survey_omega_sr: f64,
    confidence_level: f64,
    cosmo: &Cosmology,
) -> RateUpperLimit {
    let rate_per_unit = compute_rate(efficiency_vs_z, 1.0, survey_omega_sr, cosmo);
    let effective_vt = rate_per_unit * survey_duration_years;

    let n_upper = poisson_upper_limit(n_observed, confidence_level);

    let rate_upper = if effective_vt > 0.0 {
        n_upper / effective_vt
    } else {
        f64::INFINITY
    };

    RateUpperLimit {
        n_observed,
        confidence_level,
        n_upper,
        effective_vt,
        rate_upper,
        survey_duration_years,
        survey_omega_sr,
    }
}

/// Summary of expected detection rates.
#[derive(Clone, Debug)]
pub struct RateSummary {
    /// Transient type name.
    pub transient_type: String,
    /// Assumed volumetric rate in Gpc^-3 yr^-1.
    pub volumetric_rate: f64,
    /// Expected detections per year.
    pub detections_per_year: f64,
    /// Expected detections over the full survey duration.
    pub detections_total: f64,
    /// Overall detection efficiency (fraction of simulated transients detected).
    pub overall_efficiency: f64,
    /// Survey solid angle in steradians.
    pub survey_omega_sr: f64,
    /// Maximum redshift considered.
    pub z_max: f64,
    /// Rate recovery result (if computed).
    pub recovery: Option<RateRecovery>,
}

impl std::fmt::Display for RateSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Rate Summary: {}", self.transient_type)?;
        writeln!(f, "  Volumetric rate: {:.1} Gpc^-3 yr^-1", self.volumetric_rate)?;
        writeln!(f, "  z_max: {:.2}", self.z_max)?;
        writeln!(f, "  Overall efficiency: {:.4}", self.overall_efficiency)?;
        writeln!(f, "  Detections/yr: {:.1}", self.detections_per_year)?;
        writeln!(f, "  Total detections: {:.1}", self.detections_total)?;
        if let Some(ref recovery) = self.recovery {
            write!(f, "{}", recovery)?;
        }
        Ok(())
    }
}

/// Compute expected detection rate by integrating efficiency over redshift.
///
/// rate/yr = R_vol × Ω × ∫ eff(z) × dV/dz / (1+z) dz
///
/// The 1/(1+z) factor converts the comoving (source-frame) volumetric rate
/// to the observer frame, accounting for cosmological time dilation.
///
/// `efficiency_vs_z` is a list of (z, efficiency) pairs from the efficiency grid.
/// `survey_omega_sr` is the survey solid angle in steradians.
/// `volumetric_rate` is in Gpc^-3 yr^-1 (comoving, source-frame).
pub fn compute_rate(
    efficiency_vs_z: &[(f64, f64)],
    volumetric_rate: f64,
    survey_omega_sr: f64,
    cosmo: &Cosmology,
) -> f64 {
    if efficiency_vs_z.len() < 2 {
        return 0.0;
    }

    // Trapezoidal integration of eff(z) * dV/dz / (1+z) over z.
    let mut integral = 0.0;
    for i in 0..efficiency_vs_z.len() - 1 {
        let (z0, eff0) = efficiency_vs_z[i];
        let (z1, eff1) = efficiency_vs_z[i + 1];
        let dz = z1 - z0;

        let dvdz0 = cosmo.dv_dz(z0); // Gpc^3/sr
        let dvdz1 = cosmo.dv_dz(z1);

        // Include 1/(1+z) for observer-frame time dilation.
        let f0 = eff0 * dvdz0 / (1.0 + z0);
        let f1 = eff1 * dvdz1 / (1.0 + z1);
        integral += 0.5 * (f0 + f1) * dz;
    }

    // integral is in Gpc^3/sr, multiply by Omega to get volume.
    volumetric_rate * integral * survey_omega_sr
}

/// Estimate the survey solid angle from the number of unique HEALPix pixels
/// with observations.
///
/// Omega = n_pixels * pixel_area, where pixel_area = 4pi / (12 * nside^2).
pub fn estimate_survey_omega(n_pixels: usize, nside: u32) -> f64 {
    let pixel_area = 4.0 * std::f64::consts::PI / (12.0 * (nside as f64).powi(2));
    n_pixels as f64 * pixel_area
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rate_zero_efficiency() {
        let eff = vec![(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)];
        let cosmo = Cosmology::default();
        let rate = compute_rate(&eff, 1000.0, 1.0, &cosmo);
        assert!((rate).abs() < 1e-10);
    }

    #[test]
    fn test_compute_rate_full_efficiency() {
        let cosmo = Cosmology::default();
        // With 100% efficiency, rate should be R_vol * V(z_max) * (Omega/4pi).
        let eff: Vec<(f64, f64)> = (0..=20)
            .map(|i| {
                let z = i as f64 * 0.01;
                (z, 1.0)
            })
            .collect();
        let rate = compute_rate(&eff, 1000.0, 4.0 * std::f64::consts::PI, &cosmo);
        // Should be ~ R_vol * V(0.2) ~ 1000 * ~1 Gpc^3 = ~1000 /yr.
        assert!(rate > 100.0);
    }

    #[test]
    fn test_recover_rate_roundtrip() {
        let cosmo = Cosmology::default();
        // Perfect efficiency out to z=0.2.
        let eff: Vec<(f64, f64)> = (0..=20)
            .map(|i| {
                let z = i as f64 * 0.01;
                (z, 1.0)
            })
            .collect();
        let omega = 4.0 * std::f64::consts::PI; // full sky
        let input_rate = 100.0; // Gpc^-3 yr^-1
        let t_survey = 10.0; // years
        let rate_per_yr = compute_rate(&eff, input_rate, omega, &cosmo);
        let n_det = rate_per_yr * t_survey;
        let recovery = recover_rate(&eff, n_det, t_survey, omega, input_rate, &cosmo);
        // Should recover exactly (no MC noise with analytic efficiency).
        assert!(
            (recovery.recovered_rate - input_rate).abs() < 1e-6,
            "Expected {}, got {}",
            input_rate,
            recovery.recovered_rate
        );
        assert!(recovery.consistent_1sig);
        assert!(recovery.consistent_2sig);
    }

    #[test]
    fn test_estimate_survey_omega() {
        let nside = 64;
        let total_pixels = 12 * nside as usize * nside as usize;
        let omega = estimate_survey_omega(total_pixels, nside);
        // Full sky should be 4pi.
        assert!((omega - 4.0 * std::f64::consts::PI).abs() < 0.01);
    }
}
