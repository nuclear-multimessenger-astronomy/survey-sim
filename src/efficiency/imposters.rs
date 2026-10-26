//! Kilonova imposter detection efficiency and rate recovery analysis.
//!
//! In wide-field time-domain surveys, transient candidates passing fast-transient
//! detection criteria include true kilonovae (KNe) as well as "imposter" transients
//! that mimic their early photometric evolution. Major astrophysical imposters
//! include:
//! - Type IIb core-collapse supernovae with early shock-cooling emission
//!   (Barna et al. 2025, PASP / arXiv:2506.15900; Salafia et al. 2026, arXiv:2605.02639)
//! - Fast Blue Optical Transients (FBOTs; Ho et al. 2021, arXiv:2105.08811)
//! - Fast-fading GRB afterglows (Andreoni et al. 2021, ZTFReST, arXiv:2104.06352)
//! - Galactic flare stars and cataclysmic variables
//!
//! Because the volumetric rate of core-collapse supernovae (~10^5 Gpc^-3 yr^-1)
//! exceeds the kilonova rate (~10-300 Gpc^-3 yr^-1) by orders of magnitude, even a
//! small false-positive selection fraction can substantially contaminate the
//! candidate sample. This module provides tools to analyze:
//!
//! 1. **Sample Purity & Contamination**: precision, false discovery rate (FDR),
//!    and imposter-to-signal ratio.
//! 2. **Effective / Pure Detection Efficiency**: efficiency penalized by impurity,
//!    F_beta harmonic scores, and Poisson significance figures of merit.
//! 3. **Follow-up Capacity Dilution**: fraction of true kilonovae confirmed when
//!    follow-up telescope resources (e.g., spectroscopic target budget) are constrained.
//! 4. **Rate Recovery under Contamination**: naive rate overestimation bias,
//!    background-subtracted rate recovery, and confidence interval widening
//!    (variance inflation factor) due to imposter Poisson + systematic fluctuations.

use std::fmt;
use serde::{Deserialize, Serialize};

use crate::efficiency::rates::{compute_rate, RateSummary};
use crate::pipeline::SimulationResult;
use crate::types::Cosmology;

/// Representation of an imposter transient population that contaminates the
/// kilonova search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImposterPopulation {
    /// Name or classification (e.g. "Type IIb SN (Shock Cooling)").
    pub name: String,
    /// Assumed volumetric rate in Gpc^-3 yr^-1 (if cosmological), or annual survey count.
    pub rate: f64,
    /// Fractional systematic uncertainty on the imposter rate (e.g. 0.20 for 20%).
    pub rate_uncertainty_frac: f64,
    /// Expected number of imposter detections passing selection criteria over the survey.
    pub expected_detections: f64,
    /// Effective volume-time (Gpc^3 yr) if computed from an efficiency curve.
    pub effective_vt: Option<f64>,
}

impl ImposterPopulation {
    /// Create an imposter population directly from an expected detection count.
    pub fn from_count(name: &str, expected_detections: f64, rate_uncertainty_frac: f64) -> Self {
        Self {
            name: name.to_string(),
            rate: 0.0,
            rate_uncertainty_frac: rate_uncertainty_frac.max(0.0),
            expected_detections: expected_detections.max(0.0),
            effective_vt: None,
        }
    }

    /// Create an imposter population from a volumetric rate and an efficiency vs redshift curve.
    pub fn from_efficiency(
        name: &str,
        volumetric_rate: f64,
        rate_uncertainty_frac: f64,
        efficiency_vs_z: &[(f64, f64)],
        survey_duration_years: f64,
        survey_omega_sr: f64,
        cosmo: &Cosmology,
    ) -> Self {
        let rate_per_unit = compute_rate(efficiency_vs_z, 1.0, survey_omega_sr, cosmo);
        let effective_vt = rate_per_unit * survey_duration_years;
        let expected_detections = volumetric_rate * effective_vt;

        Self {
            name: name.to_string(),
            rate: volumetric_rate,
            rate_uncertainty_frac: rate_uncertainty_frac.max(0.0),
            expected_detections,
            effective_vt: Some(effective_vt),
        }
    }

    /// Create an imposter population directly from a simulated `RateSummary`.
    pub fn from_rate_summary(summary: &RateSummary, rate_uncertainty_frac: f64) -> Self {
        let effective_vt = summary.recovery.as_ref().map(|r| r.effective_vt);
        Self {
            name: summary.transient_type.clone(),
            rate: summary.volumetric_rate,
            rate_uncertainty_frac: rate_uncertainty_frac.max(0.0),
            expected_detections: summary.detections_total,
            effective_vt,
        }
    }

    /// Standard preset: Type IIb supernova shock cooling kilonova imposters.
    ///
    /// Based on Barna et al. (2025, PASP / arXiv:2506.15900) and Salafia et al. (2026).
    /// Assumes a local Type IIb volumetric rate of ~1.2e3 Gpc^-3 yr^-1 (~12% of CCSNe)
    /// with an effective shock-cooling detection horizon out to z ≈ 0.08.
    pub fn type_iib_shock_cooling(
        survey_duration_years: f64,
        survey_omega_sr: f64,
        cosmo: &Cosmology,
    ) -> Self {
        // Approximate efficiency curve: high detection locally, declining by z~0.08.
        let eff_vs_z: Vec<(f64, f64)> = (0..=10)
            .map(|i| {
                let z = i as f64 * 0.01;
                let eff = (1.0 - z / 0.08).clamp(0.0, 1.0).powi(2);
                (z, eff * 0.08) // ~8% pass strict fast-fade kilonova cuts
            })
            .collect();

        Self::from_efficiency(
            "Type IIb SN (Shock Cooling)",
            1200.0, // Gpc^-3 yr^-1
            0.25,   // 25% rate uncertainty
            &eff_vs_z,
            survey_duration_years,
            survey_omega_sr,
            cosmo,
        )
    }

    /// Standard preset: Fast Blue Optical Transients (FBOTs).
    ///
    /// Based on Ho et al. (2021, arXiv:2105.08811). Assumes volumetric rate
    /// ~500 Gpc^-3 yr^-1 with rapid decay matching fast-transient criteria.
    pub fn fbot(
        survey_duration_years: f64,
        survey_omega_sr: f64,
        cosmo: &Cosmology,
    ) -> Self {
        let eff_vs_z: Vec<(f64, f64)> = (0..=15)
            .map(|i| {
                let z = i as f64 * 0.01;
                let eff = (1.0 - z / 0.15).clamp(0.0, 1.0);
                (z, eff * 0.12)
            })
            .collect();

        Self::from_efficiency(
            "Fast Blue Optical Transient (FBOT)",
            500.0, // Gpc^-3 yr^-1
            0.30,  // 30% rate uncertainty
            &eff_vs_z,
            survey_duration_years,
            survey_omega_sr,
            cosmo,
        )
    }

    /// Standard preset: Relativistic shocked cocoon emission from choked/structured jets.
    ///
    /// Based on Gottlieb et al. (2018) and Piro & Kollmeier (2018).
    /// Assumes a local rate of ~30 Gpc^-3 yr^-1 with rapid cooling and horizon to z ≈ 0.15.
    pub fn shocked_cocoon(
        survey_duration_years: f64,
        survey_omega_sr: f64,
        cosmo: &Cosmology,
    ) -> Self {
        let eff_vs_z: Vec<(f64, f64)> = (0..=15)
            .map(|i| {
                let z = i as f64 * 0.01;
                let eff = (1.0 - z / 0.15).clamp(0.0, 1.0);
                (z, eff * 0.15)
            })
            .collect();

        Self::from_efficiency(
            "Relativistic Shocked Cocoon",
            30.0, // Gpc^-3 yr^-1
            0.35, // 35% rate uncertainty
            &eff_vs_z,
            survey_duration_years,
            survey_omega_sr,
            cosmo,
        )
    }

    /// Standard preset: Subluminous Type Iax supernovae.
    ///
    /// Based on Foley et al. (2013) and Jha (2017).
    /// Assumes a local rate of ~3000 Gpc^-3 yr^-1 (~15-30% of normal SNIa rate).
    pub fn sn_iax(
        survey_duration_years: f64,
        survey_omega_sr: f64,
        cosmo: &Cosmology,
    ) -> Self {
        let eff_vs_z: Vec<(f64, f64)> = (0..=10)
            .map(|i| {
                let z = i as f64 * 0.01;
                let eff = (1.0 - z / 0.06).clamp(0.0, 1.0).powi(2);
                (z, eff * 0.03) // ~3% pass strict rapid-fade kilonova cuts
            })
            .collect();

        Self::from_efficiency(
            "Type Iax Supernova",
            3000.0, // Gpc^-3 yr^-1
            0.30,   // 30% rate uncertainty
            &eff_vs_z,
            survey_duration_years,
            survey_omega_sr,
            cosmo,
        )
    }

    /// Standard preset: Dense circumstellar interaction (CSM) transients.
    ///
    /// Based on Chevalier & Irwin (2011) and Chatzopoulos et al. (2012).
    /// Assumes a local rate of ~250 Gpc^-3 yr^-1.
    pub fn csm_interaction(
        survey_duration_years: f64,
        survey_omega_sr: f64,
        cosmo: &Cosmology,
    ) -> Self {
        let eff_vs_z: Vec<(f64, f64)> = (0..=15)
            .map(|i| {
                let z = i as f64 * 0.01;
                let eff = (1.0 - z / 0.12).clamp(0.0, 1.0);
                (z, eff * 0.10)
            })
            .collect();

        Self::from_efficiency(
            "CSM Interaction Transient",
            250.0, // Gpc^-3 yr^-1
            0.30,  // 30% rate uncertainty
            &eff_vs_z,
            survey_duration_years,
            survey_omega_sr,
            cosmo,
        )
    }

    /// Standard preset: Orphan (off-axis) GRB optical afterglows.
    ///
    /// Based on Ghirlanda et al. (2015), Andreoni et al. (2021, ZTFReST / arXiv:2104.06352),
    /// and Sagués-Carracedo et al. (2021). Relativistic jets viewed off-axis (theta_v > theta_j)
    /// decelerate and spread, producing power-law optical decays (t^-alpha) that mimic kilonovae.
    /// Assumes an all-jet volumetric rate of ~100 Gpc^-3 yr^-1 with detectable emission to z ≈ 0.20.
    pub fn orphan_afterglow(
        survey_duration_years: f64,
        survey_omega_sr: f64,
        cosmo: &Cosmology,
    ) -> Self {
        let eff_vs_z: Vec<(f64, f64)> = (0..=20)
            .map(|i| {
                let z = i as f64 * 0.01;
                let eff = (1.0 - z / 0.20).clamp(0.0, 1.0);
                (z, eff * 0.05) // ~5% pass rapid kilonova color & fade criteria
            })
            .collect();

        Self::from_efficiency(
            "Orphan GRB Afterglow",
            100.0, // Gpc^-3 yr^-1
            0.40,  // 40% rate uncertainty
            &eff_vs_z,
            survey_duration_years,
            survey_omega_sr,
            cosmo,
        )
    }

    /// Standard preset: Empirical fast-transient contaminants (e.g., ZTFReST-like alert broker false positives).
    ///
    /// Based on Andreoni et al. (2021, arXiv:2104.06352) and Stevenson et al. (2025, arXiv:2510.12932).
    /// `annual_rate` is the number of unveted fast-transient candidates per year across the survey.
    pub fn empirical_fast_transients(
        name: &str,
        annual_rate: f64,
        survey_duration_years: f64,
        rate_uncertainty_frac: f64,
    ) -> Self {
        Self::from_count(name, annual_rate * survey_duration_years, rate_uncertainty_frac)
    }
}

/// Target kilonova signal properties.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KilonovaSignal {
    /// Population name (e.g. "BNS Kilonova").
    pub name: String,
    /// Assumed volumetric rate in Gpc^-3 yr^-1.
    pub volumetric_rate: f64,
    /// Expected total kilonova detections over the survey duration.
    pub expected_detections: f64,
    /// Effective volume-time (Gpc^3 yr).
    pub effective_vt: f64,
    /// Overall detection efficiency (fraction of injected transients detected).
    pub overall_efficiency: f64,
}

impl KilonovaSignal {
    /// Create from a simulated `RateSummary`.
    pub fn from_rate_summary(summary: &RateSummary) -> Self {
        let effective_vt = summary
            .recovery
            .as_ref()
            .map(|r| r.effective_vt)
            .unwrap_or_else(|| {
                if summary.volumetric_rate > 0.0 {
                    summary.detections_total / summary.volumetric_rate
                } else {
                    0.0
                }
            });

        Self {
            name: summary.transient_type.clone(),
            volumetric_rate: summary.volumetric_rate,
            expected_detections: summary.detections_total,
            effective_vt,
            overall_efficiency: summary.overall_efficiency,
        }
    }

    /// Create from an efficiency curve and survey parameters.
    pub fn from_efficiency(
        name: &str,
        volumetric_rate: f64,
        efficiency_vs_z: &[(f64, f64)],
        survey_duration_years: f64,
        survey_omega_sr: f64,
        cosmo: &Cosmology,
    ) -> Self {
        let rate_per_unit = compute_rate(efficiency_vs_z, 1.0, survey_omega_sr, cosmo);
        let effective_vt = rate_per_unit * survey_duration_years;
        let expected_detections = volumetric_rate * effective_vt;

        // Approximate overall efficiency as average efficiency over non-zero bins
        let non_zero_effs: Vec<f64> = efficiency_vs_z
            .iter()
            .map(|&(_, eff)| eff)
            .filter(|&eff| eff > 0.0)
            .collect();
        let overall_efficiency = if !non_zero_effs.is_empty() {
            non_zero_effs.iter().sum::<f64>() / non_zero_effs.len() as f64
        } else {
            0.0
        };

        Self {
            name: name.to_string(),
            volumetric_rate,
            expected_detections,
            effective_vt,
            overall_efficiency,
        }
    }

    /// Direct manual constructor.
    pub fn new(
        name: &str,
        volumetric_rate: f64,
        expected_detections: f64,
        effective_vt: f64,
        overall_efficiency: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            volumetric_rate,
            expected_detections,
            effective_vt,
            overall_efficiency,
        }
    }
}

/// Imposter contamination and purity metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImposterMetrics {
    /// Expected detection counts per imposter class.
    pub imposter_breakdown: Vec<(String, f64)>,
    /// Total expected imposter detections (N_imp).
    pub total_imposters: f64,
    /// Expected true kilonova detections (N_KN).
    pub signal_detections: f64,
    /// Total candidates passing cuts: N_obs = N_KN + N_imp.
    pub total_detections: f64,
    /// Imposter-to-signal ratio: r_IS = N_imp / N_KN.
    pub imposter_to_signal_ratio: f64,
    /// Sample purity (precision): P = N_KN / (N_KN + N_imp).
    pub purity: f64,
    /// False discovery rate: FDR = 1 - P = N_imp / (N_KN + N_imp).
    pub false_discovery_rate: f64,
    /// Pure / effective efficiency: ε_pure = ε_KN × P.
    pub pure_efficiency: f64,
    /// F1 score: harmonic mean of efficiency (recall) and purity (precision).
    pub f1_score: f64,
    /// Poisson signal significance above imposter background: S / sqrt(S + B).
    pub significance_fom: f64,
}

impl ImposterMetrics {
    /// Compute F_beta score with customizable beta weighting.
    ///
    /// beta > 1 weights efficiency (recall) more heavily.
    /// beta < 1 weights purity (precision) more heavily.
    pub fn f_beta(&self, beta: f64) -> f64 {
        let beta_sq = beta * beta;
        let p = self.purity;
        let r = self.pure_efficiency / self.purity.max(1e-12); // recovers raw efficiency
        let denom = beta_sq * p + r;
        if denom > 0.0 {
            (1.0 + beta_sq) * (p * r) / denom
        } else {
            0.0
        }
    }
}

/// Follow-up telescope resource constraint metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FollowupBudgetMetrics {
    /// Maximum follow-up target capacity (e.g. spectroscopic triggers over the survey).
    pub budget: f64,
    /// Total candidate alerts passing detection criteria: N_total = N_KN + N_imp.
    pub total_candidates: f64,
    /// Fraction of candidates that can be followed up: min(1.0, budget / total_candidates).
    pub allocation_fraction: f64,
    /// Expected number of true kilonovae successfully followed up and confirmed.
    pub expected_confirmed_kn: f64,
    /// Confirmed kilonova detection efficiency: ε_KN × allocation_fraction.
    pub confirmed_efficiency: f64,
    /// Follow-up slots consumed by imposter transients.
    pub wasted_imposter_followups: f64,
    /// Whether the follow-up capacity is exceeded.
    pub is_budget_exceeded: bool,
}

/// Rate recovery analysis under imposter contamination.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContaminatedRateRecovery {
    /// True injected kilonova volumetric rate in Gpc^-3 yr^-1.
    pub input_rate: f64,
    /// Effective volume-time (Gpc^3 yr).
    pub effective_vt: f64,
    /// Observed candidate detections: N_obs = N_KN + N_imp.
    pub observed_candidates: f64,
    /// Expected imposter detections: N_imp.
    pub expected_imposters: f64,
    /// Systematic uncertainty on imposter count: sqrt(sum(f_i * N_i)^2).
    pub imposter_systematic_uncertainty: f64,
    /// Naive recovered rate (ignoring imposters): N_obs / (VT)_eff.
    pub naive_recovered_rate: f64,
    /// Absolute rate overestimation bias: N_imp / (VT)_eff.
    pub rate_bias_absolute: f64,
    /// Fractional rate bias: N_imp / N_KN = r_IS.
    pub rate_bias_fractional: f64,
    /// Naive 1-sigma Poisson interval (assuming 0 background, demonstrating false precision).
    pub naive_poisson_1sig_interval: (f64, f64),
    /// Decontaminated (background-subtracted) recovered rate: max(0, N_obs - N_imp) / (VT)_eff.
    pub decontaminated_recovered_rate: f64,
    /// 1-sigma standard deviation on net signal count: sqrt(N_obs + N_imp + sigma_sys^2).
    pub decontaminated_count_sigma: f64,
    /// 1-sigma confidence interval on decontaminated rate.
    pub decontaminated_1sig_interval: (f64, f64),
    /// 2-sigma confidence interval on decontaminated rate.
    pub decontaminated_2sig_interval: (f64, f64),
    /// Variance inflation factor: Var(N_net) / Var(N_clean) = (N_obs + N_imp + sigma_sys^2) / N_KN.
    pub variance_inflation_factor: f64,
    /// Uncertainty inflation factor: sqrt(VIF).
    pub uncertainty_inflation_factor: f64,
    /// Whether true input rate is within decontaminated 1-sigma interval.
    pub consistent_1sig: bool,
    /// Whether true input rate is within decontaminated 2-sigma interval.
    pub consistent_2sig: bool,
}

/// Comprehensive analysis of kilonova detection efficiency in the presence of imposters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KilonovaImposterAnalysis {
    /// Target kilonova population properties.
    pub kilonova: KilonovaSignal,
    /// Imposter populations included in the analysis.
    pub imposters: Vec<ImposterPopulation>,
    /// Purity, contamination, and effective efficiency metrics.
    pub metrics: ImposterMetrics,
    /// Rate recovery and bias analysis.
    pub rate_recovery: ContaminatedRateRecovery,
    /// Optional follow-up budget constraint metrics.
    pub followup_budget: Option<FollowupBudgetMetrics>,
}

impl KilonovaImposterAnalysis {
    /// Perform the analysis from a target kilonova and a list of imposter populations.
    pub fn new(kilonova: KilonovaSignal, imposters: Vec<ImposterPopulation>) -> Self {
        let n_kn = kilonova.expected_detections.max(0.0);
        let mut n_imp_total = 0.0;
        let mut imposter_breakdown = Vec::new();
        let mut var_sys_total = 0.0;

        for imp in &imposters {
            let n_det = imp.expected_detections.max(0.0);
            imposter_breakdown.push((imp.name.clone(), n_det));
            n_imp_total += n_det;
            let sigma_sys = imp.rate_uncertainty_frac * n_det;
            var_sys_total += sigma_sys * sigma_sys;
        }

        let n_total = n_kn + n_imp_total;
        let r_is = if n_kn > 0.0 { n_imp_total / n_kn } else { f64::INFINITY };
        let purity = if n_total > 0.0 { n_kn / n_total } else { 1.0 };
        let fdr = 1.0 - purity;
        let pure_efficiency = kilonova.overall_efficiency * purity;

        let f1_score = if purity + kilonova.overall_efficiency > 0.0 {
            2.0 * (purity * kilonova.overall_efficiency) / (purity + kilonova.overall_efficiency)
        } else {
            0.0
        };

        let significance_fom = if n_total > 0.0 {
            n_kn / n_total.sqrt()
        } else {
            0.0
        };

        let metrics = ImposterMetrics {
            imposter_breakdown,
            total_imposters: n_imp_total,
            signal_detections: n_kn,
            total_detections: n_total,
            imposter_to_signal_ratio: r_is,
            purity,
            false_discovery_rate: fdr,
            pure_efficiency,
            f1_score,
            significance_fom,
        };

        // Rate recovery computations
        let vt = kilonova.effective_vt;
        let naive_recovered_rate = if vt > 0.0 { n_total / vt } else { 0.0 };
        let rate_bias_absolute = if vt > 0.0 { n_imp_total / vt } else { 0.0 };
        let rate_bias_fractional = r_is;

        // Naive Poisson interval (sigma = R / sqrt(N_total))
        let naive_sigma_r = if n_total > 0.0 && vt > 0.0 {
            n_total.sqrt() / vt
        } else {
            0.0
        };
        let naive_poisson_1sig_interval = (
            (naive_recovered_rate - naive_sigma_r).max(0.0),
            naive_recovered_rate + naive_sigma_r,
        );

        // Decontaminated rate: net signal count N_net = N_total - N_imp = N_KN
        let decontaminated_recovered_rate = if vt > 0.0 {
            (n_total - n_imp_total).max(0.0) / vt
        } else {
            0.0
        };

        // Net variance: Var(N_obs) + Var(N_imp) = N_obs + N_imp + sigma_sys^2
        let count_variance = n_total + n_imp_total + var_sys_total;
        let decontaminated_count_sigma = count_variance.sqrt();
        let decontam_sigma_r = if vt > 0.0 {
            decontaminated_count_sigma / vt
        } else {
            0.0
        };

        let decontaminated_1sig_interval = (
            (decontaminated_recovered_rate - decontam_sigma_r).max(0.0),
            decontaminated_recovered_rate + decontam_sigma_r,
        );
        let decontaminated_2sig_interval = (
            (decontaminated_recovered_rate - 2.0 * decontam_sigma_r).max(0.0),
            decontaminated_recovered_rate + 2.0 * decontam_sigma_r,
        );

        let variance_inflation_factor = if n_kn > 0.0 {
            count_variance / n_kn
        } else {
            f64::INFINITY
        };
        let uncertainty_inflation_factor = variance_inflation_factor.sqrt();

        let consistent_1sig = kilonova.volumetric_rate >= decontaminated_1sig_interval.0
            && kilonova.volumetric_rate <= decontaminated_1sig_interval.1;
        let consistent_2sig = kilonova.volumetric_rate >= decontaminated_2sig_interval.0
            && kilonova.volumetric_rate <= decontaminated_2sig_interval.1;

        let rate_recovery = ContaminatedRateRecovery {
            input_rate: kilonova.volumetric_rate,
            effective_vt: vt,
            observed_candidates: n_total,
            expected_imposters: n_imp_total,
            imposter_systematic_uncertainty: var_sys_total.sqrt(),
            naive_recovered_rate,
            rate_bias_absolute,
            rate_bias_fractional,
            naive_poisson_1sig_interval,
            decontaminated_recovered_rate,
            decontaminated_count_sigma,
            decontaminated_1sig_interval,
            decontaminated_2sig_interval,
            variance_inflation_factor,
            uncertainty_inflation_factor,
            consistent_1sig,
            consistent_2sig,
        };

        Self {
            kilonova,
            imposters,
            metrics,
            rate_recovery,
            followup_budget: None,
        }
    }

    /// Add a telescope follow-up budget constraint to the analysis.
    pub fn with_followup_budget(mut self, budget: f64) -> Self {
        let n_kn = self.kilonova.expected_detections;
        let n_total = self.metrics.total_detections;
        let allocation_fraction = if n_total > 0.0 {
            (budget / n_total).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let expected_confirmed_kn = n_kn * allocation_fraction;
        let confirmed_efficiency = self.kilonova.overall_efficiency * allocation_fraction;
        let wasted_imposter_followups = self.metrics.total_imposters * allocation_fraction;

        self.followup_budget = Some(FollowupBudgetMetrics {
            budget,
            total_candidates: n_total,
            allocation_fraction,
            expected_confirmed_kn,
            confirmed_efficiency,
            wasted_imposter_followups,
            is_budget_exceeded: n_total > budget,
        });

        self
    }

    /// Perform imposter analysis directly from a slice of `RateSummary` outputs.
    ///
    /// Identifies the signal population matching `signal_type` and treats all other
    /// summaries as contaminating imposter populations with `imposter_uncertainty_frac`.
    pub fn from_rate_summaries(
        rate_summaries: &[RateSummary],
        signal_type: &str,
        imposter_uncertainty_frac: f64,
    ) -> Option<Self> {
        let signal_summary = rate_summaries
            .iter()
            .find(|s| s.transient_type == signal_type)?;

        let signal = KilonovaSignal::from_rate_summary(signal_summary);

        let imposters: Vec<ImposterPopulation> = rate_summaries
            .iter()
            .filter(|s| s.transient_type != signal_type)
            .map(|s| ImposterPopulation::from_rate_summary(s, imposter_uncertainty_frac))
            .collect();

        Some(Self::new(signal, imposters))
    }

    /// Perform imposter analysis directly from an end-to-end `SimulationResult`.
    pub fn from_simulation_result(
        result: &SimulationResult,
        signal_type: &str,
        imposter_uncertainty_frac: f64,
    ) -> Option<Self> {
        Self::from_rate_summaries(&result.rate_summaries, signal_type, imposter_uncertainty_frac)
    }
}

impl fmt::Display for KilonovaImposterAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Kilonova Imposter & Detection Efficiency Analysis ===")?;
        writeln!(f, "Target Signal: {}", self.kilonova.name)?;
        writeln!(
            f,
            "  Input Rate:        {:.1} Gpc^-3 yr^-1",
            self.kilonova.volumetric_rate
        )?;
        writeln!(
            f,
            "  Effective (VT):    {:.4} Gpc^3 yr",
            self.kilonova.effective_vt
        )?;
        writeln!(
            f,
            "  Expected Signal:   {:.2} events",
            self.metrics.signal_detections
        )?;
        writeln!(
            f,
            "  Raw Detection Eff: {:.2}%",
            self.kilonova.overall_efficiency * 100.0
        )?;
        writeln!(f)?;

        writeln!(f, "Imposter Contamination:")?;
        for (name, count) in &self.metrics.imposter_breakdown {
            writeln!(f, "  - {:<28} {:.2} events", name, count)?;
        }
        writeln!(
            f,
            "  Total Imposters:   {:.2} events",
            self.metrics.total_imposters
        )?;
        writeln!(
            f,
            "  Total Candidates:  {:.2} events",
            self.metrics.total_detections
        )?;
        writeln!(
            f,
            "  Imposter/Signal:   {:.2}x",
            self.metrics.imposter_to_signal_ratio
        )?;
        writeln!(
            f,
            "  Sample Purity:     {:.1}%",
            self.metrics.purity * 100.0
        )?;
        writeln!(
            f,
            "  False Discovery:   {:.1}%",
            self.metrics.false_discovery_rate * 100.0
        )?;
        writeln!(
            f,
            "  Pure Detection Eff:{:.2}%",
            self.metrics.pure_efficiency * 100.0
        )?;
        writeln!(
            f,
            "  F1 Score:          {:.3}",
            self.metrics.f1_score
        )?;
        writeln!(
            f,
            "  Poisson FOM:       {:.2} sigma",
            self.metrics.significance_fom
        )?;
        writeln!(f)?;

        writeln!(f, "Rate Recovery Impact:")?;
        writeln!(
            f,
            "  Naive Recovered:   {:.2} Gpc^-3 yr^-1 (bias: +{:.2} Gpc^-3 yr^-1, +{:.1}%)",
            self.rate_recovery.naive_recovered_rate,
            self.rate_recovery.rate_bias_absolute,
            self.rate_recovery.rate_bias_fractional * 100.0
        )?;
        writeln!(
            f,
            "  Naive 1-sig Range: [{:.2}, {:.2}] (false overconfidence)",
            self.rate_recovery.naive_poisson_1sig_interval.0,
            self.rate_recovery.naive_poisson_1sig_interval.1
        )?;
        writeln!(
            f,
            "  Decontaminated:    {:.2} Gpc^-3 yr^-1",
            self.rate_recovery.decontaminated_recovered_rate
        )?;
        writeln!(
            f,
            "  Decontam 1-sig:    [{:.2}, {:.2}] (consistent: {})",
            self.rate_recovery.decontaminated_1sig_interval.0,
            self.rate_recovery.decontaminated_1sig_interval.1,
            if self.rate_recovery.consistent_1sig { "yes" } else { "no" }
        )?;
        writeln!(
            f,
            "  Decontam 2-sig:    [{:.2}, {:.2}] (consistent: {})",
            self.rate_recovery.decontaminated_2sig_interval.0,
            self.rate_recovery.decontaminated_2sig_interval.1,
            if self.rate_recovery.consistent_2sig { "yes" } else { "no" }
        )?;
        writeln!(
            f,
            "  Variance Inflation:{:.2}x (uncertainty widened by {:.2}x)",
            self.rate_recovery.variance_inflation_factor,
            self.rate_recovery.uncertainty_inflation_factor
        )?;

        if let Some(ref fb) = self.followup_budget {
            writeln!(f)?;
            writeln!(f, "Follow-up Resource Budget:")?;
            writeln!(f, "  Follow-up Budget:  {:.0} targets", fb.budget)?;
            writeln!(
                f,
                "  Budget Exceeded:   {}",
                if fb.is_budget_exceeded { "YES" } else { "NO" }
            )?;
            writeln!(
                f,
                "  Allocation Frac:   {:.1}%",
                fb.allocation_fraction * 100.0
            )?;
            writeln!(
                f,
                "  Confirmed KNe:     {:.2} / {:.2} events",
                fb.expected_confirmed_kn,
                self.metrics.signal_detections
            )?;
            writeln!(
                f,
                "  Confirmed Eff:     {:.2}%",
                fb.confirmed_efficiency * 100.0
            )?;
            writeln!(
                f,
                "  Wasted Imposter:   {:.2} triggers",
                fb.wasted_imposter_followups
            )?;
        }

        Ok(())
    }
}

/// Analyze simulation results containing kilonova and imposter populations.
///
/// Looks up `kn_type` and each name in `imposter_types` from `result.rate_summaries`.
pub fn analyze_simulation_imposters(
    result: &SimulationResult,
    kn_type: &str,
    imposter_types: &[&str],
    imposter_uncertainty_frac: f64,
) -> Option<KilonovaImposterAnalysis> {
    let kn_summary = result
        .rate_summaries
        .iter()
        .find(|s| s.transient_type == kn_type)?;

    let kn_signal = KilonovaSignal::from_rate_summary(kn_summary);

    let mut imposters = Vec::new();
    for &imp_name in imposter_types {
        if let Some(imp_summary) = result
            .rate_summaries
            .iter()
            .find(|s| s.transient_type == imp_name)
        {
            imposters.push(ImposterPopulation::from_rate_summary(
                imp_summary,
                imposter_uncertainty_frac,
            ));
        }
    }

    Some(KilonovaImposterAnalysis::new(kn_signal, imposters))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_imposters_ideal_case() {
        let kn = KilonovaSignal::new("BNS KN", 50.0, 10.0, 0.20, 0.40);
        let analysis = KilonovaImposterAnalysis::new(kn, Vec::new());

        assert_eq!(analysis.metrics.total_imposters, 0.0);
        assert_eq!(analysis.metrics.imposter_to_signal_ratio, 0.0);
        assert_eq!(analysis.metrics.purity, 1.0);
        assert_eq!(analysis.metrics.false_discovery_rate, 0.0);
        assert_eq!(analysis.metrics.pure_efficiency, 0.40);
        assert_eq!(analysis.rate_recovery.rate_bias_absolute, 0.0);
        assert_eq!(analysis.rate_recovery.rate_bias_fractional, 0.0);
        assert_eq!(analysis.rate_recovery.decontaminated_recovered_rate, 50.0);
        assert!((analysis.rate_recovery.variance_inflation_factor - 1.0).abs() < 1e-6);
        assert!(analysis.rate_recovery.consistent_1sig);
    }

    #[test]
    fn test_imposter_contamination_and_bias() {
        // 10 real KNe, 30 imposters (e.g. Type IIb shock cooling + FBOTs)
        let kn = KilonovaSignal::new("BNS KN", 50.0, 10.0, 0.20, 0.40);
        let imp1 = ImposterPopulation::from_count("Type IIb Shock Cooling", 20.0, 0.20);
        let imp2 = ImposterPopulation::from_count("FBOT", 10.0, 0.30);

        let analysis = KilonovaImposterAnalysis::new(kn, vec![imp1, imp2]);

        assert_eq!(analysis.metrics.total_imposters, 30.0);
        assert_eq!(analysis.metrics.total_detections, 40.0);
        assert_eq!(analysis.metrics.imposter_to_signal_ratio, 3.0); // 3 imposters per kilonova
        assert_eq!(analysis.metrics.purity, 0.25); // 10/40 = 25% purity
        assert_eq!(analysis.metrics.false_discovery_rate, 0.75); // 75% FDR
        assert_eq!(analysis.metrics.pure_efficiency, 0.10); // 40% * 25% = 10% pure efficiency

        // Naive recovery recovers (40 / 0.20) = 200 Gpc^-3 yr^-1 (4x true rate 50!)
        assert_eq!(analysis.rate_recovery.naive_recovered_rate, 200.0);
        assert_eq!(analysis.rate_recovery.rate_bias_absolute, 150.0);
        assert_eq!(analysis.rate_recovery.rate_bias_fractional, 3.0);

        // Decontaminated recovers exact true rate 50.0
        assert_eq!(analysis.rate_recovery.decontaminated_recovered_rate, 50.0);

        // Variance inflation: Var(N_net) = N_obs (40) + N_imp (30) + (0.2*20)^2 + (0.3*10)^2 = 70 + 16 + 9 = 95
        // VIF = 95 / 10 = 9.5
        assert!((analysis.rate_recovery.variance_inflation_factor - 9.5).abs() < 1e-4);
        assert!((analysis.rate_recovery.uncertainty_inflation_factor - 9.5_f64.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn test_followup_budget_constraint() {
        let kn = KilonovaSignal::new("BNS KN", 50.0, 20.0, 0.40, 0.50);
        let imp = ImposterPopulation::from_count("Imposters", 80.0, 0.20);

        // Total candidates = 100
        let analysis = KilonovaImposterAnalysis::new(kn, vec![imp])
            .with_followup_budget(25.0); // Only budget for 25 follow-up spectra!

        let fb = analysis.followup_budget.unwrap();
        assert!(fb.is_budget_exceeded);
        assert_eq!(fb.allocation_fraction, 0.25);
        // Only 25% of 20 = 5 true KNe get confirmed!
        assert_eq!(fb.expected_confirmed_kn, 5.0);
        // Confirmed efficiency drops from 50% to 12.5%!
        assert_eq!(fb.confirmed_efficiency, 0.125);
        // 20 follow-up slots wasted on imposters
        assert_eq!(fb.wasted_imposter_followups, 20.0);
    }

    #[test]
    fn test_followup_budget_unconstrained() {
        let kn = KilonovaSignal::new("BNS KN", 50.0, 10.0, 0.20, 0.40);
        let imp = ImposterPopulation::from_count("Imposters", 15.0, 0.10);

        // Total candidates = 25, budget = 50 (plenty)
        let analysis = KilonovaImposterAnalysis::new(kn, vec![imp])
            .with_followup_budget(50.0);

        let fb = analysis.followup_budget.unwrap();
        assert!(!fb.is_budget_exceeded);
        assert_eq!(fb.allocation_fraction, 1.0);
        assert_eq!(fb.expected_confirmed_kn, 10.0);
        assert_eq!(fb.confirmed_efficiency, 0.40);
        assert_eq!(fb.wasted_imposter_followups, 15.0);
    }

    #[test]
    fn test_f_beta_scoring() {
        let kn = KilonovaSignal::new("BNS KN", 50.0, 10.0, 0.20, 0.50);
        let imp = ImposterPopulation::from_count("Imposters", 10.0, 0.10);
        // Purity = 10/20 = 0.5, raw recall = 0.5
        let analysis = KilonovaImposterAnalysis::new(kn, vec![imp]);

        assert!((analysis.metrics.purity - 0.5).abs() < 1e-6);
        assert!((analysis.metrics.f1_score - 0.5).abs() < 1e-6);
        assert!((analysis.metrics.f_beta(1.0) - 0.5).abs() < 1e-6);
        assert!((analysis.metrics.f_beta(0.5) - 0.5).abs() < 1e-6);
        assert!((analysis.metrics.f_beta(2.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_from_rate_summaries() {
        let rs_kn = RateSummary {
            transient_type: "Kilonova".to_string(),
            volumetric_rate: 100.0,
            detections_per_year: 5.0,
            detections_total: 15.0,
            overall_efficiency: 0.15,
            survey_omega_sr: 1.0,
            z_max: 0.2,
            recovery: None,
        };
        let rs_fbot = RateSummary {
            transient_type: "FBOT".to_string(),
            volumetric_rate: 500.0,
            detections_per_year: 10.0,
            detections_total: 30.0,
            overall_efficiency: 0.05,
            survey_omega_sr: 1.0,
            z_max: 0.4,
            recovery: None,
        };

        let summaries = vec![rs_kn, rs_fbot];
        let analysis = KilonovaImposterAnalysis::from_rate_summaries(&summaries, "Kilonova", 0.20)
            .expect("should construct analysis");

        assert_eq!(analysis.kilonova.name, "Kilonova");
        assert_eq!(analysis.imposters.len(), 1);
        assert_eq!(analysis.imposters[0].name, "FBOT");
        assert_eq!(analysis.metrics.signal_detections, 15.0);
        assert_eq!(analysis.metrics.total_imposters, 30.0);
        assert_eq!(analysis.metrics.total_detections, 45.0);
        assert!((analysis.metrics.purity - 15.0 / 45.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_efficiency_curve() {
        let cosmo = Cosmology::default();
        let eff_vs_z = vec![(0.0, 1.0), (0.05, 0.8), (0.1, 0.0)];
        let imp = ImposterPopulation::from_efficiency(
            "Test Imposter",
            1000.0,
            0.20,
            &eff_vs_z,
            1.0,
            1.0,
            &cosmo,
        );

        assert!(imp.expected_detections > 0.0);
        assert!(imp.effective_vt.is_some());
    }

    #[test]
    fn test_display_output() {
        let kn = KilonovaSignal::new("BNS KN", 50.0, 10.0, 0.20, 0.40);
        let imp = ImposterPopulation::from_count("Type IIb Shock Cooling", 15.0, 0.20);
        let analysis = KilonovaImposterAnalysis::new(kn, vec![imp])
            .with_followup_budget(20.0);

        let disp = format!("{}", analysis);
        assert!(disp.contains("=== Kilonova Imposter & Detection Efficiency Analysis ==="));
        assert!(disp.contains("Sample Purity"));
        assert!(disp.contains("Rate Recovery Impact"));
        assert!(disp.contains("Follow-up Resource Budget"));
    }
}
