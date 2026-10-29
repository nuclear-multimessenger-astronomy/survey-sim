use std::collections::HashMap;

use lightcurve_fitting::SviModelName;

use crate::types::{Band, TransientInstance, TransientType};

use super::cosmology::{extinction_in_band, k_correction_bolometric};
use super::{LightcurveEvaluation, LightcurveModel, Result};

/// Speed of light in cm/s (for wavelength → frequency conversion).
const C_CMS: f64 = 2.998e10;
/// Mpc in cm.
const MPC_CM: f64 = 3.0857e24;

/// Map band short name to effective wavelength in Angstroms.
fn band_wavelength_angstrom(band: &str) -> Option<f64> {
    match band {
        // UV bands (from m4opt instrument configs)
        "FUV" => Some(1600.0),  // UVEX FUV: 160nm
        "NUV" => Some(2600.0),  // ULTRASAT NUV: 260nm (also UVEX NUV at 230nm)
        // Optical/NIR
        "u" => Some(3671.0),
        "g" => Some(4770.0),
        "r" => Some(6231.0),
        "i" => Some(7625.0),
        "z" => Some(8691.0),
        "y" | "Y" => Some(9712.0),
        // LSST names
        "lsstu" => Some(3671.0),
        "lsstg" => Some(4827.0),
        "lsstr" => Some(6223.0),
        "lssti" => Some(7546.0),
        "lsstz" => Some(8691.0),
        "lssty" => Some(9712.0),
        _ => None,
    }
}

/// Maps TransientType to the appropriate SviModelName for lightcurve-fitting.
fn default_model_for_type(
    transient_type: crate::types::TransientType,
) -> SviModelName {
    use crate::types::TransientType;
    match transient_type {
        TransientType::Kilonova => SviModelName::MetzgerKN,
        TransientType::SupernovaIa => SviModelName::Bazin,
        TransientType::SupernovaII => SviModelName::Villar,
        TransientType::SupernovaIbc => SviModelName::Villar,
        TransientType::SupernovaIIb => SviModelName::Villar,
        TransientType::SupernovaIax => SviModelName::Villar,
        TransientType::CsmInteraction => SviModelName::Villar,
        TransientType::ShockedCocoon => SviModelName::Bazin,
        TransientType::Tde => SviModelName::Tde,
        TransientType::Fbot => SviModelName::Bazin,
        TransientType::Afterglow => SviModelName::Afterglow,
        TransientType::Custom => SviModelName::Bazin,
    }
}

/// Parametric lightcurve model wrapping lightcurve-fitting's eval_model_flux.
///
/// For MetzgerKN: uses `metzger_kn_mags` which computes per-band apparent AB
/// magnitudes via blackbody emission (temperature from 1-zone ODE).
///
/// For other models: evaluates normalized flux via `eval_model_flux`, then
/// converts to apparent magnitudes using peak_abs_mag + distance modulus.
pub struct ParametricModel {
    /// Override the default model name (otherwise inferred from TransientType).
    pub model_override: Option<SviModelName>,
    /// Per-band color offsets relative to the reference band (magnitudes).
    pub color_offsets: HashMap<String, f64>,
    /// Optional override for maximum rest-frame phase (days) before clamping to 99.0 mag.
    pub max_phase_days: Option<f64>,
}

impl ParametricModel {
    pub fn new() -> Self {
        Self {
            model_override: None,
            color_offsets: HashMap::new(),
            max_phase_days: None,
        }
    }

    pub fn with_model(mut self, model: SviModelName) -> Self {
        self.model_override = Some(model);
        self
    }

    pub fn with_color_offset(mut self, band: &str, offset: f64) -> Self {
        self.color_offsets.insert(band.to_string(), offset);
        self
    }

    pub fn with_max_phase(mut self, max_phase_days: f64) -> Self {
        self.max_phase_days = Some(max_phase_days);
        self
    }

    pub fn default_max_phase_for_model(model: SviModelName) -> f64 {
        match model {
            SviModelName::MetzgerKN => 25.0,
            SviModelName::Tde => 365.0,
            SviModelName::Afterglow => 100.0,
            SviModelName::Bazin | SviModelName::Villar | SviModelName::Arnett => 90.0,
            SviModelName::Magnetar | SviModelName::ShockCooling => 60.0,
        }
    }

    pub fn default_max_phase_for_type(transient_type: TransientType) -> f64 {
        Self::default_max_phase_for_model(default_model_for_type(transient_type))
    }

    pub fn resolved_max_phase(&self, model: SviModelName) -> f64 {
        self.max_phase_days
            .unwrap_or_else(|| Self::default_max_phase_for_model(model))
    }

    pub fn resolved_max_phase_for_type(&self, transient_type: TransientType) -> f64 {
        let model = self
            .model_override
            .unwrap_or_else(|| default_model_for_type(transient_type));
        self.resolved_max_phase(model)
    }
}

impl Default for ParametricModel {
    fn default() -> Self {
        Self::new()
    }
}

impl LightcurveModel for ParametricModel {
    fn evaluate(
        &self,
        instance: &TransientInstance,
        times_mjd: &[f64],
        bands: &[Band],
    ) -> Result<LightcurveEvaluation> {
        let model = self
            .model_override
            .unwrap_or_else(|| default_model_for_type(instance.transient_type));
        let max_phase = self.max_phase_days_for_instance(instance);

        // Convert observer-frame MJD to rest-frame days since explosion.
        let rest_times: Vec<f64> = times_mjd
            .iter()
            .map(|&t| (t - instance.t_exp) / (1.0 + instance.z))
            .collect();

        // Get model parameters as a flat array.
        let params = extract_model_params(instance, model);

        if model == SviModelName::MetzgerKN {
            return self.evaluate_metzger(instance, times_mjd, bands, &rest_times, &params, max_phase);
        }

        // --- Generic path for all other models ---
        let norm_flux = lightcurve_fitting::eval_model_flux(model, &params, &rest_times);

        let distance_modulus = if instance.d_l > 0.0 {
            let d_l_pc = instance.d_l * 1e6;
            5.0 * (d_l_pc / 10.0).log10()
        } else {
            0.0
        };
        let k_corr = k_correction_bolometric(instance.z);

        let mut seen_bands = std::collections::HashSet::new();
        let unique_bands: Vec<&str> = bands
            .iter()
            .filter_map(|b| if seen_bands.insert(b.0.as_str()) { Some(b.0.as_str()) } else { None })
            .collect();

        let mut apparent_mags = HashMap::with_capacity(unique_bands.len());
        for band_name in unique_bands {
            let mw_ext = extinction_in_band(instance.mw_extinction_av, band_name);
            let host_ext = extinction_in_band(instance.host_extinction_av, band_name);
            let color_offset = self.color_offsets.get(band_name).copied().unwrap_or(0.0);

            let mags: Vec<f64> = norm_flux
                .iter()
                .zip(rest_times.iter())
                .map(|(&f, &t_rest)| {
                    if t_rest < 0.0 || t_rest > max_phase || f <= 0.0 {
                        return 99.0;
                    }
                    let abs_mag = instance.peak_abs_mag - 2.5 * f.log10();
                    abs_mag + distance_modulus + k_corr + mw_ext + host_ext + color_offset
                })
                .collect();

            apparent_mags.insert(band_name.to_string(), mags);
        }

        Ok(LightcurveEvaluation {
            apparent_mags,
            times_mjd: times_mjd.to_vec(),
        })
    }

    fn max_phase_days(&self) -> f64 {
        if let Some(mp) = self.max_phase_days {
            mp
        } else if let Some(model) = self.model_override {
            Self::default_max_phase_for_model(model)
        } else {
            100.0
        }
    }

    fn max_phase_days_for_type(&self, transient_type: TransientType) -> f64 {
        self.resolved_max_phase_for_type(transient_type)
    }
}

impl ParametricModel {
    /// MetzgerKN path: computes per-band apparent AB magnitudes via blackbody.
    fn evaluate_metzger(
        &self,
        instance: &TransientInstance,
        times_mjd: &[f64],
        bands: &[Band],
        rest_times: &[f64],
        params: &[f64],
        max_phase: f64,
    ) -> Result<LightcurveEvaluation> {
        // Build (band_name, frequency_Hz) pairs for unique bands.
        // Frequency in host frame: ν_host = ν_obs * (1 + z)
        let unique_bands: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            bands.iter().filter_map(|b| {
                if seen.insert(b.0.clone()) { Some(b.0.clone()) } else { None }
            }).collect()
        };

        let band_freqs: Vec<(&str, f64)> = unique_bands
            .iter()
            .filter_map(|name| {
                let lambda_a = band_wavelength_angstrom(name)?;
                // Host-frame frequency: ν = c / λ, with λ in cm (Å * 1e-8)
                let nu_obs = C_CMS / (lambda_a * 1e-8);
                let nu_host = nu_obs * (1.0 + instance.z);
                Some((name.as_str(), nu_host))
            })
            .collect();

        let d_l_cm = instance.d_l * MPC_CM;

        let mag_map = lightcurve_fitting::metzger_kn_mags(params, rest_times, &band_freqs, d_l_cm);

        // Apply extinction to the raw AB mags for unique bands
        let mut apparent_mags = HashMap::with_capacity(unique_bands.len());
        for band_name in &unique_bands {
            if let Some(raw_mags) = mag_map.get(band_name) {
                let mw_ext = extinction_in_band(instance.mw_extinction_av, band_name);
                let host_ext = extinction_in_band(instance.host_extinction_av, band_name);
                let mags: Vec<f64> = raw_mags
                    .iter()
                    .zip(rest_times.iter())
                    .map(|(&m, &t_rest)| {
                        if t_rest < 0.0 || t_rest > max_phase || m >= 90.0 {
                            99.0
                        } else {
                            m + mw_ext + host_ext
                        }
                    })
                    .collect();
                apparent_mags.insert(band_name.clone(), mags);
            } else {
                // Band not recognized — fill with faint
                apparent_mags.insert(band_name.clone(), vec![99.0; times_mjd.len()]);
            }
        }

        Ok(LightcurveEvaluation {
            apparent_mags,
            times_mjd: times_mjd.to_vec(),
        })
    }
}

/// Extract model parameters from the TransientInstance for the given model.
///
/// Maps named parameters to the positional array expected by eval_model_flux.
fn extract_model_params(instance: &TransientInstance, model: SviModelName) -> Vec<f64> {
    let get = |key: &str, default: f64| -> f64 {
        instance.model_params.get(key).copied().unwrap_or(default)
    };

    match model {
        SviModelName::MetzgerKN => {
            // MetzgerKN params: [log10(mej/Msun), log10(vej/c), log10(kappa/(cm^2/g)), t0_offset]
            // Population generators store linear values; convert to log10 here.
            vec![
                get("mej", 0.01).log10(),
                get("vej", 0.2).log10(),
                get("kappa", 1.0).log10(),
                get("t0_offset", 0.0),
            ]
        }
        SviModelName::Bazin => {
            // Bazin params: [A, t0, tfall, trise, c]
            vec![
                get("A", 1.0),
                get("t0", 0.0),
                get("tfall", 30.0),
                get("trise", 5.0),
                get("c", 0.0),
            ]
        }
        SviModelName::Villar => {
            // Villar params: [log_A, beta, log_gamma, t0, log_tau_rise, log_tau_fall, log_sigma_extra]
            // All log params are natural log (ln). Population stores log10 values
            // from superphot+ priors (Kenworthy+ 2024, Table 2); convert to ln here.
            let ln10 = std::f64::consts::LN_10;
            vec![
                get("log10_A", 0.096) * ln10,           // log₁₀(A) → ln(A)
                get("beta", 0.008),                       // linear
                get("log10_gamma", 1.43) * ln10,         // log₁₀(γ) → ln(γ)
                get("t0", 0.0),                           // days relative to explosion
                get("log10_tau_rise", 0.67) * ln10,      // log₁₀(τ_rise) → ln(τ_rise)
                get("log10_tau_fall", 1.53) * ln10,      // log₁₀(τ_fall) → ln(τ_fall)
                get("log10_sigma_extra", -1.66) * ln10,  // log₁₀(σ_extra) → ln(σ_extra)
            ]
        }
        SviModelName::Tde => {
            // TDE params: [log_a, b, t0, log_tau_rise, log_tau_fall, alpha, sigma_extra]
            // Sigmoid rise × power-law decay: a * sig(phase/tau_rise) * (1+softplus(phase)/tau_fall)^(-alpha) + b
            // Default log_a=ln(2)≈0.69 so peak flux ≈ 1.0 (sigmoid peaks at 0.5, × exp(ln2) = 1)
            vec![
                get("log_a", 0.69),
                get("b", 0.0),
                get("t0", 0.0),
                get("log_tau_rise", 2.9),   // ln(18) ≈ 18 day rise
                get("log_tau_fall", 4.1),   // ln(60) ≈ 60 day fall
                get("alpha", 1.67),         // 5/3 power-law (TDE fallback)
                0.0,                        // sigma_extra (noise, not used in forward eval)
            ]
        }
        SviModelName::Afterglow => {
            // Afterglow params: [F0, alpha, t0, c]
            vec![
                get("F0", 1.0),
                get("alpha", -1.2),
                get("t0", 0.0),
                get("c", 0.0),
            ]
        }
        SviModelName::Arnett => {
            // Arnett params: [A, t0, tau_m, tau_ni, c]
            vec![
                get("A", 1.0),
                get("t0", 0.0),
                get("tau_m", 15.0),
                get("tau_ni", 8.8),
                get("c", 0.0),
            ]
        }
        SviModelName::Magnetar => {
            // Magnetar params: [A, t0, tau_sd, tau_d, sigma, c]
            vec![
                get("A", 1.0),
                get("t0", 0.0),
                get("tau_sd", 10.0),
                get("tau_d", 5.0),
                get("sigma", 3.0),
                get("c", 0.0),
            ]
        }
        SviModelName::ShockCooling => {
            // ShockCooling params: [A, t0, a, c]
            vec![
                get("A", 1.0),
                get("t0", 0.0),
                get("a", 0.5),
                get("c", 0.0),
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SkyCoord, TransientType};

    #[test]
    fn test_parametric_max_phase_resolution() {
        let model = ParametricModel::new();

        // Check fallback when no type or override is provided
        assert_eq!(model.max_phase_days(), 100.0);

        // Check type-based resolution
        assert_eq!(model.max_phase_days_for_type(TransientType::Kilonova), 25.0);
        assert_eq!(model.max_phase_days_for_type(TransientType::Tde), 365.0);
        assert_eq!(model.max_phase_days_for_type(TransientType::SupernovaIa), 90.0);
        assert_eq!(model.max_phase_days_for_type(TransientType::SupernovaII), 90.0);
        assert_eq!(model.max_phase_days_for_type(TransientType::Afterglow), 100.0);
        assert_eq!(model.max_phase_days_for_type(TransientType::Fbot), 90.0);

        // Check model override
        let kn_override = ParametricModel::new().with_model(SviModelName::MetzgerKN);
        assert_eq!(kn_override.max_phase_days(), 25.0);
        assert_eq!(kn_override.max_phase_days_for_type(TransientType::Tde), 25.0);

        // Check explicit max_phase_days override
        let explicit = ParametricModel::new().with_max_phase(45.0);
        assert_eq!(explicit.max_phase_days(), 45.0);
        assert_eq!(explicit.max_phase_days_for_type(TransientType::Tde), 45.0);

        // Check instance with and without custom max_phase_days parameter
        let mut inst = TransientInstance {
            coord: SkyCoord { ra: 0.0, dec: 0.0 },
            z: 0.01,
            d_l: 40.0,
            t_exp: 60000.0,
            peak_abs_mag: -16.0,
            transient_type: TransientType::Tde,
            model_params: HashMap::new(),
            mw_extinction_av: 0.0,
            host_extinction_av: 0.0,
        };

        // Without parameter: uses transient_type (TDE -> 365.0)
        assert_eq!(model.max_phase_days_for_instance(&inst), 365.0);

        // With parameter in model_params
        inst.model_params.insert("max_phase_days".to_string(), 123.0);
        assert_eq!(model.max_phase_days_for_instance(&inst), 123.0);
    }
}
