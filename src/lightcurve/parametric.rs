use std::collections::HashMap;

use lightcurve_fitting::SviModelName;

use crate::types::{Band, TransientInstance};

use super::cosmology::{extinction_in_band, k_correction_bolometric};
use super::{LightcurveEvaluation, LightcurveModel, Result};

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
        TransientType::Tde => SviModelName::Tde,
        TransientType::Afterglow => SviModelName::Afterglow,
        TransientType::Custom => SviModelName::Bazin,
    }
}

/// Parametric lightcurve model wrapping lightcurve-fitting's eval_model_flux.
///
/// Pipeline:
/// 1. Convert observer-frame times to rest-frame relative times
/// 2. Evaluate normalized flux via eval_model_flux (peak ~ 1)
/// 3. Convert to absolute magnitude using peak_abs_mag
/// 4. Apply distance modulus, K-correction, and extinction
/// 5. Add per-band color offsets
pub struct ParametricModel {
    /// Override the default model name (otherwise inferred from TransientType).
    pub model_override: Option<SviModelName>,
    /// Per-band color offsets relative to the reference band (magnitudes).
    pub color_offsets: HashMap<String, f64>,
}

impl ParametricModel {
    pub fn new() -> Self {
        Self {
            model_override: None,
            color_offsets: HashMap::new(),
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

        // Convert observer-frame MJD to rest-frame days since explosion.
        let rest_times: Vec<f64> = times_mjd
            .iter()
            .map(|&t| (t - instance.t_exp) / (1.0 + instance.z))
            .collect();

        // Get model parameters as a flat array.
        let params = extract_model_params(instance, model);

        // Evaluate normalized flux (peak ~ 1).
        let norm_flux = lightcurve_fitting::eval_model_flux(model, &params, &rest_times);

        // Convert flux to magnitudes for each band.
        let distance_modulus = if instance.d_l > 0.0 {
            let d_l_pc = instance.d_l * 1e6;
            5.0 * (d_l_pc / 10.0).log10()
        } else {
            0.0
        };
        let k_corr = k_correction_bolometric(instance.z);

        let mut apparent_mags = HashMap::new();
        for band in bands {
            let mw_ext = extinction_in_band(instance.mw_extinction_av, &band.0);
            let host_ext = extinction_in_band(instance.host_extinction_av, &band.0);
            let color_offset = self.color_offsets.get(&band.0).copied().unwrap_or(0.0);

            let mags: Vec<f64> = norm_flux
                .iter()
                .map(|&f| {
                    if f <= 0.0 {
                        // No flux -> set to very faint magnitude.
                        return 99.0;
                    }
                    // Normalized flux -> absolute mag.
                    // peak_abs_mag corresponds to flux=1.
                    let abs_mag = instance.peak_abs_mag - 2.5 * f.log10();

                    // Apply distance modulus, K-correction, extinction, color.
                    abs_mag + distance_modulus + k_corr + mw_ext + host_ext + color_offset
                })
                .collect();

            apparent_mags.insert(band.0.clone(), mags);
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
            // Villar params: [A, beta, t0, tfall, trise, c]
            vec![
                get("A", 1.0),
                get("beta", 0.01),
                get("t0", 0.0),
                get("tfall", 30.0),
                get("trise", 5.0),
                get("c", 0.0),
            ]
        }
        SviModelName::Tde => {
            // TDE params: [A, t0, sigma, tau, c]
            vec![
                get("A", 1.0),
                get("t0", 0.0),
                get("sigma", 10.0),
                get("tau", 30.0),
                get("c", 0.0),
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
