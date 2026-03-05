use std::collections::HashMap;

use blastwave::afterglow::afterglow::Afterglow;
use blastwave::afterglow::eats::EATS;
use blastwave::afterglow::models::Dict;
use blastwave::constants::{C_SPEED, MASS_P, MPC};
use blastwave::hydro::config::{JetConfig, SpreadMode};
use blastwave::hydro::sim_box::SimBox;

use crate::types::{Band, TransientInstance};

use super::cosmology::extinction_in_band;
use super::{LightcurveEvaluation, LightcurveError, LightcurveModel, Result};

const SECONDS_PER_DAY: f64 = 86400.0;

/// Blastwave GRB afterglow lightcurve model.
///
/// Wraps the `blastwave` crate to compute relativistic blast wave dynamics
/// and synchrotron afterglow emission for each transient instance.
pub struct BlastwaveModel {
    pub radiation_model: String,
    pub band_frequencies: HashMap<String, f64>,
}

impl Default for BlastwaveModel {
    fn default() -> Self {
        let mut freqs = HashMap::new();
        freqs.insert("g".to_string(), 6.3e14); // Argus g-band
        Self {
            radiation_model: "sync_ssa_smooth".to_string(),
            band_frequencies: freqs,
        }
    }
}

impl BlastwaveModel {
    pub fn new(radiation_model: &str, band_frequencies: HashMap<String, f64>) -> Self {
        Self {
            radiation_model: radiation_model.to_string(),
            band_frequencies,
        }
    }
}

/// Build an arcsinh-spaced angular grid clustered around theta_c.
fn forward_jet_res(theta_c: f64, npoints: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    let max_val = (PI / theta_c).asinh();
    let mut cells: Vec<f64> = (0..npoints)
        .map(|i| (i as f64 / (npoints - 1) as f64 * max_val).sinh() * theta_c)
        .collect();
    cells[0] = 0.0;
    let last = cells.len() - 1;
    cells[last] = PI;
    cells
}

/// Configure a top-hat jet from GRB physical parameters.
fn build_jet_config(
    eiso: f64,
    gamma_0: f64,
    theta_c: f64,
    n0: f64,
    p_fwd: f64,
    eps_e: f64,
    eps_b: f64,
    p_rs: f64,
    eps_e_rs: f64,
    eps_b_rs: f64,
) -> JetConfig {
    let theta_edge = forward_jet_res(theta_c, 129);
    let ncells = theta_edge.len() - 1;

    // Cell centres from edges.
    let theta_centres: Vec<f64> = (0..ncells)
        .map(|i| 0.5 * (theta_edge[i] + theta_edge[i + 1]))
        .collect();

    // Top-hat profile: full energy inside theta_c, tail outside.
    let eiso_tail = eiso * 1e-12;
    let gamma_tail = 1.005;

    let mut e_arr = Vec::with_capacity(ncells);
    let mut gamma_arr = Vec::with_capacity(ncells);
    for &tc in &theta_centres {
        if tc <= theta_c {
            e_arr.push(eiso);
            gamma_arr.push(gamma_0);
        } else {
            e_arr.push(eiso_tail);
            gamma_arr.push(gamma_tail);
        }
    }

    // Convert (Eiso, Gamma) -> (Eb, Mej, Msw, R) at tmin.
    let tmin = 10.0;
    let tmax = 1e8;
    let nism = n0;

    let mut eb = Vec::with_capacity(ncells);
    let mut mej = Vec::with_capacity(ncells);
    let mut msw = Vec::with_capacity(ncells);
    let mut r_init = Vec::with_capacity(ncells);

    for i in 0..ncells {
        let e = e_arr[i];
        let g = gamma_arr[i];
        // beta*c * tmin
        let beta = (1.0 - 1.0 / (g * g)).sqrt();
        let r0 = beta * C_SPEED * tmin;

        // Swept mass per steradian: (4/3) * n * mp * R^3 (uniform ISM).
        let m_sw = (4.0 / 3.0) * nism * MASS_P * r0.powi(3);

        // Ejecta mass from E = (Gamma - 1) * Mej * c^2.
        let m_ej = e / ((g - 1.0) * C_SPEED * C_SPEED);

        // Blast wave energy: Eb = (Gamma - 1) * (Mej + Msw) * c^2.
        let e_b = (g - 1.0) * (m_ej + m_sw) * C_SPEED * C_SPEED;

        // Enthalpy proxy: ht = Mej * Gamma * c (matching blastwave convention).
        // blastwave uses ht for the total energy-momentum; set via Gamma*Mej*c.
        eb.push(e_b);
        mej.push(m_ej);
        msw.push(m_sw);
        r_init.push(r0);
    }

    // Enthalpy = Gamma * (Mej + Msw) * c  (momentum-like variable).
    let ht: Vec<f64> = (0..ncells)
        .map(|i| gamma_arr[i] * (mej[i] + msw[i]) * C_SPEED)
        .collect();

    let mut config = JetConfig::default();
    config.theta_edge = theta_edge;
    config.eb = eb;
    config.ht = ht;
    config.msw = msw;
    config.mej = mej;
    config.r = r_init;
    config.nwind = 0.0;
    config.nism = nism;
    config.tmin = tmin;
    config.tmax = tmax;
    config.rtol = 1e-5;
    config.spread = true;
    config.spread_mode = SpreadMode::Pde;
    config.theta_c = theta_c;
    config.include_reverse_shock = true;
    config.eps_e = eps_e;
    config.eps_b = eps_b;
    config.p_fwd = p_fwd;
    config.eps_e_rs = eps_e_rs;
    config.eps_b_rs = eps_b_rs;
    config.p_rs = p_rs;

    config
}

/// Extract a required parameter from model_params.
fn get_param(params: &HashMap<String, f64>, key: &str) -> Result<f64> {
    params
        .get(key)
        .copied()
        .ok_or_else(|| LightcurveError::InvalidParameters(format!("missing parameter: {}", key)))
}

impl LightcurveModel for BlastwaveModel {
    fn evaluate(
        &self,
        instance: &TransientInstance,
        times_mjd: &[f64],
        bands: &[Band],
    ) -> Result<LightcurveEvaluation> {
        let params = &instance.model_params;

        // Extract physical parameters.
        let eiso = get_param(params, "Eiso")?;
        let gamma_0 = get_param(params, "Gamma_0")?;
        let theta_v = get_param(params, "theta_v")?;
        let logthc = get_param(params, "logthc")?;
        let logn0 = get_param(params, "logn0")?;
        let logepse = get_param(params, "logepse")?;
        let logepsb = get_param(params, "logepsB")?;
        let p = get_param(params, "p")?;
        let av = get_param(params, "av")?;
        let p_rs = get_param(params, "p_rvs")?;
        let logepse_rs = get_param(params, "logepse_rvs")?;
        let logepsb_rs = get_param(params, "logepsB_rvs")?;

        let theta_c = 10.0_f64.powf(logthc);
        let n0 = 10.0_f64.powf(logn0);
        let eps_e = 10.0_f64.powf(logepse);
        let eps_b = 10.0_f64.powf(logepsb);
        let eps_e_rs = 10.0_f64.powf(logepse_rs);
        let eps_b_rs = 10.0_f64.powf(logepsb_rs);

        let z = instance.z;
        let d_mpc = instance.d_l;
        let d_cm = d_mpc * MPC;

        // Build jet configuration and solve hydrodynamics.
        let config = build_jet_config(
            eiso, gamma_0, theta_c, n0, p, eps_e, eps_b, p_rs, eps_e_rs, eps_b_rs,
        );

        let mut sim_box = SimBox::new(&config);
        sim_box.solve_pde();
        sim_box.solve_reverse_shock();

        let theta_data = sim_box.get_theta();
        let t_data = &sim_box.ts;
        let y_data = &sim_box.ys;
        let rs_data = sim_box.ys_rs.as_ref();

        // Configure afterglow radiation.
        let mut afterglow = Afterglow::new();

        let mut ag_params = Dict::new();
        ag_params.insert("eps_e".into(), eps_e);
        ag_params.insert("eps_b".into(), eps_b);
        ag_params.insert("p".into(), p);
        ag_params.insert("theta_v".into(), theta_v);
        ag_params.insert("d".into(), d_cm);
        ag_params.insert("z".into(), z);
        afterglow.config_parameters(ag_params);

        let mut rs_params = Dict::new();
        rs_params.insert("eps_e".into(), eps_e_rs);
        rs_params.insert("eps_b".into(), eps_b_rs);
        rs_params.insert("p".into(), p_rs);
        rs_params.insert("theta_v".into(), theta_v);
        rs_params.insert("d".into(), d_cm);
        rs_params.insert("z".into(), z);
        afterglow.config_rs_parameters(rs_params);

        afterglow.config_intensity(&self.radiation_model);

        let eats = EATS::new(theta_data, t_data);

        // Compute apparent magnitudes at each observation time/band.
        let mut apparent_mags: HashMap<String, Vec<f64>> = HashMap::new();
        let mut out_times: Vec<f64> = Vec::with_capacity(times_mjd.len());

        for (i, (&t_mjd, band)) in times_mjd.iter().zip(bands.iter()).enumerate() {
            let band_name = band.to_string();
            let nu = match self.band_frequencies.get(&band_name) {
                Some(&f) => f,
                None => continue, // skip unknown bands
            };

            // Convert MJD to seconds since explosion.
            let t_s = (t_mjd - instance.t_exp) * SECONDS_PER_DAY;
            if t_s <= 0.0 {
                // Before explosion — set to very faint.
                apparent_mags
                    .entry(band_name)
                    .or_insert_with(Vec::new)
                    .push(99.0);
                if i < out_times.len() || out_times.is_empty() || *out_times.last().unwrap_or(&0.0) != t_mjd {
                    out_times.push(t_mjd);
                }
                continue;
            }

            // Compute total luminosity (FS + RS) in erg/s/Hz.
            let lum = if let Some(rs) = rs_data {
                afterglow.luminosity_total(
                    t_s, nu, 1e-2, 50, true, &eats, y_data, rs, t_data, theta_data,
                    sim_box.tool(),
                )
            } else {
                afterglow.luminosity(
                    t_s, nu, 1e-2, 50, true, &eats, y_data, t_data, theta_data,
                    sim_box.tool(),
                )
            };

            // Convert luminosity (erg/s/Hz) to flux density (mJy).
            // F_nu = L_nu * (1+z) / (4 * pi * d_L^2)  [erg/s/cm²/Hz]
            // F_mJy = F_nu / 1e-26
            let f_nu = lum * (1.0 + z) / (4.0 * std::f64::consts::PI * d_cm * d_cm);
            let f_mjy = f_nu / 1e-26;

            // Convert to AB magnitude.
            let mag = if f_mjy > 0.0 {
                // m_AB = 23.9 - 2.5 * log10(F_mJy)
                // (since AB zero-point is 3631 Jy = 3.631e6 mJy)
                23.9 - 2.5 * f_mjy.log10()
            } else {
                99.0 // undetectable
            };

            // Apply host extinction.
            let a_band_host = extinction_in_band(av, &band_name);
            // Apply MW extinction.
            let a_band_mw = extinction_in_band(instance.mw_extinction_av, &band_name);
            let mag_ext = mag + a_band_host + a_band_mw;

            apparent_mags
                .entry(band_name)
                .or_insert_with(Vec::new)
                .push(mag_ext);
            out_times.push(t_mjd);
        }

        Ok(LightcurveEvaluation {
            apparent_mags,
            times_mjd: out_times,
        })
    }

    fn requires_gil(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_jet_res_endpoints() {
        let grid = forward_jet_res(0.1, 129);
        assert_eq!(grid.len(), 129);
        assert!((grid[0] - 0.0).abs() < 1e-15);
        assert!((grid[128] - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn test_forward_jet_res_clustering() {
        let theta_c = 0.1;
        let grid = forward_jet_res(theta_c, 129);
        // Should have more points near theta_c than near PI.
        let near_tc = grid.iter().filter(|&&t| t < 2.0 * theta_c).count();
        let near_pi = grid.iter().filter(|&&t| t > std::f64::consts::PI - 0.2).count();
        assert!(near_tc > near_pi, "Grid should cluster near theta_c");
    }
}
