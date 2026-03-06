use std::collections::HashMap;

use blastwave::afterglow::eats::EATS;
use blastwave::afterglow::forward_grid::ForwardGrid;
use blastwave::afterglow::models::{Dict, get_radiation_model};
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
pub fn build_jet_config(
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
    tmax: f64,
) -> JetConfig {
    let theta_edge = forward_jet_res(theta_c, 33);
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

        // Compute tmax from the latest observation time (with margin).
        let t_max_obs = times_mjd
            .iter()
            .map(|&t| (t - instance.t_exp) * SECONDS_PER_DAY)
            .fold(0.0_f64, f64::max);
        // Cap at 1e6 s (~12 days) — optical afterglows fade below Argus depth well before this.
        let tmax = (t_max_obs * 2.0).max(1e5).min(1e6);

        // Build jet configuration and solve hydrodynamics.
        let config = build_jet_config(
            eiso, gamma_0, theta_c, n0, p, eps_e, eps_b, p_rs, eps_e_rs, eps_b_rs, tmax,
        );

        let mut sim_box = SimBox::new(&config);
        sim_box.solve_pde();

        let theta_data = sim_box.get_theta();
        let t_data = &sim_box.ts;
        let y_data = &sim_box.ys;
        let rs_data = sim_box.ys_rs.as_ref();

        let eats = EATS::new(theta_data, t_data);
        let tool = sim_box.tool();

        let radiation_model = get_radiation_model(&self.radiation_model)
            .ok_or_else(|| LightcurveError::InvalidParameters(
                format!("unknown radiation model: {}", self.radiation_model),
            ))?;

        // FS radiation parameters.
        let mut ag_params = Dict::new();
        ag_params.insert("eps_e".into(), eps_e);
        ag_params.insert("eps_b".into(), eps_b);
        ag_params.insert("p".into(), p);

        // RS radiation parameters.
        let mut rs_ag_params = Dict::new();
        rs_ag_params.insert("eps_e".into(), eps_e_rs);
        rs_ag_params.insert("eps_b".into(), eps_b_rs);
        rs_ag_params.insert("p".into(), p_rs);

        // Group observations by band/frequency.
        // For each unique frequency, collect (index, t_seconds) pairs.
        let mut freq_groups: HashMap<String, Vec<(usize, f64, f64)>> = HashMap::new();
        for (i, (&t_mjd, band)) in times_mjd.iter().zip(bands.iter()).enumerate() {
            let band_name = band.to_string();
            if self.band_frequencies.contains_key(&band_name) {
                let t_s = (t_mjd - instance.t_exp) * SECONDS_PER_DAY;
                freq_groups
                    .entry(band_name)
                    .or_default()
                    .push((i, t_mjd, t_s));
            }
        }

        // Pre-allocate output arrays.
        let mut apparent_mags: HashMap<String, Vec<f64>> = HashMap::new();
        let mut out_times: Vec<f64> = Vec::with_capacity(times_mjd.len());
        // We'll build a flat (index, mag, t_mjd) vec and sort by original index.
        let mut results: Vec<(usize, String, f64, f64)> = Vec::with_capacity(times_mjd.len());

        let flux_factor = (1.0 + z) / (4.0 * std::f64::consts::PI * d_cm * d_cm);

        for (band_name, obs) in &freq_groups {
            let nu = self.band_frequencies[band_name];
            let nu_z = nu * (1.0 + z);

            // Separate pre-explosion and beyond-tmax observations.
            let mut valid: Vec<(usize, f64, f64)> = Vec::new(); // (orig_idx, t_mjd, t_s)
            for &(idx, t_mjd, t_s) in obs {
                if t_s <= 0.0 || t_s > tmax {
                    results.push((idx, band_name.clone(), 99.0, t_mjd));
                } else {
                    valid.push((idx, t_mjd, t_s));
                }
            }

            if valid.is_empty() {
                continue;
            }

            // Sort query times for batch evaluation.
            let mut sorted_queries: Vec<(usize, f64, f64)> = valid.clone();
            sorted_queries.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
            let t_sorted: Vec<f64> = sorted_queries.iter().map(|q| q.2 / (1.0 + z)).collect();

            // Build FS forward grid and evaluate batch.
            let fs_grid = ForwardGrid::precompute(
                nu_z, theta_v, y_data, t_data, theta_data,
                &eats, tool, &ag_params, radiation_model,
            );
            let fs_lum = fs_grid.luminosity_batch(&t_sorted);

            // Build RS forward grid if available.
            let rs_lum = if let Some(rs) = rs_data {
                let rs_grid = ForwardGrid::precompute_reverse(
                    nu_z, theta_v, y_data, rs, t_data, theta_data,
                    &eats, tool, &rs_ag_params, radiation_model,
                );
                rs_grid.luminosity_batch(&t_sorted)
            } else {
                vec![0.0; t_sorted.len()]
            };

            // Convert luminosities to magnitudes.
            for (q_idx, &(orig_idx, t_mjd, _t_s)) in sorted_queries.iter().enumerate() {
                let lum = fs_lum[q_idx] + rs_lum[q_idx];
                let f_nu = lum * flux_factor;
                let f_mjy = f_nu / 1e-26;

                let mag = if f_mjy > 0.0 {
                    23.9 - 2.5 * f_mjy.log10()
                } else {
                    99.0
                };

                let a_band_host = extinction_in_band(av, band_name);
                let a_band_mw = extinction_in_band(instance.mw_extinction_av, band_name);
                let mag_ext = mag + a_band_host + a_band_mw;

                results.push((orig_idx, band_name.clone(), mag_ext, t_mjd));
            }
        }

        // Sort by original observation index and build output.
        results.sort_by_key(|r| r.0);
        for (_idx, band_name, mag, t_mjd) in results {
            apparent_mags
                .entry(band_name)
                .or_default()
                .push(mag);
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
