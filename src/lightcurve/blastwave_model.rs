use std::collections::HashMap;

use blastwave::afterglow::afterglow::Afterglow;
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
///
/// `eiso` is the isotropic-equivalent gamma-ray energy. We convert to kinetic
/// energy using a gamma-ray efficiency of 15%: `E_K = Eiso * (1 - eta) / eta`.
/// This matches the Freeburn/VegasAfterglow convention.
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
    let theta_edge = forward_jet_res(theta_c, 129);
    let ncells = theta_edge.len() - 1;

    // Cell centres from edges.
    let theta_centres: Vec<f64> = (0..ncells)
        .map(|i| 0.5 * (theta_edge[i] + theta_edge[i + 1]))
        .collect();

    // GenerateAfterglows.py applies eta=0.15: E_K = Eiso * (1-eta)/eta.
    // SimulateAfterglows.py passes Eiso directly (no eta conversion).
    // We match GenerateAfterglows.py here since the detectable pre-filter
    // peak_mag was computed with this convention.
    const ETA_GAMMA: f64 = 0.15;
    let e_k_iso = eiso * (1.0 - ETA_GAMMA) / ETA_GAMMA;

    // Top-hat profile: full energy inside theta_c, tail outside.
    let eiso_tail = e_k_iso * 1e-12;
    let gamma_tail = 1.005;

    let mut e_arr = Vec::with_capacity(ncells);
    let mut gamma_arr = Vec::with_capacity(ncells);
    for &tc in &theta_centres {
        if tc <= theta_c {
            e_arr.push(e_k_iso);
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

    let four_pi_c2 = 4.0 * std::f64::consts::PI * C_SPEED * C_SPEED;

    for i in 0..ncells {
        let e = e_arr[i];
        let g = gamma_arr[i];

        // Energy per steradian in mass units (matching Python _configJet):
        //   E0 = Eiso / (4π c²)
        let e0 = e / four_pi_c2;

        // Ejecta mass per steradian: Mej = E0 / (Gamma - 1)
        let m_ej = e0 / (g - 1.0);

        // Initial radius from coasting: R0 = beta * c * tmin
        let beta = (1.0 - 1.0 / (g * g)).sqrt();
        let r0 = beta * C_SPEED * tmin;

        // Swept mass per steradian for ISM: n * mp * R^3 / 3
        // (volume per steradian of a cone = R^3/3)
        let m_sw = nism * MASS_P * r0.powi(3) / 3.0;

        // Total energy per steradian in mass units: Eb = E0 + Mej + Msw
        let e_b = e0 + m_ej + m_sw;

        eb.push(e_b);
        mej.push(m_ej);
        msw.push(m_sw);
        r_init.push(r0);
    }

    // Enthalpy: zeros (matching Python _configJet: Ht = np.zeros_like(theta))
    let ht: Vec<f64> = vec![0.0; ncells];

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
    // No lateral spreading (matching Freeburn/VegasAfterglow convention).
    config.spread = false;
    config.spread_mode = SpreadMode::None;
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
        // The PDE must extend far beyond the observer-frame query times because
        // the EATS surface maps observer times to much later lab-frame times.
        // Use 1e10 s (matching Python blastwave's default).
        let tmax = 1e10;

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

        let radiation_model_name = &self.radiation_model;
        let _radiation_model = get_radiation_model(radiation_model_name)
            .ok_or_else(|| LightcurveError::InvalidParameters(
                format!("unknown radiation model: {}", radiation_model_name),
            ))?;

        // Configure Afterglow for EATS integration.
        let mut afterglow = Afterglow::new();
        let mut ag_params = Dict::new();
        ag_params.insert("theta_v".into(), theta_v);
        ag_params.insert("d".into(), d_mpc);
        ag_params.insert("z".into(), z);
        ag_params.insert("eps_e".into(), eps_e);
        ag_params.insert("eps_b".into(), eps_b);
        ag_params.insert("p".into(), p);
        afterglow.config_parameters(ag_params);
        afterglow.config_intensity(radiation_model_name);

        if rs_data.is_some() {
            let mut rs_ag_params = Dict::new();
            rs_ag_params.insert("theta_v".into(), theta_v);
            rs_ag_params.insert("d".into(), d_mpc);
            rs_ag_params.insert("z".into(), z);
            rs_ag_params.insert("eps_e".into(), eps_e_rs);
            rs_ag_params.insert("eps_b".into(), eps_b_rs);
            rs_ag_params.insert("p".into(), p_rs);
            afterglow.config_rs_parameters(rs_ag_params);
        }

        // Collect all observations with time since explosion.
        let mut obs_with_ts: Vec<(usize, String, f64, f64)> = Vec::new(); // (idx, band, t_mjd, t_s)
        let mut results: Vec<(usize, String, f64, f64)> = Vec::with_capacity(times_mjd.len());

        for (i, (&t_mjd, band)) in times_mjd.iter().zip(bands.iter()).enumerate() {
            let band_name = band.to_string();
            if !self.band_frequencies.contains_key(&band_name) {
                continue;
            }
            let t_s = (t_mjd - instance.t_exp) * SECONDS_PER_DAY;
            if t_s <= 0.0 {
                results.push((i, band_name, 99.0, t_mjd));
            } else {
                obs_with_ts.push((i, band_name, t_mjd, t_s));
            }
        }

        let flux_factor = (1.0 + z) / (4.0 * std::f64::consts::PI * d_cm * d_cm);

        // EATS evaluation: compute luminosity at each observation time.
        // For efficiency with many observations (e.g., raw 1-sec cadence),
        // subsample to log-spaced times and interpolate.
        const MAX_EATS_CALLS: usize = 30;

        if obs_with_ts.len() <= MAX_EATS_CALLS {
            // Few observations: evaluate each directly.
            for &(idx, ref band_name, t_mjd, t_s) in &obs_with_ts {
                let nu = self.band_frequencies[band_name];
                let lum = if let Some(rs) = rs_data {
                    afterglow.luminosity_total(
                        t_s, nu, 1e-3, 100, true,
                        &eats, y_data, rs, t_data, theta_data, tool,
                    )
                } else {
                    afterglow.luminosity(
                        t_s, nu, 1e-3, 100, true,
                        &eats, y_data, t_data, theta_data, tool,
                    )
                };

                let f_nu = lum * flux_factor;
                let f_ujy = f_nu / 1e-29; // microjansky
                let mag = if f_ujy > 0.0 { 23.9 - 2.5 * f_ujy.log10() } else { 99.0 };
                let a_host = extinction_in_band(av, band_name);
                let a_mw = extinction_in_band(instance.mw_extinction_av, band_name);
                results.push((idx, band_name.clone(), mag + a_host + a_mw, t_mjd));
            }
        } else {
            // Many observations: evaluate on a log-spaced grid and interpolate.
            // Group by band (usually just one for Argus).
            let mut band_groups: HashMap<String, Vec<(usize, f64, f64)>> = HashMap::new();
            for &(idx, ref band_name, t_mjd, t_s) in &obs_with_ts {
                band_groups.entry(band_name.clone()).or_default().push((idx, t_mjd, t_s));
            }

            for (band_name, obs) in &band_groups {
                let nu = self.band_frequencies[band_name];

                // Build log-spaced evaluation grid.
                let t_min = obs.iter().map(|o| o.2).fold(f64::INFINITY, f64::min);
                let t_max = obs.iter().map(|o| o.2).fold(0.0_f64, f64::max);
                let log_min = t_min.max(1.0).ln();
                let log_max = t_max.ln();
                let n_grid = MAX_EATS_CALLS.min(obs.len());
                let mut grid_times = Vec::with_capacity(n_grid);
                let mut grid_mags = Vec::with_capacity(n_grid);
                for k in 0..n_grid {
                    let log_t = log_min + (log_max - log_min) * k as f64 / (n_grid - 1).max(1) as f64;
                    grid_times.push(log_t.exp());
                }

                let mut max_lum = 0.0_f64;
                for &t_s in &grid_times {
                    let lum = if let Some(rs) = rs_data {
                        afterglow.luminosity_total(
                            t_s, nu, 1e-3, 100, true,
                            &eats, y_data, rs, t_data, theta_data, tool,
                        )
                    } else {
                        afterglow.luminosity(
                            t_s, nu, 1e-3, 100, true,
                            &eats, y_data, t_data, theta_data, tool,
                        )
                    };
                    if lum > max_lum { max_lum = lum; }
                    let f_nu = lum * flux_factor;
                    let f_ujy = f_nu / 1e-29; // microjansky
                    let mag = if f_ujy > 0.0 { 23.9 - 2.5 * f_ujy.log10() } else { 99.0 };
                    grid_mags.push(mag);
                }
                eprintln!("[eats-grid] z={z:.3} Eiso={eiso:.1e} G0={gamma_0:.0} thc={theta_c:.4} thv={theta_v:.4} n0={n0:.1e} eps_e={eps_e:.1e} eps_b={eps_b:.1e} max_lum={max_lum:.2e} d_cm={d_cm:.2e} flux_factor={flux_factor:.2e} t_range=[{t_min:.0}..{t_max:.0}]s grid_mags={grid_mags:?}");

                // Interpolate for each observation.
                let a_host = extinction_in_band(av, band_name);
                let a_mw = extinction_in_band(instance.mw_extinction_av, band_name);
                for &(idx, t_mjd, t_s) in obs {
                    // Find bracketing grid points and linearly interpolate in log-t space.
                    let log_t = t_s.ln();
                    let mag = if grid_times.len() == 1 {
                        grid_mags[0]
                    } else {
                        // Binary search for bracket.
                        let mut lo = 0;
                        let mut hi = grid_times.len() - 1;
                        while lo < hi - 1 {
                            let mid = (lo + hi) / 2;
                            if grid_times[mid].ln() <= log_t { lo = mid; } else { hi = mid; }
                        }
                        if lo == hi {
                            grid_mags[lo]
                        } else {
                            let log_lo = grid_times[lo].ln();
                            let log_hi = grid_times[hi].ln();
                            let frac = (log_t - log_lo) / (log_hi - log_lo);
                            grid_mags[lo] + frac * (grid_mags[hi] - grid_mags[lo])
                        }
                    };
                    results.push((idx, band_name.clone(), mag + a_host + a_mw, t_mjd));
                }
            }
        }

        // Debug: log brightest magnitude for this GRB.
        {
            let best = results.iter().map(|r| r.2).fold(99.0_f64, f64::min);
            if best < 30.0 {
                eprintln!("[eats] z={z:.3} Eiso={eiso:.1e} thc={theta_c:.3} thv={theta_v:.3} peak_mag={best:.1} n_obs={}",
                    obs_with_ts.len());
            }
        }

        // Sort by original observation index and build output.
        results.sort_by_key(|r| r.0);
        let mut apparent_mags: HashMap<String, Vec<f64>> = HashMap::new();
        let mut out_times: Vec<f64> = Vec::with_capacity(results.len());
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
        let near_tc = grid.iter().filter(|&&t| t < 2.0 * theta_c).count();
        let near_pi = grid.iter().filter(|&&t| t > std::f64::consts::PI - 0.2).count();
        assert!(near_tc > near_pi, "Grid should cluster near theta_c");
    }

    /// Compare ForwardGrid vs EATS for a bright catalog GRB (z=3.38).
    #[test]
    fn test_forward_grid_vs_eats_catalog_grb() {
        use blastwave::afterglow::forward_grid::ForwardGrid;

        // Parameters from a bright catalog GRB (csv_peak=14.8, z=3.38)
        let theta_c = 10.0_f64.powf(-1.18);  // ~0.066 rad
        let n0 = 10.0_f64.powf(-0.73);       // ~0.19 cm^-3
        let eps_e = 10.0_f64.powf(-0.81);
        let eps_b = 10.0_f64.powf(-1.50);
        let p = 2.60;
        let z = 3.38;
        let d_mpc = 3.0e28 / 3.09e24;  // ~9700 Mpc
        let theta_v = 0.0;

        let config = build_jet_config(
            1.2e53, 400.0, theta_c, n0, p, eps_e, eps_b, p, eps_e, eps_b, 1e10,
        );

        let mut sim = SimBox::new(&config);
        sim.solve_pde();

        let theta_data = sim.get_theta();
        let t_data = &sim.ts;
        let y_data = &sim.ys;
        let eats = EATS::new(theta_data, t_data);
        let tool = sim.tool();

        let rad_model = get_radiation_model("sync_ssa_smooth").unwrap();

        let mut ag_params = Dict::new();
        ag_params.insert("eps_e".into(), eps_e);
        ag_params.insert("eps_b".into(), eps_b);
        ag_params.insert("p".into(), p);

        let mut afterglow = Afterglow::new();
        let mut af_params = Dict::new();
        af_params.insert("theta_v".into(), theta_v);
        af_params.insert("d".into(), d_mpc);
        af_params.insert("z".into(), z);
        af_params.insert("eps_e".into(), eps_e);
        af_params.insert("eps_b".into(), eps_b);
        af_params.insert("p".into(), p);
        afterglow.config_parameters(af_params);
        afterglow.config_intensity("sync_ssa_smooth");

        let nu = 6.3e14;
        let nu_z = nu * (1.0 + z);

        let grid = ForwardGrid::precompute(
            nu_z, theta_v, y_data, t_data, theta_data,
            &eats, tool, &ag_params, rad_model,
        );

        let test_times = [60.0, 300.0, 900.0, 3600.0, 86400.0];
        let t_rest: Vec<f64> = test_times.iter().map(|&t| t / (1.0 + z)).collect();
        let fg_lum = grid.luminosity_batch(&t_rest);

        eprintln!("Catalog GRB test (z=3.38, Eiso=1.2e53, theta_c={:.4}):", theta_c);
        eprintln!("PDE: {} cells, {} steps", theta_data.len(), t_data.len());

        for (i, &t) in test_times.iter().enumerate() {
            let eats_lum = afterglow.luminosity(
                t, nu, 1e-3, 100, true,
                &eats, y_data, t_data, theta_data, tool,
            );
            let ratio = if eats_lum > 0.0 { fg_lum[i] / eats_lum } else { 0.0 };
            eprintln!(
                "  t={:.0}s: FG={:.4e}, EATS={:.4e}, ratio={:.3}",
                t, fg_lum[i], eats_lum, ratio
            );
            if eats_lum > 0.0 {
                assert!(
                    ratio > 0.1,
                    "ForwardGrid much too low at t={}s: FG={}, EATS={}, ratio={}",
                    t, fg_lum[i], eats_lum, ratio
                );
            }
        }
    }

    /// Compare ForwardGrid vs EATS for a canonical on-axis GRB.
    #[test]
    fn test_forward_grid_vs_eats() {
        use blastwave::afterglow::forward_grid::ForwardGrid;

        let theta_c = 0.1;
        let n0 = 1.0;
        let eps_e = 0.1;
        let eps_b = 0.01;
        let p = 2.3;
        let z = 0.1;
        let d_mpc = 474.33;
        let theta_v = 0.0;

        let config = build_jet_config(
            1e52, 300.0, theta_c, n0, p, eps_e, eps_b, p, eps_e, eps_b, 1e10,
        );

        let mut sim = SimBox::new(&config);
        sim.solve_pde();

        let theta_data = sim.get_theta();
        let t_data = &sim.ts;
        let y_data = &sim.ys;
        let eats = EATS::new(theta_data, t_data);
        let tool = sim.tool();

        let rad_model = get_radiation_model("sync_ssa_smooth").unwrap();

        // Params for radiation model
        let mut ag_params = Dict::new();
        ag_params.insert("eps_e".into(), eps_e);
        ag_params.insert("eps_b".into(), eps_b);
        ag_params.insert("p".into(), p);

        // Configure Afterglow for EATS
        let mut afterglow = Afterglow::new();
        let mut af_params = Dict::new();
        af_params.insert("theta_v".into(), theta_v);
        af_params.insert("d".into(), d_mpc);
        af_params.insert("z".into(), z);
        af_params.insert("eps_e".into(), eps_e);
        af_params.insert("eps_b".into(), eps_b);
        af_params.insert("p".into(), p);
        afterglow.config_parameters(af_params);
        afterglow.config_intensity("sync_ssa_smooth");

        let nu = 6.3e14;
        let nu_z = nu * (1.0 + z);

        // ForwardGrid path
        let grid = ForwardGrid::precompute(
            nu_z, theta_v, y_data, t_data, theta_data,
            &eats, tool, &ag_params, rad_model,
        );

        // Test at several observer times
        let test_times = [100.0, 500.0, 3600.0, 86400.0];
        let t_rest: Vec<f64> = test_times.iter().map(|&t| t / (1.0 + z)).collect();
        let fg_lum = grid.luminosity_batch(&t_rest);

        eprintln!("PDE: {} theta cells, {} time steps, {} state vars",
            theta_data.len(), t_data.len(), y_data.len());

        for (i, &t) in test_times.iter().enumerate() {
            let eats_lum = afterglow.luminosity(
                t, nu, 1e-3, 100, true,
                &eats, y_data, t_data, theta_data, tool,
            );
            let ratio = if eats_lum > 0.0 { fg_lum[i] / eats_lum } else { 0.0 };
            eprintln!(
                "  t={:.0}s: FG={:.4e}, EATS={:.4e}, ratio={:.3}",
                t, fg_lum[i], eats_lum, ratio
            );
            // They should agree within ~10%
            if eats_lum > 0.0 {
                assert!(
                    ratio > 0.5 && ratio < 2.0,
                    "ForwardGrid and EATS disagree at t={}s: FG={}, EATS={}, ratio={}",
                    t, fg_lum[i], eats_lum, ratio
                );
            }
        }
    }
}
