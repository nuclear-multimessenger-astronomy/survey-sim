use std::collections::HashMap;
use std::path::Path;

use super::{Result, SurveyError};

/// A GW event from a LIGO/Virgo/KAGRA observing scenario simulation.
///
/// Combines injection truth parameters with skymap summary statistics.
#[derive(Clone, Debug)]
pub struct GwEvent {
    /// Simulation ID (matches FITS skymap filename).
    pub simulation_id: u64,
    /// Coincidence event ID.
    pub coinc_event_id: u64,
    /// True sky position: longitude in radians.
    pub longitude: f64,
    /// True sky position: latitude in radians.
    pub latitude: f64,
    /// True RA in degrees (derived from longitude).
    pub ra: f64,
    /// True Dec in degrees (derived from latitude).
    pub dec: f64,
    /// True luminosity distance in Mpc.
    pub distance_mpc: f64,
    /// Component mass 1 (solar masses).
    pub mass1: f64,
    /// Component mass 2 (solar masses).
    pub mass2: f64,
    /// Spin z-component of mass 1.
    pub spin1z: f64,
    /// Spin z-component of mass 2.
    pub spin2z: f64,
    /// Viewing angle / inclination in radians.
    pub inclination: f64,
    /// Network SNR.
    pub snr: f64,
    /// False alarm rate (Hz).
    pub far: f64,
    /// 90% credible sky area in sq deg.
    pub area_90: f64,
    /// 50% credible sky area in sq deg.
    pub area_50: f64,
    /// Posterior mean distance in Mpc.
    pub dist_mean: f64,
    /// Posterior distance std in Mpc.
    pub dist_std: f64,
    /// Detectors that contributed (e.g. "H1,L1,V1").
    pub ifos: String,
}

impl GwEvent {
    /// Whether this event is a BNS (both components < 3 M_sun).
    pub fn is_bns(&self) -> bool {
        self.mass1 < 3.0 && self.mass2 < 3.0
    }

    /// Whether this event is a NSBH (one component < 3, other > 3 M_sun).
    pub fn is_nsbh(&self) -> bool {
        (self.mass1 < 3.0) != (self.mass2 < 3.0)
    }

    /// Whether this event is a BBH (both components > 3 M_sun).
    pub fn is_bbh(&self) -> bool {
        self.mass1 >= 3.0 && self.mass2 >= 3.0
    }

    /// Chirp mass.
    pub fn chirp_mass(&self) -> f64 {
        let m1 = self.mass1;
        let m2 = self.mass2;
        (m1 * m2).powf(3.0 / 5.0) / (m1 + m2).powf(1.0 / 5.0)
    }
}

/// Load GW events from an observing scenario run directory.
///
/// Reads `injections.dat`, `coincs.dat`, and `allsky.dat` to build
/// a complete list of GW events with truth parameters and skymap statistics.
///
/// # Arguments
/// * `run_dir` - Path to the run directory (e.g. `runs/O5a/bgp/`)
pub fn load_observing_scenario(run_dir: &str) -> Result<Vec<GwEvent>> {
    let base = Path::new(run_dir);

    // Read injections.dat (TSV: simulation_id, longitude, latitude, inclination, distance, mass1, mass2, spin1z, spin2z)
    let injections_path = base.join("injections.dat");
    let mut inj_reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(&injections_path)?;

    let mut injections: HashMap<u64, InjRow> = HashMap::new();
    for result in inj_reader.records() {
        let rec = result?;
        let sim_id: u64 = rec[0].parse().map_err(|e| SurveyError::InvalidData(format!("bad sim_id: {}", e)))?;
        injections.insert(sim_id, InjRow {
            longitude: rec[1].parse().unwrap_or(0.0),
            latitude: rec[2].parse().unwrap_or(0.0),
            inclination: rec[3].parse().unwrap_or(0.0),
            distance: rec[4].parse().unwrap_or(0.0),
            mass1: rec[5].parse().unwrap_or(0.0),
            mass2: rec[6].parse().unwrap_or(0.0),
            spin1z: rec[7].parse().unwrap_or(0.0),
            spin2z: rec[8].parse().unwrap_or(0.0),
        });
    }

    // Read coincs.dat (TSV: coinc_event_id, ifos, snr)
    let coincs_path = base.join("coincs.dat");
    let mut coinc_reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(&coincs_path)?;

    let mut coincs: HashMap<u64, CoincRow> = HashMap::new();
    for result in coinc_reader.records() {
        let rec = result?;
        let coinc_id: u64 = rec[0].parse().unwrap_or(0);
        coincs.insert(coinc_id, CoincRow {
            ifos: rec[1].to_string(),
            snr: rec[2].parse().unwrap_or(0.0),
        });
    }

    // Read allsky.dat (TSV with header, skipping comment lines)
    let allsky_path = base.join("allsky.dat");
    let mut allsky_reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .comment(Some(b'#'))
        .from_path(&allsky_path)?;

    let mut events = Vec::new();
    for result in allsky_reader.records() {
        let rec = result?;
        let coinc_id: u64 = rec[0].parse().unwrap_or(0);
        let sim_id: u64 = rec[1].parse().unwrap_or(0);

        let inj = match injections.get(&sim_id) {
            Some(i) => i,
            None => continue,
        };

        let coinc = coincs.get(&coinc_id);

        // Convert longitude/latitude (radians) to RA/Dec (degrees)
        let ra = inj.longitude.to_degrees().rem_euclid(360.0);
        let dec = inj.latitude.to_degrees();

        events.push(GwEvent {
            simulation_id: sim_id,
            coinc_event_id: coinc_id,
            longitude: inj.longitude,
            latitude: inj.latitude,
            ra,
            dec,
            distance_mpc: inj.distance,
            mass1: inj.mass1,
            mass2: inj.mass2,
            spin1z: inj.spin1z,
            spin2z: inj.spin2z,
            inclination: inj.inclination,
            snr: coinc.map_or(rec[3].parse().unwrap_or(0.0), |c| c.snr),
            far: rec[2].parse().unwrap_or(0.0),
            area_90: rec[17].parse().unwrap_or(0.0),
            area_50: rec[16].parse().unwrap_or(0.0),
            dist_mean: rec[11].parse().unwrap_or(0.0),
            dist_std: rec[12].parse().unwrap_or(0.0),
            ifos: coinc.map_or_else(String::new, |c| c.ifos.clone()),
        });
    }

    log::info!(
        "Loaded {} GW events from {} ({} injections, {} BNS, {} NSBH, {} BBH)",
        events.len(),
        run_dir,
        injections.len(),
        events.iter().filter(|e| e.is_bns()).count(),
        events.iter().filter(|e| e.is_nsbh()).count(),
        events.iter().filter(|e| e.is_bbh()).count(),
    );

    Ok(events)
}

struct InjRow {
    longitude: f64,
    latitude: f64,
    inclination: f64,
    distance: f64,
    mass1: f64,
    mass2: f64,
    spin1z: f64,
    spin2z: f64,
}

struct CoincRow {
    ifos: String,
    snr: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gw_event_classification() {
        let bns = GwEvent {
            simulation_id: 0, coinc_event_id: 0,
            longitude: 0.0, latitude: 0.0, ra: 0.0, dec: 0.0,
            distance_mpc: 100.0, mass1: 1.4, mass2: 1.3,
            spin1z: 0.0, spin2z: 0.0, inclination: 0.0,
            snr: 10.0, far: 0.0, area_90: 30.0, area_50: 10.0,
            dist_mean: 100.0, dist_std: 20.0, ifos: "H1,L1".into(),
        };
        assert!(bns.is_bns());
        assert!(!bns.is_nsbh());
        assert!(!bns.is_bbh());

        let nsbh = GwEvent { mass1: 8.0, mass2: 1.4, ..bns.clone() };
        assert!(!nsbh.is_bns());
        assert!(nsbh.is_nsbh());
        assert!(!nsbh.is_bbh());

        let bbh = GwEvent { mass1: 30.0, mass2: 20.0, ..bns.clone() };
        assert!(!bbh.is_bns());
        assert!(!bbh.is_nsbh());
        assert!(bbh.is_bbh());
    }

    #[test]
    fn test_chirp_mass() {
        let event = GwEvent {
            simulation_id: 0, coinc_event_id: 0,
            longitude: 0.0, latitude: 0.0, ra: 0.0, dec: 0.0,
            distance_mpc: 100.0, mass1: 1.4, mass2: 1.4,
            spin1z: 0.0, spin2z: 0.0, inclination: 0.0,
            snr: 10.0, far: 0.0, area_90: 30.0, area_50: 10.0,
            dist_mean: 100.0, dist_std: 20.0, ifos: "H1,L1".into(),
        };
        // M_chirp for equal mass 1.4+1.4: (1.4*1.4)^(3/5) / (2.8)^(1/5) = 1.2187
        assert!((event.chirp_mass() - 1.2187).abs() < 0.01);
    }
}
