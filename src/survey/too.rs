use crate::instrument::InstrumentConfig;
use crate::types::{Band, SkyCoord};

use super::SurveyObservation;

/// A trigger event (e.g. GW alert) that initiates a ToO campaign.
#[derive(Clone, Debug)]
pub struct TooTrigger {
    /// Best-estimate sky position of the event.
    pub coord: SkyCoord,
    /// MJD of the trigger.
    pub trigger_mjd: f64,
    /// 90% credible localization area in sq deg.
    pub localization_area_deg2: f64,
    /// Distance estimate in Mpc (if available).
    pub distance_mpc: Option<f64>,
}

/// A single visit in a ToO observing plan.
#[derive(Clone, Debug)]
pub struct TooVisit {
    /// Offset from trigger time in days.
    pub dt_days: f64,
    /// Photometric band.
    pub band: String,
    /// Exposure time in seconds.
    pub exposure_s: f64,
    /// 5-sigma depth for this visit.
    pub depth_mag: f64,
}

/// Trait for Target-of-Opportunity follow-up strategies.
///
/// Given a trigger event, generates a sequence of `SurveyObservation`s
/// representing the follow-up campaign.
pub trait TooStrategy: Send + Sync {
    /// Name of this strategy (e.g. "Rubin Gold").
    fn name(&self) -> &str;

    /// Instrument configuration for this strategy.
    fn instrument(&self) -> InstrumentConfig;

    /// Generate the planned visit sequence for a trigger.
    fn plan_visits(&self, trigger: &TooTrigger) -> Vec<TooVisit>;

    /// Generate `SurveyObservation`s from a trigger.
    ///
    /// Tiles the localization area with pointings and applies the visit plan.
    /// The default implementation computes the number of pointings needed to
    /// cover the localization area and distributes them around the trigger
    /// position.
    fn generate_observations(
        &self,
        trigger: &TooTrigger,
        start_obs_id: u64,
    ) -> Vec<SurveyObservation> {
        let visits = self.plan_visits(trigger);
        let instrument = self.instrument();
        let fov_deg2 = instrument.detector.fov_deg2;

        // Number of pointings to tile the localization area.
        // Overlap factor ~1.2 for practical tiling efficiency.
        let n_pointings = ((trigger.localization_area_deg2 / fov_deg2) * 1.2)
            .ceil()
            .max(1.0) as usize;

        // Generate pointing centers on a grid around the trigger position.
        let pointings = tile_localization(&trigger.coord, n_pointings, fov_deg2);

        let mut observations = Vec::with_capacity(pointings.len() * visits.len());
        let mut obs_id = start_obs_id;

        for visit in &visits {
            let mjd = trigger.trigger_mjd + visit.dt_days;
            let night = mjd.floor() as i64;

            for pointing in &pointings {
                observations.push(SurveyObservation {
                    obs_id,
                    coord: pointing.clone(),
                    mjd,
                    band: Band::new(&visit.band),
                    five_sigma_depth: visit.depth_mag,
                    seeing_fwhm: instrument
                        .bands
                        .get(&visit.band)
                        .map_or(1.0, |b| b.typical_seeing_arcsec),
                    exposure_time: visit.exposure_s,
                    airmass: 1.3, // typical airmass for ToO
                    sky_brightness: instrument
                        .bands
                        .get(&visit.band)
                        .map_or(21.0, |b| b.sky_brightness),
                    night,
                });
                obs_id += 1;
            }
        }

        observations
    }
}

/// Generate a simple grid of pointing centers to tile a localization region.
fn tile_localization(center: &SkyCoord, n_pointings: usize, fov_deg2: f64) -> Vec<SkyCoord> {
    if n_pointings <= 1 {
        return vec![center.clone()];
    }

    let fov_side = fov_deg2.sqrt(); // approximate square FoV side
    let n_side = (n_pointings as f64).sqrt().ceil() as usize;
    let cos_dec = center.dec.to_radians().cos().max(0.1);

    let mut pointings = Vec::with_capacity(n_side * n_side);
    let offset_start = -(n_side as f64 - 1.0) / 2.0;

    for i in 0..n_side {
        for j in 0..n_side {
            if pointings.len() >= n_pointings {
                break;
            }
            let dra = (offset_start + i as f64) * fov_side * 0.9 / cos_dec;
            let ddec = (offset_start + j as f64) * fov_side * 0.9;
            let ra = (center.ra + dra).rem_euclid(360.0);
            let dec = (center.dec + ddec).clamp(-90.0, 90.0);
            pointings.push(SkyCoord::new(ra, dec));
        }
    }

    pointings
}

// ---------------------------------------------------------------------------
// Rubin LSST ToO strategies (from 2411.04793)
// ---------------------------------------------------------------------------

/// Rubin Gold strategy: deep 3-filter follow-up of well-localized BNS events.
///
/// - Localization: ≤100 sq deg (90% credible)
/// - Night 0: 3 scans in g,r,i × 120s each
/// - Night 1–3: r,i × 180s each
/// - Total: 15 visits over 4 nights
pub struct RubinGoldToo;

impl TooStrategy for RubinGoldToo {
    fn name(&self) -> &str {
        "Rubin Gold"
    }

    fn instrument(&self) -> InstrumentConfig {
        InstrumentConfig::rubin()
    }

    fn plan_visits(&self, _trigger: &TooTrigger) -> Vec<TooVisit> {
        let rubin = InstrumentConfig::rubin();

        // Night 0: 3 scans, each g+r+i at 120s.
        // Depth boost for 120s vs 30s default: +2.5*log10(sqrt(120/30)) = +0.75 mag
        let depth_boost_120 = 2.5 * (120.0_f64 / 30.0).sqrt().log10();
        let depth_boost_180 = 2.5 * (180.0_f64 / 30.0).sqrt().log10();

        let mut visits = Vec::new();

        // Night 0: 3 scans × gri × 120s
        for scan in 0..3 {
            let dt = scan as f64 * 0.02; // ~30 min between scans
            for band_name in &["g", "r", "i"] {
                let base_depth = rubin.bands[*band_name].single_visit_depth;
                visits.push(TooVisit {
                    dt_days: dt,
                    band: band_name.to_string(),
                    exposure_s: 120.0,
                    depth_mag: base_depth + depth_boost_120,
                });
            }
        }

        // Nights 1–3: ri × 180s
        for night in 1..=3 {
            for band_name in &["r", "i"] {
                let base_depth = rubin.bands[*band_name].single_visit_depth;
                visits.push(TooVisit {
                    dt_days: night as f64,
                    band: band_name.to_string(),
                    exposure_s: 180.0,
                    depth_mag: base_depth + depth_boost_180,
                });
            }
        }

        visits
    }
}

/// Rubin Silver strategy: wide-area 2-filter follow-up.
///
/// - Localization: ≤500 sq deg
/// - Night 0: g+i × 30s
/// - Night 1–3: 2 filters × 120s
pub struct RubinSilverToo;

impl TooStrategy for RubinSilverToo {
    fn name(&self) -> &str {
        "Rubin Silver"
    }

    fn instrument(&self) -> InstrumentConfig {
        InstrumentConfig::rubin()
    }

    fn plan_visits(&self, _trigger: &TooTrigger) -> Vec<TooVisit> {
        let rubin = InstrumentConfig::rubin();
        let depth_boost_120 = 2.5 * (120.0_f64 / 30.0).sqrt().log10();

        let mut visits = Vec::new();

        // Night 0: g+i × 30s (standard depth)
        for band_name in &["g", "i"] {
            visits.push(TooVisit {
                dt_days: 0.0,
                band: band_name.to_string(),
                exposure_s: 30.0,
                depth_mag: rubin.bands[*band_name].single_visit_depth,
            });
        }

        // Nights 1–3: g+i × 120s
        for night in 1..=3 {
            for band_name in &["g", "i"] {
                let base_depth = rubin.bands[*band_name].single_visit_depth;
                visits.push(TooVisit {
                    dt_days: night as f64,
                    band: band_name.to_string(),
                    exposure_s: 120.0,
                    depth_mag: base_depth + depth_boost_120,
                });
            }
        }

        visits
    }
}

// ---------------------------------------------------------------------------
// ZTF ToO (from 2405.12403)
// ---------------------------------------------------------------------------

/// ZTF GW follow-up strategy.
///
/// - 47 deg² FoV per pointing
/// - 30s exposures in g,r,i
/// - 2 visits per night for 3 nights
/// - Single-visit depths: g~20.8, r~20.6, i~19.9
pub struct ZtfToo;

impl TooStrategy for ZtfToo {
    fn name(&self) -> &str {
        "ZTF ToO"
    }

    fn instrument(&self) -> InstrumentConfig {
        InstrumentConfig::ztf()
    }

    fn plan_visits(&self, _trigger: &TooTrigger) -> Vec<TooVisit> {
        let ztf = InstrumentConfig::ztf();
        let mut visits = Vec::new();

        // Night 0: 2 visits in g+r (early + late in night)
        for visit_offset in &[0.0, 0.125] {
            // ~3hr apart
            for band_name in &["g", "r"] {
                visits.push(TooVisit {
                    dt_days: *visit_offset,
                    band: band_name.to_string(),
                    exposure_s: 30.0,
                    depth_mag: ztf.bands[*band_name].single_visit_depth,
                });
            }
        }

        // Nights 1–2: g+r per night
        for night in 1..=2 {
            for band_name in &["g", "r"] {
                visits.push(TooVisit {
                    dt_days: night as f64,
                    band: band_name.to_string(),
                    exposure_s: 30.0,
                    depth_mag: ztf.bands[*band_name].single_visit_depth,
                });
            }
        }

        // Night 3: i-band for photometric classification
        visits.push(TooVisit {
            dt_days: 3.0,
            band: "i".to_string(),
            exposure_s: 30.0,
            depth_mag: ztf.bands["i"].single_visit_depth,
        });

        visits
    }
}

// ---------------------------------------------------------------------------
// ULTRASAT ToO (from 2304.14482)
// ---------------------------------------------------------------------------

/// ULTRASAT GW follow-up strategy.
///
/// Parameters from m4opt and Shvartzvald et al. (2024):
/// - 14.28° × 14.28° FoV (204 deg², single pointing covers most GW localizations)
/// - NUV band centered at 260nm (σ=34nm)
/// - 33cm aperture, 5.4"/pix
/// - 3×300s exposures stacked → 22.5 mag depth
/// - <15 min response time from geosynchronous orbit
/// - Continuous monitoring for ~1 day (orbital visibility permitting)
pub struct UltrasatToo;

impl TooStrategy for UltrasatToo {
    fn name(&self) -> &str {
        "ULTRASAT ToO"
    }

    fn instrument(&self) -> InstrumentConfig {
        InstrumentConfig::ultrasat()
    }

    fn plan_visits(&self, _trigger: &TooTrigger) -> Vec<TooVisit> {
        let mut visits = Vec::new();

        // Response within 15 min, then 3×300s stacked observations
        // repeated over ~1 day with orbital gaps.
        // Orbital period ~97 min → ~5 observation windows per day

        let response_delay_days = 15.0 / (24.0 * 60.0); // 15 min

        // 5 observation windows, each 3×300s stacked
        let stacked_depth = 22.5;
        let orbital_period_days = 97.0 / (24.0 * 60.0);

        for window in 0..5 {
            visits.push(TooVisit {
                dt_days: response_delay_days + window as f64 * orbital_period_days,
                band: "NUV".to_string(),
                exposure_s: 900.0, // 3×300s
                depth_mag: stacked_depth,
            });
        }

        // Day 2–3: continued monitoring at same cadence (fewer windows)
        for day in 1..=2 {
            for window in 0..3 {
                visits.push(TooVisit {
                    dt_days: day as f64 + window as f64 * orbital_period_days,
                    band: "NUV".to_string(),
                    exposure_s: 900.0,
                    depth_mag: stacked_depth,
                });
            }
        }

        visits
    }
}

// ---------------------------------------------------------------------------
// UVEX ToO (from 2502.17560)
// ---------------------------------------------------------------------------

/// UVEX GW follow-up strategy.
///
/// Parameters from m4opt and Kulkarni et al. (2021):
/// - 3.5° × 3.5° FoV (12.25 deg²)
/// - 75cm aperture, 1"/pix
/// - NUV (230nm, σ=18nm) + FUV (160nm, σ=10nm) simultaneous
/// - Standard dwell 900s, adaptive 300s+ for ToO
/// - 2 visits per field, 30 min cadence
/// - 6-hour observation window
/// - TESS-like HEO orbit, 60s settling time
/// - M⁴OPT MILP scheduling
pub struct UvexToo;

impl TooStrategy for UvexToo {
    fn name(&self) -> &str {
        "UVEX ToO"
    }

    fn instrument(&self) -> InstrumentConfig {
        InstrumentConfig::uvex()
    }

    fn plan_visits(&self, _trigger: &TooTrigger) -> Vec<TooVisit> {
        let mut visits = Vec::new();

        // 2 visits per field, 30 min apart, within 6-hour window
        // NUV and FUV are simultaneous (both recorded per visit)
        // 900s standard dwell (m4opt): NUV ~25.0 mag, FUV ~24.5 mag

        for epoch in 0..2 {
            let dt = epoch as f64 * (30.0 / (24.0 * 60.0)); // 30 min cadence

            visits.push(TooVisit {
                dt_days: dt,
                band: "NUV".to_string(),
                exposure_s: 900.0,
                depth_mag: 25.0,
            });
            visits.push(TooVisit {
                dt_days: dt,
                band: "FUV".to_string(),
                exposure_s: 900.0,
                depth_mag: 24.5,
            });
        }

        // Follow-up epoch at +1 day
        for band in &[("NUV", 25.0), ("FUV", 24.5)] {
            visits.push(TooVisit {
                dt_days: 1.0,
                band: band.0.to_string(),
                exposure_s: 900.0,
                depth_mag: band.1,
            });
        }

        visits
    }
}

// ---------------------------------------------------------------------------
// Built-in strategy constructors
// ---------------------------------------------------------------------------

/// Get a named built-in ToO strategy.
pub fn builtin_strategy(name: &str) -> Option<Box<dyn TooStrategy>> {
    match name {
        "rubin_gold" | "Rubin Gold" => Some(Box::new(RubinGoldToo)),
        "rubin_silver" | "Rubin Silver" => Some(Box::new(RubinSilverToo)),
        "ztf" | "ZTF ToO" => Some(Box::new(ZtfToo)),
        "ultrasat" | "ULTRASAT ToO" => Some(Box::new(UltrasatToo)),
        "uvex" | "UVEX ToO" => Some(Box::new(UvexToo)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_trigger() -> TooTrigger {
        TooTrigger {
            coord: SkyCoord::new(180.0, -30.0),
            trigger_mjd: 60000.0,
            localization_area_deg2: 100.0,
            distance_mpc: Some(200.0),
        }
    }

    #[test]
    fn test_rubin_gold_visits() {
        let strategy = RubinGoldToo;
        let visits = strategy.plan_visits(&test_trigger());
        // Night 0: 3 scans × 3 bands = 9, Nights 1-3: 3 × 2 bands = 6
        assert_eq!(visits.len(), 15);
        assert!(visits.iter().all(|v| v.depth_mag > 24.0));
    }

    #[test]
    fn test_rubin_silver_visits() {
        let strategy = RubinSilverToo;
        let visits = strategy.plan_visits(&test_trigger());
        // Night 0: 2 bands, Nights 1-3: 3 × 2 bands = 6, total = 8
        assert_eq!(visits.len(), 8);
    }

    #[test]
    fn test_ztf_too_visits() {
        let strategy = ZtfToo;
        let visits = strategy.plan_visits(&test_trigger());
        // Night 0: 2×2=4, Nights 1-2: 2×2=4, Night 3: 1 = 9
        assert_eq!(visits.len(), 9);
    }

    #[test]
    fn test_ultrasat_too_visits() {
        let strategy = UltrasatToo;
        let visits = strategy.plan_visits(&test_trigger());
        assert!(visits.len() >= 5);
        assert!(visits.iter().all(|v| v.band == "NUV"));
    }

    #[test]
    fn test_uvex_too_visits() {
        let strategy = UvexToo;
        let visits = strategy.plan_visits(&test_trigger());
        assert!(visits.len() >= 4);
        let nuv_count = visits.iter().filter(|v| v.band == "NUV").count();
        let fuv_count = visits.iter().filter(|v| v.band == "FUV").count();
        assert_eq!(nuv_count, fuv_count);
    }

    #[test]
    fn test_generate_observations_rubin_gold() {
        let strategy = RubinGoldToo;
        let trigger = TooTrigger {
            coord: SkyCoord::new(180.0, -30.0),
            trigger_mjd: 60000.0,
            localization_area_deg2: 30.0, // ~3 Rubin pointings
            distance_mpc: Some(200.0),
        };
        let obs = strategy.generate_observations(&trigger, 0);
        // 15 visits × ~4 pointings
        assert!(obs.len() > 15);
        assert!(obs.iter().all(|o| o.mjd >= 60000.0));
    }

    #[test]
    fn test_tile_localization_single() {
        let center = SkyCoord::new(100.0, 20.0);
        let tiles = tile_localization(&center, 1, 9.6);
        assert_eq!(tiles.len(), 1);
    }

    #[test]
    fn test_tile_localization_grid() {
        let center = SkyCoord::new(100.0, 20.0);
        let tiles = tile_localization(&center, 9, 9.6);
        assert!(tiles.len() >= 9);
    }

    #[test]
    fn test_builtin_strategy_lookup() {
        assert!(builtin_strategy("rubin_gold").is_some());
        assert!(builtin_strategy("ztf").is_some());
        assert!(builtin_strategy("ultrasat").is_some());
        assert!(builtin_strategy("uvex").is_some());
        assert!(builtin_strategy("nonexistent").is_none());
    }
}
