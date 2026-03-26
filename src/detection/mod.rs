use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::lightcurve::LightcurveEvaluation;

/// Depth boost for stacking N Argus 1-sec exposures (read-noise dominated).
/// Interpolated from Argus website science requirements (g-band, 5σ).
fn argus_depth_boost(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    const TABLE: [(f64, f64); 5] = [
        (0.0, 0.0),       // N=1
        (1.778, 3.2),     // N=60 (1 min)
        (2.954, 4.7),     // N=900 (15 min)
        (3.556, 5.5),     // N=3600 (1 hr)
        (4.459, 6.4),     // N=28800 (1 night)
    ];
    let log_n = (n as f64).log10();
    if log_n >= TABLE[TABLE.len() - 1].0 {
        return TABLE[TABLE.len() - 1].1;
    }
    for i in 1..TABLE.len() {
        if log_n <= TABLE[i].0 {
            let frac = (log_n - TABLE[i - 1].0) / (TABLE[i].0 - TABLE[i - 1].0);
            return TABLE[i - 1].1 + frac * (TABLE[i].1 - TABLE[i - 1].1);
        }
    }
    TABLE[TABLE.len() - 1].1
}
use crate::survey::SurveyObservation;

/// Criteria for determining whether a transient is detected.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectionCriteria {
    /// Minimum total detections above threshold.
    pub min_detections: usize,
    /// Minimum number of bands with at least one detection.
    pub min_bands: usize,
    /// Minimum detections required in any single band.
    pub min_per_band: usize,
    /// Maximum timespan in days for required detections.
    pub max_timespan_days: f64,
    /// Primary SNR threshold (default 5.0, matching five_sigma_depth).
    /// At least `min_detections_primary` observations must exceed this.
    pub snr_threshold: f64,
    /// Secondary (looser) SNR threshold for additional detections.
    /// Defaults to same as `snr_threshold`. Set lower (e.g., 3.0) for
    /// two-tier detection (e.g., ZTFReST: one at 5σ, one at 3σ).
    pub snr_threshold_secondary: f64,
    /// Minimum detections at the primary (stricter) SNR threshold. Default: 1.
    pub min_detections_primary: usize,
    /// Minimum time separation in hours between first and last detection.
    /// Default: 0.0 (no requirement). ZTFReST uses 3.0 hours.
    pub min_time_separation_hours: f64,
    /// If true, require the source to be a "fast transient" based on rise/decay rates.
    /// A source qualifies if any band shows rising faster than `min_rise_rate` mag/day
    /// or fading faster than `min_fade_rate` mag/day (boom algorithm).
    pub require_fast_transient: bool,
    /// Minimum rise rate in mag/day (absolute value). Rising = brightening = negative slope.
    /// Default: 1.0 mag/day.
    pub min_rise_rate: f64,
    /// Minimum fade rate in mag/day (positive slope). Default: 0.3 mag/day.
    pub min_fade_rate: f64,
    /// Minimum detections before peak brightness. Default: 0 (no requirement).
    /// DR2-style cuts typically require >=1 pre-peak detection.
    pub min_pre_peak_detections: usize,
    /// Minimum detections after peak brightness. Default: 0.
    pub min_post_peak_detections: usize,
    /// Minimum phase range in days (last - first detection relative to peak).
    /// Default: 0.0. DR2 requires coverage from roughly -10 to +40 days.
    pub min_phase_range_days: f64,
    /// Minimum absolute galactic latitude |b| in degrees. Default: 0.0.
    /// Set to e.g. 7.0 to exclude the galactic plane (ZTF avoids |b| < 7°).
    /// This is checked against the transient sky position, not the observations.
    pub min_galactic_lat: f64,
    /// Spectroscopic completeness logistic steepness. Default: 0.0 (disabled).
    /// When > 0, applies a magnitude-dependent classification probability:
    ///   P(classified) = 1 / (1 + exp(k * (peak_mag - m0)))
    /// Fitted to ZTF BTS (Perley 2020): k≈2.4, m0≈19.5–19.9.
    /// k=2.378, m0=19.9 reproduces DR2 SN Ia counts within ~2%.
    pub spectroscopic_completeness_k: f64,
    /// Spectroscopic completeness logistic midpoint magnitude. Default: 19.46.
    pub spectroscopic_completeness_m0: f64,
    /// Stacking windows (seconds) for flux-averaging detection.
    ///
    /// When non-empty, raw observations are grouped into time bins of each
    /// window size. Within each bin, the model fluxes are averaged and the
    /// stacked noise is reduced by sqrt(N). A detection counts if the
    /// stacked flux exceeds the stacked noise threshold. This properly
    /// handles fast transients (GRB afterglows) whose brightness changes
    /// within a single stack window.
    ///
    /// Example: `vec![900.0, 3600.0, 86400.0]` for 15-min, 1-hr, 1-day.
    #[serde(default)]
    pub stack_windows_s: Vec<f64>,
}

impl Default for DetectionCriteria {
    fn default() -> Self {
        Self {
            min_detections: 2,
            min_bands: 1,
            min_per_band: 1,
            max_timespan_days: 30.0,
            snr_threshold: 5.0,
            snr_threshold_secondary: 5.0,
            min_detections_primary: 1,
            min_time_separation_hours: 0.0,
            require_fast_transient: false,
            min_rise_rate: 1.0,
            min_fade_rate: 0.3,
            min_pre_peak_detections: 0,
            min_post_peak_detections: 0,
            min_phase_range_days: 0.0,
            min_galactic_lat: 0.0,
            spectroscopic_completeness_k: 0.0,
            spectroscopic_completeness_m0: 19.46,
            stack_windows_s: Vec::new(),
        }
    }
}

impl DetectionCriteria {
    /// Construct ZTFReST-style detection criteria (Andreoni & Coughlin 2021).
    ///
    /// - At least 2 detections total (one at 5σ, one at 3σ)
    /// - Minimum 3 hour separation between detections
    /// - Fade rate > 0.3 mag/day in any band
    /// - Max timespan 14 days (kilonova duration)
    pub fn ztfrest() -> Self {
        Self {
            min_detections: 2,
            min_bands: 1,
            min_per_band: 1,
            max_timespan_days: 14.0,
            snr_threshold: 5.0,
            snr_threshold_secondary: 3.0,
            min_detections_primary: 1,
            min_time_separation_hours: 3.0,
            require_fast_transient: true,
            min_rise_rate: 1.0,
            min_fade_rate: 0.3,
            min_pre_peak_detections: 0,
            min_post_peak_detections: 0,
            min_phase_range_days: 0.0,
            min_galactic_lat: 0.0,
            spectroscopic_completeness_k: 0.0,
            spectroscopic_completeness_m0: 19.46,
            stack_windows_s: Vec::new(),
        }
    }
}

/// Rise/decay rate properties for a single band segment, following boom's WLS approach.
#[derive(Clone, Debug)]
pub struct BandRateResult {
    /// Linear rate in mag/day (negative = brightening, positive = fading).
    pub rate: f64,
    /// Uncertainty on the rate.
    pub rate_error: f64,
    /// Number of data points used.
    pub n_points: usize,
    /// Time span of the segment in days.
    pub dt: f64,
}

/// Weighted least squares fit for mag = a*t + b, centered for numerical stability.
/// Ported from boom's `weighted_least_squares_centered`.
///
/// Returns the slope (rate in mag/day) and its uncertainty.
fn weighted_least_squares(times: &[f64], mags: &[f64], mag_errs: &[f64]) -> Option<BandRateResult> {
    let n = times.len();
    if n < 2 || mags.len() != n || mag_errs.len() != n {
        return None;
    }

    let mut sum_w = 0.0;
    let mut sum_wt = 0.0;
    let mut sum_wm = 0.0;

    for i in 0..n {
        if mag_errs[i] <= 0.0 || !mag_errs[i].is_finite() {
            return None;
        }
        let w = 1.0 / (mag_errs[i] * mag_errs[i]);
        if !w.is_finite() {
            return None;
        }
        sum_w += w;
        sum_wt += w * times[i];
        sum_wm += w * mags[i];
    }

    let t_mean = sum_wt / sum_w;
    let m_mean = sum_wm / sum_w;

    let mut stt = 0.0;
    let mut stm = 0.0;

    for i in 0..n {
        let w = 1.0 / (mag_errs[i] * mag_errs[i]);
        let dt = times[i] - t_mean;
        let dm = mags[i] - m_mean;
        stt += w * dt * dt;
        stm += w * dt * dm;
    }

    if stt.abs() < 1e-10 {
        return None;
    }

    let rate = stm / stt;
    let rate_error = (1.0 / stt).sqrt();

    Some(BandRateResult {
        rate,
        rate_error,
        n_points: n,
        dt: times[n - 1] - times[0],
    })
}

/// Compute rise and fade rates for a single band from detected observations.
///
/// Splits at peak (minimum magnitude), fits WLS to each segment.
/// With only 2 points total, one segment gets both points and computes a rate.
/// No noise check is applied — in simulation the magnitudes are known, and
/// rate thresholds handle the filtering.
fn compute_band_rates(
    times: &[f64],
    mags: &[f64],
    mag_errs: &[f64],
) -> (Option<BandRateResult>, Option<BandRateResult>) {
    if times.len() < 2 {
        return (None, None);
    }

    // Find peak index (minimum magnitude = brightest).
    let peak_idx = mags
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Split: before = [0..=peak], after = [peak..end].
    let rising = if peak_idx > 0 {
        let seg_t: Vec<f64> = times[..=peak_idx].iter().map(|t| t - times[0]).collect();
        let seg_m = &mags[..=peak_idx];
        let seg_e = &mag_errs[..=peak_idx];
        let dt = seg_t.last().unwrap() - seg_t[0];
        if dt > 0.01 {
            weighted_least_squares(&seg_t, &seg_m.to_vec(), &seg_e.to_vec())
        } else {
            None
        }
    } else {
        None
    };

    let fading = if peak_idx < times.len() - 1 {
        let seg_t: Vec<f64> = times[peak_idx..].iter().map(|t| t - times[peak_idx]).collect();
        let seg_m = &mags[peak_idx..];
        let seg_e = &mag_errs[peak_idx..];
        let dt = seg_t.last().unwrap() - seg_t[0];
        if dt > 0.01 {
            weighted_least_squares(&seg_t, &seg_m.to_vec(), &seg_e.to_vec())
        } else {
            None
        }
    } else {
        None
    };

    (rising, fading)
}

/// Estimate magnitude error from apparent magnitude and five-sigma depth.
///
/// At the 5-sigma limit, SNR=5, so sigma_mag = 1.0857/5 ≈ 0.217.
/// SNR scales as 10^(0.4*(depth - mag)), so:
///   sigma_mag = 1.0857 / (5 * 10^(0.4*(depth - mag)))
fn mag_error_from_depth(apparent_mag: f64, five_sigma_depth: f64) -> f64 {
    let snr = 5.0 * 10_f64.powf(0.4 * (five_sigma_depth - apparent_mag));
    if snr > 0.0 {
        1.0857 / snr
    } else {
        99.0
    }
}

/// Result of detection evaluation for a single transient.
#[derive(Clone, Debug)]
pub struct DetectionResult {
    /// Whether the transient passed all detection criteria.
    pub detected: bool,
    /// Total number of individual detections (above secondary SNR threshold).
    pub n_detections: usize,
    /// Number of detections above the primary (stricter) SNR threshold.
    pub n_detections_primary: usize,
    /// Number of bands with at least one detection.
    pub n_bands_detected: usize,
    /// MJD of first detection (if any).
    pub first_detection_mjd: Option<f64>,
    /// MJD of last detection (if any).
    pub last_detection_mjd: Option<f64>,
    /// Detections per band.
    pub detections_per_band: HashMap<String, usize>,
    /// Best (steepest) rise rate across all bands in mag/day (negative = brightening).
    pub best_rise_rate: Option<f64>,
    /// Best (steepest) fade rate across all bands in mag/day (positive = fading).
    pub best_fade_rate: Option<f64>,
    /// Whether the source qualifies as a fast transient.
    pub is_fast_transient: bool,
    /// MJD of peak brightness (minimum apparent magnitude across all detected bands).
    pub peak_mjd: Option<f64>,
    /// Peak apparent magnitude (brightest detection).
    pub peak_mag: Option<f64>,
    /// Number of detections before peak brightness.
    pub n_pre_peak: usize,
    /// Number of detections after peak brightness.
    pub n_post_peak: usize,
    /// Phase of earliest detection relative to peak (days, negative = before peak).
    pub phase_min_days: Option<f64>,
    /// Phase of latest detection relative to peak (days, positive = after peak).
    pub phase_max_days: Option<f64>,
}

/// Evaluate detection for a transient given its lightcurve and overlapping observations.
///
/// For each observation, checks if `apparent_mag < five_sigma_depth`.
/// Then applies the detection criteria (min detections, min bands, timespan, etc.).
/// If `require_fast_transient` is set, computes rise/decay rates per band using
/// boom-style weighted least squares and checks thresholds.
pub fn evaluate_detection(
    evaluation: &LightcurveEvaluation,
    observations: &[&SurveyObservation],
    criteria: &DetectionCriteria,
) -> DetectionResult {
    let mut n_detections = 0usize;
    let mut n_detections_primary = 0usize;
    let mut bands_detected = HashSet::new();
    let mut detections_per_band: HashMap<String, usize> = HashMap::new();
    let mut detection_mjds = Vec::new();

    // Collect per-band detected photometry for rate computation.
    // Each entry: (time, mag, mag_err) sorted by time within band.
    let mut band_photometry: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::new();

    // Compute the secondary depth threshold. The five_sigma_depth corresponds to
    // SNR=5. For a different SNR threshold, depth_eff = depth + 2.5*log10(5/snr).
    let secondary_depth_boost = if criteria.snr_threshold_secondary < criteria.snr_threshold
        && criteria.snr_threshold_secondary > 0.0
    {
        2.5 * (criteria.snr_threshold / criteria.snr_threshold_secondary).log10()
    } else {
        0.0
    };

    // ── Flux-stacking detection ──
    // If stacking windows are specified, group raw observations into time bins,
    // average the linear fluxes, and detect against the stacked depth.
    // This properly handles fast transients (GRB afterglows).
    if !criteria.stack_windows_s.is_empty() {
        let seconds_per_day = 86400.0;

        for &window_s in &criteria.stack_windows_s {
            let window_days = window_s / seconds_per_day;

            // Group observations by (band, time_bin).
            let mut bins: HashMap<(String, i64), Vec<(usize, f64, f64)>> = HashMap::new();
            for (i, obs) in observations.iter().enumerate() {
                let band_name = obs.band.0.clone();
                let time_bin = (obs.mjd / window_days).floor() as i64;
                if let Some(mags) = evaluation.apparent_mags.get(&band_name) {
                    if i < mags.len() {
                        let app_mag = mags[i];
                        bins.entry((band_name, time_bin))
                            .or_default()
                            .push((i, app_mag, obs.five_sigma_depth));
                    }
                }
            }

            for ((band_name, _), entries) in &bins {
                if entries.is_empty() {
                    continue;
                }
                let n_in_bin = entries.len() as f64;

                // Freeburn stacking: average the linear flux densities within the bin,
                // then compare against the stacked depth (median + sqrt(N) boost).
                // This matches Jim's redback fork: flux_density = mean of per-frame fluxes.
                let total_flux: f64 = entries
                    .iter()
                    .map(|&(_, mag, _)| {
                        if mag < 90.0 {
                            10.0_f64.powf(-0.4 * (mag - 23.9))
                        } else {
                            0.0
                        }
                    })
                    .sum();
                let mean_flux = total_flux / n_in_bin;

                if mean_flux <= 0.0 {
                    continue;
                }

                let stacked_mag = 23.9 - 2.5 * mean_flux.log10();

                // Stacked depth: median single-frame depth + 2.5*log10(sqrt(N))
                // matching Freeburn's redback fork (line 1137).
                let mut depths: Vec<f64> = entries.iter().map(|e| e.2).collect();
                depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_depth = depths[depths.len() / 2];
                let n = entries.len();
                // sqrt(N) scaling matching Freeburn's redback fork.
                let depth_boost = 2.5 * (n as f64).sqrt().log10();
                let stacked_depth = median_depth + depth_boost;

                let depth_secondary = stacked_depth + secondary_depth_boost;

                if stacked_mag < depth_secondary {
                    n_detections += 1;
                    bands_detected.insert(band_name.clone());
                    *detections_per_band.entry(band_name.clone()).or_insert(0) += 1;

                    // Use midpoint MJD for the stacked detection.
                    let mjd_min = entries.iter().map(|e| observations[e.0].mjd).fold(f64::INFINITY, f64::min);
                    let mjd_max = entries.iter().map(|e| observations[e.0].mjd).fold(f64::NEG_INFINITY, f64::max);
                    let mjd_mid = 0.5 * (mjd_min + mjd_max);
                    detection_mjds.push(mjd_mid);

                    if stacked_mag < stacked_depth {
                        n_detections_primary += 1;
                    }

                    let mag_err = mag_error_from_depth(stacked_mag, stacked_depth);
                    band_photometry
                        .entry(band_name.clone())
                        .or_default()
                        .push((mjd_mid, stacked_mag, mag_err));
                }
            }
        }
    }

    // ── Per-observation (raw/pre-stacked) detection ──
    // Match each observation to the corresponding magnitude.
    for (i, obs) in observations.iter().enumerate() {
        let band_name = &obs.band.0;

        // Look up the apparent magnitude for this band at this time index.
        if let Some(mags) = evaluation.apparent_mags.get(band_name) {
            if i < mags.len() {
                let app_mag = mags[i];
                let depth_secondary = obs.five_sigma_depth + secondary_depth_boost;

                // Two-tier detection: check against secondary (looser) threshold first.
                if app_mag < depth_secondary {
                    n_detections += 1;
                    bands_detected.insert(band_name.clone());
                    *detections_per_band.entry(band_name.clone()).or_insert(0) += 1;
                    detection_mjds.push(obs.mjd);

                    // Also check primary (stricter) threshold.
                    if app_mag < obs.five_sigma_depth {
                        n_detections_primary += 1;
                    }

                    // Store detected photometry for rate computation.
                    let mag_err = mag_error_from_depth(app_mag, obs.five_sigma_depth);
                    band_photometry
                        .entry(band_name.clone())
                        .or_default()
                        .push((obs.mjd, app_mag, mag_err));
                }
            }
        }
    }

    let first_detection_mjd = detection_mjds.iter().copied().reduce(f64::min);
    let last_detection_mjd = detection_mjds.iter().copied().reduce(f64::max);

    // Find peak (brightest detection) across all bands.
    let mut peak_mjd: Option<f64> = None;
    let mut peak_mag: Option<f64> = None;
    // Collect all (mjd, mag) pairs for detected observations.
    let mut detected_phot: Vec<(f64, f64)> = Vec::new();

    for (i, obs) in observations.iter().enumerate() {
        let band_name = &obs.band.0;
        if let Some(mags) = evaluation.apparent_mags.get(band_name) {
            if i < mags.len() {
                let app_mag = mags[i];
                let depth_secondary = obs.five_sigma_depth
                    + if criteria.snr_threshold_secondary < criteria.snr_threshold
                        && criteria.snr_threshold_secondary > 0.0
                    {
                        2.5 * (criteria.snr_threshold / criteria.snr_threshold_secondary).log10()
                    } else {
                        0.0
                    };
                if app_mag < depth_secondary {
                    detected_phot.push((obs.mjd, app_mag));
                    match peak_mag {
                        Some(prev) if app_mag < prev => {
                            peak_mag = Some(app_mag);
                            peak_mjd = Some(obs.mjd);
                        }
                        None => {
                            peak_mag = Some(app_mag);
                            peak_mjd = Some(obs.mjd);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Count pre/post peak detections and phase range.
    let mut n_pre_peak = 0usize;
    let mut n_post_peak = 0usize;
    let mut phase_min_days: Option<f64> = None;
    let mut phase_max_days: Option<f64> = None;

    if let Some(pk_mjd) = peak_mjd {
        for &(mjd, _) in &detected_phot {
            let phase = mjd - pk_mjd;
            if phase < -0.01 {
                n_pre_peak += 1;
            } else if phase > 0.01 {
                n_post_peak += 1;
            }
            phase_min_days = Some(match phase_min_days {
                Some(prev) => prev.min(phase),
                None => phase,
            });
            phase_max_days = Some(match phase_max_days {
                Some(prev) => prev.max(phase),
                None => phase,
            });
        }
    }

    // Compute rise/decay rates per band (boom algorithm).
    let mut best_rise_rate: Option<f64> = None;
    let mut best_fade_rate: Option<f64> = None;

    for (_band, mut phot) in band_photometry {
        // Sort by time.
        phot.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Deduplicate same-night observations (within 0.5 days), keeping the
        // brightest. Multiple overlapping pointings from the same visit cover
        // the same position but are separate observations in the index.
        phot.dedup_by(|a, b| {
            if (a.0 - b.0).abs() < 0.5 {
                // Keep the brighter (lower mag) observation in b.
                if a.1 < b.1 {
                    b.1 = a.1;
                    b.2 = a.2;
                }
                true
            } else {
                false
            }
        });

        if phot.len() < 2 {
            continue;
        }

        let times: Vec<f64> = phot.iter().map(|p| p.0).collect();
        let mags: Vec<f64> = phot.iter().map(|p| p.1).collect();
        let errs: Vec<f64> = phot.iter().map(|p| p.2).collect();

        let (rising, fading) = compute_band_rates(&times, &mags, &errs);

        if let Some(ref r) = rising {
            // Rising = negative rate (brightening). We want the most negative.
            match best_rise_rate {
                Some(prev) if r.rate < prev => best_rise_rate = Some(r.rate),
                None => best_rise_rate = Some(r.rate),
                _ => {}
            }
        }
        if let Some(ref f_rate) = fading {
            // Fading = positive rate. We want the largest positive.
            match best_fade_rate {
                Some(prev) if f_rate.rate > prev => best_fade_rate = Some(f_rate.rate),
                None => best_fade_rate = Some(f_rate.rate),
                _ => {}
            }
        }
    }

    // Check fast transient criterion.
    let is_fast = {
        let rising_fast = best_rise_rate
            .map(|r| r.abs() >= criteria.min_rise_rate)
            .unwrap_or(false);
        let fading_fast = best_fade_rate
            .map(|r| r >= criteria.min_fade_rate)
            .unwrap_or(false);
        rising_fast || fading_fast
    };

    // Check timespan constraint.
    let timespan_ok = match (first_detection_mjd, last_detection_mjd) {
        (Some(first), Some(last)) => (last - first) <= criteria.max_timespan_days,
        _ => true, // No detections or single detection — timespan is trivially satisfied.
    };

    // Check minimum time separation between first and last detection.
    let time_separation_ok = if criteria.min_time_separation_hours > 0.0 {
        match (first_detection_mjd, last_detection_mjd) {
            (Some(first), Some(last)) => {
                (last - first) * 24.0 >= criteria.min_time_separation_hours
            }
            _ => false,
        }
    } else {
        true
    };

    // Check min_per_band: at least one band must have >= min_per_band detections.
    let per_band_ok = detections_per_band
        .values()
        .any(|&count| count >= criteria.min_per_band);

    // Check primary SNR tier.
    let primary_ok = n_detections_primary >= criteria.min_detections_primary;

    // Phase coverage check.
    let phase_range = match (phase_min_days, phase_max_days) {
        (Some(mn), Some(mx)) => mx - mn,
        _ => 0.0,
    };
    let phase_range_ok = phase_range >= criteria.min_phase_range_days;
    let pre_peak_ok = n_pre_peak >= criteria.min_pre_peak_detections;
    let post_peak_ok = n_post_peak >= criteria.min_post_peak_detections;

    let mut detected = n_detections >= criteria.min_detections
        && bands_detected.len() >= criteria.min_bands
        && (criteria.min_per_band == 0 || per_band_ok)
        && timespan_ok
        && time_separation_ok
        && primary_ok
        && pre_peak_ok
        && post_peak_ok
        && phase_range_ok;

    // Apply fast transient requirement.
    if criteria.require_fast_transient && detected {
        detected = is_fast;
    }

    DetectionResult {
        detected,
        n_detections,
        n_detections_primary,
        n_bands_detected: bands_detected.len(),
        first_detection_mjd,
        last_detection_mjd,
        detections_per_band,
        best_rise_rate,
        best_fade_rate,
        is_fast_transient: is_fast,
        peak_mjd,
        peak_mag,
        n_pre_peak,
        n_post_peak,
        phase_min_days,
        phase_max_days,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survey::SurveyObservation;
    use crate::types::{Band, SkyCoord};

    fn make_obs(mjd: f64, band: &str, depth: f64) -> SurveyObservation {
        SurveyObservation {
            obs_id: 0,
            coord: SkyCoord::new(0.0, 0.0),
            mjd,
            band: Band::new(band),
            five_sigma_depth: depth,
            seeing_fwhm: 1.0,
            exposure_time: 30.0,
            airmass: 1.0,
            sky_brightness: 21.0,
            night: 0,
        }
    }

    #[test]
    fn test_bright_transient_detected() {
        let obs = vec![
            make_obs(60000.0, "g", 25.0),
            make_obs(60001.0, "r", 24.5),
        ];
        let obs_refs: Vec<&SurveyObservation> = obs.iter().collect();

        // Each band's mag array is indexed by observation index.
        // Obs 0 is g-band, obs 1 is r-band.
        let mut mags = HashMap::new();
        mags.insert("g".to_string(), vec![22.0, 99.0]); // obs 0: detected, obs 1: N/A
        mags.insert("r".to_string(), vec![99.0, 23.0]); // obs 0: N/A, obs 1: detected

        let eval = LightcurveEvaluation {
            apparent_mags: mags,
            times_mjd: vec![60000.0, 60001.0],
        };

        let criteria = DetectionCriteria {
            min_detections: 2,
            min_bands: 1,
            ..Default::default()
        };

        let result = evaluate_detection(&eval, &obs_refs, &criteria);
        assert!(result.detected);
        assert_eq!(result.n_detections, 2);
        assert_eq!(result.n_bands_detected, 2);
    }

    #[test]
    fn test_fast_transient_rate_detection() {
        // Simulate a kilonova: brightens 2 mag over 1 day, then fades 0.5 mag/day.
        // 5 observations over 5 days in g-band.
        let obs = vec![
            make_obs(60000.0, "g", 25.0), // pre-peak: mag 23.0
            make_obs(60001.0, "g", 25.0), // peak: mag 21.0
            make_obs(60002.0, "g", 25.0), // +1d: mag 21.5
            make_obs(60003.0, "g", 25.0), // +2d: mag 22.0
            make_obs(60004.0, "g", 25.0), // +3d: mag 22.5
        ];
        let obs_refs: Vec<&SurveyObservation> = obs.iter().collect();

        let mut mags = HashMap::new();
        // Mag array indexed by observation index, all in g-band.
        mags.insert(
            "g".to_string(),
            vec![23.0, 21.0, 21.5, 22.0, 22.5],
        );

        let eval = LightcurveEvaluation {
            apparent_mags: mags,
            times_mjd: vec![60000.0, 60001.0, 60002.0, 60003.0, 60004.0],
        };

        // With fast transient requirement.
        let criteria = DetectionCriteria {
            require_fast_transient: true,
            ..Default::default()
        };

        let result = evaluate_detection(&eval, &obs_refs, &criteria);
        assert!(result.detected, "Expected detection with fast transient");
        assert!(result.is_fast_transient);
        // Rise rate should be ~-2 mag/day, fade rate should be ~0.5 mag/day.
        assert!(
            result.best_rise_rate.unwrap().abs() > 1.0,
            "Rise rate {:?} should exceed 1 mag/day",
            result.best_rise_rate
        );
        assert!(
            result.best_fade_rate.unwrap() > 0.3,
            "Fade rate {:?} should exceed 0.3 mag/day",
            result.best_fade_rate
        );
    }

    #[test]
    fn test_slow_transient_rejected_by_rate() {
        // A slow transient: changes 0.1 mag over 10 days.
        let obs: Vec<_> = (0..10)
            .map(|i| make_obs(60000.0 + i as f64, "r", 25.0))
            .collect();
        let obs_refs: Vec<&SurveyObservation> = obs.iter().collect();

        let mut mags = HashMap::new();
        let slow_mags: Vec<f64> = (0..10).map(|i| 22.0 - 0.01 * i as f64).collect();
        mags.insert("r".to_string(), slow_mags);

        let eval = LightcurveEvaluation {
            apparent_mags: mags,
            times_mjd: (0..10).map(|i| 60000.0 + i as f64).collect(),
        };

        let criteria = DetectionCriteria {
            require_fast_transient: true,
            ..Default::default()
        };

        let result = evaluate_detection(&eval, &obs_refs, &criteria);
        assert!(!result.is_fast_transient);
        assert!(!result.detected, "Slow transient should be rejected");
    }

    #[test]
    fn test_faint_transient_not_detected() {
        let obs = vec![
            make_obs(60000.0, "g", 25.0),
            make_obs(60001.0, "r", 24.5),
        ];
        let obs_refs: Vec<&SurveyObservation> = obs.iter().collect();

        let mut mags = HashMap::new();
        mags.insert("g".to_string(), vec![26.0, 99.0]); // fainter than 25.0
        mags.insert("r".to_string(), vec![99.0, 25.0]); // fainter than 24.5

        let eval = LightcurveEvaluation {
            apparent_mags: mags,
            times_mjd: vec![60000.0, 60001.0],
        };

        let criteria = DetectionCriteria::default();
        let result = evaluate_detection(&eval, &obs_refs, &criteria);
        assert!(!result.detected);
        assert_eq!(result.n_detections, 0);
    }
}
