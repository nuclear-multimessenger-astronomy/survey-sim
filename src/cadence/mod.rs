use std::collections::HashMap;
use std::fmt;

use crate::survey::SurveyStore;
use crate::types::SkyCoord;

/// Per-band cadence statistics aggregated over sky positions.
pub struct BandCadenceStats {
    pub band: String,
    pub n_gaps: usize,
    pub n_positions: usize,
    pub mean_days: f64,
    pub median_days: f64,
    pub std_days: f64,
    /// Percentiles: 10th, 25th, 50th, 75th, 90th.
    pub percentiles: [f64; 5],
    pub min_days: f64,
    pub max_days: f64,
}

/// Full cadence analysis result.
pub struct ReturnTimeAnalysis {
    pub band_stats: Vec<BandCadenceStats>,
    pub all_bands_stats: BandCadenceStats,
    pub n_positions_sampled: usize,
}

impl ReturnTimeAnalysis {
    /// Run return-time analysis on a `SurveyStore`.
    ///
    /// Samples unique pointing centers from the survey, queries all observations
    /// at each position, and computes inter-visit gap statistics per band and
    /// across all bands.
    ///
    /// `max_positions`: cap on sky positions to sample (0 = all unique pointings).
    pub fn analyze(survey: &SurveyStore, max_positions: usize) -> Self {
        // Step 1: Collect unique pointing centers by rounding to ~0.1° grid.
        let mut grid: HashMap<(i32, i32), SkyCoord> = HashMap::new();
        for obs in survey.observations() {
            let ra_key = (obs.coord.ra * 10.0).round() as i32;
            let dec_key = (obs.coord.dec * 10.0).round() as i32;
            grid.entry((ra_key, dec_key))
                .or_insert_with(|| obs.coord.clone());
        }

        let mut positions: Vec<SkyCoord> = grid.into_values().collect();
        // Sort for determinism (by dec then ra).
        positions.sort_by(|a, b| {
            a.dec
                .partial_cmp(&b.dec)
                .unwrap()
                .then(a.ra.partial_cmp(&b.ra).unwrap())
        });

        if max_positions > 0 && positions.len() > max_positions {
            // Subsample evenly.
            let step = positions.len() as f64 / max_positions as f64;
            positions = (0..max_positions)
                .map(|i| positions[(i as f64 * step) as usize])
                .collect();
        }

        let n_positions_sampled = positions.len();

        // Step 2-5: For each position, collect gaps per band and across all bands.
        let mut band_gaps: HashMap<String, Vec<f64>> = HashMap::new();
        let mut band_positions: HashMap<String, usize> = HashMap::new();
        let mut all_gaps: Vec<f64> = Vec::new();
        let mut all_positions_with_revisit = 0usize;

        for coord in &positions {
            let indices = survey.query(coord, survey.mjd_min, survey.mjd_max);
            if indices.len() < 2 {
                continue;
            }

            // Group by band.
            let mut by_band: HashMap<String, Vec<f64>> = HashMap::new();
            let mut all_mjds: Vec<f64> = Vec::new();

            for &idx in &indices {
                let obs = survey.get(idx);
                by_band
                    .entry(obs.band.0.clone())
                    .or_default()
                    .push(obs.mjd);
                all_mjds.push(obs.mjd);
            }

            // Per-band gaps.
            for (band, mut mjds) in by_band {
                mjds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if mjds.len() >= 2 {
                    *band_positions.entry(band.clone()).or_insert(0) += 1;
                    let gaps = band_gaps.entry(band).or_default();
                    for w in mjds.windows(2) {
                        gaps.push(w[1] - w[0]);
                    }
                }
            }

            // Any-filter gaps.
            all_mjds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if all_mjds.len() >= 2 {
                all_positions_with_revisit += 1;
                for w in all_mjds.windows(2) {
                    all_gaps.push(w[1] - w[0]);
                }
            }
        }

        // Step 6-7: Compute statistics.
        let mut band_names: Vec<String> = band_gaps.keys().cloned().collect();
        band_names.sort();

        let band_stats: Vec<BandCadenceStats> = band_names
            .iter()
            .map(|band| {
                let gaps = band_gaps.get(band).unwrap();
                let n_pos = *band_positions.get(band).unwrap_or(&0);
                compute_stats(band.clone(), gaps, n_pos)
            })
            .collect();

        let all_bands_stats = compute_stats(
            "all_bands".to_string(),
            &all_gaps,
            all_positions_with_revisit,
        );

        Self {
            band_stats,
            all_bands_stats,
            n_positions_sampled,
        }
    }
}

fn compute_stats(band: String, gaps: &[f64], n_positions: usize) -> BandCadenceStats {
    if gaps.is_empty() {
        return BandCadenceStats {
            band,
            n_gaps: 0,
            n_positions,
            mean_days: f64::NAN,
            median_days: f64::NAN,
            std_days: f64::NAN,
            percentiles: [f64::NAN; 5],
            min_days: f64::NAN,
            max_days: f64::NAN,
        };
    }

    let mut sorted = gaps.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    let percentile = |p: f64| -> f64 {
        let rank = p / 100.0 * (n - 1) as f64;
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let frac = rank - lo as f64;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    };

    BandCadenceStats {
        band,
        n_gaps: n,
        n_positions,
        mean_days: mean,
        median_days: percentile(50.0),
        std_days: std,
        percentiles: [
            percentile(10.0),
            percentile(25.0),
            percentile(50.0),
            percentile(75.0),
            percentile(90.0),
        ],
        min_days: sorted[0],
        max_days: sorted[n - 1],
    }
}

impl fmt::Display for BandCadenceStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  Band: {}", self.band)?;
        writeln!(
            f,
            "    Positions with ≥2 visits: {}, total gaps: {}",
            self.n_positions, self.n_gaps
        )?;
        writeln!(
            f,
            "    Mean: {:.2} d, Median: {:.2} d, Std: {:.2} d",
            self.mean_days, self.median_days, self.std_days
        )?;
        writeln!(
            f,
            "    Percentiles [10,25,50,75,90]: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
            self.percentiles[0],
            self.percentiles[1],
            self.percentiles[2],
            self.percentiles[3],
            self.percentiles[4],
        )?;
        write!(
            f,
            "    Min: {:.2} d, Max: {:.2} d",
            self.min_days, self.max_days
        )
    }
}

impl fmt::Display for ReturnTimeAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Return-Time Cadence Analysis ===")?;
        writeln!(f, "Sky positions sampled: {}", self.n_positions_sampled)?;
        writeln!(f)?;
        for stats in &self.band_stats {
            writeln!(f, "{}", stats)?;
        }
        writeln!(f)?;
        writeln!(f, "--- Any-filter revisit ---")?;
        write!(f, "{}", self.all_bands_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survey::{SurveyObservation, SurveyStore};
    use crate::types::{Band, SkyCoord};

    /// Create a synthetic survey with daily cadence in two bands at a single pointing.
    fn make_daily_survey() -> SurveyStore {
        let mut observations = Vec::new();
        let coord = SkyCoord::new(180.0, -30.0);

        for day in 0..30 {
            let mjd = 60000.0 + day as f64;
            // g-band observation
            observations.push(SurveyObservation {
                obs_id: (day * 2) as u64,
                coord,
                mjd,
                band: Band::new("g"),
                five_sigma_depth: 24.0,
                seeing_fwhm: 0.8,
                exposure_time: 30.0,
                airmass: 1.2,
                sky_brightness: 21.0,
                night: day as i64,
            });
            // r-band observation 0.01 days later
            observations.push(SurveyObservation {
                obs_id: (day * 2 + 1) as u64,
                coord,
                mjd: mjd + 0.01,
                band: Band::new("r"),
                five_sigma_depth: 23.5,
                seeing_fwhm: 0.7,
                exposure_time: 30.0,
                airmass: 1.2,
                sky_brightness: 20.5,
                night: day as i64,
            });
        }

        SurveyStore::new(observations, 64)
    }

    #[test]
    fn test_daily_cadence_return_time() {
        let survey = make_daily_survey();
        let analysis = ReturnTimeAnalysis::analyze(&survey, 0);

        // Should find exactly 1 position.
        assert_eq!(analysis.n_positions_sampled, 1);

        // Per-band: g and r should both have ~1 day median return time.
        for stats in &analysis.band_stats {
            assert!(
                (stats.median_days - 1.0).abs() < 0.1,
                "{}: expected ~1.0 day median, got {:.3}",
                stats.band,
                stats.median_days,
            );
            assert_eq!(stats.n_gaps, 29);
        }

        // Any-filter: alternating g/r with 0.01 day gap, so median ~0.01 or ~0.99.
        // The gaps alternate: 0.01, 0.99, 0.01, 0.99, ...
        // Median of those should be close to 0.5 (middle of sorted list).
        assert!(
            analysis.all_bands_stats.median_days < 1.0,
            "any-filter median should be < 1 day, got {:.3}",
            analysis.all_bands_stats.median_days,
        );
    }
}
