use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Location of the telescope.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "location")]
pub enum Location {
    #[serde(rename = "ground")]
    Ground {
        latitude_deg: f64,
        longitude_deg: f64,
        altitude_m: f64,
    },
    #[serde(rename = "space")]
    Space,
}

/// Telescope properties.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelescopeConfig {
    /// Effective aperture diameter in meters.
    pub aperture_m: f64,
    /// Location (ground with coordinates or space).
    #[serde(flatten)]
    pub location: Location,
}

/// Detector properties.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// Arcsec per pixel.
    pub plate_scale_arcsec: f64,
    /// Read noise in electrons.
    pub read_noise_e: f64,
    /// Dark current in electrons/s/pixel.
    pub dark_current_e_per_s: f64,
    /// Detector gain.
    pub gain: f64,
    /// Total field of view in square degrees.
    pub fov_deg2: f64,
}

/// Properties of a single photometric band.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BandConfig {
    /// Central wavelength in nanometers.
    pub central_wavelength_nm: f64,
    /// Bandwidth in nanometers.
    pub width_nm: f64,
    /// Typical seeing FWHM in arcsec.
    pub typical_seeing_arcsec: f64,
    /// Single-visit 5-sigma limiting magnitude.
    pub single_visit_depth: f64,
    /// Typical sky brightness in mag/arcsec^2.
    pub sky_brightness: f64,
}

/// Observing constraints and parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservingConfig {
    /// Standard visit exposure time in seconds.
    pub default_exposure_s: f64,
    /// Readout time in seconds.
    pub readout_s: f64,
    /// Maximum slew speed in deg/s.
    pub slew_rate_deg_per_s: f64,
    /// Settling time after slew in seconds.
    pub settle_s: f64,
    /// Minimum altitude above horizon in degrees.
    pub min_altitude_deg: f64,
    /// Maximum airmass.
    pub max_airmass: f64,
    /// Minimum Moon separation in degrees.
    pub min_moon_sep_deg: f64,
}

/// Complete instrument configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstrumentConfig {
    /// Instrument name (e.g. "Rubin LSST").
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Telescope properties.
    pub telescope: TelescopeConfig,
    /// Detector properties.
    pub detector: DetectorConfig,
    /// Band definitions, keyed by band name.
    pub bands: HashMap<String, BandConfig>,
    /// Observing constraints.
    pub observing: ObservingConfig,
    /// Extinction coefficients A_band / A_V (Cardelli R_V=3.1).
    #[serde(default)]
    pub extinction_coefficients: HashMap<String, f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum InstrumentError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

impl InstrumentConfig {
    /// Load an instrument configuration from a YAML file.
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self, InstrumentError> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    /// Parse an instrument configuration from a YAML string.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, InstrumentError> {
        let config: Self = serde_yaml::from_str(yaml)?;
        Ok(config)
    }

    /// Get the extinction coefficient (A_band / A_V) for a band.
    /// Falls back to hardcoded Cardelli values if not in the config.
    pub fn extinction_ratio(&self, band: &str) -> f64 {
        if let Some(&ratio) = self.extinction_coefficients.get(band) {
            return ratio;
        }
        // Fallback to standard Cardelli values.
        match band {
            "FUV" => 8.4,
            "NUV" => 8.0,
            "u" | "U" => 1.56,
            "g" | "B" | "bessellb" => 1.31,
            "r" | "V" | "bessellv" => 1.0,
            "i" | "R" | "bessellr" => 0.75,
            "z" => 0.55,
            "y" | "Y" => 0.47,
            "J" => 0.29,
            "H" => 0.18,
            "K" | "Ks" => 0.11,
            _ => 1.0,
        }
    }

    /// Built-in Rubin LSST instrument configuration.
    pub fn rubin() -> Self {
        Self {
            name: "Rubin LSST".to_string(),
            description: Some(
                "Vera C. Rubin Observatory Legacy Survey of Space and Time".to_string(),
            ),
            telescope: TelescopeConfig {
                aperture_m: 6.423,
                location: Location::Ground {
                    latitude_deg: -30.2446,
                    longitude_deg: -70.7494,
                    altitude_m: 2663.0,
                },
            },
            detector: DetectorConfig {
                plate_scale_arcsec: 0.2,
                read_noise_e: 8.8,
                dark_current_e_per_s: 0.2,
                gain: 1.0,
                fov_deg2: 9.6,
            },
            bands: HashMap::from([
                (
                    "u".to_string(),
                    BandConfig {
                        central_wavelength_nm: 367.0,
                        width_nm: 52.0,
                        typical_seeing_arcsec: 0.92,
                        single_visit_depth: 23.9,
                        sky_brightness: 22.95,
                    },
                ),
                (
                    "g".to_string(),
                    BandConfig {
                        central_wavelength_nm: 482.0,
                        width_nm: 140.0,
                        typical_seeing_arcsec: 0.87,
                        single_visit_depth: 25.0,
                        sky_brightness: 22.24,
                    },
                ),
                (
                    "r".to_string(),
                    BandConfig {
                        central_wavelength_nm: 622.0,
                        width_nm: 135.0,
                        typical_seeing_arcsec: 0.83,
                        single_visit_depth: 24.7,
                        sky_brightness: 21.20,
                    },
                ),
                (
                    "i".to_string(),
                    BandConfig {
                        central_wavelength_nm: 754.0,
                        width_nm: 125.0,
                        typical_seeing_arcsec: 0.80,
                        single_visit_depth: 24.0,
                        sky_brightness: 20.47,
                    },
                ),
                (
                    "z".to_string(),
                    BandConfig {
                        central_wavelength_nm: 869.0,
                        width_nm: 100.0,
                        typical_seeing_arcsec: 0.78,
                        single_visit_depth: 23.3,
                        sky_brightness: 19.60,
                    },
                ),
                (
                    "y".to_string(),
                    BandConfig {
                        central_wavelength_nm: 971.0,
                        width_nm: 100.0,
                        typical_seeing_arcsec: 0.76,
                        single_visit_depth: 22.1,
                        sky_brightness: 18.61,
                    },
                ),
            ]),
            observing: ObservingConfig {
                default_exposure_s: 30.0,
                readout_s: 2.0,
                slew_rate_deg_per_s: 1.5,
                settle_s: 1.0,
                min_altitude_deg: 20.0,
                max_airmass: 2.5,
                min_moon_sep_deg: 30.0,
            },
            extinction_coefficients: HashMap::from([
                ("u".to_string(), 1.56),
                ("g".to_string(), 1.31),
                ("r".to_string(), 1.00),
                ("i".to_string(), 0.75),
                ("z".to_string(), 0.55),
                ("y".to_string(), 0.47),
            ]),
        }
    }

    /// Built-in ZTF instrument configuration.
    pub fn ztf() -> Self {
        Self {
            name: "ZTF".to_string(),
            description: Some("Zwicky Transient Facility".to_string()),
            telescope: TelescopeConfig {
                aperture_m: 1.22,
                location: Location::Ground {
                    latitude_deg: 33.3564,
                    longitude_deg: -116.8650,
                    altitude_m: 1712.0,
                },
            },
            detector: DetectorConfig {
                plate_scale_arcsec: 1.01,
                read_noise_e: 10.0,
                dark_current_e_per_s: 0.1,
                gain: 6.2,
                fov_deg2: 47.0,
            },
            bands: HashMap::from([
                (
                    "g".to_string(),
                    BandConfig {
                        central_wavelength_nm: 472.0,
                        width_nm: 124.0,
                        typical_seeing_arcsec: 2.0,
                        single_visit_depth: 20.8,
                        sky_brightness: 22.0,
                    },
                ),
                (
                    "r".to_string(),
                    BandConfig {
                        central_wavelength_nm: 634.0,
                        width_nm: 160.0,
                        typical_seeing_arcsec: 2.0,
                        single_visit_depth: 20.6,
                        sky_brightness: 21.0,
                    },
                ),
                (
                    "i".to_string(),
                    BandConfig {
                        central_wavelength_nm: 782.0,
                        width_nm: 145.0,
                        typical_seeing_arcsec: 2.1,
                        single_visit_depth: 19.9,
                        sky_brightness: 20.3,
                    },
                ),
            ]),
            observing: ObservingConfig {
                default_exposure_s: 30.0,
                readout_s: 10.0,
                slew_rate_deg_per_s: 2.5,
                settle_s: 3.0,
                min_altitude_deg: 20.0,
                max_airmass: 2.5,
                min_moon_sep_deg: 15.0,
            },
            extinction_coefficients: HashMap::from([
                ("g".to_string(), 1.31),
                ("r".to_string(), 1.00),
                ("i".to_string(), 0.75),
            ]),
        }
    }

    /// Built-in ULTRASAT instrument configuration.
    ///
    /// Parameters from m4opt and Shvartzvald et al. (2024).
    /// Geosynchronous orbit, 33cm aperture, NUV 230–290nm.
    pub fn ultrasat() -> Self {
        Self {
            name: "ULTRASAT".to_string(),
            description: Some(
                "Ultraviolet Transient Astronomy Satellite".to_string(),
            ),
            telescope: TelescopeConfig {
                aperture_m: 0.33,
                location: Location::Space,
            },
            detector: DetectorConfig {
                plate_scale_arcsec: 5.4,
                read_noise_e: 6.0,
                dark_current_e_per_s: 0.04, // 12/300
                gain: 1.0,
                fov_deg2: 203.9, // 14.28° × 14.28°
            },
            bands: HashMap::from([
                (
                    "NUV".to_string(),
                    BandConfig {
                        central_wavelength_nm: 260.0,
                        width_nm: 68.0, // σ=34nm → FWHM≈80nm, effective width ~68nm
                        typical_seeing_arcsec: 5.4, // pixel-limited
                        single_visit_depth: 22.5, // 3×300s stacked
                        sky_brightness: 25.0, // low UV background in space
                    },
                ),
            ]),
            observing: ObservingConfig {
                default_exposure_s: 300.0,
                readout_s: 0.0, // CMOS, negligible
                slew_rate_deg_per_s: 1.0,
                settle_s: 5.0,
                min_altitude_deg: 0.0, // space — use earth limb constraint instead
                max_airmass: 99.0,     // space
                min_moon_sep_deg: 35.0,
            },
            extinction_coefficients: HashMap::from([
                ("NUV".to_string(), 8.0), // UV extinction A_NUV/A_V
            ]),
        }
    }

    /// Built-in UVEX instrument configuration.
    ///
    /// Parameters from m4opt and Kulkarni et al. (2021).
    /// TESS-like HEO orbit, 75cm aperture, FUV+NUV simultaneous.
    pub fn uvex() -> Self {
        Self {
            name: "UVEX".to_string(),
            description: Some(
                "UltraViolet EXplorer — NASA Medium Explorer".to_string(),
            ),
            telescope: TelescopeConfig {
                aperture_m: 0.75,
                location: Location::Space,
            },
            detector: DetectorConfig {
                plate_scale_arcsec: 1.0,
                read_noise_e: 2.0,
                dark_current_e_per_s: 0.001,
                gain: 0.85,
                fov_deg2: 12.25, // 3.5° × 3.5°
            },
            bands: HashMap::from([
                (
                    "FUV".to_string(),
                    BandConfig {
                        central_wavelength_nm: 160.0,
                        width_nm: 20.0, // σ=10nm → FWHM≈24nm
                        typical_seeing_arcsec: 1.0, // pixel-limited
                        single_visit_depth: 24.5, // 5σ at 900s dwell (m4opt)
                        sky_brightness: 27.0,
                    },
                ),
                (
                    "NUV".to_string(),
                    BandConfig {
                        central_wavelength_nm: 230.0,
                        width_nm: 36.0, // σ=18nm → FWHM≈42nm
                        typical_seeing_arcsec: 1.0,
                        single_visit_depth: 25.0, // 5σ at 900s dwell (m4opt)
                        sky_brightness: 26.0,
                    },
                ),
            ]),
            observing: ObservingConfig {
                default_exposure_s: 900.0, // standard dwell
                readout_s: 0.0,
                slew_rate_deg_per_s: 0.6,
                settle_s: 60.0,
                min_altitude_deg: 0.0,
                max_airmass: 99.0,
                min_moon_sep_deg: 25.0,
            },
            extinction_coefficients: HashMap::from([
                ("FUV".to_string(), 8.4), // A_FUV/A_V
                ("NUV".to_string(), 8.0), // A_NUV/A_V
            ]),
        }
    }

    /// Built-in Argus Array instrument configuration.
    pub fn argus() -> Self {
        Self {
            name: "Argus Array".to_string(),
            description: Some(
                "Argus Array — 1200-telescope array at UNC Chapel Hill".to_string(),
            ),
            telescope: TelescopeConfig {
                aperture_m: 8.0,
                location: Location::Ground {
                    latitude_deg: 35.9132,
                    longitude_deg: -79.0558,
                    altitude_m: 150.0,
                },
            },
            detector: DetectorConfig {
                plate_scale_arcsec: 1.0,
                read_noise_e: 1.4,
                dark_current_e_per_s: 0.1,
                gain: 1.0,
                fov_deg2: 8000.0,
            },
            bands: HashMap::from([
                (
                    "b".to_string(),
                    BandConfig {
                        central_wavelength_nm: 445.0,
                        width_nm: 160.0,
                        typical_seeing_arcsec: 2.0,
                        single_visit_depth: 20.5,
                        sky_brightness: 22.0,
                    },
                ),
                (
                    "r".to_string(),
                    BandConfig {
                        central_wavelength_nm: 630.0,
                        width_nm: 140.0,
                        typical_seeing_arcsec: 2.0,
                        single_visit_depth: 20.0,
                        sky_brightness: 21.0,
                    },
                ),
            ]),
            observing: ObservingConfig {
                default_exposure_s: 60.0,
                readout_s: 0.001,
                slew_rate_deg_per_s: 0.0,
                settle_s: 50.0,
                min_altitude_deg: 20.0,
                max_airmass: 2.5,
                min_moon_sep_deg: 15.0,
            },
            extinction_coefficients: HashMap::from([
                ("b".to_string(), 1.40),
                ("r".to_string(), 0.87),
            ]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rubin_builtin() {
        let rubin = InstrumentConfig::rubin();
        assert_eq!(rubin.name, "Rubin LSST");
        assert_eq!(rubin.bands.len(), 6);
        assert!(rubin.bands.contains_key("u"));
        assert!(rubin.bands.contains_key("y"));
        assert!((rubin.detector.fov_deg2 - 9.6).abs() < 1e-10);
    }

    #[test]
    fn test_ztf_builtin() {
        let ztf = InstrumentConfig::ztf();
        assert_eq!(ztf.name, "ZTF");
        assert_eq!(ztf.bands.len(), 3);
        assert!((ztf.detector.fov_deg2 - 47.0).abs() < 1e-10);
    }

    #[test]
    fn test_ultrasat_builtin() {
        let ultrasat = InstrumentConfig::ultrasat();
        assert_eq!(ultrasat.name, "ULTRASAT");
        assert_eq!(ultrasat.bands.len(), 1);
        assert!(ultrasat.bands.contains_key("NUV"));
        assert!((ultrasat.detector.fov_deg2 - 203.9).abs() < 1e-1);
        assert!((ultrasat.telescope.aperture_m - 0.33).abs() < 1e-3);
        match &ultrasat.telescope.location {
            Location::Space => {}
            _ => panic!("ULTRASAT should be a space telescope"),
        }
    }

    #[test]
    fn test_uvex_builtin() {
        let uvex = InstrumentConfig::uvex();
        assert_eq!(uvex.name, "UVEX");
        assert_eq!(uvex.bands.len(), 2);
        assert!(uvex.bands.contains_key("FUV"));
        assert!(uvex.bands.contains_key("NUV"));
        assert!((uvex.detector.fov_deg2 - 12.25).abs() < 1e-2);
        assert!((uvex.telescope.aperture_m - 0.75).abs() < 1e-3);
    }

    #[test]
    fn test_argus_builtin() {
        let argus = InstrumentConfig::argus();
        assert_eq!(argus.name, "Argus Array");
        assert_eq!(argus.bands.len(), 2);
        assert!(argus.bands.contains_key("b"));
        assert!(argus.bands.contains_key("r"));
        assert!((argus.detector.fov_deg2 - 8000.0).abs() < 1e-10);
        assert!((argus.telescope.aperture_m - 8.0).abs() < 1e-10);
        assert!((argus.detector.read_noise_e - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_extinction_ratio_from_config() {
        let rubin = InstrumentConfig::rubin();
        assert!((rubin.extinction_ratio("u") - 1.56).abs() < 1e-10);
        assert!((rubin.extinction_ratio("r") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_extinction_ratio_fallback() {
        let rubin = InstrumentConfig::rubin();
        // J is not in rubin's extinction_coefficients, should fall back to Cardelli
        assert!((rubin.extinction_ratio("J") - 0.29).abs() < 1e-10);
    }

    #[test]
    fn test_yaml_roundtrip() {
        let rubin = InstrumentConfig::rubin();
        let yaml = serde_yaml::to_string(&rubin).unwrap();
        let parsed = InstrumentConfig::from_yaml_str(&yaml).unwrap();
        assert_eq!(parsed.name, rubin.name);
        assert_eq!(parsed.bands.len(), rubin.bands.len());
    }

    #[test]
    fn test_ground_location() {
        let rubin = InstrumentConfig::rubin();
        match &rubin.telescope.location {
            Location::Ground {
                latitude_deg,
                longitude_deg,
                altitude_m,
            } => {
                assert!((*latitude_deg - (-30.2446)).abs() < 1e-4);
                assert!((*longitude_deg - (-70.7494)).abs() < 1e-4);
                assert!((*altitude_m - 2663.0).abs() < 1e-1);
            }
            Location::Space => panic!("Rubin should be a ground telescope"),
        }
    }
}
