use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Sky coordinate in equatorial (ICRS) frame.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SkyCoord {
    /// Right ascension in degrees [0, 360).
    pub ra: f64,
    /// Declination in degrees [-90, 90].
    pub dec: f64,
}

impl SkyCoord {
    pub fn new(ra: f64, dec: f64) -> Self {
        Self { ra, dec }
    }

    /// Right ascension in radians.
    pub fn ra_rad(&self) -> f64 {
        self.ra.to_radians()
    }

    /// Declination in radians.
    pub fn dec_rad(&self) -> f64 {
        self.dec.to_radians()
    }

    /// Galactic latitude b in degrees [-90, 90].
    ///
    /// Uses the standard ICRS → Galactic transformation with the NGP at
    /// (ra, dec) = (192.85948°, 27.12825°) and l_NCP = 122.93192°.
    pub fn galactic_lat(&self) -> f64 {
        let d2r = std::f64::consts::PI / 180.0;
        // North Galactic Pole in ICRS
        let ra_ngp = 192.85948 * d2r;
        let dec_ngp = 27.12825 * d2r;

        let ra = self.ra * d2r;
        let dec = self.dec * d2r;

        let sin_b = dec.sin() * dec_ngp.sin()
            + dec.cos() * dec_ngp.cos() * (ra - ra_ngp).cos();
        sin_b.clamp(-1.0, 1.0).asin() / d2r
    }

    /// Angular separation to another coordinate in degrees.
    pub fn separation(&self, other: &SkyCoord) -> f64 {
        let d2r = std::f64::consts::PI / 180.0;
        let dec1 = self.dec * d2r;
        let dec2 = other.dec * d2r;
        let dra = (self.ra - other.ra) * d2r;
        let cos_sep = dec1.sin() * dec2.sin() + dec1.cos() * dec2.cos() * dra.cos();
        cos_sep.clamp(-1.0, 1.0).acos() / d2r
    }
}

/// Photometric band identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Band(pub String);

impl Band {
    pub fn new(name: &str) -> Self {
        Self(name.to_string())
    }
}

impl fmt::Display for Band {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Classification of transient type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransientType {
    Kilonova,
    SupernovaIa,
    SupernovaII,
    SupernovaIbc,
    Tde,
    Fbot,
    Afterglow,
    Custom,
}

impl fmt::Display for TransientType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kilonova => write!(f, "Kilonova"),
            Self::SupernovaIa => write!(f, "SNIa"),
            Self::SupernovaII => write!(f, "SNII"),
            Self::SupernovaIbc => write!(f, "SNIbc"),
            Self::Tde => write!(f, "TDE"),
            Self::Fbot => write!(f, "FBOT"),
            Self::Afterglow => write!(f, "Afterglow"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Flat LCDM cosmology parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cosmology {
    pub h: f64,
    pub omega_m: f64,
    pub omega_lambda: f64,
}

impl Default for Cosmology {
    fn default() -> Self {
        Self {
            h: 0.674,
            omega_m: 0.315,
            omega_lambda: 0.685,
        }
    }
}

/// A single transient instance drawn from a population.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransientInstance {
    /// Sky position.
    pub coord: SkyCoord,
    /// Cosmological redshift.
    pub z: f64,
    /// Luminosity distance in Mpc.
    pub d_l: f64,
    /// Explosion epoch in MJD.
    pub t_exp: f64,
    /// Peak absolute magnitude (used for normalization).
    pub peak_abs_mag: f64,
    /// Transient type.
    pub transient_type: TransientType,
    /// Model-specific parameters (e.g., ejecta mass, velocity, opacity).
    pub model_params: HashMap<String, f64>,
    /// Milky Way extinction A_V along this line of sight.
    pub mw_extinction_av: f64,
    /// Host galaxy extinction A_V.
    pub host_extinction_av: f64,
}
