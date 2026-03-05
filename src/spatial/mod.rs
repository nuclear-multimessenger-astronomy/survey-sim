use std::collections::HashMap;

/// HEALPix-based spatial index for fast sky-position lookups.
///
/// Maps each HEALPix pixel (at the configured NSIDE) to the list of
/// observation indices whose pointing centers fall within that pixel.
pub struct SpatialIndex {
    nside: u32,
    depth: u8,
    /// Map from HEALPix pixel hash to list of observation indices.
    pixel_map: HashMap<u64, Vec<usize>>,
}

impl SpatialIndex {
    /// Build a spatial index from a list of (ra_deg, dec_deg) coordinates.
    ///
    /// Each coordinate is mapped to its HEALPix nested pixel at the given NSIDE.
    pub fn new(coords: &[(f64, f64)], nside: u32) -> Self {
        let depth = nside_to_depth(nside);
        let mut pixel_map: HashMap<u64, Vec<usize>> = HashMap::new();

        for (idx, &(ra_deg, dec_deg)) in coords.iter().enumerate() {
            let pixel = lonlat_to_hash(ra_deg, dec_deg, depth);
            pixel_map.entry(pixel).or_default().push(idx);
        }

        log::debug!(
            "Built HEALPix spatial index: nside={}, depth={}, {} pixels populated",
            nside,
            depth,
            pixel_map.len()
        );

        Self {
            nside,
            depth,
            pixel_map,
        }
    }

    /// Query all observation indices in the same HEALPix pixel as the given position.
    pub fn query(&self, ra_deg: f64, dec_deg: f64) -> Vec<usize> {
        let pixel = lonlat_to_hash(ra_deg, dec_deg, self.depth);
        self.pixel_map
            .get(&pixel)
            .cloned()
            .unwrap_or_default()
    }

    /// Query observation indices in the pixel and its immediate neighbors.
    pub fn query_with_neighbors(&self, ra_deg: f64, dec_deg: f64) -> Vec<usize> {
        let pixel = lonlat_to_hash(ra_deg, dec_deg, self.depth);
        let neighbors = nested_neighbors(pixel, self.depth);

        let mut result = self.pixel_map.get(&pixel).cloned().unwrap_or_default();
        for nb in neighbors {
            if let Some(indices) = self.pixel_map.get(&nb) {
                result.extend(indices);
            }
        }
        result
    }

    /// Query observation indices within a cone of given radius around a position.
    ///
    /// Uses `cdshealpix::nested::cone_coverage_approx` to compute the MOC
    /// (Multi-Order Coverage) of the cone, then looks up all observations
    /// whose pointing centers fall in any covered pixel. This correctly
    /// handles arbitrary FoV sizes regardless of NSIDE.
    ///
    /// Falls back to neighbor queries at very low depth (< 3) where the
    /// cone coverage algorithm has numerical issues.
    pub fn query_cone(&self, ra_deg: f64, dec_deg: f64, radius_deg: f64) -> Vec<usize> {
        // At very low depth, cone_coverage_approx can panic on boundary cases.
        // Fall back to neighbor queries (sufficient when pixels are large).
        if self.depth < 3 {
            return self.query_with_neighbors(ra_deg, dec_deg);
        }

        use cdshealpix::nested::cone_coverage_approx;

        let lon_rad = ra_deg.to_radians();
        // Clamp latitude to avoid edge cases at exact poles.
        let lat_rad = dec_deg.to_radians().clamp(
            -std::f64::consts::FRAC_PI_2 + 1e-10,
            std::f64::consts::FRAC_PI_2 - 1e-10,
        );
        let radius_rad = radius_deg.to_radians();

        let cone_bmoc = cone_coverage_approx(self.depth, lon_rad, lat_rad, radius_rad);

        let mut result = Vec::new();
        for pixel_hash in cone_bmoc.flat_iter() {
            if let Some(indices) = self.pixel_map.get(&pixel_hash) {
                result.extend(indices);
            }
        }
        result
    }

    /// Number of populated pixels.
    pub fn n_pixels(&self) -> usize {
        self.pixel_map.len()
    }

    /// NSIDE parameter.
    pub fn nside(&self) -> u32 {
        self.nside
    }
}

/// Convert NSIDE to HEALPix depth (order). NSIDE = 2^depth.
fn nside_to_depth(nside: u32) -> u8 {
    assert!(nside.is_power_of_two(), "NSIDE must be a power of 2");
    nside.trailing_zeros() as u8
}

/// Convert (lon_deg, lat_deg) to a HEALPix nested pixel hash.
///
/// Uses the cdshealpix crate for the actual computation.
fn lonlat_to_hash(ra_deg: f64, dec_deg: f64, depth: u8) -> u64 {
    use cdshealpix::nested;
    let ra_rad = ra_deg.to_radians();
    let dec_rad = dec_deg.to_radians();
    nested::hash(depth, ra_rad, dec_rad)
}

/// Get the neighboring pixel indices for a nested HEALPix pixel.
fn nested_neighbors(hash: u64, depth: u8) -> Vec<u64> {
    use cdshealpix::compass_point::MainWind;
    use cdshealpix::nested;
    let main_map = nested::neighbours(depth, hash, false);
    let mut result = Vec::with_capacity(8);
    // The MainWind enum has 8 cardinal/intercardinal directions.
    let directions = [
        MainWind::S,
        MainWind::SE,
        MainWind::E,
        MainWind::SW,
        MainWind::NE,
        MainWind::W,
        MainWind::NW,
        MainWind::N,
    ];
    for dir in directions {
        if let Some(nb) = main_map.get(dir) {
            result.push(*nb);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nside_to_depth() {
        assert_eq!(nside_to_depth(1), 0);
        assert_eq!(nside_to_depth(2), 1);
        assert_eq!(nside_to_depth(64), 6);
        assert_eq!(nside_to_depth(1024), 10);
    }

    #[test]
    fn test_spatial_index_basic() {
        // Create a few coordinates and verify they're indexed.
        let coords = vec![
            (10.0, 20.0),
            (10.1, 20.1),
            (180.0, -45.0),
        ];
        let index = SpatialIndex::new(&coords, 64);
        assert!(index.n_pixels() >= 1);

        // Query the first coordinate — should return at least index 0.
        let results = index.query(10.0, 20.0);
        assert!(results.contains(&0));
    }

    #[test]
    fn test_cone_query_covers_fov() {
        // Place observations at RA=0, Dec=0 and RA=1.5, Dec=0 (1.5° apart).
        // A cone query at RA=0 with radius 2° should find both.
        let coords = vec![
            (0.0, 0.0),
            (1.5, 0.0),
            (10.0, 0.0), // far away, should NOT be found
        ];
        let index = SpatialIndex::new(&coords, 64);
        let results = index.query_cone(0.0, 0.0, 2.0);
        assert!(results.contains(&0), "Center point should be found");
        assert!(results.contains(&1), "Point 1.5° away should be found");
        assert!(!results.contains(&2), "Point 10° away should NOT be found");
    }

    #[test]
    fn test_query_returns_nearby() {
        // Two identical points should be in the same pixel.
        let coords = vec![
            (45.0, 30.0),
            (45.0, 30.0),
        ];
        let index = SpatialIndex::new(&coords, 64);
        let results = index.query(45.0, 30.0);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
    }
}
