pub mod rates;

use std::collections::HashMap;

/// N-dimensional binned efficiency grid.
///
/// Each bin stores (n_detected, n_total) for computing detection efficiency.
/// Axes can include: redshift, sky pixel, intrinsic parameters, time-of-year, etc.
pub struct EfficiencyGrid {
    /// Axis definitions: name -> bin edges.
    pub axes: Vec<GridAxis>,
    /// Flat storage of (n_detected, n_total) for each bin.
    /// Indexed by multi-dimensional bin index flattened in row-major order.
    data: Vec<(u64, u64)>,
    /// Total number of bins.
    n_bins: usize,
    /// Strides for each axis (for computing flat index).
    strides: Vec<usize>,
}

/// Definition of a single grid axis.
#[derive(Clone, Debug)]
pub struct GridAxis {
    pub name: String,
    /// Bin edges (n_bins + 1 values).
    pub edges: Vec<f64>,
}

impl GridAxis {
    pub fn new(name: &str, edges: Vec<f64>) -> Self {
        Self {
            name: name.to_string(),
            edges,
        }
    }

    /// Create an axis with uniform bins.
    pub fn uniform(name: &str, lo: f64, hi: f64, n_bins: usize) -> Self {
        let step = (hi - lo) / n_bins as f64;
        let edges: Vec<f64> = (0..=n_bins).map(|i| lo + i as f64 * step).collect();
        Self::new(name, edges)
    }

    /// Number of bins.
    pub fn n_bins(&self) -> usize {
        self.edges.len() - 1
    }

    /// Find the bin index for a value. Returns None if out of range.
    pub fn bin_index(&self, value: f64) -> Option<usize> {
        if value < self.edges[0] || value >= *self.edges.last().unwrap() {
            return None;
        }
        // Binary search for the right bin.
        let pos = self
            .edges
            .partition_point(|&edge| edge <= value)
            .saturating_sub(1);
        if pos < self.n_bins() {
            Some(pos)
        } else {
            None
        }
    }
}

impl EfficiencyGrid {
    /// Create a new efficiency grid with the given axes.
    pub fn new(axes: Vec<GridAxis>) -> Self {
        let n_bins: usize = axes.iter().map(|a| a.n_bins()).product();
        let mut strides = vec![1usize; axes.len()];
        for i in (0..axes.len() - 1).rev() {
            strides[i] = strides[i + 1] * axes[i + 1].n_bins();
        }
        Self {
            axes,
            data: vec![(0, 0); n_bins],
            n_bins,
            strides,
        }
    }

    /// Record a transient in the grid.
    ///
    /// `values` maps axis name to the value for that axis.
    /// `detected` is whether the transient was detected.
    pub fn record(&mut self, values: &HashMap<String, f64>, detected: bool) {
        if let Some(idx) = self.flat_index(values) {
            self.data[idx].1 += 1; // n_total
            if detected {
                self.data[idx].0 += 1; // n_detected
            }
        }
    }

    /// Compute the flat index from axis values.
    fn flat_index(&self, values: &HashMap<String, f64>) -> Option<usize> {
        let mut idx = 0;
        for (i, axis) in self.axes.iter().enumerate() {
            let val = values.get(&axis.name)?;
            let bin = axis.bin_index(*val)?;
            idx += bin * self.strides[i];
        }
        if idx < self.n_bins {
            Some(idx)
        } else {
            None
        }
    }

    /// Get efficiency (n_detected / n_total) for a given bin.
    pub fn efficiency_at(&self, values: &HashMap<String, f64>) -> Option<f64> {
        let idx = self.flat_index(values)?;
        let (detected, total) = self.data[idx];
        if total == 0 {
            Some(0.0)
        } else {
            Some(detected as f64 / total as f64)
        }
    }

    /// Marginalize over one axis, returning a map from bin centers to efficiency.
    ///
    /// Sums (n_detected, n_total) over all other axes for each bin of the target axis.
    pub fn marginalize_over(&self, axis_name: &str) -> Option<Vec<(f64, f64)>> {
        let axis_idx = self.axes.iter().position(|a| a.name == axis_name)?;
        let axis = &self.axes[axis_idx];
        let n_bins = axis.n_bins();

        let mut sums = vec![(0u64, 0u64); n_bins];

        for (flat_idx, &(det, tot)) in self.data.iter().enumerate() {
            // Recover the bin index for the target axis.
            let bin = (flat_idx / self.strides[axis_idx]) % n_bins;
            sums[bin].0 += det;
            sums[bin].1 += tot;
        }

        Some(
            sums.iter()
                .enumerate()
                .map(|(i, &(det, tot))| {
                    let center = (axis.edges[i] + axis.edges[i + 1]) / 2.0;
                    let eff = if tot == 0 {
                        0.0
                    } else {
                        det as f64 / tot as f64
                    };
                    (center, eff)
                })
                .collect(),
        )
    }

    /// Get raw data as (n_detected, n_total) pairs in flat row-major order.
    pub fn raw_data(&self) -> &[(u64, u64)] {
        &self.data
    }

    /// Get the shape of the grid (number of bins per axis).
    pub fn shape(&self) -> Vec<usize> {
        self.axes.iter().map(|a| a.n_bins()).collect()
    }

    /// Total transients recorded.
    pub fn total_recorded(&self) -> u64 {
        self.data.iter().map(|&(_, t)| t).sum()
    }

    /// Total transients detected.
    pub fn total_detected(&self) -> u64 {
        self.data.iter().map(|&(d, _)| d).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_basic() {
        let axes = vec![
            GridAxis::uniform("z", 0.0, 0.3, 3),
            GridAxis::uniform("sky", 0.0, 360.0, 4),
        ];
        let mut grid = EfficiencyGrid::new(axes);
        assert_eq!(grid.shape(), vec![3, 4]);

        let mut vals = HashMap::new();
        vals.insert("z".to_string(), 0.05);
        vals.insert("sky".to_string(), 45.0);

        grid.record(&vals, true);
        grid.record(&vals, false);
        grid.record(&vals, true);

        assert_eq!(grid.efficiency_at(&vals), Some(2.0 / 3.0));
        assert_eq!(grid.total_recorded(), 3);
        assert_eq!(grid.total_detected(), 2);
    }

    #[test]
    fn test_marginalize() {
        let axes = vec![
            GridAxis::uniform("z", 0.0, 0.3, 3),
        ];
        let mut grid = EfficiencyGrid::new(axes);

        let mut vals = HashMap::new();
        // Low z bin: 2/2 detected.
        vals.insert("z".to_string(), 0.05);
        grid.record(&vals, true);
        grid.record(&vals, true);

        // High z bin: 0/2 detected.
        vals.insert("z".to_string(), 0.25);
        grid.record(&vals, false);
        grid.record(&vals, false);

        let marg = grid.marginalize_over("z").unwrap();
        assert_eq!(marg.len(), 3);
        assert!((marg[0].1 - 1.0).abs() < 1e-10); // low z: 100%
        assert!((marg[2].1 - 0.0).abs() < 1e-10); // high z: 0%
    }
}
