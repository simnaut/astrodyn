//! 2D array with row rotation and downsampling.
//!
//! Port of JEOD's `TwoDArray<double>` (two_d_array.hh).
//! Uses `Vec<Vec<f64>>` for O(1) row swap operations (rotate, downsample).

use glam::DVec3;

/// 2D array of f64 with efficient row rotation and downsampling.
///
/// JEOD: `DoubleTwoDArray` / `TwoDArray<double>` in `two_d_array.hh`.
/// Rows are stored as separate `Vec<f64>` to enable O(1) pointer-swap
/// operations matching JEOD's `row_array` design.
#[derive(Debug, Clone)]
pub(crate) struct TwoDArray {
    rows: Vec<Vec<f64>>,
    cols: usize,
}

impl TwoDArray {
    /// Create an empty array.
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            cols: 0,
        }
    }

    /// Allocate (or reallocate) an n×m array, zero-filled.
    /// JEOD: `TwoDArray::allocate(N, M)`.
    pub fn allocate(&mut self, nrows: usize, ncols: usize) {
        self.cols = ncols;
        self.rows = vec![vec![0.0; ncols]; nrows];
    }

    /// Copy a DVec3 into row `n`.
    pub fn set_dvec3(&mut self, n: usize, v: DVec3) {
        let row = &mut self.rows[n];
        row[0] = v.x;
        row[1] = v.y;
        row[2] = v.z;
    }

    /// Read row `n` as a DVec3.
    pub fn get_dvec3(&self, n: usize) -> DVec3 {
        let row = &self.rows[n];
        DVec3::new(row[0], row[1], row[2])
    }

    /// Rotate rows 0..=limit downward.
    /// Row 0 moves to position `limit`; others shift down by one.
    ///
    /// JEOD: `TwoDArray::rotate_down(limit)`.
    /// ```text
    /// Before: [A, B, C, D, ...]
    /// After:  [B, C, D, A, ...]  (for limit=3)
    /// ```
    pub fn rotate_down(&mut self, limit: usize) {
        // JEOD implementation: save row 0, shift 1..=limit down, put saved at limit.
        // Using Vec::swap this is just a series of swaps, or we can rotate the slice.
        // Vec rotation: rows[0..=limit].rotate_left(1) does exactly this.
        self.rows[..=limit].rotate_left(1);
    }

    /// Downsample by swapping pointers for every-other row.
    /// After downsample, rows 0..limit contain every-other original row.
    ///
    /// JEOD: `TwoDArray::downsample(limit)`.
    /// ```text
    /// For i in 1..limit: swap(rows[i], rows[2*i])
    /// ```
    pub fn downsample(&mut self, limit: usize) {
        for i in 1..limit {
            self.rows.swap(i, 2 * i);
        }
    }

    /// Get a pointer to row pointers, matching JEOD's `operator const double* const*()`.
    /// Used for offset arithmetic: `acc_hist + offset` in JEOD becomes
    /// `acc_hist.offset_rows(offset)`.
    pub fn offset_rows(&self, offset: usize) -> OffsetRows<'_> {
        OffsetRows {
            array: self,
            offset,
        }
    }
}

/// A view into a TwoDArray starting at a given row offset.
/// Equivalent to JEOD's `acc_hist + history_length - (order+1)` pointer arithmetic.
pub(crate) struct OffsetRows<'a> {
    array: &'a TwoDArray,
    offset: usize,
}

impl<'a> OffsetRows<'a> {
    pub fn get_dvec3(&self, i: usize) -> DVec3 {
        self.array.get_dvec3(self.offset + i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_access() {
        let mut arr = TwoDArray::new();
        arr.allocate(3, 3);
        arr.set_dvec3(0, DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(arr.get_dvec3(0), DVec3::new(1.0, 2.0, 3.0));
        assert_eq!(arr.get_dvec3(1), DVec3::ZERO);
    }

    #[test]
    fn test_rotate_down() {
        let mut arr = TwoDArray::new();
        arr.allocate(4, 1);
        arr.rows[0][0] = 0.0;
        arr.rows[1][0] = 1.0;
        arr.rows[2][0] = 2.0;
        arr.rows[3][0] = 3.0;
        arr.rotate_down(3);
        assert_eq!(arr.rows[0][0], 1.0);
        assert_eq!(arr.rows[1][0], 2.0);
        assert_eq!(arr.rows[2][0], 3.0);
        assert_eq!(arr.rows[3][0], 0.0);
    }

    #[test]
    fn test_downsample() {
        let mut arr = TwoDArray::new();
        arr.allocate(5, 1);
        for i in 0..5 {
            arr.rows[i][0] = i as f64;
        }
        // Before: [0, 1, 2, 3, 4]
        // downsample(3): swap(1,2), swap(2,4)
        // After: [0, 2, 4, 3, 1]
        arr.downsample(3);
        assert_eq!(arr.rows[0][0], 0.0);
        assert_eq!(arr.rows[1][0], 2.0);
        assert_eq!(arr.rows[2][0], 4.0);
    }
}
