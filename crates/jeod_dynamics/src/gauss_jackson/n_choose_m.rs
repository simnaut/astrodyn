//! Cached binomial coefficient calculator.
//!
//! Port of JEOD's `NChooseM` (er7_utils/math/include/n_choose_m.hh).
//! Stores Pascal's triangle for fast lookup of C(n, m).
//! Uses symmetry: C(n, m) = C(n, n-m) to store only floor((n+1)/2) entries per row.

/// Cached binomial coefficient calculator using Pascal's triangle.
///
/// JEOD: `NChooseM` in `er7_utils/math/include/n_choose_m.hh`.
pub(crate) struct NChooseM {
    /// Pascal's triangle stored as a flat array.
    /// Row n starts at index `row_index(n)` and contains `floor((n+1)/2)` elements.
    triangle: Vec<u64>,
    /// Number of rows currently computed.
    nrows: usize,
}

/// Starting index in `triangle` for row `n`.
/// JEOD: `NChooseM::row_index(n)`.
fn row_index(n: usize) -> usize {
    // Sum of floor((k+1)/2) for k=0..n-1
    // = floor(n^2/4) for n >= 0
    (n * n) / 4
}

impl NChooseM {
    /// Create a new calculator, optionally pre-computing rows.
    pub fn new() -> Self {
        Self {
            triangle: Vec::new(),
            nrows: 0,
        }
    }

    /// Compute C(n, m).
    /// JEOD: `NChooseM::operator()(n, m)`.
    /// Auto-resizes the triangle if needed.
    pub fn get(&mut self, n: usize, m: usize) -> u64 {
        assert!(m <= n, "NChooseM: m ({m}) > n ({n})");

        // Use symmetry: C(n, m) = C(n, n-m)
        let m = m.min(n - m);

        if m == 0 {
            return 1;
        }

        // Ensure triangle is large enough
        if n >= self.nrows {
            self.resize(n + 1);
        }

        let idx = row_index(n) + m - 1; // -1 because we don't store C(n,0)=1
        self.triangle[idx]
    }

    /// Resize the triangle to hold at least `new_nrows` rows.
    /// JEOD: `NChooseM::resize(new_size)`.
    fn resize(&mut self, new_nrows: usize) {
        if new_nrows <= self.nrows {
            return;
        }

        let new_size = row_index(new_nrows);
        self.triangle.resize(new_size, 0);

        for n in self.nrows..new_nrows {
            self.construct_row(n);
        }
        self.nrows = new_nrows;
    }

    /// Construct row `n` of Pascal's triangle.
    /// JEOD: `NChooseM::construct_row(n, last_known_row)`.
    fn construct_row(&mut self, n: usize) {
        let half = n / 2; // number of entries stored for row n (excluding C(n,0)=1)
        let base = row_index(n);

        if n == 0 {
            return; // Row 0 has no stored entries (C(0,0) = 1 is implicit)
        }
        if n == 1 {
            return; // Row 1: C(1,0)=1, C(1,1)=1 — both handled by m==0 or symmetry
        }

        // C(n, m) = C(n-1, m-1) + C(n-1, m)
        // Store entries for m = 1..half
        for m in 1..=half {
            let left = if m == 1 {
                // C(n-1, 0) = 1
                1u64
            } else {
                let prev_m_use = (m - 1).min(n - 1 - (m - 1));
                if prev_m_use == 0 {
                    1u64
                } else {
                    self.triangle[row_index(n - 1) + prev_m_use - 1]
                }
            };
            let right = {
                let m_use = m.min(n - 1 - m);
                if m_use == 0 {
                    1u64
                } else {
                    self.triangle[row_index(n - 1) + m_use - 1]
                }
            };
            self.triangle[base + m - 1] = left + right;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_binomial() {
        let mut ncm = NChooseM::new();
        assert_eq!(ncm.get(0, 0), 1);
        assert_eq!(ncm.get(1, 0), 1);
        assert_eq!(ncm.get(1, 1), 1);
        assert_eq!(ncm.get(5, 2), 10);
        assert_eq!(ncm.get(10, 3), 120);
        assert_eq!(ncm.get(10, 5), 252);
    }

    #[test]
    fn test_symmetry() {
        let mut ncm = NChooseM::new();
        for n in 0..20 {
            for m in 0..=n {
                assert_eq!(ncm.get(n, m), ncm.get(n, n - m));
            }
        }
    }

    #[test]
    fn test_pascals_rule() {
        let mut ncm = NChooseM::new();
        for n in 1..15 {
            for m in 1..n {
                assert_eq!(ncm.get(n, m), ncm.get(n - 1, m - 1) + ncm.get(n - 1, m));
            }
        }
    }
}
