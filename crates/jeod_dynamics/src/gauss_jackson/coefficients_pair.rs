//! Summed Adams / Gauss-Jackson coefficient pair.
//!
//! Port of JEOD's `GaussJacksonCoefficientsPair`
//! (`gauss_jackson_coefficients_pair.hh/cc`).
//!
//! Each pair holds two coefficient vectors (Summed Adams for velocity,
//! Gauss-Jackson/Störmer-Cowell for position) in ordinate form. The
//! `apply` method computes weighted sums of acceleration history using
//! both coefficient sets simultaneously.

use super::two_d_array::{OffsetRows, TwoDArray};
use glam::DVec3;

/// A pair of coefficient vectors: Summed Adams (1st integral) and
/// Gauss-Jackson (2nd integral), both in ordinate form.
///
/// JEOD: `GaussJacksonCoefficientsPair` in
/// `gauss_jackson_coefficients_pair.hh`.
#[derive(Debug, Clone)]
pub(crate) struct CoefficientsPair {
    /// Summed Adams coefficients in ordinate form.
    /// JEOD: `sa_coefs`.
    pub sa_coefs: Vec<f64>,
    /// Gauss-Jackson (Störmer-Cowell) coefficients in ordinate form.
    /// JEOD: `gj_coefs`.
    pub gj_coefs: Vec<f64>,
}

impl CoefficientsPair {
    /// Allocate coefficient arrays for the given max order.
    /// JEOD: `GaussJacksonCoefficientsPair::configure(max_order)`.
    pub fn configure(max_order: usize) -> Self {
        Self {
            sa_coefs: vec![0.0; max_order + 1],
            gj_coefs: vec![0.0; max_order + 1],
        }
    }

    /// Apply both coefficient sets to acceleration history (TwoDArray, no offset).
    ///
    /// JEOD: `GaussJacksonCoefficientsPair::apply(nelem, ncoeff, acc_hist, state_sum)`.
    ///
    /// Returns (vel_sum, pos_sum) where:
    ///   vel_sum = Σ sa_coefs[i] * acc_hist[i]
    ///   pos_sum = Σ gj_coefs[i] * acc_hist[i]
    /// for i in 0..ncoeff.
    pub fn apply(&self, acc_hist: &TwoDArray, ncoeff: usize) -> (DVec3, DVec3) {
        // JEOD: Initialize from first history point
        let acc_0 = acc_hist.get_dvec3(0);
        let mut vel_sum = acc_0 * self.sa_coefs[0];
        let mut pos_sum = acc_0 * self.gj_coefs[0];

        // JEOD: Accumulate remaining history points
        for icoeff in 1..ncoeff {
            let acc_i = acc_hist.get_dvec3(icoeff);
            vel_sum += acc_i * self.sa_coefs[icoeff];
            pos_sum += acc_i * self.gj_coefs[icoeff];
        }

        (vel_sum, pos_sum)
    }

    /// Apply both coefficient sets to acceleration history with row offset.
    ///
    /// Used during BootstrapStep where the relevant history subset starts
    /// at `acc_hist + history_length - (order+1)`.
    pub fn apply_offset(&self, acc_hist: &OffsetRows<'_>, ncoeff: usize) -> (DVec3, DVec3) {
        let acc_0 = acc_hist.get_dvec3(0);
        let mut vel_sum = acc_0 * self.sa_coefs[0];
        let mut pos_sum = acc_0 * self.gj_coefs[0];

        for icoeff in 1..ncoeff {
            let acc_i = acc_hist.get_dvec3(icoeff);
            vel_sum += acc_i * self.sa_coefs[icoeff];
            pos_sum += acc_i * self.gj_coefs[icoeff];
        }

        (vel_sum, pos_sum)
    }

    /// Apply both coefficient sets, skipping the first row of acc_hist.
    ///
    /// JEOD: `coeff->corrector[order].apply(size, order, ahist + 1, corrector_sum)`
    /// Used in `integrate_gj()` to pre-compute the corrector sum excluding
    /// the latest acceleration.
    pub fn apply_skip_first(&self, acc_hist: &TwoDArray, ncoeff: usize) -> (DVec3, DVec3) {
        let acc_0 = acc_hist.get_dvec3(1); // Start from row 1
        let mut vel_sum = acc_0 * self.sa_coefs[0];
        let mut pos_sum = acc_0 * self.gj_coefs[0];

        for icoeff in 1..ncoeff {
            let acc_i = acc_hist.get_dvec3(1 + icoeff);
            vel_sum += acc_i * self.sa_coefs[icoeff];
            pos_sum += acc_i * self.gj_coefs[icoeff];
        }

        (vel_sum, pos_sum)
    }

    /// Apply both coefficient sets, skipping first row from offset view.
    pub fn apply_offset_skip_first(
        &self,
        acc_hist: &OffsetRows<'_>,
        ncoeff: usize,
    ) -> (DVec3, DVec3) {
        let acc_0 = acc_hist.get_dvec3(1);
        let mut vel_sum = acc_0 * self.sa_coefs[0];
        let mut pos_sum = acc_0 * self.gj_coefs[0];

        for icoeff in 1..ncoeff {
            let acc_i = acc_hist.get_dvec3(1 + icoeff);
            vel_sum += acc_i * self.sa_coefs[icoeff];
            pos_sum += acc_i * self.gj_coefs[icoeff];
        }

        (vel_sum, pos_sum)
    }
}
