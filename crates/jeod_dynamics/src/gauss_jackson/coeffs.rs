//! Gauss-Jackson coefficient computation pipeline.
//!
//! Port of JEOD's `GaussJacksonCoeffs` (`gauss_jackson_coeffs.hh/cc`).
//!
//! Generates predictor and corrector coefficient pairs (Summed Adams +
//! Störmer-Cowell) from rational arithmetic, converts to ordinate form,
//! and computes displaced corrector sets for mid-correction during bootstrap.

use super::coefficients_pair::CoefficientsPair;
use super::n_choose_m::NChooseM;
use super::rational_coeffs::RationalCoefficients;

/// Complete set of Gauss-Jackson predictor and corrector coefficients.
///
/// JEOD: `GaussJacksonCoeffs` in `gauss_jackson_coeffs.hh`.
#[derive(Debug, Clone)]
pub(crate) struct GaussJacksonCoeffs {
    /// Predictor coefficient pair (SA + GJ).
    pub predictor: CoefficientsPair,
    /// Corrector coefficient pairs indexed by displacement.
    /// `corrector[order]` is the undisplaced corrector.
    /// `corrector[order-i]` is displaced back by `i` steps via `(1-∇)^i`.
    /// JEOD: `GaussJacksonCoefficientsPair* corrector` (array of size max_order+1).
    pub corrector: Vec<CoefficientsPair>,
    pub max_order: usize,
    pub order: usize,
}

impl GaussJacksonCoeffs {
    /// Allocate coefficient storage for the given max order.
    /// JEOD: `GaussJacksonCoeffs::configure(max_order_in)`.
    pub fn configure(max_order: usize) -> Self {
        let predictor = CoefficientsPair::configure(max_order);
        let corrector = (0..=max_order)
            .map(|_| CoefficientsPair::configure(max_order))
            .collect();

        Self {
            predictor,
            corrector,
            max_order,
            order: 0,
        }
    }

    /// Compute coefficients for the given order.
    ///
    /// JEOD: `GaussJacksonCoeffs::compute_coeffs(order_in)`.
    /// Pipeline:
    /// 1. Adams corrector in difference form (order + 3 elements)
    /// 2. Störmer-Cowell corrector via convolution
    /// 3. Predictor from corrector via cumulative sum
    /// 4. Discard extra terms (Adams: 1 front + 1 back, S-C: 2 front + 0 back)
    /// 5. Convert to ordinate form via binomial transform
    /// 6. Displaced correctors via repeated (1-∇) application
    pub fn compute_coeffs(&mut self, order_in: usize) {
        assert!(order_in <= self.max_order);
        self.order = order_in;

        let mut n_choose_m = NChooseM::new();

        // Step 1: Adams corrector in difference form.
        // JEOD: `ac.configure_adams_corrector(order + 3)`
        let mut ac = RationalCoefficients::configure_adams_corrector(order_in + 3);

        // Step 2: Störmer-Cowell corrector via convolution.
        // JEOD: `sc = ac.construct_stormer_cowell_corrector()`
        let mut sc = ac.construct_stormer_cowell_corrector();

        // Step 3: Predictor from corrector via cumulative sum.
        // JEOD: `ap = ac.construct_predictor()`, `sp = sc.construct_predictor()`
        let mut ap = ac.construct_predictor();
        let mut sp = sc.construct_predictor();

        // Step 4: Discard extra terms.
        // Adams: 1 front + 1 back = 2 total
        // Störmer-Cowell: 2 front + 0 back = 2 total
        // JEOD: `ac.discard_extra_terms(1, 1)`, etc.
        ac.discard_extra_terms(1, 1);
        ap.discard_extra_terms(1, 1);
        sc.discard_extra_terms(2, 0);
        sp.discard_extra_terms(2, 0);

        // Step 5: Convert to ordinate form.
        // JEOD: `ap.convert_to_ordinate_form(n_choose_m, predictor.sa_coefs)`
        ap.convert_to_ordinate_form(&mut n_choose_m, &mut self.predictor.sa_coefs);
        sp.convert_to_ordinate_form(&mut n_choose_m, &mut self.predictor.gj_coefs);
        ac.convert_to_ordinate_form(&mut n_choose_m, &mut self.corrector[order_in].sa_coefs);
        sc.convert_to_ordinate_form(&mut n_choose_m, &mut self.corrector[order_in].gj_coefs);

        // Step 6: Displaced correctors.
        // JEOD: for ii in 1..=order: displace_back(), convert, store in corrector[order-ii]
        for ii in 1..=order_in {
            ac.displace_back();
            ac.convert_to_ordinate_form(
                &mut n_choose_m,
                &mut self.corrector[order_in - ii].sa_coefs,
            );
            sc.displace_back();
            sc.convert_to_ordinate_form(
                &mut n_choose_m,
                &mut self.corrector[order_in - ii].gj_coefs,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_coeffs_order4() {
        let mut coeffs = GaussJacksonCoeffs::configure(14);
        coeffs.compute_coeffs(4);

        // Predictor SA coefficients should be non-zero
        assert!(
            coeffs.predictor.sa_coefs[..5]
                .iter()
                .any(|c| c.abs() > 1e-15),
            "Predictor SA coefficients should be non-zero"
        );

        // Corrector[4] (undisplaced) should be non-zero
        assert!(
            coeffs.corrector[4].sa_coefs[..5]
                .iter()
                .any(|c| c.abs() > 1e-15),
            "Corrector[4] SA coefficients should be non-zero"
        );

        // Corrector[0] (most displaced) should be non-zero
        assert!(
            coeffs.corrector[0].sa_coefs[..5]
                .iter()
                .any(|c| c.abs() > 1e-15),
            "Corrector[0] SA coefficients should be non-zero"
        );
    }

    #[test]
    fn test_compute_coeffs_order8() {
        let mut coeffs = GaussJacksonCoeffs::configure(14);
        coeffs.compute_coeffs(8);

        // Verify predictor GJ coefficients exist
        let sum: f64 = coeffs.predictor.gj_coefs[..9].iter().sum();
        assert!(sum.abs() > 1e-10, "GJ predictor coefs should sum non-zero");
    }

    #[test]
    fn test_compute_coeffs_order12() {
        let mut coeffs = GaussJacksonCoeffs::configure(14);
        coeffs.compute_coeffs(12);

        // All 13 corrector sets (0..=12) should be populated
        for ii in 0..=12 {
            let sa_sum: f64 = coeffs.corrector[ii].sa_coefs[..13]
                .iter()
                .map(|c| c.abs())
                .sum();
            assert!(sa_sum > 1e-15, "corrector[{ii}] SA should be non-zero");
        }
    }
}
