//! Adams and Störmer-Cowell coefficient generation in rational arithmetic.
//!
//! Port of JEOD's `GaussJacksonRationalCoefficients`
//! (`gauss_jackson_rational_coeffs.cc`).
//!
//! Generates Adams corrector, Störmer-Cowell corrector, and predictor
//! coefficients in difference form using exact rational arithmetic, then
//! converts to ordinate form via binomial transform.

use super::n_choose_m::NChooseM;
use super::ratio128::Ratio128;

/// A set of Adams or Störmer-Cowell coefficients in difference form.
///
/// JEOD: `GaussJacksonRationalCoefficients` in
/// `gauss_jackson_rational_coeffs.hh`.
#[derive(Clone, Debug)]
pub(crate) struct RationalCoefficients {
    pub coefficients: Vec<Ratio128>,
}

impl RationalCoefficients {
    /// Configure as Adams corrector coefficients in difference form.
    ///
    /// JEOD: `GaussJacksonRationalCoefficients::configure_adams_corrector(nelem)`.
    /// Uses the recurrence:
    ///   c_0 = 1
    ///   c_n = -Σ_{i=0}^{n-1} c_i / (n+1-i)
    /// See Berry eqn 2.37 and 2.38.
    pub fn configure_adams_corrector(nelem: usize) -> Self {
        let mut coefficients = vec![Ratio128::ZERO; nelem];

        // c_0 = 1
        coefficients[0] = Ratio128::ONE;

        for nn in 1..nelem {
            let mut sum = Ratio128::ZERO;
            for (ii, coeff) in coefficients.iter().enumerate().take(nn) {
                sum -= *coeff / Ratio128::from((nn + 1 - ii) as i64);
            }
            coefficients[nn] = sum;
        }

        Self { coefficients }
    }

    /// Construct Störmer-Cowell corrector coefficients from Adams coefficients.
    ///
    /// JEOD: `GaussJacksonRationalCoefficients::construct_stormer_cowell_corrector()`.
    /// Convolution: q_i = Σ_{k=0}^{i} c_k * c_{i-k}
    /// See Berry eqn 2.55.
    pub fn construct_stormer_cowell_corrector(&self) -> Self {
        let nelem = self.coefficients.len();
        let mut result = vec![Ratio128::ZERO; nelem];

        for (ii, res) in result.iter_mut().enumerate().take(nelem) {
            let mut sum = Ratio128::ZERO;
            for kk in 0..=ii {
                sum += self.coefficients[kk] * self.coefficients[ii - kk];
            }
            *res = sum;
        }

        Self {
            coefficients: result,
        }
    }

    /// Construct predictor coefficients from corrector coefficients.
    ///
    /// JEOD: `GaussJacksonRationalCoefficients::construct_predictor()`.
    /// Cumulative sum: γ_i = Σ_{k=0}^{i} c_k
    /// See Berry eqn 2.43.
    /// Works for both Adams and Störmer-Cowell coefficients.
    pub fn construct_predictor(&self) -> Self {
        let nelem = self.coefficients.len();
        let mut result = vec![Ratio128::ZERO; nelem];

        let mut sum = Ratio128::ZERO;
        for (res, coeff) in result.iter_mut().zip(self.coefficients.iter()).take(nelem) {
            sum += *coeff;
            *res = sum;
        }

        Self {
            coefficients: result,
        }
    }

    /// Convert to ordinate form via binomial transform.
    ///
    /// JEOD: `GaussJacksonRationalCoefficients::convert_to_ordinate_form()`.
    /// z_{Nm} = (-1)^m * Σ_{i=m}^{N} C(i,m) * z'_i
    /// Stored in reverse order: result[nelem-1-m] = z_{Nm}
    /// See Berry eqn 2.79.
    pub fn convert_to_ordinate_form(&self, n_choose_m: &mut NChooseM, result: &mut [f64]) {
        let nelem = self.coefficients.len();
        let mut rational_coefs = vec![Ratio128::ZERO; nelem];

        for mm in 0..nelem {
            let mut sum = Ratio128::ZERO;
            for ii in mm..nelem {
                sum += Ratio128::from(n_choose_m.get(ii, mm) as i64) * self.coefficients[ii];
            }
            if mm % 2 != 0 {
                sum = -sum;
            }
            // Store in reverse order to simplify calculations.
            // JEOD: `rational_coefs[nelem - 1 - mm] = sum`
            rational_coefs[nelem - 1 - mm] = sum;
        }

        // Convert rational coefficients to doubles.
        // JEOD: `std::copy(rational_coefs.begin(), rational_coefs.end(), result)`
        Ratio128::slice_to_f64(&rational_coefs, result);
    }

    /// Discard extra terms from front and back.
    ///
    /// JEOD: `GaussJacksonRationalCoefficients::discard_extra_terms(nfront, nback)`.
    /// Adams: discard 1 front + 1 back = 2 total.
    /// Störmer-Cowell: discard 2 front + 0 back = 2 total.
    pub fn discard_extra_terms(&mut self, nfront: usize, nback: usize) {
        assert_eq!(nfront + nback, 2);
        // Remove from front
        self.coefficients.drain(..nfront);
        // Remove from back
        let new_len = self.coefficients.len() - nback;
        self.coefficients.truncate(new_len);
    }

    /// Apply the (1-∇) backward difference operator.
    ///
    /// JEOD: `GaussJacksonRationalCoefficients::displace_back()`.
    /// c'_0 = c_0, c'_i = c_i - c_{i-1}
    pub fn displace_back(&mut self) {
        let nelem = self.coefficients.len();
        let mut new_coefficients = vec![Ratio128::ZERO; nelem];

        new_coefficients[0] = self.coefficients[0];
        for (nc, w) in new_coefficients
            .iter_mut()
            .skip(1)
            .zip(self.coefficients.windows(2))
        {
            *nc = w[1] - w[0];
        }

        self.coefficients = new_coefficients;
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "rational-coefficient tests assert bit-exact equality of computed values against exact rationals"
)]
mod tests {
    use super::*;

    #[test]
    fn test_adams_corrector_order3() {
        // For order 3 (nelem=3+3=6), verify first few coefficients.
        // c_0 = 1
        // c_1 = -1/2
        // c_2 = -1/12
        // c_3 = -1/24
        let ac = RationalCoefficients::configure_adams_corrector(6);
        assert_eq!(ac.coefficients[0].to_f64(), 1.0);
        assert!((ac.coefficients[1].to_f64() - (-0.5)).abs() < 1e-15);
        assert!((ac.coefficients[2].to_f64() - (-1.0 / 12.0)).abs() < 1e-15);
        assert!((ac.coefficients[3].to_f64() - (-1.0 / 24.0)).abs() < 1e-15);
    }

    #[test]
    fn test_stormer_cowell_is_convolution() {
        // Verify q_0 = c_0^2 = 1
        // q_1 = 2*c_0*c_1 = -1
        let ac = RationalCoefficients::configure_adams_corrector(6);
        let sc = ac.construct_stormer_cowell_corrector();
        assert_eq!(sc.coefficients[0].to_f64(), 1.0);
        assert!((sc.coefficients[1].to_f64() - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn test_predictor_is_cumulative_sum() {
        let ac = RationalCoefficients::configure_adams_corrector(4);
        let ap = ac.construct_predictor();
        // γ_0 = c_0 = 1
        // γ_1 = c_0 + c_1 = 1 - 1/2 = 1/2
        assert_eq!(ap.coefficients[0].to_f64(), 1.0);
        assert!((ap.coefficients[1].to_f64() - 0.5).abs() < 1e-15);
    }
}
