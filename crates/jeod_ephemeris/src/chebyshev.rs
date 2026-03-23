/// Evaluate Chebyshev polynomial expansion for position and velocity.
///
/// Ported from JEOD `de4xx_file_update.cc` lines 296-358.
///
/// # Arguments
/// * `coeffs` — Chebyshev coefficients for one component (nterms values)
/// * `nterms` — number of Chebyshev terms
/// * `x` — normalized time in [-1, 1] (Chebyshev domain)
///
/// # Returns
/// `(position, velocity_unnormalized)` — velocity needs scaling by
/// `2 * npoly / (interval_days * 86400)` to get m/s.
pub fn chebyshev_evaluate(coeffs: &[f64], nterms: usize, x: f64) -> (f64, f64) {
    debug_assert!(nterms >= 2);
    debug_assert!(coeffs.len() >= nterms);

    // Chebyshev polynomials: T[0]=1, T[1]=x, T[k]=2*x*T[k-1]-T[k-2]
    // Derivatives: dT[0]=0, dT[1]=1, dT[k]=2*T[k-1]+2*x*dT[k-1]-dT[k-2]
    let mut t_prev2 = 1.0; // T[0]
    let mut t_prev1 = x;   // T[1]
    let mut dt_prev2 = 0.0; // dT[0]
    let mut dt_prev1 = 1.0; // dT[1]

    let mut pos = coeffs[0] + coeffs[1] * x;
    let mut vel = coeffs[1]; // dT[1] * coeffs[1]

    for coeff in &coeffs[2..nterms] {
        let t_k = 2.0 * x * t_prev1 - t_prev2;
        let dt_k = 2.0 * t_prev1 + 2.0 * x * dt_prev1 - dt_prev2;

        pos += coeff * t_k;
        vel += coeff * dt_k;

        t_prev2 = t_prev1;
        t_prev1 = t_k;
        dt_prev2 = dt_prev1;
        dt_prev1 = dt_k;
    }

    (pos, vel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_polynomial() {
        // T[0] = 1, so coeffs = [5.0] gives position=5.0, velocity=0.0
        let (pos, vel) = chebyshev_evaluate(&[5.0, 0.0], 2, 0.0);
        assert!((pos - 5.0).abs() < 1e-15);
        assert!(vel.abs() < 1e-15);
    }

    #[test]
    fn linear_polynomial() {
        // coeffs = [0.0, 3.0]: position = 3*x, velocity = 3
        let (pos, vel) = chebyshev_evaluate(&[0.0, 3.0], 2, 0.5);
        assert!((pos - 1.5).abs() < 1e-15);
        assert!((vel - 3.0).abs() < 1e-15);
    }

    #[test]
    fn quadratic_t2() {
        // T[2] = 2x^2 - 1. coeffs = [0, 0, 1]: pos = 2x^2-1, vel = 4x
        let x = 0.3;
        let (pos, vel) = chebyshev_evaluate(&[0.0, 0.0, 1.0], 3, x);
        let expected_pos = 2.0 * x * x - 1.0;
        let expected_vel = 4.0 * x; // d/dx(2x^2-1) = 4x
        assert!((pos - expected_pos).abs() < 1e-14);
        assert!((vel - expected_vel).abs() < 1e-14);
    }
}
