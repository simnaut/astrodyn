//! Gottlieb (1993) spherical harmonics gravity computation.
//!
//! Direct port of JEOD `spherical_harmonics_calc_nonspherical.cc`.
//! The caller must provide position in planet-fixed coordinates and
//! rotate the result back to inertial.

use glam::{DMat3, DVec3};
use jeod_dynamics::GravityAcceleration;

use crate::spherical_harmonics_gravity_source::SphericalHarmonicsData;

/// sqrt(f64::MIN_POSITIVE) — underflow guard matching JEOD's SQRT_DBL_MIN.
const SQRT_DBL_MIN: f64 = 1.4916681462400413e-154;

/// Compute nonspherical gravity using the Gottlieb (1993) algorithm.
///
/// Direct port of JEOD `SphericalHarmonicsGravityControls::calc_nonspherical()`.
///
/// # Arguments
/// * `data` — precomputed spherical harmonics source data
/// * `posn_pf` — position in planet-fixed coordinates (m)
/// * `degree` — max degree for this computation (<= data.degree)
/// * `order` — max order for this computation (<= data.order)
/// * `compute_gradient` — whether to compute the gravity gradient tensor
/// * `gradient_degree` — max degree for gradient (0 = no gradient)
/// * `gradient_order` — max order for gradient
///
/// # Returns
/// `GravityAcceleration` with acceleration, gradient, and potential in
/// planet-fixed coordinates.
pub fn compute_nonspherical_gravity(
    data: &SphericalHarmonicsData,
    posn_pf: DVec3,
    degree: usize,
    order: usize,
    compute_gradient: bool,
    gradient_degree: usize,
    gradient_order: usize,
) -> GravityAcceleration {
    let degree = degree.min(data.degree);
    let order = order.min(data.order).min(degree);

    // If degree < 2, there are no harmonics to compute (only point-mass).
    // Return zero perturbation.
    if degree < 2 {
        return GravityAcceleration {
            accel: DVec3::ZERO,
            gradient: DMat3::ZERO,
            potential: 0.0,
        };
    }
    let gradient_degree = if compute_gradient {
        gradient_degree.min(degree)
    } else {
        0
    };
    let gradient_order = if compute_gradient {
        gradient_order.min(order).min(gradient_degree)
    } else {
        0
    };

    // Compute position vector magnitude
    let r_mag = posn_pf.length();
    let r_mag_inv = 1.0 / r_mag;

    // Define terms (page 33 of Gottlieb 1993)
    let x_div_r = posn_pf.x * r_mag_inv;
    let y_div_r = posn_pf.y * r_mag_inv;
    let z_div_r = posn_pf.z * r_mag_inv;
    let epilson = z_div_r;

    let rad_div_r = data.radius * r_mag_inv;
    let mut rad_div_r_nth = rad_div_r;
    let mu_div_r = data.mu * r_mag_inv;
    let mu_div_rsq = mu_div_r * r_mag_inv;

    // Compute magnitude of projection on equatorial plane
    let mut rho_sq = 0.0;
    if posn_pf.x < -SQRT_DBL_MIN || posn_pf.x > SQRT_DBL_MIN {
        rho_sq += posn_pf.x * posn_pf.x;
    }
    if posn_pf.y < -SQRT_DBL_MIN || posn_pf.y > SQRT_DBL_MIN {
        rho_sq += posn_pf.y * posn_pf.y;
    }

    // Modification to Gottlieb: equations 3-18 underflow near poles
    let rho = rho_sq.sqrt();
    let cos_phi = rho * r_mag_inv;
    let mut cos_phi_nth = cos_phi;

    // Recursive cos(m*lambda), sin(m*lambda)
    let mut cos_mlambda = vec![0.0; degree + 1];
    let mut sin_mlambda = vec![0.0; degree + 1];
    cos_mlambda[0] = 1.0;
    sin_mlambda[0] = 0.0;
    if rho_sq > 0.0 {
        cos_mlambda[1] = posn_pf.x / rho;
        sin_mlambda[1] = posn_pf.y / rho;
    } else {
        cos_mlambda[1] = 1.0;
        sin_mlambda[1] = 0.0;
    }

    // Initialize sums (perturbing gravity only: zeros)
    let mut sum_v = 0.0;
    let mut sum_gam = 0.0;
    let mut sum_gam_grad = 0.0;
    let mut sum_l = 0.0;
    let mut sum_h = 0.0;
    let mut sum_h_grad = 0.0;
    let mut sum_j = 0.0;
    let mut sum_k = 0.0;
    let mut sum_m = 0.0;
    let mut sum_n = 0.0;
    let mut sum_o = 0.0;
    let mut sum_p = 0.0;
    let mut sum_q = 0.0;
    let mut sum_r = 0.0;
    let mut sum_s = 0.0;
    let mut sum_t = 0.0;

    // C_tilde, S_tilde (equation 3-18, modified for underflow)
    let mut c_tilde = vec![0.0; degree + 1];
    let mut s_tilde = vec![0.0; degree + 1];
    c_tilde[0] = 1.0;
    c_tilde[1] = x_div_r;
    s_tilde[0] = 0.0;
    s_tilde[1] = y_div_r;

    // Pnm: working array for Legendre polynomials (per-call scratch)
    // Pnm[ii] has ii+3 elements
    // The diagonal elements Pnm[ii][ii] are constant (independent of position)
    // and are precomputed during initialize_control in JEOD. We compute them here.
    let mut pnm: Vec<Vec<f64>> = Vec::with_capacity(degree + 1);
    for ii in 0..=degree {
        pnm.push(vec![0.0; ii + 3]);
    }
    // Initialize Pnm[0] and Pnm[1] (from JEOD initialize_control, lines 118-123)
    pnm[0][0] = 1.0;
    // pnm[0][1] = 0.0; pnm[0][2] = 0.0; (already zero)
    if degree >= 1 {
        pnm[1][1] = 3.0_f64.sqrt();
        // pnm[1][2] = 0.0; pnm[1][3] = 0.0; (already zero)
    }
    // Precompute diagonal elements Pnm[ii][ii] (equation 7-8)
    // These are position-independent and used in the inner loop.
    for ii in 2..=degree {
        let ii_f = data.int_to_double[ii];
        pnm[ii][ii] = ((2.0 * ii_f + 1.0) / (2.0 * ii_f)).sqrt() * pnm[ii - 1][ii - 1];
        // pnm[ii][ii+1] = 0.0; pnm[ii][ii+2] = 0.0; (already zero)
    }
    // Set position-dependent P(1,0)
    pnm[1][0] = 3.0_f64.sqrt() * epilson;

    let i2d = &data.int_to_double;

    // Store local copy of Cnm[2] to avoid modifying source data
    // (JEOD protects against writes to harmonics body Cnm array)
    let local_cnm2: Vec<f64> = if degree >= 2 {
        data.cnm[2].clone()
    } else {
        vec![]
    };

    for ii in 2..=degree {
        let ii_grad_deg_nonzero = ii <= gradient_degree && gradient_degree > 0;

        // Get coefficient pointers for this degree
        let c_ii: &[f64] = if ii == 2 { &local_cnm2 } else { &data.cnm[ii] };
        let s_ii: &[f64] = &data.snm[ii];

        rad_div_r_nth *= rad_div_r;

        // Protect for underflow
        if rad_div_r_nth < 1.0e-299 {
            rad_div_r_nth = 0.0;
        }

        let dbl_iip1 = i2d[ii + 1];

        // P(n,0) term, equation (7-14)
        pnm[ii][0] = data.alpha[ii] * epilson * pnm[ii - 1][0]
            - data.beta[ii] * pnm[ii - 2][0];

        // P(n,n-1) term, equation (7-16)
        pnm[ii][ii - 1] = epilson * data.nrdiag[ii];

        // P(n,1) term, equation (7-12)
        pnm[ii][1] = data.xi[ii][1] * epilson * pnm[ii - 1][1]
            - data.eta[ii][1] * pnm[ii - 2][1];

        let mut sum_v_n = pnm[ii][0] * c_ii[0];
        let mut sum_h_n = pnm[ii][1] * c_ii[0] * data.zeta[ii][0];
        let mut sum_gam_n = sum_v_n * dbl_iip1;

        // Equation (7-12) for jj=2..ii-2
        for jj in 2..=(ii.saturating_sub(2)) {
            pnm[ii][jj] = data.xi[ii][jj] * epilson * pnm[ii - 1][jj]
                - data.eta[ii][jj] * pnm[ii - 2][jj];
        }

        let mut sum_h_grad_n = 0.0;
        let mut sum_gam_grad_n = 0.0;
        let mut sum_m_n = 0.0;
        let mut sum_p_n = 0.0;
        let mut sum_l_n = 0.0;

        if ii_grad_deg_nonzero {
            sum_h_grad_n = pnm[ii][1] * c_ii[0] * data.zeta[ii][0];
            sum_gam_grad_n = sum_v_n * dbl_iip1;
            sum_m_n = pnm[ii][2] * c_ii[0] * data.upsilon[ii][0];
            sum_p_n = sum_h_grad_n * dbl_iip1;
            sum_l_n = sum_gam_grad_n * (dbl_iip1 + 1.0);
        }

        if order > 0 {
            let grad_order_nonzero = gradient_order > 0;

            let mut sum_j_n = 0.0;
            let mut sum_k_n = 0.0;
            let mut sum_n_n = 0.0;
            let mut sum_o_n = 0.0;
            let mut sum_q_n = 0.0;
            let mut sum_r_n = 0.0;
            let mut sum_s_n = 0.0;
            let mut sum_t_n = 0.0;

            if cos_phi_nth > SQRT_DBL_MIN {
                cos_phi_nth *= cos_phi;
            } else {
                cos_phi_nth = 0.0;
            }
            cos_mlambda[ii] =
                cos_mlambda[1] * cos_mlambda[ii - 1] - sin_mlambda[1] * sin_mlambda[ii - 1];
            sin_mlambda[ii] =
                sin_mlambda[1] * cos_mlambda[ii - 1] + cos_mlambda[1] * sin_mlambda[ii - 1];

            // Equation (3-18), modified for underflow
            c_tilde[ii] = cos_phi_nth * cos_mlambda[ii];
            s_tilde[ii] = cos_phi_nth * sin_mlambda[ii];

            let jj_max = order.min(ii);
            for jj in 1..=jj_max {
                let jj_lt_grad_order = jj <= gradient_order;

                let dbl_jj = i2d[jj];
                let dbl_jjp1 = i2d[jj + 1];
                let dbl_jjm1 = if jj > 0 { i2d[jj - 1] } else { 0.0 };

                let c_iijj = c_ii[jj];
                let s_iijj = s_ii[jj];

                let jj_x_piijj = dbl_jj * pnm[ii][jj];
                let b_tilde = c_iijj * c_tilde[jj] + s_iijj * s_tilde[jj];

                // Equation (3-9)
                let b_tilde_m1 =
                    c_iijj * c_tilde[jj - 1] + s_iijj * s_tilde[jj - 1];
                let a_tilde_m1 =
                    c_iijj * s_tilde[jj - 1] - s_iijj * c_tilde[jj - 1];
                let piijj_x_btilde = pnm[ii][jj] * b_tilde;
                sum_v_n += piijj_x_btilde;

                if jj < ii {
                    let zetaiijj_x_piijjp1 =
                        data.zeta[ii][jj] * pnm[ii][jj + 1];
                    sum_h_n += zetaiijj_x_piijjp1 * b_tilde;
                    if ii_grad_deg_nonzero && grad_order_nonzero && jj_lt_grad_order {
                        sum_h_grad_n += zetaiijj_x_piijjp1 * b_tilde;
                        sum_p_n += (dbl_jj + dbl_iip1) * zetaiijj_x_piijjp1 * b_tilde;
                        sum_q_n += dbl_jj * zetaiijj_x_piijjp1 * b_tilde_m1;
                        sum_r_n -= dbl_jj * zetaiijj_x_piijjp1 * a_tilde_m1;
                    }
                }

                sum_j_n += jj_x_piijj * b_tilde_m1;
                sum_k_n -= jj_x_piijj * a_tilde_m1;
                sum_gam_n += (dbl_jj + dbl_iip1) * piijj_x_btilde;

                if ii_grad_deg_nonzero && grad_order_nonzero && jj_lt_grad_order {
                    sum_gam_grad_n += (dbl_jj + dbl_iip1) * piijj_x_btilde;
                    sum_l_n +=
                        (dbl_jj + dbl_iip1) * (dbl_jjp1 + dbl_iip1) * piijj_x_btilde;
                    sum_m_n += pnm[ii][jj + 2] * b_tilde * data.upsilon[ii][jj];
                    sum_s_n += (dbl_jj + dbl_iip1) * jj_x_piijj * b_tilde_m1;
                    sum_t_n -= (dbl_jj + dbl_iip1) * jj_x_piijj * a_tilde_m1;
                }

                if jj >= 2 && ii_grad_deg_nonzero && grad_order_nonzero && jj_lt_grad_order {
                    sum_n_n += dbl_jjm1
                        * jj_x_piijj
                        * (c_iijj * c_tilde[jj - 2] + s_iijj * s_tilde[jj - 2]);
                    sum_o_n += dbl_jjm1
                        * jj_x_piijj
                        * (c_iijj * s_tilde[jj - 2] - s_iijj * c_tilde[jj - 2]);
                }
            } // next m

            sum_j += rad_div_r_nth * sum_j_n;
            sum_k += rad_div_r_nth * sum_k_n;

            if ii_grad_deg_nonzero && grad_order_nonzero {
                sum_n += rad_div_r_nth * sum_n_n;
                sum_o += rad_div_r_nth * sum_o_n;
                sum_q += rad_div_r_nth * sum_q_n;
                sum_r += rad_div_r_nth * sum_r_n;
                sum_s += rad_div_r_nth * sum_s_n;
                sum_t += rad_div_r_nth * sum_t_n;
            }
        } // end if order > 0

        sum_v += rad_div_r_nth * sum_v_n;
        sum_h += rad_div_r_nth * sum_h_n;
        sum_gam += rad_div_r_nth * sum_gam_n;

        if ii_grad_deg_nonzero {
            sum_h_grad += rad_div_r_nth * sum_h_grad_n;
            sum_gam_grad += rad_div_r_nth * sum_gam_grad_n;
            sum_l += rad_div_r_nth * sum_l_n;
            sum_m += rad_div_r_nth * sum_m_n;
            sum_p += rad_div_r_nth * sum_p_n;
        }
    } // next n

    // Gravitational potential
    let pot = mu_div_r * sum_v;
    let lambda = sum_gam + epilson * sum_h;

    // Equation (4-13): acceleration in planet-fixed coordinates
    let accel = DVec3::new(
        -mu_div_rsq * (lambda * x_div_r - sum_j),
        -mu_div_rsq * (lambda * y_div_r - sum_k),
        -mu_div_rsq * (lambda * z_div_r - sum_h),
    );

    // Compute gravity gradient if requested
    let gradient = if compute_gradient && gradient_degree > 0 {
        let lambda_grad = sum_gam_grad + epilson * sum_h_grad;
        let gg = -(sum_m * epilson + sum_p + sum_h_grad);
        let ff = sum_l + lambda_grad + epilson * (sum_p + sum_h_grad - gg);
        let d1 = epilson * sum_q + sum_s;
        let d2 = epilson * sum_r + sum_t;

        let mu_div_r3 = mu_div_rsq * r_mag_inv;
        let g00 = mu_div_r3 * ((ff * x_div_r - 2.0 * d1) * x_div_r - lambda_grad + sum_n);
        let g11 = mu_div_r3 * ((ff * y_div_r - 2.0 * d2) * y_div_r - lambda_grad - sum_n);
        let g22 = mu_div_r3 * ((ff * z_div_r + 2.0 * gg) * z_div_r - lambda_grad + sum_m);
        let g01 = mu_div_r3 * ((ff * y_div_r - d2) * x_div_r - d1 * y_div_r - sum_o);
        let g02 = mu_div_r3 * ((ff * x_div_r - d1) * z_div_r + gg * x_div_r + sum_q);
        let g12 = mu_div_r3 * ((ff * y_div_r - d2) * z_div_r + gg * y_div_r + sum_r);

        // Symmetric gradient tensor in planet-fixed coords
        DMat3::from_cols(
            DVec3::new(g00, g01, g02),
            DVec3::new(g01, g11, g12),
            DVec3::new(g02, g12, g22),
        )
    } else {
        DMat3::ZERO
    };

    GravityAcceleration {
        accel,
        gradient,
        potential: pot,
    }
}
