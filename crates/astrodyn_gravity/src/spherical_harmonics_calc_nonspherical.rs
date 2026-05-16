//! Gottlieb (1993) spherical harmonics gravity computation.
//!
//! Direct port of JEOD `spherical_harmonics_calc_nonspherical.cc`.
//! The caller must provide position in planet-fixed coordinates and
//! rotate the result back to inertial.

use astrodyn_dynamics::forces::GravityAccelerationTyped;
use astrodyn_dynamics::GravityAcceleration;
use astrodyn_quantities::aliases::{HarmonicDegree, Position};
use astrodyn_quantities::frame::{Planet, PlanetFixed};
use glam::{DMat3, DVec3};
// `PlanetFixed<P>: Frame` follows from the blanket
// `impl<P: Planet> Frame for PlanetFixed<P>` in astrodyn_quantities.

use crate::spherical_harmonics_gravity_source::{tri_idx, SphericalHarmonicsData};

/// sqrt(f64::MIN_POSITIVE) — underflow guard matching JEOD's SQRT_DBL_MIN.
const SQRT_DBL_MIN: f64 = 1.4916681462400413e-154;

thread_local! {
    /// Per-thread cached [`GottliebScratch`] so `gravitation`-style
    /// callers can reuse one buffer across calls without plumbing it
    /// through a long signature chain. Grows monotonically: a request
    /// for a degree higher than the cached buffer reallocates once and
    /// the larger buffer covers all subsequent calls (including
    /// lower-degree ones — the kernel only writes to the prefix it
    /// needs). Bit-identical to per-call `::new` because the kernel
    /// overwrites the scratch on entry.
    static GOTTLIEB_SCRATCH: std::cell::RefCell<GottliebScratch> =
        std::cell::RefCell::new(GottliebScratch::new(2));
}

/// Borrow a thread-local [`GottliebScratch`] grown to at least
/// `degree`, then run `f` with a `&mut GottliebScratch` argument.
///
/// Use this from any per-call gravity wrapper that would otherwise
/// allocate a fresh scratch — most importantly the [`crate::compute::gravitation`]
/// wrapper invoked from `GravityControl::evaluate_inner` in the RK4
/// inner loop.
pub fn with_scratch<R>(degree: usize, f: impl FnOnce(&mut GottliebScratch) -> R) -> R {
    GOTTLIEB_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        // Grow on demand. `degree.max(2)` mirrors the prior wrapper's
        // floor, ensuring the scratch always satisfies the kernel's
        // `degree >= 2` precondition when `degree == 0` callers
        // accidentally enter this path.
        let needed = degree.max(2);
        if scratch.degree < needed {
            *scratch = GottliebScratch::new(needed);
        }
        f(&mut scratch)
    })
}

/// Pre-allocated scratch buffers for `calc_nonspherical`.
///
/// Avoids per-call heap allocations in the RK4 inner loop. Create once
/// per degree/order and reuse across calls.
pub struct GottliebScratch {
    cos_mlambda: Vec<f64>,
    sin_mlambda: Vec<f64>,
    c_tilde: Vec<f64>,
    s_tilde: Vec<f64>,
    /// `Pnm[ii]` has `ii+3` elements; stored as a flat Vec with offsets.
    pnm_flat: Vec<f64>,
    /// `pnm_offsets[ii]` = start index of row `ii` in `pnm_flat`.
    pnm_offsets: Vec<usize>,
    degree: usize,
}

impl GottliebScratch {
    /// Allocate scratch buffers for a given maximum degree.
    pub fn new(degree: usize) -> Self {
        let n = degree + 1;
        // Compute total Pnm storage: sum of (ii+3) for ii=0..=degree
        let mut pnm_offsets = Vec::with_capacity(n);
        let mut total = 0usize;
        for ii in 0..n {
            pnm_offsets.push(total);
            total += ii + 3;
        }
        Self {
            cos_mlambda: vec![0.0; n],
            sin_mlambda: vec![0.0; n],
            c_tilde: vec![0.0; n],
            s_tilde: vec![0.0; n],
            pnm_flat: vec![0.0; total],
            pnm_offsets,
            degree,
        }
    }

    /// Maximum degree this scratch buffer can serve.
    pub fn max_degree(&self) -> usize {
        self.degree
    }

    #[inline]
    fn pnm(&self, ii: usize, jj: usize) -> f64 {
        self.pnm_flat[self.pnm_offsets[ii] + jj]
    }

    #[inline]
    fn pnm_mut(&mut self, ii: usize, jj: usize) -> &mut f64 {
        let idx = self.pnm_offsets[ii] + jj;
        &mut self.pnm_flat[idx]
    }
}

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
pub fn calc_nonspherical(
    data: &SphericalHarmonicsData,
    posn_pf: DVec3,
    degree: usize,
    order: usize,
    compute_gradient: bool,
    gradient_degree: usize,
    gradient_order: usize,
) -> GravityAcceleration {
    // JEOD_INV: GV.04 — degree ≤ source degree. Belt-and-suspenders guard
    // matching the validation site in `gravity_controls.rs::check_validity`;
    // protects standalone callers that bypass the manager-level check.
    assert!(
        degree <= data.degree,
        "Requested degree ({degree}) exceeds source max degree ({})",
        data.degree
    );
    let mut scratch = GottliebScratch::new(degree);
    calc_nonspherical_with_scratch(
        data,
        posn_pf,
        degree,
        order,
        compute_gradient,
        gradient_degree,
        gradient_order,
        &mut scratch,
        0.0,   // no tidal delta in standalone mode
        false, // no delta coefficients configured
    )
}

/// Compute nonspherical gravity with a reusable scratch workspace.
///
/// Same algorithm as [`calc_nonspherical`] but avoids heap
/// allocation by reusing pre-allocated buffers. The scratch workspace must
/// have been created with `degree >= ` the requested degree.
///
/// # Panics
///
/// All of the following preconditions are checked and panic immediately on
/// violation (in both debug and release builds):
///
/// - `degree > data.degree` — requested harmonics exceed the source's
///   stored coefficient table (`JEOD_INV: GV.04`).
/// - `order > data.order` — requested order exceeds the source max
///   (`JEOD_INV: GV.05`).
/// - `order > degree` — order must not exceed degree (`JEOD_INV: GV.06`).
/// - `scratch.degree < degree` — the reusable buffer was sized for a
///   smaller harmonic expansion (`JEOD_INV: GV.21`).
/// - `posn_pf` is the zero vector — the Gottlieb recurrence divides by
///   `r`, so zero-position input would silently produce NaN gravity
///   (`JEOD_INV: GV.22`).
#[allow(clippy::too_many_arguments)]
pub fn calc_nonspherical_with_scratch(
    data: &SphericalHarmonicsData,
    posn_pf: DVec3,
    degree: usize,
    order: usize,
    compute_gradient: bool,
    gradient_degree: usize,
    gradient_order: usize,
    scratch: &mut GottliebScratch,
    delta_c20: f64,
    has_delta_coeffs: bool,
) -> GravityAcceleration {
    // Matching JEOD's check_validity(): these are fatal errors, not silent
    // clamps. The kernel-side defense duplicates the validation site in
    // `gravity_controls.rs::check_validity` so standalone callers (and any
    // future skip-the-manager path) still fail loudly.
    // JEOD_INV: GV.04 — requested degree must not exceed the source's stored
    // coefficient table max degree (kernel-side defense for standalone callers).
    assert!(
        degree <= data.degree,
        "Requested degree ({degree}) exceeds source max degree ({})",
        data.degree
    );
    // JEOD_INV: GV.05 — requested order must not exceed the source's stored
    // coefficient table max order (kernel-side defense for standalone callers).
    assert!(
        order <= data.order,
        "Requested order ({order}) exceeds source max order ({})",
        data.order
    );
    // JEOD_INV: GV.06 — requested order must not exceed requested degree
    // (kernel-side defense for standalone callers; mirrors gravity_controls).
    assert!(
        order <= degree,
        "Requested order ({order}) exceeds requested degree ({degree})"
    );
    // JEOD_INV: GV.21 — Gottlieb scratch workspace must be sized at least
    // as large as the requested degree (the recurrence indexes by degree).
    assert!(
        scratch.degree >= degree,
        "GottliebScratch degree ({}) must be >= requested degree ({degree})",
        scratch.degree
    );

    // JEOD_INV: GV.22 — position must be non-zero (division by `r` is
    // unavoidable in the Gottlieb recurrence; zero-vector input would
    // silently produce NaN gravity).
    assert!(posn_pf.length_squared() > 0.0, "position must be non-zero");

    // If degree < 2, there are no harmonics to compute (only point-mass).
    // Return zero perturbation.
    if degree < 2 {
        return GravityAcceleration {
            grav_accel: DVec3::ZERO,
            grav_grad: DMat3::ZERO,
            grav_pot: 0.0,
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

    // No full scratch.reset() needed: every element is written before it is
    // read. Pnm diagonals, cos/sin_mlambda, c/s_tilde are all initialized
    // explicitly below before the degree loop accesses them.

    // Compute position vector magnitude
    let r_mag = posn_pf.length();
    let r_mag_inv = 1.0 / r_mag;

    // Define terms (page 33 of Gottlieb 1993)
    let x_div_r = posn_pf.x * r_mag_inv;
    let y_div_r = posn_pf.y * r_mag_inv;
    let z_div_r = posn_pf.z * r_mag_inv;
    let epsilon = z_div_r;

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
    scratch.cos_mlambda[0] = 1.0;
    scratch.sin_mlambda[0] = 0.0;
    if rho_sq > 0.0 {
        scratch.cos_mlambda[1] = posn_pf.x / rho;
        scratch.sin_mlambda[1] = posn_pf.y / rho;
    } else {
        scratch.cos_mlambda[1] = 1.0;
        scratch.sin_mlambda[1] = 0.0;
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
    scratch.c_tilde[0] = 1.0;
    scratch.c_tilde[1] = x_div_r;
    scratch.s_tilde[0] = 0.0;
    scratch.s_tilde[1] = y_div_r;

    // Initialize Pnm[0] and Pnm[1] (from JEOD initialize_control, lines 118-123)
    *scratch.pnm_mut(0, 0) = 1.0;
    if degree >= 1 {
        *scratch.pnm_mut(1, 1) = 3.0_f64.sqrt();
    }
    // Precompute diagonal elements Pnm[ii][ii] (equation 7-8)
    // These are position-independent and used in the inner loop.
    for ii in 2..=degree {
        let ii_f = data.int_to_double[ii];
        let prev = scratch.pnm(ii - 1, ii - 1);
        *scratch.pnm_mut(ii, ii) = ((2.0 * ii_f + 1.0) / (2.0 * ii_f)).sqrt() * prev;
    }
    // Set position-dependent P(1,0)
    *scratch.pnm_mut(1, 0) = 3.0_f64.sqrt() * epsilon;

    let i2d = &data.int_to_double;

    // Store local copy of Cnm[2] to avoid modifying source data.
    // JEOD's calc_nonspherical does this because variational tidal effects
    // (delta C20) are added to local_Cnm[0] per-call. We preserve the
    // pattern for when tidal corrections are ported. The copy is small
    // (degree-2 row has only 3 elements regardless of model degree).
    // Stack-allocated to avoid heap allocation in the hot path.
    let mut local_cnm = [0.0_f64; 3];
    if degree >= 2 {
        let src = data.cnm_row(2);
        let n = src.len().min(3);
        local_cnm[..n].copy_from_slice(&src[..n]);
    }
    // Apply tidal ΔC20 and permanent tide corrections.
    // JEOD: spherical_harmonics_calc_nonspherical.cc:91-105.
    // Gate on has_delta_coeffs (JEOD: n_deltacoeffs > 0), which indicates
    // tidal delta coefficient sets are configured on this source — NOT on
    // whether total_dC20 is nonzero.
    if has_delta_coeffs && degree >= 2 {
        local_cnm[0] += delta_c20;
        // Correct permanent tide if already included in C20 coefficient.
        // JEOD: if(!harmonics_source->tide_free) local_Cnm[0] += tide_free_delta
        if !data.tide_free {
            local_cnm[0] += data.tide_free_delta;
        }
    }

    for ii in 2..=degree {
        let ii_grad_deg_nonzero = ii <= gradient_degree && gradient_degree > 0;

        // Borrow contiguous rows once per `ii`. Flat triangular storage
        // means each row is a single cache-friendly slice; the inner
        // `jj` loop then indexes a `&[f64]` directly instead of paying
        // the row-pointer load that `Vec<Vec<f64>>` would impose.
        //
        // `row_start..row_end` is derived from `tri_idx(ii, 0)` once per
        // `ii` and reused for every per-row slice below. Reusing it
        // (instead of calling `cnm_row(ii)` / `snm_row(ii)`) avoids the
        // redundant `tri_idx` recomputation and the per-call release
        // `assert!` bound check on each accessor — those bounds are
        // already established by the `row_start..row_end` derivation,
        // and the slice itself panics on an out-of-range index.
        let row_start = tri_idx(ii, 0);
        let row_end = row_start + ii + 1;
        let c_ii: &[f64] = if ii == 2 {
            &local_cnm
        } else {
            &data.cnm[row_start..row_end]
        };
        let s_ii: &[f64] = &data.snm[row_start..row_end];
        let xi_ii: &[f64] = &data.xi[row_start..row_end];
        let eta_ii: &[f64] = &data.eta[row_start..row_end];
        let zeta_ii: &[f64] = &data.zeta[row_start..row_end];
        let upsilon_ii: &[f64] = &data.upsilon[row_start..row_end];

        rad_div_r_nth *= rad_div_r;

        // Protect for underflow
        if rad_div_r_nth < 1.0e-299 {
            rad_div_r_nth = 0.0;
        }

        let dbl_iip1 = i2d[ii + 1];

        // P(n,0) term, equation (7-14)
        *scratch.pnm_mut(ii, 0) = data.alpha[ii] * epsilon * scratch.pnm(ii - 1, 0)
            - data.beta[ii] * scratch.pnm(ii - 2, 0);

        // P(n,n-1) term, equation (7-16)
        *scratch.pnm_mut(ii, ii - 1) = epsilon * data.nrdiag[ii];

        // P(n,1) term, equation (7-12)
        *scratch.pnm_mut(ii, 1) =
            xi_ii[1] * epsilon * scratch.pnm(ii - 1, 1) - eta_ii[1] * scratch.pnm(ii - 2, 1);

        let mut sum_v_n = scratch.pnm(ii, 0) * c_ii[0];
        let mut sum_h_n = scratch.pnm(ii, 1) * c_ii[0] * zeta_ii[0];
        let mut sum_gam_n = sum_v_n * dbl_iip1;

        // Equation (7-12) for jj=2..ii-2
        for jj in 2..=(ii.saturating_sub(2)) {
            *scratch.pnm_mut(ii, jj) = xi_ii[jj] * epsilon * scratch.pnm(ii - 1, jj)
                - eta_ii[jj] * scratch.pnm(ii - 2, jj);
        }

        let mut sum_h_grad_n = 0.0;
        let mut sum_gam_grad_n = 0.0;
        let mut sum_m_n = 0.0;
        let mut sum_p_n = 0.0;
        let mut sum_l_n = 0.0;

        if ii_grad_deg_nonzero {
            sum_h_grad_n = scratch.pnm(ii, 1) * c_ii[0] * zeta_ii[0];
            sum_gam_grad_n = sum_v_n * dbl_iip1;
            sum_m_n = scratch.pnm(ii, 2) * c_ii[0] * upsilon_ii[0];
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
            scratch.cos_mlambda[ii] = scratch.cos_mlambda[1] * scratch.cos_mlambda[ii - 1]
                - scratch.sin_mlambda[1] * scratch.sin_mlambda[ii - 1];
            scratch.sin_mlambda[ii] = scratch.sin_mlambda[1] * scratch.cos_mlambda[ii - 1]
                + scratch.cos_mlambda[1] * scratch.sin_mlambda[ii - 1];

            // Equation (3-18), modified for underflow
            scratch.c_tilde[ii] = cos_phi_nth * scratch.cos_mlambda[ii];
            scratch.s_tilde[ii] = cos_phi_nth * scratch.sin_mlambda[ii];

            let jj_max = order.min(ii);
            for jj in 1..=jj_max {
                let jj_lt_grad_order = jj <= gradient_order;

                let dbl_jj = i2d[jj];
                let dbl_jjp1 = i2d[jj + 1];
                let dbl_jjm1 = i2d[jj - 1]; // jj >= 1 always in this loop

                let c_iijj = c_ii[jj];
                let s_iijj = s_ii[jj];

                let jj_x_piijj = dbl_jj * scratch.pnm(ii, jj);
                let b_tilde = c_iijj * scratch.c_tilde[jj] + s_iijj * scratch.s_tilde[jj];

                // Equation (3-9)
                let b_tilde_m1 =
                    c_iijj * scratch.c_tilde[jj - 1] + s_iijj * scratch.s_tilde[jj - 1];
                let a_tilde_m1 =
                    c_iijj * scratch.s_tilde[jj - 1] - s_iijj * scratch.c_tilde[jj - 1];
                let piijj_x_btilde = scratch.pnm(ii, jj) * b_tilde;
                sum_v_n += piijj_x_btilde;

                if jj < ii {
                    let zetaiijj_x_piijjp1 = zeta_ii[jj] * scratch.pnm(ii, jj + 1);
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
                    sum_l_n += (dbl_jj + dbl_iip1) * (dbl_jjp1 + dbl_iip1) * piijj_x_btilde;
                    sum_m_n += scratch.pnm(ii, jj + 2) * b_tilde * upsilon_ii[jj];
                    sum_s_n += (dbl_jj + dbl_iip1) * jj_x_piijj * b_tilde_m1;
                    sum_t_n -= (dbl_jj + dbl_iip1) * jj_x_piijj * a_tilde_m1;
                }

                if jj >= 2 && ii_grad_deg_nonzero && grad_order_nonzero && jj_lt_grad_order {
                    sum_n_n += dbl_jjm1
                        * jj_x_piijj
                        * (c_iijj * scratch.c_tilde[jj - 2] + s_iijj * scratch.s_tilde[jj - 2]);
                    sum_o_n += dbl_jjm1
                        * jj_x_piijj
                        * (c_iijj * scratch.s_tilde[jj - 2] - s_iijj * scratch.c_tilde[jj - 2]);
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
    let lambda = sum_gam + epsilon * sum_h;

    // Equation (4-13): acceleration in planet-fixed coordinates
    let accel = DVec3::new(
        -mu_div_rsq * (lambda * x_div_r - sum_j),
        -mu_div_rsq * (lambda * y_div_r - sum_k),
        -mu_div_rsq * (lambda * z_div_r - sum_h),
    );

    // Compute gravity gradient if requested
    let gradient = if compute_gradient && gradient_degree > 0 {
        let lambda_grad = sum_gam_grad + epsilon * sum_h_grad;
        let gg = -(sum_m * epsilon + sum_p + sum_h_grad);
        let ff = sum_l + lambda_grad + epsilon * (sum_p + sum_h_grad - gg);
        let d1 = epsilon * sum_q + sum_s;
        let d2 = epsilon * sum_r + sum_t;

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
        grav_accel: accel,
        grav_grad: gradient,
        grav_pot: pot,
    }
}

/// Typed sibling of [`calc_nonspherical`].
///
/// Accepts a [`Position<PlanetFixed<P>>`] for the field-evaluation
/// point, takes [`HarmonicDegree`] ordinal indices, and returns a
/// [`GravityAccelerationTyped<PlanetFixed<P>>`] for the field at that
/// point. The Gottlieb kernel is invoked through the existing
/// untyped [`calc_nonspherical`] — no new arithmetic in the boundary —
/// so the J2 regression hash and Tier 3 SH-using trajectories produce
/// bit-identical output.
pub fn calc_nonspherical_typed<P: Planet>(
    data: &SphericalHarmonicsData,
    posn_pf: Position<PlanetFixed<P>>,
    degree: HarmonicDegree,
    order: HarmonicDegree,
    compute_gradient: bool,
    gradient_degree: HarmonicDegree,
    gradient_order: HarmonicDegree,
) -> GravityAccelerationTyped<PlanetFixed<P>> {
    let untyped = calc_nonspherical(
        data,
        posn_pf.raw_si(),
        degree.get(),
        order.get(),
        compute_gradient,
        gradient_degree.get(),
        gradient_order.get(),
    );
    GravityAccelerationTyped::<PlanetFixed<P>>::from_untyped_unchecked(&untyped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spherical_harmonics_gravity_source::SphericalHarmonicsData;

    /// Build a small SH source for kernel-precondition tests. The
    /// coefficient arrays follow the triangular `cnm[n].len() == n+1`
    /// shape that `SphericalHarmonicsData::new` expects. Numerics are
    /// irrelevant — these tests drive the panic preconditions before
    /// the recurrence runs.
    fn dummy_sh(degree: usize, order: usize) -> SphericalHarmonicsData {
        let mu = 3.986_004_415e14;
        let radius = 6_378_137.0;
        let cnm: Vec<Vec<f64>> = (0..=degree).map(|n| vec![0.0_f64; n + 1]).collect();
        let snm: Vec<Vec<f64>> = (0..=degree).map(|n| vec![0.0_f64; n + 1]).collect();
        SphericalHarmonicsData::new(degree, order, radius, mu, cnm, snm, true, 0.0)
    }

    /// Scratch sized smaller than the requested degree: the recurrence
    /// indexes `pnm[ii]` for `ii <= degree`, so a buffer allocated for
    /// (say) degree-2 cannot serve a degree-8 request without writing
    /// out of bounds. JEOD's `calc_nonspherical` keeps its own scratch
    /// owned by the source, so the size invariant is structural there;
    /// our `calc_nonspherical_with_scratch` exposes the buffer to
    /// callers (RK4 reuse), so the invariant must be runtime-checked
    /// at the kernel boundary.
    // JEOD_INV: GV.21 — negative test: scratch smaller than requested degree
    #[test]
    #[should_panic(expected = "GottliebScratch degree (2) must be >= requested degree (8)")]
    fn gv_21_panics_on_scratch_smaller_than_degree() {
        let data = dummy_sh(8, 8);
        let mut scratch = GottliebScratch::new(2);
        let _ = calc_nonspherical_with_scratch(
            &data,
            DVec3::new(7e6, 0.0, 0.0),
            8, // requested degree exceeds scratch.degree
            8,
            false,
            0,
            0,
            &mut scratch,
            0.0,
            false,
        );
    }

    /// Zero-vector position: the Gottlieb recurrence normalizes by
    /// `r = |posn_pf|`, so a zero input would silently produce
    /// `NaN`/`inf` acceleration. JEOD's documentation calls out the
    /// origin singularity in `spherical_harmonics_calc_nonspherical.cc`;
    /// we assert at the kernel boundary so the caller sees the
    /// configuration error rather than a quietly-poisoned trajectory.
    // JEOD_INV: GV.22 — negative test: zero-vector input position
    #[test]
    #[should_panic(expected = "position must be non-zero")]
    fn gv_22_panics_on_zero_position() {
        let data = dummy_sh(4, 4);
        let mut scratch = GottliebScratch::new(4);
        let _ = calc_nonspherical_with_scratch(
            &data,
            DVec3::ZERO, // ← zero-length input
            4,
            4,
            false,
            0,
            0,
            &mut scratch,
            0.0,
            false,
        );
    }
}
