//! Flat-plate aerodynamic surface model.
//!
//! Port of JEOD `FlatPlateAeroFacet::aerodrag_force()` from
//! `models/interactions/aerodynamics/src/flat_plate_aero_facet.cc`.
//!
//! In the free-molecular flow regime (orbital altitudes), aerodynamic force on
//! each facet depends on the angle of attack, the speed ratio (vehicle speed /
//! mean thermal speed), and the gas-surface interaction model (specular,
//! diffuse, mixed, or calculated coefficients).
//!
//! Per-facet forces are summed to produce total force and torque in the
//! structural frame.

use glam::DVec3;

/// Error function approximation using Abramowitz & Stegun formula 7.1.26.
/// Maximum error |epsilon(x)| <= 1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();

    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;

    let t = 1.0 / (1.0 + P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let result = 1.0 - (A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5) * (-x * x).exp();
    sign * result
}

/// Aerodynamic coefficient computation method.
///
/// Port of JEOD `AeroDragEnum::CoefCalcMethod`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AeroCoeffMethod {
    /// Specular reflection: molecules "bounce" off the plate, producing force
    /// normal to the surface only.
    /// JEOD assumes epsilon=1, sin_alpha=1 for coefficient calculation.
    Specular,
    /// Diffuse reflection: molecules "stick" to the plate, producing force
    /// parallel to the relative velocity vector.
    /// JEOD assumes epsilon=0, sin_alpha=1 for coefficient calculation.
    Diffuse,
    /// Mixed specular + diffuse with configurable fraction `epsilon`.
    /// `epsilon` is the fraction of molecules that "bounce" (specular).
    /// JEOD assumes sin_alpha=1 for coefficient calculation.
    ///
    /// Prefer constructing via [`AeroCoeffMethod::mixed`], which clamps
    /// `epsilon` into the physical range `[0, 1]`.
    Mixed {
        /// Fraction of molecules that bounce (specular).
        /// Expected physical range `[0, 1]`.
        epsilon: f64,
    },
    /// Full coefficient calculation based on angle of attack, material
    /// properties, wall temperature, and freestream conditions.
    /// Uses the exact formulas from JEOD with normal and tangential
    /// drag coefficients computed per-facet.
    ///
    /// Prefer constructing via [`AeroCoeffMethod::calc_coef`], which clamps
    /// `epsilon` into the physical range `[0, 1]`.
    CalcCoef {
        /// Fraction of molecules that "bounce" (specular).
        /// Expected physical range `[0, 1]`.
        epsilon: f64,
    },
}

impl AeroCoeffMethod {
    #[inline]
    fn clamp_epsilon(epsilon: f64) -> f64 {
        epsilon.clamp(0.0, 1.0)
    }

    /// Construct a mixed specular/diffuse coefficient model, clamping
    /// `epsilon` into the physical range `[0, 1]`.
    #[inline]
    pub fn mixed(epsilon: f64) -> Self {
        Self::Mixed {
            epsilon: Self::clamp_epsilon(epsilon),
        }
    }

    /// Construct a full coefficient-calculation model, clamping `epsilon`
    /// into the physical range `[0, 1]`.
    #[inline]
    pub fn calc_coef(epsilon: f64) -> Self {
        Self::CalcCoef {
            epsilon: Self::clamp_epsilon(epsilon),
        }
    }
}

/// Parameters for the freestream gas, shared across all facets.
///
/// Port of JEOD `AeroDragParameters` (gas_const, temp_free_stream fields).
#[derive(Debug, Clone, Copy)]
pub struct AeroGasParams {
    /// Gas constant R in N*m/(kg*K). For air, R = 287.
    /// Used to compute the speed ratio s = v / sqrt(2*R*T).
    pub gas_const: f64,
    /// Temperature of the incident freestream in K.
    pub temp_free_stream: f64,
}

/// Single flat-plate aerodynamic surface facet.
///
/// Port of JEOD `FlatPlateAeroFacet` fields.
#[derive(Debug, Clone, Copy)]
pub struct AeroFacet {
    /// Plate area in m^2.
    pub area: f64,
    /// Outward-facing unit normal in the structural frame.
    pub normal: DVec3,
    /// Center of pressure in the structural frame (m), for torque computation.
    pub center_pressure: DVec3,
    /// Coefficient calculation method.
    pub coeff_method: AeroCoeffMethod,
    /// Temperature of the plate surface in K.
    /// Used by Diffuse, Mixed, and CalcCoef methods for reflected molecule
    /// temperature calculation.
    pub temperature: f64,
}

/// Result of flat-plate aero computation for all facets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatPlateAeroResult {
    /// Total aerodynamic force in the structural frame (N).
    pub force: DVec3,
    /// Total aerodynamic torque about the body CoM in the structural frame (N*m).
    pub torque: DVec3,
}

impl Default for FlatPlateAeroResult {
    fn default() -> Self {
        Self {
            force: DVec3::ZERO,
            torque: DVec3::ZERO,
        }
    }
}

/// 2/sqrt(pi), matching C's M_2_SQRTPI.
const M_2_SQRTPI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// Compute flat-plate aerodynamic force and torque for a set of facets.
///
/// Port of JEOD `AerodynamicDrag::aero_drag()` (plate loop) +
/// `FlatPlateAeroFacet::aerodrag_force()`.
///
/// # Arguments
/// * `facets` - Flat-plate facets in the structural frame.
/// * `rel_vel_struct` - Relative velocity (vehicle - atmosphere) in structural frame (m/s).
/// * `dynamic_pressure` - 0.5 * rho * |v_rel|^2 (N/m^2). Pre-computed by caller.
/// * `gas_params` - Freestream gas properties (gas constant, temperature).
/// * `center_grav` - Center of gravity in the structural frame (m), for torque arm.
///
/// # Returns
/// Total force and torque in the structural frame.
pub fn compute_flat_plate_aero(
    facets: &[AeroFacet],
    rel_vel_struct: DVec3,
    dynamic_pressure: f64,
    gas_params: &AeroGasParams,
    center_grav: DVec3,
) -> FlatPlateAeroResult {
    let rel_vel_mag = rel_vel_struct.length();
    if rel_vel_mag < 1e-30 || dynamic_pressure <= 0.0 {
        return FlatPlateAeroResult::default();
    }

    let rel_vel_hat = rel_vel_struct / rel_vel_mag;

    let mut total_force = DVec3::ZERO;
    let mut total_torque = DVec3::ZERO;

    for facet in facets {
        let facet_force = compute_single_facet(
            facet,
            rel_vel_mag,
            rel_vel_hat,
            dynamic_pressure,
            gas_params,
        );

        if facet_force == DVec3::ZERO {
            continue;
        }

        // Torque = (center_pressure - center_grav) x force
        // JEOD flat_plate_aero_facet.cc line 306-311
        let cpres = facet.center_pressure - center_grav;
        let facet_torque = cpres.cross(facet_force);

        total_force += facet_force;
        total_torque += facet_torque;
    }

    FlatPlateAeroResult {
        force: total_force,
        torque: total_torque,
    }
}

/// Compute force on a single facet.
///
/// Port of JEOD `FlatPlateAeroFacet::aerodrag_force()`.
fn compute_single_facet(
    facet: &AeroFacet,
    rel_vel_mag: f64,
    rel_vel_hat: DVec3,
    dynamic_pressure: f64,
    gas_params: &AeroGasParams,
) -> DVec3 {
    // sin_alpha = dot(normal, rel_vel_hat)
    // alpha is the angle between the plate surface and the velocity vector.
    // sin_alpha > 0 means the plate is windward-facing.
    // JEOD flat_plate_aero_facet.cc line 78
    let sin_alpha = facet.normal.dot(rel_vel_hat);

    if sin_alpha <= 0.0 {
        return DVec3::ZERO;
    }

    // force_base is negative because force opposes motion.
    // JEOD flat_plate_aero_facet.cc line 223
    let force_base = -dynamic_pressure * facet.area;

    // Speed ratio: s = v / sqrt(2 * R * T)
    // JEOD flat_plate_aero_facet.cc line 117
    debug_assert!(
        gas_params.gas_const > 0.0 && gas_params.temp_free_stream > 0.0,
        "gas_const and temp_free_stream must be positive"
    );
    let denom = (2.0 * gas_params.gas_const * gas_params.temp_free_stream).sqrt();
    if denom <= 0.0 {
        return DVec3::ZERO;
    }
    let s = rel_vel_mag / denom;
    let s_2 = s * s;

    match facet.coeff_method {
        AeroCoeffMethod::Specular => {
            // JEOD: assumes epsilon=1, sin_alpha=1 for coeff, then applies
            // sin_alpha^2 scaling in force computation.
            // JEOD flat_plate_aero_facet.cc lines 175-178
            let exp_s2 = (-s_2).exp();
            let drag_coef_spec = (2.0 * M_2_SQRTPI * s * exp_s2 + 2.0 + 4.0 * s_2) / s_2;

            // Force = force_base * drag_coef_spec * sin_alpha^2, along normal
            // JEOD flat_plate_aero_facet.cc lines 253-256
            let force_n = force_base * drag_coef_spec * sin_alpha * sin_alpha;
            facet.normal * force_n
        }

        AeroCoeffMethod::Diffuse => {
            // JEOD: assumes epsilon=0, sin_alpha=1 for coeff.
            // JEOD flat_plate_aero_facet.cc lines 182-196
            let exp_s2 = (-s_2).exp();
            let temp_ratio = facet.temperature / gas_params.temp_free_stream;
            let drag_coef_diff = (M_2_SQRTPI * s * exp_s2
                + temp_ratio.sqrt() * (2.0 / M_2_SQRTPI) * s
                + 1.0
                + 2.0 * s_2)
                / s_2;

            // Force = force_base * drag_coef_diff * sin_alpha, along velocity
            // JEOD flat_plate_aero_facet.cc lines 269-273
            let force_t = force_base * drag_coef_diff * sin_alpha;
            rel_vel_hat * force_t
        }

        AeroCoeffMethod::Mixed { epsilon } => {
            // JEOD: computes both specular and diffuse coefficients, then mixes.
            // JEOD flat_plate_aero_facet.cc lines 199-207
            let exp_s2 = (-s_2).exp();
            let drag_coef_spec = (2.0 * M_2_SQRTPI * s * exp_s2 + 2.0 + 4.0 * s_2) / s_2;
            let temp_ratio = facet.temperature / gas_params.temp_free_stream;
            let drag_coef_diff = (M_2_SQRTPI * s * exp_s2
                + temp_ratio.sqrt() * (2.0 / M_2_SQRTPI) * s
                + 1.0
                + 2.0 * s_2)
                / s_2;

            // Specular component: along normal, scaled by epsilon * sin_alpha^2
            // Diffuse component: along velocity, scaled by (1-epsilon) * sin_alpha
            // JEOD flat_plate_aero_facet.cc lines 287-293
            let force_n = epsilon * force_base * drag_coef_spec * sin_alpha * sin_alpha;
            let force_t = (1.0 - epsilon) * force_base * drag_coef_diff * sin_alpha;

            let vec_force_n = facet.normal * force_n;
            let vec_force_t = rel_vel_hat * force_t;
            vec_force_n + vec_force_t
        }

        AeroCoeffMethod::CalcCoef { epsilon } => {
            // Full coefficient calculation with angle-dependent normal and
            // tangential drag coefficients.
            // JEOD flat_plate_aero_facet.cc lines 124-166
            let one_p_epsilon = 1.0 + epsilon;
            let one_m_epsilon = 1.0 - epsilon;
            let s_sinalpha = s * sin_alpha;
            let s_sa2 = s_sinalpha * s_sinalpha;
            let exp_ssa2 = (-s_sa2).exp();
            let erf_ssa = erf(s_sinalpha);

            let local_temp_reflect =
                one_m_epsilon * facet.temperature + epsilon * gas_params.temp_free_stream;
            let temp_ratio = local_temp_reflect / gas_params.temp_free_stream;

            let drag_coef_norm = (M_2_SQRTPI * one_p_epsilon * s_sinalpha * exp_ssa2
                + one_m_epsilon * temp_ratio.sqrt() * (2.0 / M_2_SQRTPI) * s_sinalpha
                + one_p_epsilon * (1.0 + 2.0 * s_sa2) * erf_ssa)
                / s_2;

            let force_n = force_base * drag_coef_norm;

            // Check if plate is full-on (alpha = 90 degrees => sin_alpha = 1).
            // sin_alpha comes from a dot product of unit vectors, so it can be
            // slightly above 1.0 due to floating-point error. Clamp to [0, 1]
            // before computing cos_alpha to avoid NaN from sqrt of a negative.
            // JEOD flat_plate_aero_facet.cc line 231
            let sin_alpha_clamped = sin_alpha.min(1.0);
            let cos_alpha_sq = 1.0 - sin_alpha_clamped * sin_alpha_clamped;
            if cos_alpha_sq < 1e-20 {
                // No tangential component when plate faces directly into flow
                facet.normal * force_n
            } else {
                let cos_alpha = cos_alpha_sq.sqrt();

                // Tangent vector: projection of velocity onto plate surface
                // tangent = (rel_vel_hat - normal * sin_alpha) / cos_alpha
                // JEOD flat_plate_aero_facet.cc lines 154-157
                let tangent = (rel_vel_hat - facet.normal * sin_alpha) / cos_alpha;

                let drag_coef_tang = ((2.0 * one_m_epsilon / (2.0 / M_2_SQRTPI))
                    * s
                    * cos_alpha
                    * (exp_ssa2 + (2.0 / M_2_SQRTPI) * s_sinalpha * erf_ssa))
                    / s_2;
                let drag_coef_tang = drag_coef_tang.abs();

                let force_t = force_base * drag_coef_tang;
                facet.normal * force_n + tangent * force_t
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: standard gas params for LEO (~400 km altitude).
    fn leo_gas_params() -> AeroGasParams {
        AeroGasParams {
            gas_const: 287.0,
            temp_free_stream: 900.0, // K, typical exospheric temperature
        }
    }

    /// Helper: standard conditions for a flat plate test.
    fn standard_conditions() -> (f64, f64, DVec3) {
        let density = 1e-12; // kg/m^3
        let velocity = 7600.0; // m/s
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        (dyn_press, velocity, rel_vel)
    }

    // ── Back-facing facet ────────────────────────────────────────

    /// Back-facing facet produces zero force for all methods.
    #[test]
    fn back_facing_zero_force() {
        let (dyn_press, _, rel_vel) = standard_conditions();
        let gas = leo_gas_params();

        for method in [
            AeroCoeffMethod::Specular,
            AeroCoeffMethod::Diffuse,
            AeroCoeffMethod::Mixed { epsilon: 0.5 },
            AeroCoeffMethod::CalcCoef { epsilon: 0.5 },
        ] {
            let facet = AeroFacet {
                area: 10.0,
                normal: DVec3::new(-1.0, 0.0, 0.0), // faces away from flow
                center_pressure: DVec3::ZERO,
                coeff_method: method,
                temperature: 300.0,
            };

            let result = compute_flat_plate_aero(&[facet], rel_vel, dyn_press, &gas, DVec3::ZERO);
            assert_eq!(
                result.force,
                DVec3::ZERO,
                "Back-facing should give zero force for {:?}",
                method
            );
        }
    }

    // ── Specular: normal incidence ────────────────────────────────

    /// Specular at normal incidence: force is purely along the plate normal,
    /// opposing motion. Magnitude = dyn_press * A * Cd_spec * sin^2(alpha).
    /// At normal incidence sin(alpha)=1 so F = dyn_press * A * Cd_spec.
    #[test]
    fn specular_normal_incidence() {
        let density = 1e-12;
        let velocity = 7600.0;
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        let facet = AeroFacet {
            area: 10.0,
            normal: DVec3::new(1.0, 0.0, 0.0), // faces into flow
            center_pressure: DVec3::ZERO,
            coeff_method: AeroCoeffMethod::Specular,
            temperature: 300.0,
        };

        let result = compute_flat_plate_aero(&[facet], rel_vel, dyn_press, &gas, DVec3::ZERO);

        // Compute expected Cd_spec
        let s = velocity / (2.0 * gas.gas_const * gas.temp_free_stream).sqrt();
        let s_2 = s * s;
        let exp_s2 = (-s_2).exp();
        let cd_spec = (2.0 * M_2_SQRTPI * s * exp_s2 + 2.0 + 4.0 * s_2) / s_2;

        // force_base = -dyn_press * area, force_n = force_base * cd_spec * 1.0
        let expected_force_x = -dyn_press * 10.0 * cd_spec;

        let rel_err = (result.force.x - expected_force_x).abs() / expected_force_x.abs();
        assert!(
            rel_err < 1e-12,
            "Specular normal force: expected {expected_force_x:.6e}, got {:.6e}",
            result.force.x
        );
        // Force should oppose motion (negative X)
        assert!(result.force.x < 0.0, "Force should oppose motion");
        assert!(result.force.y.abs() < 1e-20);
        assert!(result.force.z.abs() < 1e-20);
    }

    // ── Specular at 45 degrees ────────────────────────────────────

    /// Specular at 45 degrees: force is along normal, scaled by sin^2(alpha).
    #[test]
    fn specular_45_degrees() {
        let density = 1e-12;
        let velocity = 7600.0;
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        // Normal at 45 degrees to flow: normal = [1/sqrt(2), 1/sqrt(2), 0]
        let n = DVec3::new(1.0, 1.0, 0.0).normalize();
        let facet = AeroFacet {
            area: 10.0,
            normal: n,
            center_pressure: DVec3::ZERO,
            coeff_method: AeroCoeffMethod::Specular,
            temperature: 300.0,
        };

        let result = compute_flat_plate_aero(&[facet], rel_vel, dyn_press, &gas, DVec3::ZERO);

        // sin_alpha = normal.dot(vel_hat) = 1/sqrt(2)
        let sin_alpha = 1.0 / 2.0_f64.sqrt();

        let s = velocity / (2.0 * gas.gas_const * gas.temp_free_stream).sqrt();
        let s_2 = s * s;
        let exp_s2 = (-s_2).exp();
        let cd_spec = (2.0 * M_2_SQRTPI * s * exp_s2 + 2.0 + 4.0 * s_2) / s_2;

        let force_n = -dyn_press * 10.0 * cd_spec * sin_alpha * sin_alpha;
        let expected_force = n * force_n;

        let err = (result.force - expected_force).length();
        let rel_err = err / expected_force.length();
        assert!(
            rel_err < 1e-12,
            "Specular 45-deg: expected ({:.6e}, {:.6e}, {:.6e}), got ({:.6e}, {:.6e}, {:.6e})",
            expected_force.x,
            expected_force.y,
            expected_force.z,
            result.force.x,
            result.force.y,
            result.force.z
        );
    }

    // ── Diffuse: force along velocity ────────────────────────────

    /// Diffuse at normal incidence: force is purely along the velocity direction.
    #[test]
    fn diffuse_normal_incidence() {
        let density = 1e-12;
        let velocity = 7600.0;
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        let facet = AeroFacet {
            area: 10.0,
            normal: DVec3::new(1.0, 0.0, 0.0),
            center_pressure: DVec3::ZERO,
            coeff_method: AeroCoeffMethod::Diffuse,
            temperature: 300.0,
        };

        let result = compute_flat_plate_aero(&[facet], rel_vel, dyn_press, &gas, DVec3::ZERO);

        let s = velocity / (2.0 * gas.gas_const * gas.temp_free_stream).sqrt();
        let s_2 = s * s;
        let exp_s2 = (-s_2).exp();
        let temp_ratio = facet.temperature / gas.temp_free_stream;
        let cd_diff = (M_2_SQRTPI * s * exp_s2
            + temp_ratio.sqrt() * (2.0 / M_2_SQRTPI) * s
            + 1.0
            + 2.0 * s_2)
            / s_2;

        // force_t = force_base * cd_diff * sin_alpha (sin_alpha=1)
        let expected_force_x = -dyn_press * 10.0 * cd_diff;

        let rel_err = (result.force.x - expected_force_x).abs() / expected_force_x.abs();
        assert!(
            rel_err < 1e-12,
            "Diffuse normal force: expected {expected_force_x:.6e}, got {:.6e}",
            result.force.x
        );
        assert!(result.force.x < 0.0, "Force should oppose motion");
    }

    // ── Mixed: combination of specular and diffuse ───────────────

    /// Mixed with epsilon=1 should match pure specular.
    #[test]
    fn mixed_epsilon_1_matches_specular() {
        let density = 1e-12;
        let velocity = 7600.0;
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        let n = DVec3::new(1.0, 1.0, 0.0).normalize();
        let facet_spec = AeroFacet {
            area: 10.0,
            normal: n,
            center_pressure: DVec3::ZERO,
            coeff_method: AeroCoeffMethod::Specular,
            temperature: 300.0,
        };
        let facet_mixed = AeroFacet {
            coeff_method: AeroCoeffMethod::Mixed { epsilon: 1.0 },
            ..facet_spec
        };

        let r_spec = compute_flat_plate_aero(&[facet_spec], rel_vel, dyn_press, &gas, DVec3::ZERO);
        let r_mixed =
            compute_flat_plate_aero(&[facet_mixed], rel_vel, dyn_press, &gas, DVec3::ZERO);

        let err = (r_spec.force - r_mixed.force).length();
        assert!(
            err < 1e-20,
            "Mixed(epsilon=1) should match Specular: err = {err:.6e}"
        );
    }

    /// Mixed with epsilon=0 should match pure diffuse.
    #[test]
    fn mixed_epsilon_0_matches_diffuse() {
        let density = 1e-12;
        let velocity = 7600.0;
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        let n = DVec3::new(1.0, 1.0, 0.0).normalize();
        let facet_diff = AeroFacet {
            area: 10.0,
            normal: n,
            center_pressure: DVec3::ZERO,
            coeff_method: AeroCoeffMethod::Diffuse,
            temperature: 300.0,
        };
        let facet_mixed = AeroFacet {
            coeff_method: AeroCoeffMethod::Mixed { epsilon: 0.0 },
            ..facet_diff
        };

        let r_diff = compute_flat_plate_aero(&[facet_diff], rel_vel, dyn_press, &gas, DVec3::ZERO);
        let r_mixed =
            compute_flat_plate_aero(&[facet_mixed], rel_vel, dyn_press, &gas, DVec3::ZERO);

        let err = (r_diff.force - r_mixed.force).length();
        assert!(
            err < 1e-20,
            "Mixed(epsilon=0) should match Diffuse: err = {err:.6e}"
        );
    }

    // ── Multi-facet accumulation and torque ───────────────────────

    /// Multiple facets: forces accumulate, torque computed from offset.
    #[test]
    fn multi_facet_accumulation_and_torque() {
        let density = 1e-12;
        let velocity = 7600.0;
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        // Two identical facets, one offset in +Y, one in -Y.
        // Total force should be 2x single facet, torques should cancel.
        let facet_up = AeroFacet {
            area: 10.0,
            normal: DVec3::new(1.0, 0.0, 0.0),
            center_pressure: DVec3::new(0.0, 5.0, 0.0),
            coeff_method: AeroCoeffMethod::Specular,
            temperature: 300.0,
        };
        let facet_down = AeroFacet {
            center_pressure: DVec3::new(0.0, -5.0, 0.0),
            ..facet_up
        };

        let single = compute_flat_plate_aero(&[facet_up], rel_vel, dyn_press, &gas, DVec3::ZERO);
        let both = compute_flat_plate_aero(
            &[facet_up, facet_down],
            rel_vel,
            dyn_press,
            &gas,
            DVec3::ZERO,
        );

        // Force should be 2x
        let force_ratio = both.force.length() / single.force.length();
        assert!(
            (force_ratio - 2.0).abs() < 1e-12,
            "Two facets should give 2x force, got ratio {force_ratio}"
        );

        // Torques from symmetric facets should cancel (about origin CoM)
        assert!(
            both.torque.length() < 1e-20,
            "Symmetric facets should produce zero net torque, got {:?}",
            both.torque
        );
    }

    /// Single offset facet produces torque.
    #[test]
    fn single_facet_produces_torque() {
        let density = 1e-12;
        let velocity = 7600.0;
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        let facet = AeroFacet {
            area: 10.0,
            normal: DVec3::new(1.0, 0.0, 0.0),
            center_pressure: DVec3::new(0.0, 3.0, 0.0), // offset in +Y
            coeff_method: AeroCoeffMethod::Specular,
            temperature: 300.0,
        };

        let result = compute_flat_plate_aero(&[facet], rel_vel, dyn_press, &gas, DVec3::ZERO);

        // Force is along -X (specular normal = +X, force_n < 0)
        // Torque = [0, 3, 0] x [Fx, 0, 0] = [0, 0, -3*Fx]
        // Since Fx < 0, torque_z = -3*Fx > 0
        assert!(result.force.x < 0.0, "Force should be along -X");
        assert!(
            result.torque.z > 0.0,
            "Torque Z should be positive (right-hand rule)"
        );
        assert!(result.torque.x.abs() < 1e-20);
        assert!(result.torque.y.abs() < 1e-20);
    }

    // ── Ballistic vs flat-plate comparison ────────────────────────

    /// For a flat plate normal to flow, specular should give approximately
    /// F = dyn_press * A * Cd_spec (where Cd_spec depends on speed ratio).
    /// Compare with ballistic F = dyn_press * A * Cd.
    /// At LEO speeds, s is large (~10), so Cd_spec approaches 4 because the
    /// exp term becomes negligible and (2 + 4*s^2) / s^2 = 4 + 2/s^2 ≈ 4.
    #[test]
    fn specular_vs_ballistic_comparison() {
        let density = 1e-12;
        let velocity = 7600.0;
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        let s = velocity / (2.0 * gas.gas_const * gas.temp_free_stream).sqrt();
        let s_2 = s * s;
        let exp_s2 = (-s_2).exp();
        let cd_spec = (2.0 * M_2_SQRTPI * s * exp_s2 + 2.0 + 4.0 * s_2) / s_2;

        // At LEO speeds, s >> 1, so cd_spec ≈ 4 + 2/s^2 ≈ 4
        // Typical ballistic Cd = 2.0-2.5
        // Flat-plate specular gives roughly 2x ballistic because molecules
        // bounce back (momentum transfer is doubled).
        assert!(
            cd_spec > 3.5 && cd_spec < 5.0,
            "At LEO speeds, specular Cd should be ~4, got {cd_spec}"
        );

        let area = 10.0;
        let flat_plate_force_mag = dyn_press * area * cd_spec;
        let ballistic_force_mag = dyn_press * area * 2.2;

        assert!(
            flat_plate_force_mag > ballistic_force_mag,
            "Specular flat-plate should exceed ballistic (Cd=2.2): {flat_plate_force_mag:.6e} vs {ballistic_force_mag:.6e}"
        );
    }

    // ── CalcCoef: full coefficient method ─────────────────────────

    /// CalcCoef at normal incidence (alpha=90 deg): should have no tangential
    /// component.
    #[test]
    fn calc_coef_normal_incidence_no_tangential() {
        let density = 1e-12;
        let velocity = 7600.0;
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        let facet = AeroFacet {
            area: 10.0,
            normal: DVec3::new(1.0, 0.0, 0.0),
            center_pressure: DVec3::ZERO,
            coeff_method: AeroCoeffMethod::CalcCoef { epsilon: 0.5 },
            temperature: 300.0,
        };

        let result = compute_flat_plate_aero(&[facet], rel_vel, dyn_press, &gas, DVec3::ZERO);

        // At normal incidence, force should be purely along the normal (+X, but negative)
        assert!(result.force.x < 0.0, "Force should oppose motion");
        assert!(result.force.y.abs() < 1e-20, "No Y component");
        assert!(result.force.z.abs() < 1e-20, "No Z component");
    }

    /// CalcCoef at oblique incidence: should have both normal and tangential
    /// components.
    #[test]
    fn calc_coef_oblique_incidence() {
        let density = 1e-12;
        let velocity = 7600.0;
        let rel_vel = DVec3::new(velocity, 0.0, 0.0);
        let dyn_press = 0.5 * density * velocity * velocity;
        let gas = leo_gas_params();

        let facet = AeroFacet {
            area: 10.0,
            normal: DVec3::new(1.0, 1.0, 0.0).normalize(),
            center_pressure: DVec3::ZERO,
            coeff_method: AeroCoeffMethod::CalcCoef { epsilon: 0.5 },
            temperature: 300.0,
        };

        let result = compute_flat_plate_aero(&[facet], rel_vel, dyn_press, &gas, DVec3::ZERO);

        // Force should have both X and Y components
        assert!(result.force.x < 0.0, "Force X should oppose motion");
        assert!(
            result.force.y.abs() > 1e-30,
            "Force Y should be non-zero for oblique incidence"
        );
    }

    // ── Zero velocity / zero dynamic pressure ────────────────────

    /// Zero velocity gives zero force.
    #[test]
    fn zero_velocity_zero_force() {
        let gas = leo_gas_params();
        let facet = AeroFacet {
            area: 10.0,
            normal: DVec3::new(1.0, 0.0, 0.0),
            center_pressure: DVec3::ZERO,
            coeff_method: AeroCoeffMethod::Specular,
            temperature: 300.0,
        };

        let result = compute_flat_plate_aero(&[facet], DVec3::ZERO, 0.0, &gas, DVec3::ZERO);
        assert_eq!(result.force, DVec3::ZERO);
        assert_eq!(result.torque, DVec3::ZERO);
    }
}
