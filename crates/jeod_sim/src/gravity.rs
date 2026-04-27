use glam::{DMat3, DVec3};
use jeod_dynamics::forces::GravityAccelerationTyped;
use jeod_dynamics::GravityAcceleration;
use jeod_gravity::{GravityControls, GravitySource};
use jeod_quantities::aliases::{Acceleration, Position};
use jeod_quantities::frame::Inertial;

/// Information about a gravity source resolved from a source identifier.
///
/// Returned by the `source_lookup` closure passed to [`accumulate_gravity`].
pub struct ResolvedSource<'a> {
    /// Physical gravity source (mu, model data).
    pub source: &'a GravitySource,
    /// Optional inertial-to-planet-fixed rotation (required for spherical harmonics).
    pub rotation: Option<&'a DMat3>,
    /// Source center position in the inertial frame (m). Used for differential
    /// (third-body) acceleration computation.
    pub position: DVec3,
    /// Tidal ΔC20 to add to the base C20 coefficient during spherical harmonics
    /// evaluation. Zero when no tidal effects are configured.
    pub delta_c20: f64,
    /// Whether this source has tidal delta coefficient sets configured.
    /// Matches JEOD's `n_deltacoeffs > 0` (`harmonics_source->delta_coeffs.size() > 0`).
    /// Gates permanent tide correction (`tide_free_delta`) in the spherical
    /// harmonics computation.
    pub has_delta_coeffs: bool,
}

/// Accumulate gravity from all sources for a single body.
///
/// Iterates over the body's gravity controls, looks up each source via
/// `source_lookup`, and accumulates acceleration, gradient, and potential.
///
/// For sources with `differential == true` (third-body perturbations), the
/// acceleration is computed as the differential: acceleration of the vehicle
/// toward the source minus acceleration of the integration frame origin toward
/// the source. This matches JEOD's `GravityIntegFrame::is_third_body` logic
/// in `gravity_controls.cc:calc_spherical()`.
///
/// # Type parameter
/// - `S`: source identifier type. In Bevy: `Entity`. In `Simulation`: `usize`.
///
/// The `source_lookup` closure resolves it to a [`ResolvedSource`] containing
/// the gravity model, optional planet-fixed rotation, and inertial position.
///
/// # Arguments
/// - `position`: body position in the inertial frame (m)
/// - `controls`: per-body gravity controls specifying which sources to use
/// - `integration_origin`: position of the integration frame origin in the
///   inertial frame (m). Typically `DVec3::ZERO` for Earth-centered integration.
/// - `source_lookup`: resolves source identifiers to physical data
///
/// # Panics
/// - Source not found (JEOD_INV: GV.12)
/// - Non-spherical gravity without planet-fixed rotation
// JEOD_INV: GV.12 — gravity source must exist for control
pub fn accumulate_gravity<'a, S: Copy + std::fmt::Debug>(
    position: DVec3,
    controls: &GravityControls<S>,
    integration_origin: DVec3,
    source_lookup: impl Fn(S) -> Option<ResolvedSource<'a>>,
) -> GravityAcceleration {
    let mut total = GravityAcceleration::default();

    for ctrl in &controls.controls {
        // JEOD_INV: GV.12 — gravity source must exist for control.
        // JEOD: GravityControls::initialize_control() calls MessageHandler::error()
        // (non-fatal, severity 0) when find_grav_source() returns nullptr. We
        // escalate to a panic because silently omitting a gravity source would
        // produce incorrect physics.
        let resolved = source_lookup(ctrl.source_name).unwrap_or_else(|| {
            panic!(
                "GravityControl references source {:?} which does not exist. \
                 JEOD logs a non-fatal error and skips; we panic to prevent \
                 silently wrong physics.",
                ctrl.source_name
            );
        });

        // Pre-check: only panic if non-spherical evaluation against this
        // *specific source* will actually require the rotation (i.e.,
        // `effective_orders().0 >= 2`). `is_nonspherical()` is config-only
        // and over-approximates: a control with `degree=1` or one paired
        // with a `PointMass` source collapses to spherical at runtime and
        // doesn't need rotation.
        if ctrl.requires_planet_fixed_rotation(resolved.source) && resolved.rotation.is_none() {
            panic!(
                "Non-spherical GravityControl (degree={}, order={}) references \
                 source {:?} which has no planet-fixed rotation matrix.",
                ctrl.degree, ctrl.order, ctrl.source_name
            );
        }

        // Compute position relative to source center.
        // JEOD: posn = integ_pos + grav_source_frame.pos (where
        // grav_source_frame.pos = integration frame origin - source center).
        // In our coordinates: pos_rel = vehicle_inertial - source_inertial.
        let pos_relative_to_source = position - resolved.position;

        let result = ctrl.evaluate(
            resolved.source,
            pos_relative_to_source,
            resolved.rotation,
            resolved.delta_c20,
            resolved.has_delta_coeffs,
        );

        // JEOD_INV: GV.14 — third-body sources use differential acceleration
        // JEOD gravity_controls.cc:306-347: if is_third_body, subtract the
        // acceleration of the integration frame origin toward the source.
        // Original method: a_diff = -mu/|d|^3 * d + mu/|rho|^3 * rho
        // where d = vehicle->source, rho = frame_origin->source.
        if ctrl.differential {
            let frame_pos_relative_to_source = integration_origin - resolved.position;
            assert!(
                frame_pos_relative_to_source.length_squared() > 0.0,
                "Differential (third-body) gravity source {:?} is at the integration \
                 frame origin. Third-body sources must be distinct from the central \
                 body (e.g., Sun/Moon when integrating in an Earth-centered frame).",
                ctrl.source_name
            );

            // Battin's method applies only to the spherical (point-mass) term of
            // third-body gravity. In JEOD, calc_spherical() contains the Battin
            // logic while calc_nonspherical() is evaluated separately without any
            // third-body differential. We gate Battin to point-mass-only controls
            // to match JEOD's architecture — a non-spherical control with
            // battin_method would silently skip the harmonics contribution.
            if ctrl.battin_method && !ctrl.is_nonspherical() && !ctrl.perturbing_only {
                // Battin's method for improved third-body numerical accuracy.
                // Avoids catastrophic cancellation when the vehicle is close to
                // the integration frame origin relative to the source distance.
                // JEOD ref: gravity_controls.cc:317-331
                //
                // JEOD uses `grav_source_frame.pos` which points FROM source TO
                // frame origin (= integration_origin - source_position), which is
                // directionally opposite to `rho` in Battin's documentation.
                // We follow JEOD's variable naming exactly to avoid sign errors.
                let mu = resolved.source.mu;
                // vehicle relative to frame origin
                let integ_pos = position - integration_origin;
                // Reuse the already-computed frame-relative vector so the
                // Battin and direct-subtraction paths stay consistent.
                // grav_pos = frame_origin - source = -rho (matches JEOD's grav_source_frame.pos)
                let grav_pos = frame_pos_relative_to_source;
                let rho_sq = grav_pos.length_squared();
                let rho_mag = rho_sq.sqrt();
                let rho_3rd = rho_sq * rho_mag;

                // JEOD: r_dot_rho = dot(integ_pos, grav_source_frame.pos)
                let r_dot_rho = integ_pos.dot(grav_pos);
                let q = (integ_pos.length_squared() + 2.0 * r_dot_rho) / rho_sq;

                if q > -0.38 {
                    // Battin polynomial: f(q) = q * (3 + q * (3 + q)) / (1 + sqrt(1 + q))
                    let fq = q * (3.0 + q * (3.0 + q)) / (1.0 + (1.0 + q).sqrt());
                    let scale = -mu / (rho_3rd * (1.0 + fq));
                    // JEOD: scale_decr(grav_source_frame.pos, q, integ_pos)
                    // = integ_pos - fq * grav_source_frame.pos
                    // = integ_pos - fq * grav_pos
                    let accel = (integ_pos - grav_pos * fq) * scale;
                    total.grav_accel += accel;
                } else {
                    // Fallback to direct subtraction when q <= -0.38, i.e. when
                    // integ_pos points sufficiently opposite grav_pos (commonly:
                    // vehicle between the frame origin and the source).
                    let frame_accel = ctrl.evaluate_accel_only(
                        resolved.source,
                        frame_pos_relative_to_source,
                        resolved.rotation,
                        resolved.delta_c20,
                        resolved.has_delta_coeffs,
                    );
                    total.grav_accel += result.grav_accel - frame_accel.grav_accel;
                }
            } else {
                let frame_accel = ctrl.evaluate_accel_only(
                    resolved.source,
                    frame_pos_relative_to_source,
                    resolved.rotation,
                    resolved.delta_c20,
                    resolved.has_delta_coeffs,
                );
                total.grav_accel += result.grav_accel - frame_accel.grav_accel;
            }
        } else {
            total.grav_accel += result.grav_accel;
        }

        if ctrl.gradient {
            total.grav_grad += result.grav_grad;
        }
        total.grav_pot += result.grav_pot;
    }

    total
}

/// Resolved source for relativistic correction computation.
///
/// Provides the mu and position of a gravity source, plus its velocity
/// for the post-Newtonian coordinate velocity term.
pub struct ResolvedRelativisticSource {
    /// Gravitational parameter (m³/s²).
    pub mu: f64,
    /// Position of source in the inertial frame (m).
    pub position: DVec3,
    /// Velocity of source in the inertial frame (m/s).
    pub velocity: DVec3,
}

/// Compute post-Newtonian relativistic corrections for all gravity controls
/// that have `relativistic: true`.
///
/// Returns the total relativistic acceleration correction to be added to
/// the Newtonian gravity acceleration.
///
/// This function iterates over the gravity controls, and for each relativistic
/// source, builds the "other sources" list and calls
/// [`jeod_gravity::relativistic::compute_relativistic_correction`].
///
/// `source_lookup` resolves source identifiers (index or Entity) to
/// `ResolvedRelativisticSource` values, the same pattern as `accumulate_gravity`.
pub fn accumulate_relativistic_corrections<S: Copy + std::fmt::Debug + PartialEq>(
    body_position: DVec3,
    body_velocity: DVec3,
    controls: &GravityControls<S>,
    source_lookup: impl Fn(S) -> Option<ResolvedRelativisticSource>,
) -> DVec3 {
    let mut total_correction = DVec3::ZERO;

    for ctrl in &controls.controls {
        if !ctrl.relativistic {
            continue;
        }
        let Some(src) = source_lookup(ctrl.source_name) else {
            continue;
        };
        let other: Vec<jeod_gravity::relativistic::RelativisticSource> = controls
            .controls
            .iter()
            .filter(|c| c.source_name != ctrl.source_name)
            .filter_map(|c| {
                source_lookup(c.source_name).map(|s| {
                    jeod_gravity::relativistic::RelativisticSource {
                        mu: s.mu,
                        position: s.position,
                    }
                })
            })
            .collect();

        total_correction += jeod_gravity::relativistic::compute_relativistic_correction(
            src.mu,
            src.position,
            body_position,
            body_velocity,
            src.velocity,
            &other,
        );
    }

    total_correction
}

/// Typed sibling of [`accumulate_gravity`].
///
/// Identical numerics — wraps the untyped kernel by extracting the SI
/// values via [`Position::raw_si`] at entry and reconstructing the typed
/// [`GravityAccelerationTyped<Inertial>`] at exit. The `source_lookup`
/// closure still returns the existing untyped [`ResolvedSource`].
// JEOD_INV: GV.12 — gravity source must exist for control
pub fn accumulate_gravity_typed<'a, S: Copy + std::fmt::Debug>(
    position: Position<Inertial>,
    controls: &GravityControls<S>,
    integration_origin: Position<Inertial>,
    source_lookup: impl Fn(S) -> Option<ResolvedSource<'a>>,
) -> GravityAccelerationTyped<Inertial> {
    let raw = accumulate_gravity(
        position.raw_si(),
        controls,
        integration_origin.raw_si(),
        source_lookup,
    );
    GravityAccelerationTyped::<Inertial>::from_untyped_unchecked(&raw)
}

/// Typed sibling of [`accumulate_relativistic_corrections`].
///
/// Same kernel; entry/exit boundary types are
/// `Position<Inertial>` / `Velocity<Inertial>` / `Acceleration<Inertial>`.
pub fn accumulate_relativistic_corrections_typed<S: Copy + std::fmt::Debug + PartialEq>(
    body_position: Position<Inertial>,
    body_velocity: jeod_quantities::aliases::Velocity<Inertial>,
    controls: &GravityControls<S>,
    source_lookup: impl Fn(S) -> Option<ResolvedRelativisticSource>,
) -> Acceleration<Inertial> {
    let raw = accumulate_relativistic_corrections(
        body_position.raw_si(),
        body_velocity.raw_si(),
        controls,
        source_lookup,
    );
    Acceleration::<Inertial>::from_raw_si(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeod_gravity::gravity_source::{GravityModel, GravitySource};
    use jeod_gravity::{GravityControl, GravityControls};

    /// Helper: create a point-mass gravity source with the given mu.
    fn point_mass_source(mu: f64) -> GravitySource {
        GravitySource {
            mu,
            model: GravityModel::PointMass,
        }
    }

    /// Helper: build a single third-body control with or without Battin's method.
    fn third_body_control(source_id: usize, battin: bool) -> GravityControl<usize> {
        let mut ctrl = GravityControl::new_third_body(source_id);
        ctrl.battin_method = battin;
        ctrl
    }

    /// Battin ON vs OFF should produce results that agree to within the
    /// precision limits of the direct subtraction method.
    ///
    /// Uses two geometries:
    /// 1. A close source (10,000 km) where both methods are well-conditioned
    ///    and should agree to a tight tolerance.
    /// 2. A distant source (Sun at 1 AU) where direct subtraction loses ~5
    ///    digits but both methods should still agree in direction and magnitude.
    #[test]
    fn battin_vs_direct_agree_for_leo() {
        // --- Case 1: Well-conditioned geometry (close source) ---
        // Source at 10,000 km, vehicle at 6,778 km off-axis. The vehicle-to-source
        // distance is comparable to the origin-to-source distance, so direct
        // subtraction loses only ~1 digit → both methods should agree tightly.
        let mu = 1.0e15;
        let source_pos = DVec3::new(1.0e7, 0.0, 0.0); // 10,000 km
        let source = point_mass_source(mu);
        let vehicle_pos = DVec3::new(6_000_000.0, 3_000_000.0, 1_000_000.0);
        let integration_origin = DVec3::ZERO;

        let controls_direct = GravityControls {
            controls: vec![third_body_control(0, false)],
        };
        let result_direct =
            accumulate_gravity(vehicle_pos, &controls_direct, integration_origin, |_| {
                Some(ResolvedSource {
                    source: &source,
                    rotation: None,
                    position: source_pos,
                    delta_c20: 0.0,
                    has_delta_coeffs: false,
                })
            });

        let controls_battin = GravityControls {
            controls: vec![third_body_control(0, true)],
        };
        let result_battin =
            accumulate_gravity(vehicle_pos, &controls_battin, integration_origin, |_| {
                Some(ResolvedSource {
                    source: &source,
                    rotation: None,
                    position: source_pos,
                    delta_c20: 0.0,
                    has_delta_coeffs: false,
                })
            });

        // Both produce non-zero differential acceleration
        assert!(result_direct.grav_accel.length() > 0.0);
        assert!(result_battin.grav_accel.length() > 0.0);

        // Tight per-component check: well-conditioned geometry means
        // both methods agree to near machine epsilon.
        for (i, axis) in ["x", "y", "z"].iter().enumerate() {
            let d = result_direct.grav_accel[i];
            let b = result_battin.grav_accel[i];
            let rel_err = (b - d).abs() / d.abs().max(1e-30);
            assert!(
                rel_err < 1e-12,
                "Case 1: Battin vs direct {axis}-component relative error too large: \
                 {rel_err:.3e} (direct={d:.6e}, battin={b:.6e})"
            );
        }

        // --- Case 2: Sun geometry (cancellation-prone) ---
        // Verify both methods agree in direction and order of magnitude even
        // when direct subtraction loses ~5 digits.
        let mu_sun = 1.327_124_400_18e20;
        let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
        let sun_source = point_mass_source(mu_sun);

        let result_direct_sun =
            accumulate_gravity(vehicle_pos, &controls_direct, integration_origin, |_| {
                Some(ResolvedSource {
                    source: &sun_source,
                    rotation: None,
                    position: sun_pos,
                    delta_c20: 0.0,
                    has_delta_coeffs: false,
                })
            });
        let result_battin_sun =
            accumulate_gravity(vehicle_pos, &controls_battin, integration_origin, |_| {
                Some(ResolvedSource {
                    source: &sun_source,
                    rotation: None,
                    position: sun_pos,
                    delta_c20: 0.0,
                    has_delta_coeffs: false,
                })
            });

        // With ~5 digits lost in direct method, allow up to 1e-4 relative error
        // (this is the direct method's rounding, not a Battin bug).
        let rel_err = (result_battin_sun.grav_accel - result_direct_sun.grav_accel).length()
            / result_direct_sun.grav_accel.length();
        assert!(
            rel_err < 1e-4,
            "Case 2: Sun geometry relative error too large: {rel_err:.3e}"
        );

        // Same direction (dot product > 0)
        assert!(
            result_battin_sun
                .grav_accel
                .dot(result_direct_sun.grav_accel)
                > 0.0,
            "Case 2: Battin and direct should point in the same direction"
        );
    }

    /// Battin's method must handle q = 0 correctly (vehicle at integration
    /// frame origin). In this case integ_pos = 0, so q = 0 and fq = 0,
    /// giving zero differential acceleration (as expected: if the vehicle
    /// is at the frame origin, the differential is zero).
    #[test]
    fn battin_q_zero_vehicle_at_origin() {
        let mu_sun = 1.327_124_400_18e20;
        let sun_pos = DVec3::new(1.496e11, 0.0, 0.0);
        let source = point_mass_source(mu_sun);

        // Vehicle exactly at integration frame origin
        let vehicle_pos = DVec3::ZERO;
        let integration_origin = DVec3::ZERO;

        let controls = GravityControls {
            controls: vec![third_body_control(0, true)],
        };
        let result = accumulate_gravity(vehicle_pos, &controls, integration_origin, |_| {
            Some(ResolvedSource {
                source: &source,
                rotation: None,
                position: sun_pos,
                delta_c20: 0.0,
                has_delta_coeffs: false,
            })
        });

        // Differential acceleration should be zero when vehicle is at frame origin
        assert!(
            result.grav_accel.length() < 1e-30,
            "Expected zero accel at frame origin, got {:?}",
            result.grav_accel
        );
    }

    /// When q <= -0.38 (vehicle between the frame origin and the source),
    /// Battin's method falls back to direct subtraction.
    /// Verify the fallback path produces the same result as non-Battin.
    #[test]
    fn battin_fallback_q_below_threshold() {
        // Construct a geometry where q < -0.38.
        // With JEOD's convention: grav_pos = origin - source, so
        // q = (|integ_pos|^2 + 2 * dot(integ_pos, grav_pos)) / |grav_pos|^2
        // Place vehicle between origin and source so integ_pos opposes grav_pos.
        let mu = 1.0e15;
        let source_pos = DVec3::new(1000.0, 0.0, 0.0); // source at +x
        let source = point_mass_source(mu);
        let integration_origin = DVec3::ZERO;

        // Vehicle at (500, 0, 0): integ_pos = (500, 0, 0), grav_pos = (-1000, 0, 0)
        // r_dot_rho = 500 * (-1000) = -5e5
        // q = (2.5e5 + 2*(-5e5)) / 1e6 = -0.75, which is < -0.38
        let vehicle_pos = DVec3::new(500.0, 0.0, 0.0);

        let controls_direct = GravityControls {
            controls: vec![third_body_control(0, false)],
        };
        let result_direct =
            accumulate_gravity(vehicle_pos, &controls_direct, integration_origin, |_| {
                Some(ResolvedSource {
                    source: &source,
                    rotation: None,
                    position: source_pos,
                    delta_c20: 0.0,
                    has_delta_coeffs: false,
                })
            });

        let controls_battin = GravityControls {
            controls: vec![third_body_control(0, true)],
        };
        let result_battin =
            accumulate_gravity(vehicle_pos, &controls_battin, integration_origin, |_| {
                Some(ResolvedSource {
                    source: &source,
                    rotation: None,
                    position: source_pos,
                    delta_c20: 0.0,
                    has_delta_coeffs: false,
                })
            });

        // Fallback path should give identical results to direct method
        let diff = (result_battin.grav_accel - result_direct.grav_accel).length();
        assert!(
            diff < 1e-30,
            "Battin fallback should match direct method exactly, diff = {:.3e}",
            diff
        );
    }

    /// Battin's method is gated to point-mass-only controls. When a control
    /// has `battin_method = true` but `perturbing_only = true` (which skips
    /// the spherical term in JEOD), Battin should fall back to direct
    /// subtraction. This matches JEOD's architecture where Battin is only
    /// inside `calc_spherical`, which is guarded by `!perturbing_only`.
    #[test]
    fn battin_falls_back_for_perturbing_only() {
        let mu = 1.0e15;
        let source_pos = DVec3::new(1e8, 0.0, 0.0);
        let source = point_mass_source(mu);
        let vehicle_pos = DVec3::new(6_778_000.0, 0.0, 0.0);
        let integration_origin = DVec3::ZERO;

        // Spherical control with perturbing_only + battin_method = true
        let mut ctrl = GravityControl::new_third_body(0);
        ctrl.battin_method = true;
        ctrl.perturbing_only = true;

        // Direct (no battin) with perturbing_only
        let mut ctrl_direct = GravityControl::new_third_body(0);
        ctrl_direct.battin_method = false;
        ctrl_direct.perturbing_only = true;

        let controls_battin = GravityControls {
            controls: vec![ctrl],
        };
        let controls_direct = GravityControls {
            controls: vec![ctrl_direct],
        };

        let result_battin =
            accumulate_gravity(vehicle_pos, &controls_battin, integration_origin, |_| {
                Some(ResolvedSource {
                    source: &source,
                    rotation: None,
                    position: source_pos,
                    delta_c20: 0.0,
                    has_delta_coeffs: false,
                })
            });
        let result_direct =
            accumulate_gravity(vehicle_pos, &controls_direct, integration_origin, |_| {
                Some(ResolvedSource {
                    source: &source,
                    rotation: None,
                    position: source_pos,
                    delta_c20: 0.0,
                    has_delta_coeffs: false,
                })
            });

        let diff = (result_battin.grav_accel - result_direct.grav_accel).length();
        assert!(
            diff < 1e-30,
            "Battin with perturbing_only should fall back to direct, diff = {diff:.3e}"
        );
    }

    /// Verify `battin_method` defaults to false in all constructors.
    #[test]
    fn battin_method_defaults_false() {
        let spherical: GravityControl<usize> = GravityControl::new_spherical(0, false);
        assert!(!spherical.battin_method);

        let nonspherical: GravityControl<usize> = GravityControl::new_nonspherical(0, 4, 4, false);
        assert!(!nonspherical.battin_method);

        let third_body: GravityControl<usize> = GravityControl::new_third_body(0);
        assert!(!third_body.battin_method);
    }
}
