//! Gravity stage: per-body gravity accumulation across multiple sources
//! (point-mass, spherical-harmonics with optional tides, plus optional
//! relativistic post-Newtonian corrections).

use crate::{GravityControls, GravitySource};
use astrodyn_dynamics::forces::GravityAccelerationTyped;
use astrodyn_dynamics::GravityAcceleration;
use astrodyn_quantities::aliases::{Acceleration, Position, Velocity};
use astrodyn_quantities::frame::{IntegrationFrame, RootInertial};
use astrodyn_quantities::integ_origin::IntegOrigin;
use glam::{DMat3, DVec3};

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
/// [`crate::relativistic::compute_relativistic_correction`].
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
        let other: Vec<crate::relativistic::RelativisticSource> = controls
            .controls
            .iter()
            .filter(|c| c.source_name != ctrl.source_name)
            .filter_map(|c| {
                source_lookup(c.source_name).map(|s| crate::relativistic::RelativisticSource {
                    mu: s.mu,
                    position: s.position,
                })
            })
            .collect();

        total_correction += crate::relativistic::compute_relativistic_correction(
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
/// [`GravityAccelerationTyped<RootInertial>`] at exit. The `source_lookup`
/// closure still returns the existing untyped [`ResolvedSource`].
// JEOD_INV: GV.12 — gravity source must exist for control
pub fn accumulate_gravity_typed<'a, S: Copy + std::fmt::Debug>(
    position: Position<RootInertial>,
    controls: &GravityControls<S>,
    integration_origin: Position<RootInertial>,
    source_lookup: impl Fn(S) -> Option<ResolvedSource<'a>>,
) -> GravityAccelerationTyped<RootInertial> {
    let raw = accumulate_gravity(
        position.raw_si(),
        controls,
        integration_origin.raw_si(),
        source_lookup,
    );
    GravityAccelerationTyped::<RootInertial>::from_untyped_unchecked(&raw)
}

/// Typed sibling of [`accumulate_relativistic_corrections`].
///
/// Same kernel; entry/exit boundary types are
/// `Position<RootInertial>` / `Velocity<RootInertial>` / `Acceleration<RootInertial>`.
pub fn accumulate_relativistic_corrections_typed<S: Copy + std::fmt::Debug + PartialEq>(
    body_position: Position<RootInertial>,
    body_velocity: Velocity<RootInertial>,
    controls: &GravityControls<S>,
    source_lookup: impl Fn(S) -> Option<ResolvedRelativisticSource>,
) -> Acceleration<RootInertial> {
    let raw = accumulate_relativistic_corrections(
        body_position.raw_si(),
        body_velocity.raw_si(),
        controls,
        source_lookup,
    );
    Acceleration::<RootInertial>::from_raw_si(raw)
}

/// Per-body gravity composition: applies the RF.10 integ-origin shift,
/// accumulates Newtonian gravity, and adds the post-Newtonian
/// relativistic correction in a single call.
///
/// Both ECS adapters (the `astrodyn_bevy` root crate's
/// `gravity_computation_system` and `astrodyn_runner`'s
/// `Simulation::update_environment`) call this kernel inside their
/// per-body loops. Consolidating the shift and the
/// Newtonian/relativistic sum here means a new resolved-source field, a
/// new accumulator term, or a refinement to the RF.10 shift edits one
/// site instead of two.
///
/// The body's position and velocity arrive in the integration frame;
/// the kernel emits a `GravityAccelerationTyped<RootInertial>` after
/// the shift. Callers that store body state typed against
/// `PlanetInertial<P>` (the typical Bevy storage shape) re-label via
/// `from_raw_si` at the call site — relabel-then-shift is the canonical
/// path the type system endorses (see `IntegOrigin` rustdoc).
///
/// # Arguments
/// - `body_position`: body position in its integration frame (m)
/// - `body_velocity`: body velocity in its integration frame (m/s)
/// - `integ_origin`: integration-frame origin in root-inertial coords; use
///   [`IntegOrigin::zero`] for root-integrated bodies
/// - `controls`: per-body gravity controls
/// - `resolve_source`: source resolver for Newtonian gravity (same closure
///   shape as [`accumulate_gravity_typed`])
/// - `resolve_rel_source`: source resolver for the relativistic correction
///   (same closure shape as [`accumulate_relativistic_corrections_typed`])
// JEOD_INV: RF.10 — integ-frame to root-inertial shift is encapsulated here.
pub fn evaluate_body_gravity_typed<'a, S: Copy + std::fmt::Debug + PartialEq>(
    body_position: Position<IntegrationFrame>,
    body_velocity: Velocity<IntegrationFrame>,
    integ_origin: IntegOrigin,
    controls: &GravityControls<S>,
    resolve_source: impl Fn(S) -> Option<ResolvedSource<'a>>,
    resolve_rel_source: impl Fn(S) -> Option<ResolvedRelativisticSource>,
) -> GravityAccelerationTyped<RootInertial> {
    let abs_pos = integ_origin.shift_position(body_position);
    let abs_vel = integ_origin.shift_velocity(body_velocity);

    let mut accel =
        accumulate_gravity_typed(abs_pos, controls, integ_origin.position, resolve_source);
    let rel =
        accumulate_relativistic_corrections_typed(abs_pos, abs_vel, controls, resolve_rel_source);
    accel.grav_accel += rel;
    accel
}

/// Per-body inputs the gravity stage driver consumes.
///
/// Bundled together so adapter `.map(...)` projections over a `Query`
/// row or a `&mut SimBody` can yield a single record per body. The
/// `controls` field carries the per-row borrow lifetime; both the
/// driver's `BodyIter` items and its `resolve_source` closure share
/// that lifetime via the `'a` parameter on
/// [`run_gravity_stage`].
pub struct GravityBodyInputs<'a, S> {
    /// Body position in its integration frame.
    pub position: Position<IntegrationFrame>,
    /// Body velocity in its integration frame.
    pub velocity: Velocity<IntegrationFrame>,
    /// Integration-frame origin in root-inertial coordinates; use
    /// [`IntegOrigin::zero`] for root-integrated bodies.
    pub integ_origin: IntegOrigin,
    /// Per-body gravity controls. Borrowed because controls live in
    /// adapter-owned storage (Bevy `Query<&GravityControlsC>` row,
    /// `SimBody.gravity_controls` field) and cloning is unnecessary.
    pub controls: &'a GravityControls<S>,
}

/// Drive the gravity stage across an adapter-supplied iterator of
/// bodies.
///
/// `bodies` yields one `(key, inputs, store)` triple per body. The
/// driver:
///
/// 1. Calls [`evaluate_body_gravity_typed`] (RF.10 shift + Newtonian
///    accumulation + post-Newtonian correction) for each body.
/// 2. Hands the result to the per-body `store` closure, which writes
///    it back into adapter-owned storage. Both adapters now store
///    `GravityAccelerationTyped<RootInertial>` (the Bevy
///    `GravityAccelerationC` newtype and `SimBody.gravity_accel`),
///    so the closure is a direct move.
///
/// `resolve_source` and `resolve_rel_source` receive the body key
/// alongside the source identifier so adapters that want to surface
/// body context in diagnostic panics (e.g. *"Entity {body:?}: gravity
/// source {src:?} not found"*) keep the reference they need.
///
/// The closure-based writer lets each adapter use whatever mutable
/// handle its iterator yields (`Mut<'_, GravityAccelerationC>` for
/// Bevy, `&mut GravityAccelerationTyped<RootInertial>` for the runner)
/// without needing a trait impl on a foreign type — orphan rules block
/// trait impls on `Mut<'_, T>` from outside `bevy_ecs`.
///
/// Both ECS adapters today route gravity through this function; future
/// stage-shape changes (a new pre-step, a new gating condition, new
/// telemetry) edit one site rather than two.
pub fn run_gravity_stage<'a, S, K, Store, BodyIter, FS, FR>(
    bodies: BodyIter,
    resolve_source: FS,
    resolve_rel_source: FR,
) where
    S: Copy + std::fmt::Debug + PartialEq + 'a,
    K: Copy,
    Store: FnOnce(GravityAccelerationTyped<RootInertial>),
    BodyIter: IntoIterator<Item = (K, GravityBodyInputs<'a, S>, Store)>,
    FS: Fn(K, S) -> Option<ResolvedSource<'a>>,
    FR: Fn(K, S) -> Option<ResolvedRelativisticSource>,
{
    for (key, inputs, store) in bodies {
        let result = evaluate_body_gravity_typed(
            inputs.position,
            inputs.velocity,
            inputs.integ_origin,
            inputs.controls,
            |s| resolve_source(key, s),
            |s| resolve_rel_source(key, s),
        );
        store(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_source::{GravityModel, GravitySource};
    use crate::{GravityControl, GravityControls, GravityRole};

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
        let mut ctrl: GravityControl<usize> = GravityControl::new_third_body(0_usize);
        ctrl.battin_method = true;
        ctrl.perturbing_only = true;

        // Direct (no battin) with perturbing_only
        let mut ctrl_direct: GravityControl<usize> = GravityControl::new_third_body(0_usize);
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
        let spherical: GravityControl<usize> =
            GravityControl::new_spherical(0_usize, GravityRole::Central);
        assert!(!spherical.battin_method);

        let nonspherical: GravityControl<usize> =
            GravityControl::new_nonspherical(0_usize, 4, 4, GravityRole::Central);
        assert!(!nonspherical.battin_method);

        let third_body: GravityControl<usize> = GravityControl::new_third_body(0_usize);
        assert!(!third_body.battin_method);
    }

    /// `evaluate_body_gravity_typed` is numerically equivalent to the
    /// open-coded shift-then-accumulate sequence the runner has carried
    /// since before the kernel existed: lift body coordinates to root
    /// inertial via the integ-origin, then call `accumulate_gravity` and
    /// `accumulate_relativistic_corrections` and sum into `grav_accel`.
    #[test]
    fn evaluate_body_gravity_matches_manual_composition() {
        let mu = 3.986e14;
        let source = point_mass_source(mu);
        let source_pos = DVec3::ZERO;

        // Body at LEO altitude in an integration frame offset 1 AU
        // along x from root-inertial — exercises the RF.10 shift.
        let body_pos_integ = DVec3::new(7e6, 0.0, 0.0);
        let body_vel_integ = DVec3::new(0.0, 7500.0, 0.0);
        let integ_origin = IntegOrigin {
            position: Position::<RootInertial>::from_raw_si(DVec3::new(1.5e11, 0.0, 0.0)),
            velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, 30_000.0, 0.0)),
        };

        let controls = GravityControls {
            controls: vec![GravityControl::new_spherical(0_usize, GravityRole::Central)],
        };
        let resolve_source = |_: usize| {
            Some(ResolvedSource {
                source: &source,
                rotation: None,
                position: source_pos,
                delta_c20: 0.0,
                has_delta_coeffs: false,
            })
        };
        let resolve_rel_source = |_: usize| {
            Some(ResolvedRelativisticSource {
                mu,
                position: source_pos,
                velocity: DVec3::ZERO,
            })
        };

        // New kernel.
        let kernel = evaluate_body_gravity_typed(
            Position::<IntegrationFrame>::from_raw_si(body_pos_integ),
            Velocity::<IntegrationFrame>::from_raw_si(body_vel_integ),
            integ_origin,
            &controls,
            resolve_source,
            resolve_rel_source,
        );

        // Manual composition — exactly what the runner did before the
        // kernel landed.
        let abs_pos = body_pos_integ + integ_origin.position.raw_si();
        let abs_vel = body_vel_integ + integ_origin.velocity.raw_si();
        let mut manual = accumulate_gravity(
            abs_pos,
            &controls,
            integ_origin.position.raw_si(),
            resolve_source,
        );
        manual.grav_accel +=
            accumulate_relativistic_corrections(abs_pos, abs_vel, &controls, resolve_rel_source);

        assert_eq!(kernel.grav_accel.raw_si(), manual.grav_accel);
        assert_eq!(kernel.grav_pot, manual.grav_pot);
        assert_eq!(kernel.grav_grad, manual.grav_grad);
    }

    /// `run_gravity_stage` over an iterator of two bodies produces the
    /// same per-body accelerations as calling `evaluate_body_gravity_typed`
    /// directly. The driver is a thin loop over the kernel, so this is a
    /// shape check — no math regression possible if the per-body kernel
    /// tests pass.
    #[test]
    fn run_gravity_stage_drives_kernel_per_body() {
        let mu = 3.986e14;
        let source = point_mass_source(mu);
        let source_pos = DVec3::ZERO;
        let controls = GravityControls {
            controls: vec![GravityControl::new_spherical(0_usize, GravityRole::Central)],
        };

        let body_a_pos = Position::<IntegrationFrame>::from_raw_si(DVec3::new(7e6, 0.0, 0.0));
        let body_a_vel = Velocity::<IntegrationFrame>::from_raw_si(DVec3::new(0.0, 7500.0, 0.0));
        let body_b_pos = Position::<IntegrationFrame>::from_raw_si(DVec3::new(0.0, 7e6, 1e5));
        let body_b_vel = Velocity::<IntegrationFrame>::from_raw_si(DVec3::new(-7500.0, 0.0, 50.0));
        let integ_origin_a = IntegOrigin::zero();
        let integ_origin_b = IntegOrigin {
            position: Position::<RootInertial>::from_raw_si(DVec3::new(1.5e11, 0.0, 0.0)),
            velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, 30_000.0, 0.0)),
        };

        let resolve_source = |_body: usize, _src: usize| {
            Some(ResolvedSource {
                source: &source,
                rotation: None,
                position: source_pos,
                delta_c20: 0.0,
                has_delta_coeffs: false,
            })
        };
        let resolve_rel_source = |_body: usize, _src: usize| {
            Some(ResolvedRelativisticSource {
                mu,
                position: source_pos,
                velocity: DVec3::ZERO,
            })
        };

        // Per-body baseline.
        let baseline_a = evaluate_body_gravity_typed(
            body_a_pos,
            body_a_vel,
            integ_origin_a,
            &controls,
            |s| resolve_source(0_usize, s),
            |s| resolve_rel_source(0_usize, s),
        );
        let baseline_b = evaluate_body_gravity_typed(
            body_b_pos,
            body_b_vel,
            integ_origin_b,
            &controls,
            |s| resolve_source(1_usize, s),
            |s| resolve_rel_source(1_usize, s),
        );

        // Driven by `run_gravity_stage`. Two bodies, two writer
        // closures. Each closure captures a `&mut` to its own output
        // slot — emulates the adapter pattern where the iterator's
        // `.map(...)` projects each row into a writer that closes over
        // that row's mutable storage handle.
        let mut out_a = GravityAccelerationTyped::<RootInertial>::default();
        let mut out_b = GravityAccelerationTyped::<RootInertial>::default();
        let entries = [
            (
                0_usize,
                GravityBodyInputs {
                    position: body_a_pos,
                    velocity: body_a_vel,
                    integ_origin: integ_origin_a,
                    controls: &controls,
                },
                Box::new(|r: GravityAccelerationTyped<RootInertial>| out_a = r)
                    as Box<dyn FnOnce(_)>,
            ),
            (
                1_usize,
                GravityBodyInputs {
                    position: body_b_pos,
                    velocity: body_b_vel,
                    integ_origin: integ_origin_b,
                    controls: &controls,
                },
                Box::new(|r: GravityAccelerationTyped<RootInertial>| out_b = r)
                    as Box<dyn FnOnce(_)>,
            ),
        ];
        run_gravity_stage(entries, resolve_source, resolve_rel_source);

        assert_eq!(out_a.grav_accel.raw_si(), baseline_a.grav_accel.raw_si());
        assert_eq!(out_a.grav_pot, baseline_a.grav_pot);
        assert_eq!(out_b.grav_accel.raw_si(), baseline_b.grav_accel.raw_si());
        assert_eq!(out_b.grav_pot, baseline_b.grav_pot);
    }

    /// `IntegOrigin::zero()` collapses the kernel to a plain
    /// `accumulate_gravity` + `accumulate_relativistic_corrections` —
    /// the root-integrated common case is bit-identical to the
    /// pre-kernel call sequence.
    #[test]
    fn evaluate_body_gravity_zero_origin_matches_plain_accumulate() {
        let mu = 3.986e14;
        let source = point_mass_source(mu);
        let source_pos = DVec3::ZERO;

        let body_pos = DVec3::new(7e6, 0.0, 0.0);
        let body_vel = DVec3::new(0.0, 7500.0, 0.0);
        let integ_origin = IntegOrigin::zero();
        let controls = GravityControls {
            controls: vec![GravityControl::new_spherical(0_usize, GravityRole::Central)],
        };
        let resolve_source = |_: usize| {
            Some(ResolvedSource {
                source: &source,
                rotation: None,
                position: source_pos,
                delta_c20: 0.0,
                has_delta_coeffs: false,
            })
        };
        let resolve_rel_source = |_: usize| {
            Some(ResolvedRelativisticSource {
                mu,
                position: source_pos,
                velocity: DVec3::ZERO,
            })
        };

        let kernel = evaluate_body_gravity_typed(
            Position::<IntegrationFrame>::from_raw_si(body_pos),
            Velocity::<IntegrationFrame>::from_raw_si(body_vel),
            integ_origin,
            &controls,
            resolve_source,
            resolve_rel_source,
        );
        let mut plain = accumulate_gravity(body_pos, &controls, DVec3::ZERO, resolve_source);
        plain.grav_accel +=
            accumulate_relativistic_corrections(body_pos, body_vel, &controls, resolve_rel_source);

        assert_eq!(kernel.grav_accel.raw_si(), plain.grav_accel);
        assert_eq!(kernel.grav_pot, plain.grav_pot);
        assert_eq!(kernel.grav_grad, plain.grav_grad);
    }
}
