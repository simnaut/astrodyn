use glam::{DMat3, DVec3};
use jeod_dynamics::GravityAcceleration;
use jeod_gravity::{GravityControls, GravitySource};

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

        // Pre-check: non-spherical gravity requires planet-fixed rotation
        if ctrl.is_nonspherical() && resolved.rotation.is_none() {
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
            let frame_accel = ctrl.evaluate_accel_only(
                resolved.source,
                frame_pos_relative_to_source,
                resolved.rotation,
                resolved.delta_c20,
                resolved.has_delta_coeffs,
            );
            total.grav_accel += result.grav_accel - frame_accel.grav_accel;
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
