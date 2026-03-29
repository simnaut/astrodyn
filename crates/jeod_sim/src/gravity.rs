use glam::{DMat3, DVec3};
use jeod_dynamics::GravityAcceleration;
use jeod_gravity::{GravityControls, GravitySource};

/// Accumulate gravity from all sources for a single body.
///
/// Iterates over the body's gravity controls, looks up each source via
/// `source_lookup`, and accumulates acceleration, gradient, and potential.
///
/// # Type parameter
/// - `S`: source identifier type. In Bevy: `Entity`. In `Simulation`: `usize`.
///
/// The `source_lookup` closure resolves it to the source's `GravitySource` and
/// optional planet-fixed rotation matrix (for spherical harmonics).
///
/// # Panics
/// - Source not found (JEOD_INV: GV.12)
/// - Non-spherical gravity without planet-fixed rotation
// JEOD_INV: GV.12 — gravity source must exist for control
pub fn accumulate_gravity<'a, S: Copy + std::fmt::Debug>(
    position: DVec3,
    controls: &GravityControls<S>,
    source_lookup: impl Fn(S) -> Option<(&'a GravitySource, Option<&'a DMat3>)>,
) -> GravityAcceleration {
    let mut total = GravityAcceleration::default();

    for ctrl in &controls.controls {
        // JEOD_INV: GV.12 — gravity source must exist for control.
        // JEOD: GravityControls::initialize_control() calls MessageHandler::error()
        // (non-fatal, severity 0) when find_grav_source() returns nullptr. We
        // escalate to a panic because silently omitting a gravity source would
        // produce incorrect physics.
        let (source, rot) = source_lookup(ctrl.source_name).unwrap_or_else(|| {
            panic!(
                "GravityControl references source {:?} which does not exist. \
                 JEOD logs a non-fatal error and skips; we panic to prevent \
                 silently wrong physics.",
                ctrl.source_name
            );
        });

        // Pre-check: non-spherical gravity requires planet-fixed rotation
        if ctrl.is_nonspherical() && rot.is_none() {
            panic!(
                "Non-spherical GravityControl (degree={}, order={}) references \
                 source {:?} which has no planet-fixed rotation matrix.",
                ctrl.degree, ctrl.order, ctrl.source_name
            );
        }

        let result = ctrl.evaluate(source, position, rot);
        total.grav_accel += result.grav_accel;
        if ctrl.gradient {
            total.grav_grad += result.grav_grad;
        }
        total.grav_pot += result.grav_pot;
    }

    total
}
