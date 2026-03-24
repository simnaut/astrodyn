use bevy::prelude::*;
use glam::DVec3;

use crate::components::{
    DynamicsConfigC, GravityAccelerationC, GravityControlsC, GravitySourceC, MassPropertiesC,
    TotalForceC, TranslationalStateC,
};

/// Phase 2 scaffolding: collects gravity into TotalForce for future use by
/// non-gravity forces (aero, SRP, gravity torque). Currently `TotalForceC` is
/// not read by `integration_system`, which recomputes gravity directly at each
/// RK4 stage for 4th-order accuracy. In Phase 2, the integrator will sum
/// per-stage gravity with constant non-gravity forces from `TotalForceC`.
///
/// Torque is zeroed since gravity acts through the center of mass for a
/// point-mass model.
pub fn force_collection_system(
    mut query: Query<(&GravityAccelerationC, &MassPropertiesC, &mut TotalForceC)>,
) {
    for (grav, mass, mut total) in &mut query {
        total.force = grav.grav_accel * mass.mass;
        total.torque = DVec3::ZERO;
    }
}

/// Advances translational state via RK4 integration with gravity re-evaluation.
///
/// At each of the four RK4 stages, point-mass gravity is recomputed at the
/// intermediate position. This gives true 4th-order accuracy for Keplerian
/// orbits, unlike a simpler approach that holds acceleration constant over the
/// timestep.
///
/// The system reads `GravityControlsC` on each body to determine which gravity
/// sources affect it, then queries `GravitySourceC` on those source entities for
/// the gravitational parameter (mu).
///
/// **Phase 1 assumption**: gravity sources are at the origin of the integration
/// frame (body position is relative to the source center). In Phase 2, source
/// positions will be obtained from `TranslationalStateC` on the source entity,
/// not from `GlobalTransform` (which is f32 and insufficient for orbital
/// precision).
pub fn integration_system(
    mut bodies: Query<(
        &DynamicsConfigC,
        &mut TranslationalStateC,
        &GravityControlsC,
    )>,
    sources: Query<&GravitySourceC>,
    time: Res<Time<Fixed>>,
) {
    let dt = time.delta_secs_f64();
    if dt == 0.0 {
        return;
    }

    for (config, mut state, controls) in &mut bodies {
        if !config.translational_dynamics {
            continue;
        }

        let new_state = jeod_dynamics::rk4_translational_step(
            &state.0,
            |s| {
                let mut accel = DVec3::ZERO;
                for ctrl in &controls.0.controls {
                    if let Ok(source) = sources.get(ctrl.source_name) {
                        // TODO(Phase 3): obtain T_parent_this from planet-fixed frame entity;
                        // switch to gravitation_with_scratch to avoid per-stage allocation
                        accel +=
                            jeod_gravity::gravitation(&source.0, s.position, &glam::DMat3::IDENTITY).grav_accel;
                    }
                }
                accel
            },
            dt,
        );
        state.0 = new_state;
    }
}
