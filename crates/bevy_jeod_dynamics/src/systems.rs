use bevy::prelude::*;
use glam::DVec3;
use jeod_dynamics::SixDofState;

use crate::components::{
    DynamicsConfigC, GravityAccelerationC, GravityControlsC, GravitySourceC, MassPropertiesC,
    RotationalStateC, TotalForceC, TranslationalStateC,
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

/// Advances translational (and optionally rotational) state via RK4 integration
/// with gravity re-evaluation.
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
/// When `DynamicsConfig::rotational_dynamics` is enabled and the entity has a
/// `RotationalStateC` component, the system uses `rk4_sixdof_step` to integrate
/// all 13 state variables (position[3], velocity[3], quaternion[4], angular
/// velocity[3]) simultaneously. Otherwise it falls back to the 3-DOF
/// `rk4_translational_step`.
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
        Option<&mut RotationalStateC>,
        &MassPropertiesC,
        &GravityControlsC,
    )>,
    sources: Query<&GravitySourceC>,
    time: Res<Time<Fixed>>,
) {
    let dt = time.delta_secs_f64();
    if dt == 0.0 {
        return;
    }

    for (config, mut state, mut rot_state, mass, controls) in &mut bodies {
        if !config.translational_dynamics {
            continue;
        }

        // Closure: compute gravitational acceleration at a given position.
        // Used by both 3-DOF and 6-DOF paths so gravity is re-evaluated at
        // each RK4 stage for 4th-order accuracy.
        let compute_grav_accel = |position: DVec3| -> DVec3 {
            let mut accel = DVec3::ZERO;
            for ctrl in &controls.0.controls {
                if let Ok(source) = sources.get(ctrl.source_name) {
                    // TODO(Phase 4): obtain T_parent_this from planet-fixed frame entity;
                    // switch to gravitation_with_scratch to avoid per-stage allocation
                    accel += jeod_gravity::gravitation(
                        &source.0, position, &glam::DMat3::IDENTITY,
                        ctrl.degree, ctrl.order, ctrl.perturbing_only,
                        false, None, None,
                    ).grav_accel;
                }
            }
            accel
        };

        // 6-DOF path: rotational dynamics enabled AND entity has RotationalStateC
        if config.rotational_dynamics {
            if let Some(ref mut rot) = rot_state {
                let six_state = SixDofState {
                    trans: state.0,
                    rot: rot.0,
                };
                let new_state = jeod_dynamics::rk4_sixdof_step(
                    &six_state,
                    |s| compute_grav_accel(s.trans.position),
                    |_s| DVec3::ZERO, // No external torque in Phase 3 (gravity torque is Phase 4)
                    &mass.0,
                    dt,
                );
                state.0 = new_state.trans;
                rot.0 = new_state.rot;
                continue;
            }
        }

        // 3-DOF path: translational only
        let new_state = jeod_dynamics::rk4_translational_step(
            &state.0,
            |s| compute_grav_accel(s.position),
            dt,
        );
        state.0 = new_state;
    }
}
