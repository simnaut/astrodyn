use glam::{DMat3, DVec3};
use jeod_dynamics::forces::{FrameDerivativesTyped, GravityAccelerationTyped, TotalForceTyped};
use jeod_dynamics::{
    ForceContributions, FrameDerivatives, MassProperties, RotationalState, TotalForce,
};
use jeod_interactions::{AerodynamicForce, RadiationForce};
use jeod_quantities::frame::{Inertial, Vehicle};

/// Collect all interaction forces and torques, resolve frame transforms,
/// and compute frame derivatives.
///
/// This encapsulates the orchestration logic from `force_collection_system`:
/// 1. Build `ForceContributions` from optional interaction results
/// 2. Compute inertial-to-body rotation from quaternion
/// 3. Delegate to `jeod_dynamics::collect_forces()` for frame-aware aggregation
/// 4. Compute translational and rotational accelerations
///
/// # Frame pipeline (matching JEOD `dyn_body_collect.cc`)
///
/// **Force** (collected in structural frame, rotated to inertial):
///   `T_inertial_struct^T * (aero_force_struct + ...)` + SRP (already inertial)
///
/// **Torque** (collected in body frame):
///   `T_struct_body * aero_torque_struct` + gravity_torque (already body) +
///   `T_struct_body * srp_torque_struct`
///
/// Gravity is **excluded** from `TotalForce.force` — it is added by the
/// integration function for correct RK4 treatment.
///
/// # Arguments
/// - `aero`: aerodynamic force/torque in structural frame (`None` = zero)
/// - `srp`: radiation force (inertial) and torque (structural) (`None` = zero)
/// - `gravity_torque`: gravity gradient torque in body frame (`None` = zero)
/// - `rot_state`: rotational state for frame transform (`None` = identity)
/// - `t_struct_body`: structural-to-body rotation (identity if absent)
/// - `mass`: mass properties for acceleration computation (`None` = gravity-only)
/// - `gravity_accel`: pre-computed gravitational acceleration
///
/// # Panics
/// Non-zero aerodynamic force without rotational state (JEOD_INV: IN.15).
// JEOD_INV: DB.28 — forces collected in structural frame, rotated to inertial at root
// JEOD_INV: DB.29 — torques collected in structural frame, rotated to body at root
// JEOD_INV: IN.15 — aero drag requires body orientation (T_inertial_struct)
pub fn collect_and_resolve_forces(
    aero: Option<&AerodynamicForce>,
    srp: Option<&RadiationForce>,
    gravity_torque: Option<DVec3>,
    rot_state: Option<&RotationalState>,
    t_struct_body: DMat3,
    mass: Option<&MassProperties>,
    gravity_accel: DVec3,
) -> (TotalForce, FrameDerivatives) {
    // JEOD_INV: IN.15 — aero drag requires body orientation
    if let Some(aero) = aero {
        if aero.force != DVec3::ZERO && rot_state.is_none() {
            panic!(
                "Non-zero aerodynamic force but no RotationalState. \
                 In JEOD, T_inertial_struct is a mandatory parameter of aero_drag(). \
                 Provide RotationalState for any body with aerodynamic forces."
            );
        }
    }

    // Build ForceContributions from optional interaction results
    let mut contributions = ForceContributions::default();
    if let Some(aero) = aero {
        contributions.aero_force_struct = aero.force;
        contributions.aero_torque_struct = aero.torque;
    }
    if let Some(srp) = srp {
        contributions.srp_force_inertial = srp.force;
        contributions.srp_torque_struct = srp.torque;
    }
    if let Some(gt) = gravity_torque {
        contributions.gravity_torque_body = gt;
    }

    // Rotation matrix from attitude quaternion (identity if no rotational state)
    let t_inertial_body = rot_state.map_or(DMat3::IDENTITY, |r| {
        r.quaternion.left_quat_to_transformation()
    });

    // Delegate frame-aware force/torque collection to jeod_dynamics (DB.28, DB.29)
    let collected = jeod_dynamics::collect_forces(&contributions, &t_struct_body, &t_inertial_body);

    // Compute frame derivatives (FD.01, FD.02)
    let derivs = if let (Some(rot), Some(m)) = (rot_state, mass) {
        jeod_dynamics::compute_frame_derivatives(
            &collected,
            m.inverse_mass,
            gravity_accel,
            &m.inertia,
            &m.inverse_inertia,
            rot.ang_vel_body,
        )
    } else if let Some(m) = mass {
        jeod_dynamics::compute_translational_derivatives(
            collected.force,
            m.inverse_mass,
            gravity_accel,
        )
    } else {
        FrameDerivatives {
            trans_accel: gravity_accel,
            rot_accel: DVec3::ZERO,
        }
    };

    (collected, derivs)
}

/// Typed sibling of [`collect_and_resolve_forces`].
///
/// Identical kernel — entry boundary takes
/// [`GravityAccelerationTyped<Inertial>`] for the gravity
/// acceleration, exit boundary returns
/// [`TotalForceTyped<V, Inertial>`] / [`FrameDerivativesTyped<Inertial,
/// V>`]. The aerodynamic / radiation / gravity-torque / rotation-state /
/// mass inputs remain untyped because Phase 3 left those struct
/// surfaces untyped at their `jeod_dynamics` / `jeod_interactions`
/// home crates.
#[allow(clippy::too_many_arguments)]
pub fn collect_and_resolve_forces_typed<V: Vehicle>(
    aero: Option<&AerodynamicForce>,
    srp: Option<&RadiationForce>,
    gravity_torque: Option<DVec3>,
    rot_state: Option<&RotationalState>,
    t_struct_body: DMat3,
    mass: Option<&MassProperties>,
    gravity_accel: GravityAccelerationTyped<Inertial>,
) -> (
    TotalForceTyped<V, Inertial>,
    FrameDerivativesTyped<Inertial, V>,
) {
    let (force, derivs) = collect_and_resolve_forces(
        aero,
        srp,
        gravity_torque,
        rot_state,
        t_struct_body,
        mass,
        gravity_accel.grav_accel.raw_si(),
    );
    (
        TotalForceTyped::<V, Inertial>::from_untyped_unchecked(&force),
        FrameDerivativesTyped::<Inertial, V>::from_untyped_unchecked(&derivs),
    )
}
