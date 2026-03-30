use glam::{DMat3, DVec3};
use jeod_dynamics::RotationalState;
use jeod_interactions::{
    AerodynamicForce, DragConfig, FlatPlate, FlatPlateParams, FlatPlateThermal,
};

/// Flat-plate SRP configuration with mutable thermal state.
///
/// Bundles plate geometry/optical/thermal properties with per-plate temperature
/// state. Used by both the `Simulation` runner and Bevy adapter so that
/// temperature integration logic is shared.
#[derive(Debug, Clone)]
pub struct FlatPlateState {
    /// Per-plate geometry, optical, and thermal properties.
    pub plates: Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)>,
    /// Per-plate temperatures (K). Same length as `plates`.
    pub temperatures: Vec<f64>,
    /// Cached T^4 per plate from previous step (for thermal emission).
    /// Same length as `plates`.
    pub t_pow4_cached: Vec<f64>,
}

impl FlatPlateState {
    /// Integrate plate temperatures (forward Euler) and update the T^4 cache.
    ///
    /// Called after `compute_flat_plate_srp_thermal` returns `temp_dots`.
    /// Clamps temperatures to non-negative.
    pub fn integrate_temperatures(&mut self, temp_dots: &[f64], dt: f64) {
        for (i, temp) in self.temperatures.iter_mut().enumerate() {
            *temp += temp_dots[i] * dt;
            if *temp < 0.0 {
                *temp = 0.0;
            }
        }
        for (i, cached) in self.t_pow4_cached.iter_mut().enumerate() {
            *cached = self.temperatures[i].powi(4);
        }
    }
}

/// Compute aerodynamic drag for a body, handling the frame transform.
///
/// Computes `T_inertial_struct` from the body's quaternion and structural
/// transform, then delegates to `jeod_interactions::compute_ballistic_drag`.
///
/// # Arguments
/// - `drag_config`: Cd and area
/// - `atmos`: atmospheric state (density, wind)
/// - `velocity`: body velocity in inertial frame
/// - `rot`: rotational state (for frame transform). `None` = identity.
/// - `t_struct_body`: structural-to-body rotation. `DMat3::IDENTITY` when structure = body.
pub fn compute_drag(
    drag_config: &DragConfig,
    atmos: &jeod_atmosphere::AtmosphereState,
    velocity: DVec3,
    rot: Option<&RotationalState>,
    t_struct_body: DMat3,
) -> AerodynamicForce {
    let t_inertial_body = rot.map_or(DMat3::IDENTITY, |r| {
        r.quaternion.left_quat_to_transformation()
    });
    let t_inertial_struct =
        jeod_dynamics::compute_t_inertial_struct(&t_struct_body, &t_inertial_body);

    jeod_interactions::compute_ballistic_drag(drag_config, atmos, velocity, &t_inertial_struct)
}

/// Compute gravity gradient torque for a body, handling the quaternion-to-matrix conversion.
///
/// Converts the body's quaternion to a rotation matrix, then delegates to
/// `jeod_interactions::compute_gravity_torque`.
///
/// # Arguments
/// - `grav_grad`: gravity gradient tensor from `GravityAcceleration`
/// - `rot`: rotational state (for body attitude matrix)
/// - `inertia`: body inertia tensor
pub fn compute_gravity_torque(grav_grad: &DMat3, rot: &RotationalState, inertia: &DMat3) -> DVec3 {
    let t_parent_this = rot.quaternion.left_quat_to_transformation();
    jeod_interactions::compute_gravity_torque(grav_grad, &t_parent_this, inertia)
}
