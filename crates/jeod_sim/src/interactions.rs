use glam::{DMat3, DVec3};
use jeod_dynamics::RotationalState;
use jeod_interactions::{
    AerodynamicForce, DragConfig, FlatPlate, FlatPlateParams, FlatPlateThermal, STEFAN_BOLTZMANN,
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
    /// Integrate plate temperatures via RK4 with overshoot clamping.
    ///
    /// Port of JEOD `ThermalIntegrableObject::integrate()` (thermal_integrable_object.cc:98-124).
    /// Uses RK4 (matching the orbital state integrator order) with overshoot
    /// detection: if the integrated temperature crosses the radiative equilibrium
    /// value, it is clamped to equilibrium.
    ///
    /// `temp_dots_k1` is the per-plate temperature derivative from the current
    /// step's `compute_flat_plate_srp_thermal` call. The absorbed power is
    /// recovered from k1 and held constant over the RK4 sub-steps (solar flux
    /// changes negligibly over one timestep).
    ///
    /// Called after `compute_flat_plate_srp_thermal` returns `temp_dots`.
    pub fn integrate_temperatures(&mut self, temp_dots_k1: &[f64], dt: f64) {
        for (i, (plate, _params, thermal)) in self.plates.iter().enumerate() {
            let old_temp = self.temperatures[i];
            let old_t_pow4 = self.t_pow4_cached[i];

            let rad_constant = plate.area * thermal.emissivity * STEFAN_BOLTZMANN;
            let heat_cap = thermal.heat_capacity_per_area * plate.area;
            if heat_cap <= 0.0 {
                continue;
            }

            // Recover power_absorb from k1 (constant over the RK4 step).
            // temp_dot = (power_absorb - power_emit) / heat_capacity
            // power_absorb = temp_dot * heat_capacity + rad_constant * T^4
            let power_absorb = temp_dots_k1[i] * heat_cap + rad_constant * old_t_pow4;

            // Temperature derivative at a given temperature (power_absorb is constant).
            let tdot = |temp: f64| -> f64 {
                let t4 = temp * temp * temp * temp;
                (power_absorb - rad_constant * t4) / heat_cap
            };

            // RK4 stages
            let k1 = temp_dots_k1[i];
            let k2 = tdot(old_temp + k1 * dt * 0.5);
            let k3 = tdot(old_temp + k2 * dt * 0.5);
            let k4 = tdot(old_temp + k3 * dt);

            let mut new_temp = old_temp + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (dt / 6.0);
            new_temp = new_temp.max(0.0);

            let new_t_pow4 = new_temp * new_temp * new_temp * new_temp;

            // JEOD overshoot clamping (thermal_integrable_object.cc:106-121).
            // If temp_dot and (T_eq^4 - T^4) have opposite signs, the temperature
            // crossed the radiative equilibrium asymptote — clamp to equilibrium.
            if rad_constant > 0.0 {
                let t_eq_pow4 = power_absorb / rad_constant;
                if k1 * (t_eq_pow4 - new_t_pow4) < 0.0 {
                    self.t_pow4_cached[i] = t_eq_pow4.max(0.0);
                    self.temperatures[i] = self.t_pow4_cached[i].sqrt().sqrt();
                    continue;
                }
            }

            self.temperatures[i] = new_temp;
            self.t_pow4_cached[i] = new_t_pow4;
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

/// Compute cannonball SRP force using JEOD's `RadiationDefaultSurface` formula.
///
/// Force = (flux/c) * cx_area * [1 + albedo*diffuse*(4/9)] * flux_hat * illum_factor.
///
/// Returns the force vector in the inertial frame (N). Torque is always zero
/// for the cannonball model (force acts through center of mass).
///
/// # Arguments
/// - `body_pos`: vehicle position in inertial frame (m)
/// - `sun_pos`: Sun position in inertial frame (m)
/// - `cx_area`: cross-section area * Cr (m²)
/// - `albedo`: surface albedo (0–1)
/// - `diffuse`: diffuse reflection fraction (0–1)
/// - `illum_factor`: illumination factor from shadow computation (0–1)
pub fn compute_cannonball_srp(
    body_pos: DVec3,
    sun_pos: DVec3,
    cx_area: f64,
    albedo: f64,
    diffuse: f64,
    illum_factor: f64,
) -> DVec3 {
    let sun_to_vehicle = body_pos - sun_pos;
    let distance = sun_to_vehicle.length();
    if distance < 1.0 {
        return DVec3::ZERO;
    }
    let flux_hat = sun_to_vehicle / distance;
    let flux_mag = crate::solar_flux_at_distance(distance);
    let coeff = 1.0 + albedo * diffuse * (4.0 / 9.0);
    let force_mag = cx_area * flux_mag / crate::SPEED_OF_LIGHT * coeff * illum_factor;
    flux_hat * force_mag
}
