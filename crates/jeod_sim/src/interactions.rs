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
    /// Integrate plate temperatures via Forward Euler with overshoot clamping.
    ///
    /// Port of JEOD `ThermalIntegrableObject::integrate()` (thermal_integrable_object.cc:98-124).
    /// JEOD's standard `radiation.sm` schedules `compute_temp_dot()` as a
    /// "scheduled" job (once per step, not per integrator stage), so the
    /// derivative is constant across all RK4 stages. This produces
    /// `new_temp = old_temp + temp_dot * dt` — effectively Forward Euler.
    /// We match this behavior for parity with JEOD's standard SIM_3_ORBIT.
    ///
    /// Overshoot detection: if the integrated temperature crosses the radiative
    /// equilibrium value, it is clamped to equilibrium.
    ///
    /// `temp_dots_k1` is the per-plate temperature derivative from the current
    /// step's `compute_flat_plate_srp_thermal` call.
    ///
    /// Called after `compute_flat_plate_srp_thermal` returns `temp_dots`.
    /// For true RK4 thermal integration (derivative-class SRP), use
    /// [`finalize_rk4_temperatures`](Self::finalize_rk4_temperatures) via
    /// [`crate::integration::integrate_body_coupled`] instead.
    pub fn integrate_temperatures(&mut self, temp_dots_k1: &[f64], dt: f64) {
        for (i, (plate, _params, thermal)) in self.plates.iter().enumerate() {
            let (new_temp, new_t_pow4) = jeod_interactions::integrate_plate_temperature_euler(
                self.temperatures[i],
                self.t_pow4_cached[i],
                temp_dots_k1[i],
                plate.area,
                thermal.emissivity,
                thermal.heat_capacity_per_area,
                dt,
            );
            self.temperatures[i] = new_temp;
            self.t_pow4_cached[i] = new_t_pow4;
        }
    }

    /// Finalize RK4 thermal integration from four stage derivatives.
    ///
    /// Combines k1–k4 temperature derivatives via the standard RK4 formula
    /// and applies JEOD overshoot clamping (thermal_integrable_object.cc:106-121).
    /// Called by [`crate::integration::integrate_body_coupled`] after all four
    /// stages have been evaluated.
    ///
    /// This is the coupled-integration counterpart of [`Self::integrate_temperatures`]:
    /// instead of deriving k2–k4 internally from k1's `power_absorb`, the four
    /// derivatives were computed at intermediate orbital positions by the stage
    /// closure.
    #[allow(clippy::too_many_arguments)]
    pub fn finalize_rk4_temperatures(
        &mut self,
        temps0: &[f64],
        t_pow4_0: &[f64],
        k1_tdots: &[f64],
        k2_tdots: &[f64],
        k3_tdots: &[f64],
        k4_tdots: &[f64],
        dt: f64,
        n_plates: usize,
    ) {
        for i in 0..n_plates {
            let (plate, _, thermal) = &self.plates[i];
            let (new_temp, new_t_pow4) = jeod_interactions::integrate_plate_temperature_rk4(
                temps0[i],
                t_pow4_0[i],
                k1_tdots[i],
                k2_tdots[i],
                k3_tdots[i],
                k4_tdots[i],
                plate.area,
                thermal.emissivity,
                thermal.heat_capacity_per_area,
                dt,
            );
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
/// Delegates to [`jeod_interactions::compute_cannonball_srp`].
pub fn compute_cannonball_srp(
    body_pos: DVec3,
    sun_pos: DVec3,
    cx_area: f64,
    albedo: f64,
    diffuse: f64,
    illum_factor: f64,
) -> DVec3 {
    jeod_interactions::compute_cannonball_srp(
        body_pos,
        sun_pos,
        cx_area,
        albedo,
        diffuse,
        illum_factor,
    )
}
