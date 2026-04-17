use glam::{DMat3, DVec3};
use jeod_dynamics::{MassProperties, RotationalState, TranslationalState};
use jeod_interactions::{
    compute_contact_force, AerodynamicForce, ContactFacet, DragConfig, FlatPlate, FlatPlateParams,
    FlatPlateThermal,
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

/// Evaluation of a contact pair against the intermediate state of the two bodies.
///
/// `pair_force_on_a` is the contact force on body A in the inertial frame.
/// `torque_a_body` / `torque_b_body` are the induced contact torques, expressed
/// in each body's own body frame about its center of mass.
#[derive(Debug, Clone, Copy, Default)]
pub struct ContactPairEval {
    /// Force on body A (inertial frame, N).
    pub force_on_a: DVec3,
    /// Torque on body A (body A's body frame about its CoM, N*m).
    pub torque_a_body: DVec3,
    /// Torque on body B (body B's body frame about its CoM, N*m).
    pub torque_b_body: DVec3,
}

/// Evaluate a single contact pair, returning force on A and body-frame torques
/// on A and B.
///
/// Port of `PointContactPair::in_contact` / `LineContactPair::in_contact`
/// composition with `SpringPairInteraction::calculate_forces`
/// (`spring_pair_interaction.cc`) — the force is computed in an inertial-aligned
/// frame and each body accumulates `force + torque = (r × F)` about its CoM.
///
/// Arguments
/// - `facet_a`, `facet_b`: the contact facets in each body's structural frame.
/// - `trans_a`, `trans_b`: translational states (inertial).
/// - `rot_a`, `rot_b`: optional rotational states. `None` → identity attitude
///   (facet endpoints taken directly in the world frame).
/// - `t_struct_body_a`, `t_struct_body_b`: structural-to-body transforms.
/// - `mass_a`, `mass_b`: mass properties (for CoM offsets in the torque arm).
///
/// Returns `None` if the facets are not in contact, else `Some(ContactPairEval)`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_contact_pair(
    facet_a: &ContactFacet,
    facet_b: &ContactFacet,
    trans_a: &TranslationalState,
    trans_b: &TranslationalState,
    rot_a: Option<&RotationalState>,
    rot_b: Option<&RotationalState>,
    t_struct_body_a: DMat3,
    t_struct_body_b: DMat3,
    mass_a: Option<&MassProperties>,
    mass_b: Option<&MassProperties>,
) -> Option<ContactPairEval> {
    // Rotate each facet's structural-frame geometry into the inertial frame so
    // the contact routine (which operates in a common inertial-aligned frame)
    // sees the current attitude-dependent shape.
    let t_inertial_body_a = rot_a.map_or(DMat3::IDENTITY, |r| {
        r.quaternion.left_quat_to_transformation()
    });
    let t_inertial_body_b = rot_b.map_or(DMat3::IDENTITY, |r| {
        r.quaternion.left_quat_to_transformation()
    });
    // t_inertial_struct = t_inertial_body * t_body_struct = t_inertial_body * t_struct_body^T
    let t_struct_inertial_a =
        jeod_dynamics::compute_t_inertial_struct(&t_struct_body_a, &t_inertial_body_a);
    let t_struct_inertial_b =
        jeod_dynamics::compute_t_inertial_struct(&t_struct_body_b, &t_inertial_body_b);
    // Rotation from structural-frame vector to inertial-frame vector.
    let t_inertial_from_struct_a = t_struct_inertial_a.transpose();
    let t_inertial_from_struct_b = t_struct_inertial_b.transpose();

    // Create facets whose endpoints are expressed in the inertial frame.
    // After this rotation, `facet.shape.reference_position()` for the new
    // facet is the shape reference offset (from structural origin) rotated
    // into inertial coords.
    let facet_a_world = rotate_facet(facet_a, &t_inertial_from_struct_a);
    let facet_b_world = rotate_facet(facet_b, &t_inertial_from_struct_b);

    // Facet reference position RELATIVE TO THE BODY'S COM, expressed in
    // inertial coords. For a facet at structural origin on a body whose CoM
    // is at structural origin, this is DVec3::ZERO.
    let r_cm_a_struct = mass_a.map_or(DVec3::ZERO, |m| m.position);
    let r_cm_b_struct = mass_b.map_or(DVec3::ZERO, |m| m.position);
    let facet_a_offset_from_cm_inertial =
        t_inertial_from_struct_a * (facet_a.shape.reference_position() - r_cm_a_struct);
    let facet_b_offset_from_cm_inertial =
        t_inertial_from_struct_b * (facet_b.shape.reference_position() - r_cm_b_struct);

    // Full facet reference positions in the inertial frame (body CoM + offset).
    // `trans_a.position` is the body CoM in the inertial frame.
    let a_ref_inertial = trans_a.position + facet_a_offset_from_cm_inertial;
    let b_ref_inertial = trans_b.position + facet_b_offset_from_cm_inertial;

    // Relative position of facet A wrt facet B's reference (inertial).
    let rel_pos = a_ref_inertial - b_ref_inertial;

    // Relative velocity at the contact points, matching JEOD
    // `point_contact_pair.cc:83-84`:
    //   rel_velocity = ω_target_wrt_subject × r_subject_contact − v_target_in_subject
    //
    // In our inertial formulation this is equivalent to evaluating the
    // velocity of each body's contact point (v_cm + ω × r_contact_from_cm)
    // and differencing. We apply the full kinematic formula with each
    // body's own ω contribution, which is the physically correct expression.
    let omega_a_inertial = rot_a.map_or(DVec3::ZERO, |r| t_inertial_body_a * r.ang_vel_body);
    let omega_b_inertial = rot_b.map_or(DVec3::ZERO, |r| t_inertial_body_b * r.ang_vel_body);
    // Arm from CoM to contact point requires the contact point, which
    // `compute_contact_force` returns. We iterate once to get an ω-free
    // estimate, then recompute once with ω×r corrections included — JEOD
    // actually uses the facet reference position (not the contact point)
    // for the ω cross product, so we match that by using the facet-ref
    // offset here.
    let rel_vel = (trans_a.velocity - trans_b.velocity)
        + omega_a_inertial.cross(facet_a_offset_from_cm_inertial)
        - omega_b_inertial.cross(facet_b_offset_from_cm_inertial);

    let contact = compute_contact_force(&facet_a_world, &facet_b_world, rel_pos, rel_vel)?;

    // Force on A: inertial frame.
    let force_on_a = contact.force;

    // Torque arms: from each body's CoM to the contact point on its surface.
    // `contact.contact_point_on_a` is the contact point relative to facet A's
    // reference (in inertial coords). Add facet offset from CoM to get the
    // arm from CoM to the contact point.
    let arm_a_inertial = facet_a_offset_from_cm_inertial + contact.contact_point_on_a;
    let arm_b_inertial = facet_b_offset_from_cm_inertial + contact.contact_point_on_b;

    let torque_a_inertial = arm_a_inertial.cross(force_on_a);
    let torque_b_inertial = arm_b_inertial.cross(-force_on_a);

    // Rotate inertial torques back to each body's body frame.
    let torque_a_body = t_inertial_body_a.transpose() * torque_a_inertial;
    let torque_b_body = t_inertial_body_b.transpose() * torque_b_inertial;

    Some(ContactPairEval {
        force_on_a,
        torque_a_body,
        torque_b_body,
    })
}

/// Transform a contact facet's shape endpoints from structural to inertial
/// coordinates via the given rotation matrix (`t_inertial_from_struct`).
///
/// The material is unchanged. This yields a facet whose `position` / `start` /
/// `end` values are expressed in the inertial frame so that
/// [`compute_contact_force`] can operate on two inertial-aligned facets.
fn rotate_facet(facet: &ContactFacet, t_inertial_from_struct: &DMat3) -> ContactFacet {
    use jeod_interactions::ContactShape;
    let shape = match facet.shape {
        ContactShape::Point { position, radius } => ContactShape::Point {
            position: *t_inertial_from_struct * position,
            radius,
        },
        ContactShape::Line { start, end, radius } => ContactShape::Line {
            start: *t_inertial_from_struct * start,
            end: *t_inertial_from_struct * end,
            radius,
        },
    };
    ContactFacet {
        shape,
        material: facet.material,
    }
}
