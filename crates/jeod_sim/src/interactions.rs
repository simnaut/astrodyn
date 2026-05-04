//! Interaction stage: the per-body force / torque computations between
//! the gravity and force-collection stages — drag, flat-plate and
//! cannonball SRP, gravity-gradient torque, and contact pairs.

use glam::{DMat3, DVec3};
use jeod_dynamics::{MassProperties, RotationalState, TranslationalState};
use jeod_interactions::{
    compute_contact_force_from_geometry, compute_contact_geometry, AerodynamicForce,
    AerodynamicForceTyped, ContactFacet, DragConfig, DragConfigTyped, FlatPlate, FlatPlateParams,
    FlatPlateThermal,
};
use jeod_quantities::aliases::{Force, InertiaTensor, Position, Torque, Velocity};
use jeod_quantities::frame::{BodyFrame, RootInertial, Vehicle};
use uom::si::f64::{Area, Ratio};

use crate::integrable::IntegrableObject;

/// Step-constant SRP inputs cached at step start for the derivative-class
/// thermal paths.
///
/// Populated by Stage 5 (interactions) when
/// [`ThermalIntegrationOrder::DerivativeFirstOrder`] or
/// [`ThermalIntegrationOrder::DerivativeRk4`] is selected; consumed by
/// Stage 8 (integration)'s per-stage closure inside
/// [`crate::integrate_body_coupled`]. `None` for
/// [`ThermalIntegrationOrder::Scheduled`] — the Stage 5 Euler fast path
/// remains active.
///
/// Shared by the standalone `Simulation` runner and the Bevy adapter so
/// both drive the coupled integrator with identical step-start data.
///
/// # RF.10 structural guard
///
/// The original buggy expression that slipped past the type system on
/// PR #258 was
///
/// ```text
/// let sun_to_vehicle = stage_trans.position - srp_inputs.sun_position;
/// ```
///
/// where `stage_trans.position` is the integrator's `DVec3` in the
/// body's integration frame and `srp_inputs.sun_position` was *also*
/// `DVec3`. After typing `sun_position` as
/// [`Position<RootInertial>`](Position), the same expression no longer
/// typechecks. A future contributor cannot reintroduce the bug.
///
/// **`DVec3 - Position<RootInertial>` does not compile:**
///
/// ```compile_fail
/// use jeod_sim::FlatPlateStageInputs;
/// use glam::DVec3;
///
/// let inputs = FlatPlateStageInputs::default();
/// let stage_pos: DVec3 = DVec3::new(7.0e6, 0.0, 0.0); // raw integration-frame
/// // No `Sub<Position<RootInertial>>` impl for `DVec3`:
/// let _bug = stage_pos - inputs.sun_position;
/// ```
///
/// **Building the field from a raw `DVec3` does not compile either** —
/// the only way to populate it is with a typed `Position<RootInertial>`
/// (e.g. `Position::<RootInertial>::from_raw_si(...)` at a documented
/// boundary, or by passing through a structural shift like
/// [`IntegOrigin::shift_position`](jeod_quantities::IntegOrigin::shift_position)):
///
/// ```compile_fail
/// use jeod_sim::FlatPlateStageInputs;
/// use glam::DVec3;
///
/// let raw_sun: DVec3 = DVec3::new(1.5e11, 0.0, 0.0);
/// // Field is typed; raw DVec3 doesn't typecheck:
/// let _bug = FlatPlateStageInputs {
///     sun_position: raw_sun,
///     illum_factor: 1.0,
///     center_grav: DVec3::ZERO,
/// };
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatPlateStageInputs {
    /// Sun position in the simulation's root inertial frame, captured at
    /// step start. Typed so the closure inside `integrate_body_coupled`
    /// cannot subtract it from a raw integration-frame `DVec3` — the
    /// `Position<RootInertial> - DVec3` mismatch refuses to compile and
    /// forces the caller to lift the stage's `TranslationalState.position`
    /// through a typed shift first (RF.10).
    ///
    /// Matches JEOD's derivative-class `RadiationSource::calculate_flux`
    /// which reads the intermediate vehicle structure frame at every
    /// call (see
    /// `models/interactions/radiation_pressure/src/radiation_source.cc`).
    pub sun_position: Position<RootInertial>,
    /// Shadow illumination factor from step-start shadow evaluation
    /// (constant across RK4 stages — matches JEOD scheduled-class
    /// shadow in SIM_3_ORBIT; third-body frames are not propagated
    /// between RK4 stages within JEOD either).
    pub illum_factor: f64,
    /// Center of gravity in structural frame.
    pub center_grav: DVec3,
}

/// Which integrator drives plate temperatures, and at what scheduling
/// class.
///
/// JEOD's `rad_pressure.update()` can be wired in three ways (observed in
/// `models/interactions/radiation_pressure/verif/S_modules/*.sm`):
///
/// * **Scheduled** (`radiation.sm` — SIM_3_ORBIT): `rad_pressure.update` is
///   a `("scheduled")` job that runs once per DYNAMICS interval, outside
///   the integration loop. Temperature is advanced by first-order Euler
///   over the full step, and the SRP force fed to the orbital integrator
///   is step-constant.
/// * **DerivativeFirstOrder** (`radiation_1st_order.sm` — SIM_3_ORBIT_1st_ORDER):
///   `rad_pressure.update` is `("derivative")` and runs at every RK4
///   derivative evaluation, so the SRP force varies with intermediate
///   orbital position. The thermal integrator is ER7_Utils first-order —
///   i.e., temperature is advanced using the step-start derivative only.
/// * **DerivativeRk4** (no JEOD Tier 3 reference): `rad_pressure.update`
///   runs per stage AND temperature integrates via full RK4 (all four
///   derivatives). Not required to match any current JEOD reference CSV;
///   exposed as an API option for mission-level missions that want the
///   more accurate coupling going forward.
///
/// Default is [`ThermalIntegrationOrder::Scheduled`] — backward-compatible
/// with the pre-Commit-3 runner behavior and matches the SIM_3_ORBIT
/// reference without any opt-in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThermalIntegrationOrder {
    /// SRP force + Euler T update computed once per step (step-constant
    /// SRP across RK4 stages). Matches JEOD's `(DYNAMICS, "scheduled")`
    /// radiation module in SIM_3_ORBIT.
    #[default]
    Scheduled,
    /// SRP force recomputed per RK4 stage (varies with intermediate
    /// orbital state); temperature advanced at step end using only the
    /// stage-1 derivative. Matches JEOD's derivative-class + first-order
    /// ER7_Utils integrator in SIM_3_ORBIT_1st_ORDER.
    DerivativeFirstOrder,
    /// SRP force and temperature both recomputed per RK4 stage; full RK4
    /// thermal integration inside the orbital RK4 loop. No current JEOD
    /// reference uses this pattern — provided for mission code that wants
    /// the tighter coupling.
    DerivativeRk4,
}

/// Flat-plate SRP configuration with mutable thermal state.
///
/// Bundles plate geometry/optical/thermal properties with per-plate temperature
/// state. Used by both the `Simulation` runner and Bevy adapter so that
/// temperature integration logic is shared.
///
/// Implements [`IntegrableObject`] so temperatures can integrate through the
/// RK4 stage loop alongside orbital state (see
/// [`crate::integrate_body_coupled`]). Construct with `..Default::default()`
/// so the trait's snapshot scratch initializes empty:
///
/// // reason: pattern fragment — `plates` and `n` are placeholders for mission-specific values; the snippet documents the construction shape.
/// ```ignore
/// FlatPlateState {
///     plates,
///     temperatures: vec![300.0; n],
///     t_pow4_cached: vec![300.0_f64.powi(4); n],
///     ..Default::default()
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct FlatPlateState {
    /// Per-plate geometry, optical, and thermal properties.
    pub plates: Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)>,
    /// Per-plate temperatures (K). Same length as `plates`.
    pub temperatures: Vec<f64>,
    /// Cached T^4 per plate from previous step (for thermal emission).
    /// Same length as `plates`.
    pub t_pow4_cached: Vec<f64>,
    /// Which integrator drives the temperature ODE.
    ///
    /// Defaults to [`ThermalIntegrationOrder::Scheduled`], which matches
    /// JEOD's scheduled-class `SIM_3_ORBIT` behavior (SRP force + Euler T
    /// computed once per step).
    ///
    /// Use [`ThermalIntegrationOrder::DerivativeFirstOrder`] to match
    /// JEOD's derivative-class `SIM_3_ORBIT_1st_ORDER`: SRP force is
    /// recomputed per RK4 stage (varying with intermediate orbital
    /// attitude) while T is integrated via ER7_Utils first-order from
    /// the stage-1 derivative.
    ///
    /// Use [`ThermalIntegrationOrder::DerivativeRk4`] for tighter
    /// per-stage SRP and thermal coupling; no current JEOD Tier 3
    /// reference uses that mode.
    ///
    /// Both derivative-class modes run through
    /// [`crate::integrate_body_coupled`].
    pub integration_order: ThermalIntegrationOrder,
    /// Step-start SRP inputs populated by the Stage-5 interactions code
    /// when `integration_order` is derivative-class. `None` in
    /// `Scheduled` mode. Consumed by the Stage-8 integration closure;
    /// cleared at the start of the next step's Stage 5.
    pub stage_inputs: Option<FlatPlateStageInputs>,
    /// RK4 kernel scratch: step-start temperature and T^4 snapshots.
    ///
    /// Populated by [`IntegrableObject::snapshot`] and consumed by
    /// [`IntegrableObject::advance_intermediate`] /
    /// [`IntegrableObject::finalize_rk4`]. Inner fields are crate-private
    /// so external callers cannot accidentally corrupt mid-step RK4
    /// state. The field itself is `pub` so struct-update syntax
    /// (`FlatPlateState { plates, ..Default::default() }`) keeps working
    /// at literal sites — those callers receive a default-constructed
    /// `FlatPlateScratch` they can hold but not mutate.
    pub scratch: FlatPlateScratch,
}

/// Step-start RK4 snapshot vectors for [`FlatPlateState`].
///
/// Inner storage is private. Construct via `Default::default()`; mutate
/// only through `IntegrableObject::snapshot` (called by the RK4 kernel).
#[derive(Debug, Clone, Default)]
pub struct FlatPlateScratch {
    temps_snapshot: Vec<f64>,
    t_pow4_snapshot: Vec<f64>,
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
    /// Reads the step-start snapshots from `scratch.temps_snapshot` /
    /// `scratch.t_pow4_snapshot`, which [`IntegrableObject::snapshot`]
    /// populated at the start of the step.
    ///
    /// This is the coupled-integration counterpart of
    /// [`Self::integrate_temperatures`]: instead of deriving k2–k4 internally
    /// from k1's `power_absorb`, the four derivatives were computed at
    /// intermediate orbital positions by the stage closure.
    pub fn finalize_rk4_temperatures(
        &mut self,
        k1_tdots: &[f64],
        k2_tdots: &[f64],
        k3_tdots: &[f64],
        k4_tdots: &[f64],
        dt: f64,
    ) {
        let n = self.temperatures.len();
        for i in 0..n {
            let area = self.plates[i].0.area;
            let emissivity = self.plates[i].2.emissivity;
            let heat_capacity_per_area = self.plates[i].2.heat_capacity_per_area;
            let (new_temp, new_t_pow4) = jeod_interactions::integrate_plate_temperature_rk4(
                self.scratch.temps_snapshot[i],
                self.scratch.t_pow4_snapshot[i],
                k1_tdots[i],
                k2_tdots[i],
                k3_tdots[i],
                k4_tdots[i],
                area,
                emissivity,
                heat_capacity_per_area,
                dt,
            );
            self.temperatures[i] = new_temp;
            self.t_pow4_cached[i] = new_t_pow4;
        }
    }
}

// JEOD_INV: IN.32 — IntegrableObject per-stage snapshot/advance/finalize protocol.
// Port of JEOD er7_utils::IntegrableObject (trick_source/er7_utils/integration/
// core/include/integrable_object.hh) as consumed by DynamicsIntegrationGroup.
// FlatPlateState's temperature vector is the only sub-state that uses this
// protocol today; orbital translational/rotational state is integrated
// directly by the enclosing RK4 kernel.
impl IntegrableObject for FlatPlateState {
    fn n_states(&self) -> usize {
        self.temperatures.len()
    }

    fn snapshot(&mut self) {
        self.scratch.temps_snapshot.clear();
        self.scratch
            .temps_snapshot
            .extend_from_slice(&self.temperatures);
        self.scratch.t_pow4_snapshot.clear();
        self.scratch
            .t_pow4_snapshot
            .extend_from_slice(&self.t_pow4_cached);
    }

    fn advance_intermediate(&mut self, deriv: &[f64], h: f64) {
        // Length parity across every backing vector must hold in all
        // builds: iterating with `zip()` truncates silently on mismatch
        // and would leave the state partially updated. Use `assert_eq!`
        // so release builds also fail fast, then index directly.
        let n = self.temperatures.len();
        assert_eq!(
            self.t_pow4_cached.len(),
            n,
            "t_pow4_cached length must equal temperatures length",
        );
        assert_eq!(
            self.scratch.temps_snapshot.len(),
            n,
            "temps_snapshot length must equal temperatures length; \
             call IntegrableObject::snapshot before advance_intermediate",
        );
        assert_eq!(
            deriv.len(),
            n,
            "advance_intermediate derivative length must equal temperature count",
        );
        // `zip()` would truncate silently on length mismatch; the asserts
        // above guarantee parity so this indexed loop is safe. Allow the
        // range-loop lint because the alternative (nested `zip`s) is
        // precisely the pattern we're trying to avoid.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let new_t = (self.scratch.temps_snapshot[i] + deriv[i] * h).max(0.0);
            self.temperatures[i] = new_t;
            self.t_pow4_cached[i] = new_t * new_t * new_t * new_t;
        }
    }

    fn finalize_rk4(&mut self, k1: &[f64], k2: &[f64], k3: &[f64], k4: &[f64], dt: f64) {
        self.finalize_rk4_temperatures(k1, k2, k3, k4, dt);
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
    atmos: &jeod_atmosphere::AtmosphereState<jeod_quantities::frame::SelfPlanet>,
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
///
/// # Panics
/// Panics if `facet_a.material != facet_b.material`. JEOD pairs a single
/// `SpringPairInteraction` to each facet pair, so both facets must carry
/// identical stiffness / damping / friction parameters; this is enforced
/// by the downstream
/// [`jeod_interactions::compute_contact_force_from_geometry`] call and
/// propagates out of `evaluate_contact_pair`. Callers that build pairs
/// through `Simulation::register_contact_pair` in `jeod_runner` get the
/// same check at registration time.
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
    // t_inertial_struct = t_struct_body^T * t_inertial_body (inertial → struct;
    // see `compute_t_inertial_struct` in jeod_dynamics::forces).
    let t_inertial_struct_a =
        jeod_dynamics::compute_t_inertial_struct(&t_struct_body_a, &t_inertial_body_a);
    let t_inertial_struct_b =
        jeod_dynamics::compute_t_inertial_struct(&t_struct_body_b, &t_inertial_body_b);
    // Rotation from structural-frame vector to inertial-frame vector
    // (i.e., struct → inertial), the inverse of `t_inertial_struct`.
    let t_inertial_from_struct_a = t_inertial_struct_a.transpose();
    let t_inertial_from_struct_b = t_inertial_struct_b.transpose();

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

    // Compute the detection-only contact geometry first, so we can build
    // JEOD's relative-velocity term using the actual contact point on A
    // (not merely A's facet reference). Returns `None` if the pair isn't
    // interpenetrating at this stage — in that case there's no force.
    let geom = compute_contact_geometry(&facet_a_world, &facet_b_world, rel_pos)?;

    // Arm from body A's CoM to the contact point on A's surface (inertial).
    // This is JEOD's `subject_contact_point` expressed about the CoM
    // (JEOD's facet `vehicle_point` and `composite_body` positions add up
    // to the same thing: see `point_contact_facet::calculate_torque`).
    let contact_arm_a_inertial = facet_a_offset_from_cm_inertial + geom.contact_point_on_a;

    // Relative velocity at the contact point, faithful port of JEOD
    // `point_contact_pair.cc:83-84`:
    //
    //   rel_velocity = (ω_target − ω_subject) × r_subject_contact
    //                  − (v_target − v_subject)
    //
    // i.e., a *single* cross product between the relative angular velocity
    // and the SUBJECT's contact-point arm from the CoM (not a separate
    // ω × r per body). An earlier version of this code used the textbook
    // two-body kinematic formula `(v_A−v_B) + ω_A×r_A − ω_B×r_B`, which
    // differs from JEOD when the contact point isn't at the same arm on
    // both bodies (e.g., SIM_contact RUN_point_off_center) and produced a
    // tangential-friction discrepancy of ~5 % during the contact event.
    // Per project policy we match JEOD's formulation exactly.
    //
    // `t_inertial_body` is inertial→body (see
    // `jeod_dynamics::compute_t_inertial_struct` docs), so going
    // body→inertial requires the transpose.
    let omega_a_inertial = rot_a.map_or(DVec3::ZERO, |r| {
        t_inertial_body_a.transpose() * r.ang_vel_body
    });
    let omega_b_inertial = rot_b.map_or(DVec3::ZERO, |r| {
        t_inertial_body_b.transpose() * r.ang_vel_body
    });
    let omega_rel_inertial = omega_b_inertial - omega_a_inertial;
    let rel_vel =
        (trans_a.velocity - trans_b.velocity) + omega_rel_inertial.cross(contact_arm_a_inertial);

    // Reuse the geometry from above — avoid repeating closest-point math
    // inside the RK4 inner loop.
    let contact =
        compute_contact_force_from_geometry(&facet_a_world, &facet_b_world, &geom, rel_vel);

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

    // Rotate inertial torques back to each body's body frame. `t_inertial_body`
    // is inertial→body, so it applies directly: v_body = t_inertial_body * v_inertial.
    let torque_a_body = t_inertial_body_a * torque_a_inertial;
    let torque_b_body = t_inertial_body_b * torque_b_inertial;

    Some(ContactPairEval {
        force_on_a,
        torque_a_body,
        torque_b_body,
    })
}

/// Evaluation of a ground-contact pair against the intermediate state of
/// a single body.
///
/// Symmetric to [`ContactPairEval`] but with no body-B side: the ground
/// has no dynamics, no torque accumulator, and exerts no Newton's-third-
/// law reaction we need to track.
#[derive(Debug, Clone, Copy, Default)]
pub struct GroundContactPairEval {
    /// Force on the vehicle in the inertial frame (N).
    pub force_on_a: DVec3,
    /// Torque on the vehicle in its body frame about its CoM (N*m).
    pub torque_a_body: DVec3,
}

/// Evaluate a single ground-contact pair, returning the force on the
/// vehicle (inertial) and the body-frame torque about the vehicle's CoM.
///
/// Mirrors [`evaluate_contact_pair`] but for a vehicle facet contacting
/// a [`jeod_interactions::GroundFacet`] anchored on a planet.
/// Composition:
/// [`jeod_interactions::compute_ground_contact_geometry`] (port of
/// `point_ground_interaction.cc::in_contact` /
/// `line_ground_interaction.cc::in_contact`) followed by
/// [`compute_contact_force_from_geometry`] (port of
/// `spring_pair_interaction.cc::calculate_forces`).
///
/// Arguments:
/// - `vehicle_facet` — vehicle-side facet in the body's structural frame.
/// - `ground_facet` — ground-side facet anchored on a planet via
///   [`jeod_interactions::Terrain`].
/// - `trans_a` — translational state of the vehicle (inertial).
/// - `rot_a` — rotational state of the vehicle.
/// - `t_struct_body_a` — structural→body rotation.
/// - `mass_a` — mass properties (for the torque arm and CoM offset).
/// - `t_inertial_pfix` — inertial→planet-fixed rotation. For
///   [`jeod_interactions::SphericalTerrain`] the pfix rotation cancels
///   in the ground-point computation, so callers may pass
///   [`DMat3::IDENTITY`]. Other terrain shapes (e.g., DEM) require the
///   actual pfix rotation.
///
/// Returns `None` if the vehicle is not in contact with the ground.
///
/// # Panics
/// Panics if `vehicle_facet.material != ground_facet.material` — JEOD pairs
/// a single `SpringPairInteraction` per facet pair, so both sides must
/// carry identical stiffness/damping/friction. Also panics when
/// `ground_facet.active` is false (JEOD_INV: IN.35).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_ground_contact_pair(
    vehicle_facet: &ContactFacet,
    ground_facet: &jeod_interactions::GroundFacet,
    trans_a: &TranslationalState,
    rot_a: &RotationalState,
    t_struct_body_a: DMat3,
    mass_a: &MassProperties,
    t_inertial_pfix: DMat3,
    phase: jeod_interactions::Phase,
) -> Option<GroundContactPairEval> {
    // Build inertial→body rotation from the body's left-quaternion.
    let t_inertial_body_a = rot_a.quaternion.left_quat_to_transformation();

    // Detection + body-frame geometry. Faithful port of JEOD's
    // ground-interaction algorithm, which operates entirely in body frame.
    let geom = jeod_interactions::compute_ground_contact_geometry(
        vehicle_facet,
        ground_facet,
        trans_a.position,
        t_inertial_body_a,
        t_struct_body_a,
        t_inertial_pfix,
        phase,
    )?;

    // CoM-to-contact arm in body frame. `geom.contact_point_on_a` is
    // contact point relative to the facet's *shape reference*; the
    // shape reference is at structural offset
    // `shape.reference_position()` from the structural origin, which is
    // at CoM offset `mass.position` from the body's CoM. The full
    // CoM-to-contact arm threads both terms through `t_struct_body_a`
    // to land in body-frame coords. We compute it before the contact
    // force eval so the same arm feeds both `rel_vel_body` and the
    // torque calculation below.
    let r_cm_struct = mass_a.position;
    let facet_offset_from_cm_body =
        t_struct_body_a * (vehicle_facet.shape.reference_position() - r_cm_struct);
    let arm_body = facet_offset_from_cm_body + geom.contact_point_on_a;

    // JEOD's relative-velocity term. `point_ground_interaction.cc:87-93`:
    //   rel_velocity = ω_body × subject_contact_point + T_inertial_body * v_inertial
    // The `// TODO: worry about planetary rotation` comment in JEOD
    // indicates the pfix-frame ground velocity is omitted; we mirror
    // that. JEOD's `subject_contact_point` arm is from the body CoM
    // to the contact point on the surface — we use the CoM-to-contact
    // arm assembled above (NOT `geom.contact_point_on_a`, which is
    // relative to the facet shape reference and would give wrong
    // damping/friction whenever `mass.position != 0` or the facet
    // reference is offset).
    let v_inertial_in_body = t_inertial_body_a * trans_a.velocity;
    let rel_vel_body = rot_a.ang_vel_body.cross(arm_body) + v_inertial_in_body;

    // Synthesize a "ground" ContactFacet so we can reuse the existing
    // force law (which takes two facets to enforce material equality).
    // The ground facet's `shape` is irrelevant to
    // `compute_contact_force_from_geometry` once geometry is precomputed —
    // we use a Point sphere of zero radius as a benign placeholder.
    let ground_facet_synth = ContactFacet::point(DVec3::ZERO, 0.0, ground_facet.material);

    // Spring + damping + Coulomb friction — same kernel as vehicle/vehicle.
    let contact = compute_contact_force_from_geometry(
        vehicle_facet,
        &ground_facet_synth,
        &geom,
        rel_vel_body,
    );

    // Force is currently expressed in body frame (the geometry was
    // computed in body frame). Rotate to inertial for the integrator.
    let force_on_a_inertial = t_inertial_body_a.transpose() * contact.force;

    // Torque = arm × force in body frame, using the same CoM-to-contact
    // arm assembled above for consistency with the ω×r term.
    let torque_a_body = arm_body.cross(contact.force);

    Some(GroundContactPairEval {
        force_on_a: force_on_a_inertial,
        torque_a_body,
    })
}

/// Typed sibling of [`compute_drag`].
///
/// Generic over the atmosphere planet `P`: drag uses the vehicle's
/// velocity in the planet's inertial frame (matching the corotation
/// wind in `atmos.wind`), so the typed velocity argument is
/// `Velocity<PlanetInertial<P>>` — not root-inertial. RF.10.
pub fn compute_drag_typed<P: jeod_quantities::frame::Planet, V: Vehicle>(
    drag_config: &DragConfigTyped,
    atmos: &jeod_atmosphere::AtmosphereState<P>,
    velocity: Velocity<jeod_quantities::frame::PlanetInertial<P>>,
    rot: Option<&RotationalState>,
    t_struct_body: DMat3,
) -> AerodynamicForceTyped<V> {
    // The kernel reads `atmos.density` and `atmos.wind.raw_si()`; both
    // are bit-identical to a `<SelfPlanet>`-tagged equivalent. Synthesize
    // the planet-erased view at the call site to reuse the numeric
    // path. The compile-time guard is the function signature: a caller
    // cannot pass `&AtmosphereState<Mars>` together with a
    // `Velocity<PlanetInertial<Earth>>`.
    let atmos_self =
        jeod_atmosphere::AtmosphereState::<jeod_quantities::frame::SelfPlanet>::from_raw(
            atmos.density,
            atmos.temperature,
            atmos.pressure,
            atmos.wind.raw_si(),
        );
    let raw = compute_drag(
        &drag_config.to_untyped(),
        &atmos_self,
        velocity.raw_si(),
        rot,
        t_struct_body,
    );
    AerodynamicForceTyped::<V>::from_untyped_unchecked(&raw)
}

/// Typed sibling of [`compute_gravity_torque`].
///
/// Returns the gravity gradient torque in [`BodyFrame<V>`].
pub fn compute_gravity_torque_typed<V: Vehicle>(
    grav_grad: &DMat3,
    rot: &RotationalState,
    inertia: InertiaTensor<BodyFrame<V>>,
) -> Torque<BodyFrame<V>> {
    let inertia_dmat = inertia.as_dmat3();
    let raw = compute_gravity_torque(grav_grad, rot, &inertia_dmat);
    Torque::<BodyFrame<V>>::from_raw_si(raw)
}

/// Typed sibling of [`compute_cannonball_srp`].
///
/// Identical kernel — wraps positions and uom dimensionless ratios.
/// Returns the cannonball SRP force in [`RootInertial`].
pub fn compute_cannonball_srp_typed(
    body_pos: Position<RootInertial>,
    sun_pos: Position<RootInertial>,
    cx_area: Area,
    albedo: Ratio,
    diffuse: Ratio,
    illum_factor: Ratio,
) -> Force<RootInertial> {
    use uom::si::area::square_meter;
    use uom::si::ratio::ratio;
    let raw = compute_cannonball_srp(
        body_pos.raw_si(),
        sun_pos.raw_si(),
        cx_area.get::<square_meter>(),
        albedo.get::<ratio>(),
        diffuse.get::<ratio>(),
        illum_factor.get::<ratio>(),
    );
    Force::<RootInertial>::from_raw_si(raw)
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

#[cfg(test)]
mod tests {
    use super::*;
    use jeod_interactions::ContactMaterial;
    use jeod_math::JeodQuat;

    // Two point facets at body origins, radius 1 each, with a material
    // that has spring stiffness only (no damping or friction).
    fn scenario_material(stiffness: f64, damping: f64, mu: f64) -> ContactMaterial {
        ContactMaterial::jeod_spring(stiffness, damping, mu)
    }

    /// Covers the ω × r_contact_arm term in `evaluate_contact_pair`.
    ///
    /// Two equal-mass, equal-inertia spheres at rest translationally but
    /// with a non-zero angular velocity on A only. The pair is at an
    /// off-centre geometry (veh2 offset in +y), so `r_A_contact` has a
    /// non-zero tangential component in the y direction. With
    /// translational `v_rel = 0`, any non-zero friction force proves that
    /// the `(ω_B − ω_A) × r_A_contact` kinematic term flowed through
    /// into `rel_vel` (matching JEOD `point_contact_pair.cc:83`). This
    /// locks the kinematic plumbing against regressions that Tier 3
    /// doesn't catch (the SIM_contact scenarios produce ω_rel = 0 by
    /// symmetry).
    #[test]
    fn evaluate_contact_pair_uses_omega_cross_contact_arm_in_rel_vel() {
        // Identity attitudes and t_struct_body so all frames coincide
        // with inertial — keeps the algebra inspectable.
        let mat = scenario_material(1000.0, 0.0, 0.5);
        let facet = jeod_interactions::ContactFacet::point(DVec3::ZERO, 1.0, mat);
        let mass = MassProperties::with_inertia(
            100.0,
            DMat3::from_cols(
                DVec3::new(40.0, 0.0, 0.0),
                DVec3::new(0.0, 40.0, 0.0),
                DVec3::new(0.0, 0.0, 40.0),
            ),
            DVec3::ZERO,
        );

        // veh1 at origin; veh2 at (1.8, 0.5, 0) — centres ~1.868 m apart,
        // so spheres interpenetrate (sum of radii = 2). Contact point on
        // A is ~radius · direction_from_A_to_B ≈ (0.964, 0.268, 0).
        let trans_a = TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        };
        let trans_b = TranslationalState {
            position: DVec3::new(1.8, 0.5, 0.0),
            velocity: DVec3::ZERO,
        };

        // Case 1: both ω = 0. v_rel = 0 ⇒ tangential slip = 0 ⇒ no
        // friction (only normal spring force, so force ∥ normal).
        let rot_zero = RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        };
        let baseline = evaluate_contact_pair(
            &facet,
            &facet,
            &trans_a,
            &trans_b,
            Some(&rot_zero),
            Some(&rot_zero),
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            Some(&mass),
            Some(&mass),
        )
        .expect("in contact");
        // Force on A is along the (negated) approach direction — it is
        // purely the spring force along `-u_from_A_to_B` (i.e., points
        // away from B back toward A). Crucially, it lies in the
        // plane-of-contact, not perpendicular to a tangent we would expect
        // for friction.
        let u_ab = DVec3::new(1.8, 0.5, 0.0).normalize();
        let force_parallel_to_normal = baseline.force_on_a.dot(-u_ab);
        assert!(
            force_parallel_to_normal > 0.0 && baseline.force_on_a.cross(-u_ab).length() < 1e-9,
            "baseline (ω=0) force should be purely along the normal; got {:?}",
            baseline.force_on_a,
        );

        // Case 2: give veh1 a non-zero ω about +z; veh2 still at rest
        // rotationally. rel_vel = (ω_B − ω_A) × r_A_contact
        //                      = -(0,0,ω_z) × r_A_contact
        // where r_A_contact ≈ radius · u_ab = (0.964, 0.268, 0).
        // This is a purely tangential contribution (it lies in the plane
        // perpendicular to r_A_contact, which is the contact-tangent
        // plane at this point), so the evaluator should add a friction
        // force perpendicular to `u_ab`.
        let omega_z = 1.0_f64;
        let rot_a_spinning = RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::new(0.0, 0.0, omega_z),
        };
        let spinning = evaluate_contact_pair(
            &facet,
            &facet,
            &trans_a,
            &trans_b,
            Some(&rot_a_spinning),
            Some(&rot_zero),
            DMat3::IDENTITY,
            DMat3::IDENTITY,
            Some(&mass),
            Some(&mass),
        )
        .expect("in contact");

        // The normal component must be unchanged (same penetration, same
        // normal v_rel): project onto the A→B direction. (The spring
        // force on A is along −u_ab, so we expect baseline.force_on_a · −u_ab
        // ≈ spinning.force_on_a · −u_ab.)
        let normal_component_baseline = baseline.force_on_a.dot(-u_ab);
        let normal_component_spinning = spinning.force_on_a.dot(-u_ab);
        assert!(
            (normal_component_baseline - normal_component_spinning).abs() < 1e-9,
            "spinning case should have the same normal force; \
             baseline={normal_component_baseline}, spinning={normal_component_spinning}",
        );

        // The friction force is the part of `spinning.force_on_a` tangent to
        // the normal. It must be non-zero — the ω term is the only
        // source of tangential slip in this scenario.
        let tangent_component = spinning.force_on_a - normal_component_spinning * (-u_ab);
        assert!(
            tangent_component.length() > 1e-3,
            "ω × r_arm kinematic term did not propagate into rel_vel: \
             spinning force is still axial with tangent magnitude {}",
            tangent_component.length(),
        );

        // And it must oppose the slip direction. rel_vel_a_wrt_b =
        // (ω_B − ω_A) × r_A_contact ≈ -(0,0,1) × (0.964, 0.268, 0)
        //   = (0.268, -0.964, 0). Friction direction is the negation of
        // the tangential slip, so expect friction ∝ (-0.268, +0.964, 0)
        // (projected onto the tangent plane).
        let r_arm = 1.0 * u_ab;
        let rel_vel = -DVec3::new(0.0, 0.0, omega_z).cross(r_arm);
        // rel_vel is approximately tangent (normal dot ≈ 0 since ω ⊥ r).
        let expected_tangent_dir = (-rel_vel).normalize();
        let got_tangent_dir = tangent_component.normalize();
        assert!(
            got_tangent_dir.dot(expected_tangent_dir) > 0.99,
            "friction direction should oppose ω-induced slip; \
             expected {expected_tangent_dir:?}, got {got_tangent_dir:?}",
        );
    }
}
