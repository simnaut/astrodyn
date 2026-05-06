//! Supporting types for [`super::Simulation`].
//!
//! - Public surface declared here: [`VehicleOutput`],
//!   [`ContactPairConfig`] (re-exported through `simulation::mod` and
//!   `crate::lib` for API stability).
//! - Crate-internal: [`SimBody`], [`GravityData`].
//!
//! `SourceFrameIds` was lifted to `jeod_sim::source_frames` (issue #71)
//! so the Bevy adapter can build source frames against the same
//! structure `jeod_runner` uses. `DetachedSubtreeState` lives in
//! `jeod_dynamics::subtree` (issue #253 Task C) — pure rigid-body
//! kinematics, no `Simulation` dependency. It is re-exported from
//! `simulation::mod` so consumers reach it via
//! `jeod_runner::DetachedSubtreeState` regardless.

use glam::{DMat3, DVec3};

use jeod_dynamics::{MassBodyId, MassPointState};
use jeod_frames::FrameId;
use jeod_interactions::{ContactFacet, GroundFacet};
use jeod_quantities::frame::IntegrationFrame;
use jeod_sim::{
    AerodynamicForce, AtmosphereState, DragConfig, DynamicsConfig, EulerSequence, FrameDerivatives,
    FrameSwitchConfig, GeodeticState, GravityAcceleration, GravityControls, GravitySource,
    LvlhFrame, MassProperties, OrbitalElements, RadiationForce, RotationModel, RotationalState,
    SelfPlanet, SrpModel, TotalForce, TranslationalState, TranslationalStateTyped, VehicleConfig,
};

/// Registration of a contact interaction between two bodies.
///
/// Port of JEOD's `ContactPair` — the two facets are registered with the
/// `Contact` manager, which runs `check_contact()` at each derivative
/// evaluation (see `contact.sm`). In this runner, registered pairs are
/// evaluated at every RK4 stage inside [`super::Simulation::step`] when
/// any pairs are present.
#[derive(Debug, Clone)]
pub struct ContactPairConfig {
    /// Index of body A (the "subject" in JEOD terminology).
    pub body_a: usize,
    /// Facet on body A (shape positions in A's structural frame).
    pub facet_a: ContactFacet,
    /// Index of body B (the "target" in JEOD terminology).
    pub body_b: usize,
    /// Facet on body B (shape positions in B's structural frame).
    pub facet_b: ContactFacet,
}

/// Registration of a ground-contact interaction between a vehicle and a
/// planetary surface.
///
/// Symmetric to [`ContactPairConfig`] but bodyless on the ground side —
/// the ground doesn't integrate, has no Newton's-third-law reaction
/// applied to it, and is queried per-step from the [`GroundFacet`]'s
/// terrain model. The vehicle facet/material must match the ground
/// facet's material exactly (single-pair `SpringPairInteraction`
/// semantics).
#[derive(Debug, Clone)]
pub struct GroundContactPairConfig {
    /// Index of the vehicle body.
    pub body_a: usize,
    /// Vehicle facet (shape positions in the body's structural frame).
    pub vehicle_facet: ContactFacet,
    /// Ground facet (terrain, alt_offset, material).
    pub ground_facet: GroundFacet,
    /// JEOD initialization-time impulse, computed at registration via
    /// the `Phase::Initialization` evaluator (`facet_pos_body == 0`).
    /// Consumed at stage 1 of the first integration step (RK4 weight
    /// 1/6) and cleared to `None` thereafter — mirrors
    /// `ContactSurface::collect_forces_torques` zeroing `facet.force`
    /// after stage 1 in JEOD.
    ///
    /// Note that the init-phase evaluator essentially always reports
    /// contact for any realistic planet radius (because `|rel_state|`
    /// is O(1–2 m) — the facet's body-frame surface extent — while
    /// `|ground|` is O(R)). So this field is `Some(...)` for every
    /// successfully registered pair on initialization, regardless of
    /// the vehicle's actual altitude; the impulsive launch JEOD's CSV
    /// records is precisely this initialization-state effect. The
    /// field becomes `None` only after the first integration step
    /// consumes it.
    pub pending_initial_impulse: Option<GroundContactImpulse>,
}

/// Impulsive contact contribution from JEOD's pre-propagation
/// `GroundInteraction::initialize` call (force on the vehicle in
/// inertial coords + body-frame torque about CoM).
#[derive(Debug, Clone, Copy)]
pub struct GroundContactImpulse {
    pub force_inertial: DVec3,
    pub torque_body: DVec3,
}

/// Attachment of a body to a non-body reference frame (port of JEOD's
/// `DynBody::frame_attach` member, populated by `attach_to_frame`).
///
/// Captures the parent ref-frame ID and the rigid-body offset between
/// the parent frame and the attached body's composite-body frame at the
/// instant of attach. The runner's per-step pass derives the body's
/// state by composing the parent frame's current state with this fixed
/// offset (see `Simulation::propagate_frame_attached_state`); the body
/// stays glued to the parent frame as the frame moves under ephemeris,
/// planet rotation, or kinematic-joint drives.
///
/// JEOD source:
/// `models/dynamics/dyn_body/src/dyn_body_attach.cc:271-379` (the
/// three `attach_to_frame` overloads); the captured offset corresponds
/// to JEOD's `frame_attach.attach_offset` (`X_pframe_to_struct`).
#[derive(Debug, Clone, Copy)]
pub struct FrameAttachState {
    /// `FrameId` of the parent reference frame. Resolved from a
    /// frame-tree lookup at attach time and never reparented while the
    /// attachment holds.
    pub parent_frame_id: FrameId,
    /// Rigid-body offset from the parent frame to this body's
    /// composite-body frame, expressed in parent-frame coordinates.
    /// Frozen at attach time; changes only when the attachment is
    /// released and re-established.
    pub attach_offset: MassPointState,
}

/// Gravity-specific data associated with a source (decoupled from frame tree).
///
/// The frame tree stores position/velocity/rotation state; this struct stores
/// the physical gravity model data that lives alongside it. The `velocity`
/// field stores source velocity for relativistic corrections — for central
/// bodies at the root frame, the tree node has zero velocity but the source
/// may still have physical velocity (e.g., Sun orbiting the barycenter).
pub(crate) struct GravityData {
    /// Physical gravity source (mu, model: PointMass or SphericalHarmonics).
    pub source: GravitySource,
    /// Source velocity in the inertial frame (m/s). Used for relativistic
    /// corrections. Stored here rather than in the tree because the root
    /// frame's velocity must be zero (it's the reference origin).
    pub velocity: DVec3,
    /// Tidal ΔC20 to add to the base C20 coefficient. Updated each step.
    pub delta_c20: f64,
    /// Tidal configuration. When `Some`, the simulation computes ΔC20 each step.
    pub tidal_config: Option<jeod_gravity::tides::TidalConfig>,
    /// Rotation model for updating planet-fixed frame each step.
    pub rotation_model: RotationModel,
    /// Sidereal angular velocity (rad/s) for the planet-fixed frame's
    /// `ang_vel_this`. Sourced from `PlanetConfig::omega` at setup time.
    /// JEOD sets this as `[0, 0, planet_omega]` in `planet_rnp.cc`.
    pub planet_omega: f64,
}

/// Read-only view of vehicle state after stepping.
///
/// Returned by [`super::Simulation::body`]. Contains the current integrated state
/// plus any derived states that were configured.
#[derive(Debug, Clone)]
pub struct VehicleOutput {
    /// Current translational state (position, velocity) in the integration frame.
    pub trans: TranslationalState,
    /// Frame ID of the current integration frame in the simulation's frame tree.
    pub integ_frame_id: FrameId,
    /// Current rotational state (quaternion, angular velocity). `None` for 3-DOF.
    pub rot: Option<RotationalState>,
    /// Total translational acceleration in the integration frame (m/s²) at the
    /// end of the last `step()`. Sum of gravity and non-gravity contributions —
    /// mirrors JEOD's `derivs.trans_accel`. Zero before the first `step()`.
    pub trans_accel: DVec3,
    /// Total rotational acceleration in the body frame (rad/s²) at the end of
    /// the last `step()` — mirrors JEOD's `derivs.rot_accel`. `None` for 3-DOF
    /// bodies; zero before the first `step()`.
    pub rot_accel: Option<DVec3>,
    /// Orbital elements from the latest step.
    pub orbital_elements: Option<OrbitalElements<SelfPlanet>>,
    /// Euler angles `[phi, theta, psi]` from the latest step.
    pub euler_angles: Option<[f64; 3]>,
    /// LVLH frame from the latest step.
    pub lvlh_frame: Option<LvlhFrame>,
    /// Geodetic state (latitude, longitude, altitude).
    pub geodetic_state: Option<GeodeticState>,
    /// Solar beta angle (radians).
    pub solar_beta: Option<f64>,
    /// Earth lighting state (sun/moon occlusion, albedo).
    pub earth_lighting: Option<jeod_interactions::earth_lighting::EarthLightingState>,
}

/// Internal per-body simulation state. Combines user config with bookkeeping
/// and output fields. Not exposed publicly — users interact through
/// [`VehicleConfig`] (input) and [`VehicleOutput`] (output).
pub(crate) struct SimBody {
    // ── Config (from VehicleConfig) ──
    /// Translational state in this body's integration frame.
    ///
    /// `IntegrationFrame` is *kind-distinct* from `RootInertial` so that
    /// root-inertial consumers (gravity, relativistic, SRP, solar beta,
    /// earth lighting — the *shift sites*, which mix body state with
    /// root-inertial source positions for Sun, Moon, or gravity sources)
    /// cannot silently take the integration-frame value — they must call
    /// `body.trans.to_inertial(&integ_origin)` first. Planet-inertial
    /// consumers (atmosphere, drag velocity, LVLH, geodetic, orbital
    /// elements — *non-shift sites*, which operate within a single
    /// planet's inertial frame) take `body.trans.position.raw_si()`
    /// directly: the body's integration frame is that planet's inertial
    /// frame in realistic configs, so applying the shift would change the
    /// planet-relative coordinates and produce wrong physics. See
    /// issue #255 and `JEOD_invariants.md` RF.10 for the split.
    pub trans: TranslationalStateTyped<IntegrationFrame>,
    pub rot: Option<RotationalState>,
    pub mass: Option<MassProperties>,
    /// If this body participates in a mass tree, its node ID.
    pub mass_body_id: Option<MassBodyId>,
    /// When `true`, this body's `trans`/`rot` are derived each step from
    /// its mass-tree parent via `propagate_state_via_storage` rather
    /// than integrated. Mirrors the Bevy adapter's `KinematicChildC`
    /// marker. The body must be a non-root node in the mass tree;
    /// integration is skipped, force accumulation still runs (matching
    /// JEOD's `compute_point_derivative` flag, which lets a child body
    /// accumulate forces without integrating them — useful for
    /// post-step acceleration introspection).
    ///
    /// JEOD precedent: `DynBody::propagate_state_from_structure`
    /// (`models/dynamics/dyn_body/src/dyn_body_propagate_state.cc`)
    /// derives child states from the parent's structure each step;
    /// only the root integrates (`DB.17`).
    pub kinematic_only: bool,
    /// When `Some`, this body is attached to a non-body reference frame
    /// (port of JEOD `DynBody::attach_to_frame`,
    /// `models/dynamics/dyn_body/src/dyn_body_attach.cc:271-379`). The
    /// body's `trans` / `rot` are derived each step from the parent
    /// reference frame's state composed with the captured offset, and
    /// translational + rotational integration is suppressed (mirrors
    /// `dyn_body_integration.cc:309-333`'s `frame_attach.isAttached()`
    /// branch). Distinct from [`kinematic_only`](Self::kinematic_only),
    /// which targets a parent **body** in the mass tree;
    /// `frame_attach` targets a parent **reference frame**. JEOD_INV:
    /// DB.21 — only unattached bodies integrate.
    pub frame_attach: Option<FrameAttachState>,
    pub config: DynamicsConfig,
    pub gravity_controls: GravityControls<usize>,
    pub integrator: jeod_dynamics::IntegratorType,
    pub drag: Option<DragConfig>,
    pub flat_plate_state: Option<jeod_sim::FlatPlateState<jeod_sim::SelfRef>>,
    pub cannonball_srp: Option<(f64, f64, f64)>,
    pub shadow_body: Option<(usize, f64)>,
    pub t_struct_body: DMat3,
    pub compute_gravity_torque: bool,
    pub atmospheric_state: Option<AtmosphereState<SelfPlanet>>,
    pub external_force: DVec3,
    pub external_torque: DVec3,
    /// Externally applied force in the body's structural frame (N).
    ///
    /// JEOD's `Force` is collected in the body's structural frame and
    /// rotated to inertial at force-collection time
    /// (`models/dynamics/dyn_body/src/dyn_body_collect.cc:219-221`).
    /// Tier 3 sims that schedule struct-frame force events
    /// (`SIM_verif_attach_detach`'s `RUN_compute_child_derivative`) need
    /// this entry point so the inertial-frame contribution tracks the
    /// body's current attitude across each integration step.
    pub external_force_struct: DVec3,
    /// Externally applied torque in the body's structural frame (N·m).
    ///
    /// Mirrors [`external_force_struct`](Self::external_force_struct);
    /// rotated to body frame at force-collection time via the body's
    /// structural-to-body transform.
    pub external_torque_struct: DVec3,

    // ── Frame switching ──
    pub integ_frame_id: FrameId,
    pub body_frame_id: FrameId,
    pub frame_switches: Vec<FrameSwitchConfig>,

    // ── Bookkeeping (written each step, not user-visible) ──
    pub gravity_accel: GravityAcceleration,
    pub total_force: TotalForce,
    pub frame_derivs: FrameDerivatives,
    pub aero_force: Option<AerodynamicForce>,
    pub radiation_force: Option<RadiationForce>,
    pub gravity_torque: Option<DVec3>,

    // ── Derived state config ──
    pub orbital_elements_source: Option<usize>,
    pub euler_sequence: Option<EulerSequence>,
    pub compute_lvlh: bool,
    pub geodetic_planet: Option<(usize, f64, f64)>,
    pub compute_solar_beta: bool,
    pub earth_lighting_config: Option<(f64, f64, f64)>,

    // ── Derived state outputs ──
    pub orbital_elements: Option<OrbitalElements<SelfPlanet>>,
    pub euler_angles: Option<[f64; 3]>,
    pub lvlh_frame: Option<LvlhFrame>,
    pub geodetic_state: Option<GeodeticState>,
    pub solar_beta: Option<f64>,
    pub earth_lighting: Option<jeod_interactions::earth_lighting::EarthLightingState>,

    // ── Integrator state ──
    pub gj_state: Option<jeod_dynamics::GaussJacksonState>,
    pub abm4_state: Option<jeod_dynamics::Abm4State>,
}

impl SimBody {
    /// Convert a user-facing VehicleConfig into an internal SimBody.
    pub(crate) fn from_config(
        config: VehicleConfig,
        integ_frame_id: FrameId,
        body_frame_id: FrameId,
    ) -> Self {
        let has_rot = config.rot.is_some();
        let dynamics_config = DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: has_rot,
            three_dof: !has_rot,
        };

        let (flat_plate_state, cannonball_srp) = match config.srp {
            Some(SrpModel::FlatPlate(fps)) => (Some(fps), None),
            Some(SrpModel::Cannonball {
                cx_area,
                albedo,
                diffuse,
            }) => (None, Some((cx_area, albedo, diffuse))),
            None => (None, None),
        };

        let shadow_body = config.shadow_body.map(|sb| (sb.source_idx, sb.radius));

        let has_drag = config.drag.is_some();
        let atmospheric_state = if has_drag {
            Some(AtmosphereState::<SelfPlanet>::default())
        } else {
            None
        };

        Self {
            // VehicleConfig::trans is documented as integration-frame; wrap
            // the untyped storage with the IntegrationFrame phantom so
            // root-inertial consumers must shift via `to_inertial`. See #255.
            trans: TranslationalStateTyped::<IntegrationFrame>::from_untyped_unchecked(
                &config.trans,
            ),
            rot: config.rot,
            mass: config.mass,
            mass_body_id: None,
            kinematic_only: false,
            frame_attach: None,
            config: dynamics_config,
            gravity_controls: config.gravity_controls,
            integrator: config.integrator,
            drag: config.drag,
            flat_plate_state,
            cannonball_srp,
            shadow_body,
            t_struct_body: config.t_struct_body,
            compute_gravity_torque: config.compute_gravity_gradient,
            atmospheric_state,
            external_force: config.external_force,
            external_torque: config.external_torque,
            external_force_struct: DVec3::ZERO,
            external_torque_struct: DVec3::ZERO,

            integ_frame_id,
            body_frame_id,
            frame_switches: config.frame_switches,

            gravity_accel: GravityAcceleration::default(),
            total_force: TotalForce::default(),
            frame_derivs: FrameDerivatives::default(),
            aero_force: None,
            radiation_force: None,
            gravity_torque: None,

            orbital_elements_source: config.derived.orbital_elements_source,
            euler_sequence: config.derived.euler_sequence,
            compute_lvlh: config.derived.lvlh,
            geodetic_planet: config
                .derived
                .geodetic
                .map(|g| (g.source_idx, g.r_eq, g.r_pol)),
            compute_solar_beta: config.derived.solar_beta,
            earth_lighting_config: config
                .derived
                .earth_lighting
                .map(|e| (e.earth_radius, e.moon_radius, e.sun_radius)),

            orbital_elements: None,
            euler_angles: None,
            lvlh_frame: None,
            geodetic_state: None,
            solar_beta: None,
            earth_lighting: None,

            gj_state: None,
            abm4_state: None,
        }
    }

    /// Create a VehicleOutput view of the current state.
    pub(crate) fn output(&self) -> VehicleOutput {
        VehicleOutput {
            // VehicleOutput::trans is the public, untyped integration-frame
            // storage form. Drop the IntegrationFrame phantom at the API
            // boundary; the values are bit-identical.
            trans: self.trans.to_untyped(),
            integ_frame_id: self.integ_frame_id,
            rot: self.rot,
            trans_accel: self.frame_derivs.trans_accel,
            rot_accel: self.rot.map(|_| self.frame_derivs.rot_accel),
            orbital_elements: self.orbital_elements.clone(),
            euler_angles: self.euler_angles,
            lvlh_frame: self.lvlh_frame,
            geodetic_state: self.geodetic_state,
            solar_beta: self.solar_beta,
            earth_lighting: self.earth_lighting.clone(),
        }
    }
}
