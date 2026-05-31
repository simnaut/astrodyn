// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! ECS-agnostic body actions: queueable mutations to a vehicle's
//! translational state, rotational state, and mass properties.
//!
//! Port of JEOD's
//! [`models/dynamics/body_action/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/body_action/)
//! family of `BodyAction` subclasses. The variants currently ported are
//! `MassBodyInit`, `DynBodyInitTransState`, `DynBodyInitRotState`,
//! `DynBodyInitOrbit`, `DynBodyInitLvlhTransState`,
//! `DynBodyInitLvlhRotState`, and `DynBodyInitNedTransState`.
//!
//! The common JEOD pattern is:
//!
//! - construct an action object with subject + parameters,
//! - hand it to `DynManager::add_body_action`,
//! - on each `perform_actions` pass `is_ready()` is consulted, and
//!   ready actions have their `apply()` invoked, mutating the subject;
//! - actions can be removed by name via
//!   `DynManager::remove_body_action` before they fire.
//!
//! [`BodyAction`] is a single enum that carries the parameters for
//! every variant. Pure construction-by-data: no subject pointers, no
//! cross-body dependencies — adapters resolve the subject (an ECS
//! entity in the Bevy adapter, or an arena index in the runner) and
//! call the action's per-substate apply method
//! ([`BodyAction::apply_translational`], [`BodyAction::apply_rotational`],
//! [`BodyAction::apply_mass`]) to obtain the resulting state values.
//!
//! The Bevy wiring lives in `astrodyn_bevy::body_action` (a `Commands`
//! extension that queues actions and a system that drains the queue
//! each tick before [`PipelineStage::Environment`](crate::PipelineStage)).

use glam::{DMat3, DVec3};

pub use astrodyn_dynamics::body_init::LvlhAngularVelocityFrame;
use astrodyn_dynamics::body_init::{
    init_from_lvlh, init_from_mean_anomaly, init_from_ned, init_from_orbital_elements,
    init_from_time_periapsis, init_rot_from_lvlh, init_rot_relative_to_frame,
    init_trans_relative_to_frame,
};
use astrodyn_dynamics::{MassProperties, RotationalState, TranslationalState};
use astrodyn_frames::RefFrameState;
use astrodyn_math::{GeodeticState, JeodQuat, OrbitalElements};
use astrodyn_quantities::frame::SelfPlanet;

/// Selects which Keplerian element subset parameterizes an
/// [`BodyAction::InitTransOrbital`].
///
/// Mirrors JEOD `DynBodyInitOrbit::set` enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitalElementSet {
    /// `(a, e, i, Ω, ω, ν)` — true anomaly.
    SmaEccIncAscnodeArgperTanom,
    /// `(a, e, i, Ω, ω, M)` — mean anomaly.
    SmaEccIncAscnodeArgperManom,
    /// `(a, e, i, Ω, ω, t_peri)` — time elapsed since periapsis (s).
    SmaEccIncAscnodeArgperTimeperi,
}

/// One queueable body action. Each variant carries the parameters
/// needed to compute the new state — never a pointer to the subject
/// (the subject is resolved by the calling adapter).
///
/// **Mutation semantics**: an action that targets a sub-state (e.g.
/// only translational, only rotational, only mass) replaces *only that
/// sub-state* on the subject. Other components are untouched. This
/// matches JEOD's per-action semantics: `MassBodyInit` only writes
/// mass properties, `DynBodyInitTransState` only writes translational
/// state, etc.
/// `#[non_exhaustive]`: this enum is an internal initialization-recipe
/// dispatch type whose roadmap is to grow one variant per ported JEOD
/// body-action kind. It is `pub` only for the Bevy adapter and tests —
/// the user-facing API is the typestate `VehicleBuilder` / `VehicleConfig`,
/// not raw `BodyAction` construction. Marking it non-exhaustive makes
/// that expected additive growth a non-breaking change, so porting the
/// next body-action kind no longer trips a public-API stability fence.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BodyAction {
    /// Replace the subject's mass properties.
    ///
    /// JEOD analog: `MassBodyInit::apply` (mass / position / inertia
    /// install with the configured `inertia_spec`). We accept a
    /// pre-resolved `MassProperties` because the JEOD spec
    /// transformations (`Body`, `StructCG`, `Struct`, `Spec`,
    /// `SpecCG`) are already factored into
    /// [`astrodyn_dynamics::MassProperties`] constructors and the
    /// existing test scaffolding (`tier3_sim_attach_mass`).
    InitMass {
        /// New mass properties for the subject body.
        mass: MassProperties,
    },

    /// Replace the subject's translational state directly.
    ///
    /// JEOD analog: `DynBodyInitTransState`.
    InitTrans {
        /// New translational state.
        state: TranslationalState,
    },

    /// Replace the subject's rotational state directly.
    ///
    /// JEOD analog: `DynBodyInitRotState`.
    InitRot {
        /// Attitude quaternion (scalar-first, left-multiply, parent →
        /// body).
        quaternion: JeodQuat,
        /// Angular velocity expressed in the body frame (rad/s).
        ang_vel_body: DVec3,
    },

    /// Replace the subject's rotational state from an LVLH-relative
    /// attitude + angular velocity plus a reference orbit.
    ///
    /// JEOD analog: `DynBodyInitLvlhRotState`. Delegates to
    /// [`astrodyn_dynamics::body_init::init_rot_from_lvlh`], which
    /// constructs the reference-orbit LVLH frame from
    /// (`reference_position`, `reference_velocity`) and composes the
    /// user-supplied LVLH→body attitude / LVLH-relative angular
    /// velocity with the LVLH frame's own orientation and angular
    /// velocity wrt inertial.
    InitLvlhRot {
        /// LVLH→body attitude quaternion (scalar-first,
        /// left-transformation; JEOD convention).
        q_lvlh_body: JeodQuat,
        /// Angular velocity of the body wrt the LVLH frame (rad/s),
        /// expressed in the frame indicated by `ang_vel_frame`.
        ang_vel_lvlh_to_body: DVec3,
        /// Coordinate frame of `ang_vel_lvlh_to_body`. JEOD's
        /// `rate_in_parent` flag picks between body-frame and
        /// parent-frame interpretation; we expose the equivalent
        /// choice through this enum and apply the same transform
        /// JEOD's `apply_user_inputs` does.
        ang_vel_frame: LvlhAngularVelocityFrame,
        /// Reference orbit position in the central body's
        /// planet-inertial frame (m).
        reference_position: DVec3,
        /// Reference orbit velocity in the central body's
        /// planet-inertial frame (m/s).
        reference_velocity: DVec3,
    },

    /// Replace the subject's translational state from Keplerian
    /// orbital elements.
    ///
    /// JEOD analog: `DynBodyInitOrbit`. Delegates to
    /// [`astrodyn_dynamics::body_init::init_from_orbital_elements`] /
    /// [`astrodyn_dynamics::body_init::init_from_mean_anomaly`] /
    /// [`astrodyn_dynamics::body_init::init_from_time_periapsis`]
    /// depending on `set`.
    InitTransOrbital {
        /// Element subset and the anomaly-vs-time interpretation of
        /// the sixth element.
        set: OrbitalElementSet,
        /// Keplerian elements. The fields actually consumed depend
        /// on `set`:
        ///
        /// - `SmaEccIncAscnodeArgperTanom`: `semi_major_axis`,
        ///   `e_mag`, `inclination`, `long_asc_node`,
        ///   `arg_periapsis`, `true_anom`.
        /// - `SmaEccIncAscnodeArgperManom`: same five plus
        ///   `mean_anom`.
        /// - `SmaEccIncAscnodeArgperTimeperi`: same five plus
        ///   `time_periapsis` carried in the
        ///   [`OrbitalElementSet::SmaEccIncAscnodeArgperTimeperi`]
        ///   sibling.
        ///
        /// Tagged with [`SelfPlanet`] because `BodyAction` is the
        /// planet-erased boundary type used by both the Bevy adapter
        /// (`OrbitalElementsC` is `<SelfPlanet>`) and the runner queue:
        /// the central body identity is carried alongside in `mu`, and
        /// only the bare scalar fields of `OrbitalElements` are read
        /// here (the planet phantom never crosses into the f64 init
        /// kernel).
        elements: OrbitalElements<SelfPlanet>,
        /// Time elapsed since periapsis (s) — only consumed when
        /// `set = SmaEccIncAscnodeArgperTimeperi`.
        time_periapsis: f64,
        /// Gravitational parameter of the central body (m³/s²).
        mu: f64,
    },

    /// Replace the subject's translational state from LVLH-relative
    /// position + velocity plus a reference orbit.
    ///
    /// JEOD analog: `DynBodyInitLvlhTransState`.
    InitTransLvlh {
        /// LVLH-frame position offset from the reference orbit (m).
        lvlh_position: DVec3,
        /// LVLH-frame velocity offset from the reference orbit (m/s).
        lvlh_velocity: DVec3,
        /// Reference orbit position in the central body's
        /// planet-inertial frame (m).
        reference_position: DVec3,
        /// Reference orbit velocity in the central body's
        /// planet-inertial frame (m/s).
        reference_velocity: DVec3,
    },

    /// Replace the subject's translational state from an offset
    /// expressed in a reference frame `B` that is a child of the
    /// inertial integration frame (a target vehicle's composite-body
    /// frame, or a vehicle-centred LVLH / NED frame).
    ///
    /// JEOD analog: the translation-only vehicle-relative inits
    /// (`DynBodyInitTransState` with a body reference frame,
    /// `DynBodyInitLvlhTransState` / `DynBodyInitNedTransState` with
    /// `ref_body_name` set). The reference frame's full inertial state
    /// is resolved by the caller (which already holds the target
    /// vehicle's state) and passed as `reference_frame`; the offset is
    /// composed up to the inertial frame via
    /// [`astrodyn_frames::RefFrameState::incr_left`], including the
    /// reference frame's `ω × r` velocity term.
    InitTransRelativeFrame {
        /// Reference frame `B`'s state wrt the inertial frame (origin
        /// position/velocity in inertial coordinates, `T_inertial_B`,
        /// and `ω_inertial_B` in `B`).
        reference_frame: RefFrameState,
        /// Subject position offset expressed in frame `B` (m).
        offset_position: DVec3,
        /// Subject velocity offset expressed in frame `B` (m/s).
        offset_velocity: DVec3,
    },

    /// Replace the subject's full state (translational + rotational)
    /// from an offset + attitude/rate expressed in a reference frame
    /// `B` that is a child of the inertial integration frame.
    ///
    /// JEOD analog: the full-state vehicle-relative init
    /// (`DynBodyInitLvlhState` with `set_items = Pos_Vel_Att_Rate` and
    /// `ref_body_name` set). Both substates are composed up to the
    /// inertial frame via
    /// [`astrodyn_frames::RefFrameState::incr_left`]:
    /// translation including the `ω × r` velocity term, attitude via
    /// `Q_A:C = Q_B:C · Q_A:B`, and rate via
    /// `w_A:C = T_B:C · w_A:B + w_B:C`.
    InitFullRelativeFrame {
        /// Reference frame `B`'s state wrt the inertial frame.
        reference_frame: RefFrameState,
        /// Subject position offset expressed in frame `B` (m).
        offset_position: DVec3,
        /// Subject velocity offset expressed in frame `B` (m/s).
        offset_velocity: DVec3,
        /// B→subject attitude (scalar-first, left-transformation; JEOD
        /// convention).
        q_frame_subject: JeodQuat,
        /// Angular velocity of the subject wrt frame `B`, expressed in
        /// the subject body frame (rad/s) — JEOD `rate_in_parent =
        /// false`.
        ang_vel_frame_to_subject: DVec3,
    },

    /// Replace the subject's translational state from NED
    /// (North-East-Down) position + velocity.
    ///
    /// JEOD analog: `DynBodyInitNedTransState`.
    InitTransNed {
        /// Geodetic position (latitude, longitude, altitude).
        geodetic: GeodeticState,
        /// Planet-fixed velocity expressed in the NED frame (m/s).
        ned_velocity: DVec3,
        /// Equatorial radius of the central body (m).
        r_equatorial: f64,
        /// Polar radius of the central body (m).
        r_polar: f64,
        /// Rotation matrix from ECI to PCPF (planet-fixed) frame.
        t_eci_pcpf: DMat3,
        /// Planet angular velocity in the ECI frame (rad/s).
        omega_planet: DVec3,
    },
}

impl BodyAction {
    /// Whether this action is ready to be applied right now.
    ///
    /// JEOD's `BodyAction::is_ready` defaults to `true`; subclasses
    /// override when an action depends on another body's state being
    /// resolved first (e.g. an LVLH-relative init waiting for the
    /// reference body to be initialized). All variants here carry
    /// concrete numeric parameters at construction, so they are
    /// always ready as long as the subject is resolvable. Adapters
    /// that need cross-body dependencies should compose their own
    /// readiness gate ahead of this call.
    // JEOD_INV: BA.09 — is_ready() consulted before apply on every pass
    #[inline]
    pub fn is_ready(&self) -> bool {
        true
    }

    /// Returns the new translational state if this action targets
    /// translational state. Returns `None` for actions that don't
    /// touch translational state (e.g. `InitMass`, `InitRot`).
    pub fn apply_translational(&self) -> Option<TranslationalState> {
        match self {
            BodyAction::InitTrans { state } => Some(*state),
            BodyAction::InitTransOrbital {
                set,
                elements,
                time_periapsis,
                mu,
            } => Some(match set {
                OrbitalElementSet::SmaEccIncAscnodeArgperTanom => init_from_orbital_elements(
                    elements.semi_major_axis,
                    elements.e_mag,
                    elements.inclination,
                    elements.long_asc_node,
                    elements.arg_periapsis,
                    elements.true_anom,
                    *mu,
                ),
                OrbitalElementSet::SmaEccIncAscnodeArgperManom => init_from_mean_anomaly(
                    elements.semi_major_axis,
                    elements.e_mag,
                    elements.inclination,
                    elements.long_asc_node,
                    elements.arg_periapsis,
                    elements.mean_anom,
                    *mu,
                ),
                OrbitalElementSet::SmaEccIncAscnodeArgperTimeperi => init_from_time_periapsis(
                    elements.semi_major_axis,
                    elements.e_mag,
                    elements.inclination,
                    elements.long_asc_node,
                    elements.arg_periapsis,
                    *time_periapsis,
                    *mu,
                ),
            }),
            BodyAction::InitTransLvlh {
                lvlh_position,
                lvlh_velocity,
                reference_position,
                reference_velocity,
            } => Some(init_from_lvlh(
                *lvlh_position,
                *lvlh_velocity,
                *reference_position,
                *reference_velocity,
            )),
            BodyAction::InitTransRelativeFrame {
                reference_frame,
                offset_position,
                offset_velocity,
            } => Some(init_trans_relative_to_frame(
                reference_frame,
                *offset_position,
                *offset_velocity,
            )),
            BodyAction::InitFullRelativeFrame {
                reference_frame,
                offset_position,
                offset_velocity,
                ..
            } => Some(init_trans_relative_to_frame(
                reference_frame,
                *offset_position,
                *offset_velocity,
            )),
            BodyAction::InitTransNed {
                geodetic,
                ned_velocity,
                r_equatorial,
                r_polar,
                t_eci_pcpf,
                omega_planet,
            } => Some(init_from_ned(
                geodetic,
                *ned_velocity,
                *r_equatorial,
                *r_polar,
                t_eci_pcpf,
                *omega_planet,
            )),
            BodyAction::InitMass { .. }
            | BodyAction::InitRot { .. }
            | BodyAction::InitLvlhRot { .. } => None,
        }
    }

    /// Returns the new rotational state if this action targets
    /// rotational state. Returns `None` for actions that don't touch
    /// rotational state.
    pub fn apply_rotational(&self) -> Option<RotationalState> {
        match self {
            BodyAction::InitRot {
                quaternion,
                ang_vel_body,
            } => Some(RotationalState {
                quaternion: *quaternion,
                ang_vel_body: *ang_vel_body,
            }),
            BodyAction::InitLvlhRot {
                q_lvlh_body,
                ang_vel_lvlh_to_body,
                ang_vel_frame,
                reference_position,
                reference_velocity,
            } => Some(init_rot_from_lvlh(
                *q_lvlh_body,
                *ang_vel_lvlh_to_body,
                *ang_vel_frame,
                *reference_position,
                *reference_velocity,
            )),
            BodyAction::InitFullRelativeFrame {
                reference_frame,
                q_frame_subject,
                ang_vel_frame_to_subject,
                ..
            } => Some(init_rot_relative_to_frame(
                reference_frame,
                *q_frame_subject,
                *ang_vel_frame_to_subject,
            )),
            BodyAction::InitMass { .. }
            | BodyAction::InitTrans { .. }
            | BodyAction::InitTransOrbital { .. }
            | BodyAction::InitTransLvlh { .. }
            | BodyAction::InitTransRelativeFrame { .. }
            | BodyAction::InitTransNed { .. } => None,
        }
    }

    /// Returns the new mass properties if this action targets mass
    /// properties.
    pub fn apply_mass(&self) -> Option<MassProperties> {
        match self {
            BodyAction::InitMass { mass } => Some(*mass),
            BodyAction::InitTrans { .. }
            | BodyAction::InitRot { .. }
            | BodyAction::InitLvlhRot { .. }
            | BodyAction::InitTransOrbital { .. }
            | BodyAction::InitTransLvlh { .. }
            | BodyAction::InitTransRelativeFrame { .. }
            | BodyAction::InitFullRelativeFrame { .. }
            | BodyAction::InitTransNed { .. } => None,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "body-action recipe tests assert bit-exact recovery of literal-built mass values"
)]
mod tests {
    use super::*;

    #[test]
    fn init_mass_is_ready_and_returns_mass() {
        let mp = MassProperties::new(100.0);
        let action = BodyAction::InitMass { mass: mp };
        assert!(action.is_ready());
        let out = action.apply_mass().expect("mass present");
        assert_eq!(out.mass, mp.mass);
        assert!(action.apply_translational().is_none());
        assert!(action.apply_rotational().is_none());
    }

    #[test]
    fn init_trans_returns_state() {
        let state = TranslationalState {
            position: DVec3::new(7_000_000.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 7500.0, 0.0),
        };
        let action = BodyAction::InitTrans { state };
        assert!(action.is_ready());
        let out = action.apply_translational().expect("trans present");
        assert_eq!(out, state);
        assert!(action.apply_mass().is_none());
        assert!(action.apply_rotational().is_none());
    }

    #[test]
    fn init_rot_returns_state() {
        let action = BodyAction::InitRot {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        };
        let out = action.apply_rotational().expect("rot present");
        assert_eq!(out.quaternion, JeodQuat::identity());
        assert_eq!(out.ang_vel_body, DVec3::ZERO);
    }

    #[test]
    fn init_lvlh_rot_dispatches_to_kernel() {
        // Smoke test for the variant dispatch: identity LVLH→body with
        // zero LVLH-relative rate must produce a non-None
        // `RotationalState`, leave `apply_translational` / `apply_mass`
        // returning None, and report `is_ready() == true`. Kernel
        // correctness is covered by
        // `astrodyn_dynamics::body_init::tests::lvlh_rot_*`.
        const EARTH_MU: f64 = 3.986_004_415e14;
        let r = 6_778_137.0;
        let v = (EARTH_MU / r).sqrt();
        let action = BodyAction::InitLvlhRot {
            q_lvlh_body: JeodQuat::identity(),
            ang_vel_lvlh_to_body: DVec3::ZERO,
            ang_vel_frame: LvlhAngularVelocityFrame::Body,
            reference_position: DVec3::new(r, 0.0, 0.0),
            reference_velocity: DVec3::new(0.0, v, 0.0),
        };
        assert!(action.is_ready());
        let out = action.apply_rotational().expect("rot present");
        // `LvlhFrame::compute` produces a non-trivial
        // inertial→LVLH rotation; the identity LVLH→body input must
        // therefore *not* yield identity inertial→body.
        assert_ne!(out.quaternion, JeodQuat::identity());
        // LVLH frame is rotating wrt inertial at the orbital rate;
        // identity LVLH→body forwards that rate into the body frame.
        assert!(out.ang_vel_body.length() > 0.0);
        assert!(action.apply_translational().is_none());
        assert!(action.apply_mass().is_none());
    }

    #[test]
    fn init_trans_orbital_true_anom_round_trip() {
        const MU: f64 = 3.986_004_415e14;
        let mut elements = OrbitalElements::default();
        elements.semi_major_axis = 7.0e6;
        elements.e_mag = 0.001;
        elements.inclination = 51.6_f64.to_radians();
        elements.long_asc_node = 0.0;
        elements.arg_periapsis = 0.0;
        elements.true_anom = 0.5;
        let a = elements.semi_major_axis;
        let e = elements.e_mag;
        let action = BodyAction::InitTransOrbital {
            set: OrbitalElementSet::SmaEccIncAscnodeArgperTanom,
            elements,
            time_periapsis: 0.0,
            mu: MU,
        };
        let trans = action.apply_translational().expect("trans present");
        // Periapsis distance check: r >= a*(1-e) and r <= a*(1+e).
        let r = trans.position.length();
        let r_p = a * (1.0 - e);
        let r_a = a * (1.0 + e);
        assert!(r >= r_p - 1e-6);
        assert!(r <= r_a + 1e-6);
    }
}
