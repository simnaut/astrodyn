//! ECS-agnostic body actions: queueable mutations to a vehicle's
//! translational state, rotational state, and mass properties.
//!
//! Port of JEOD's
//! [`models/dynamics/body_action/`](https://github.com/nasa/jeod/blob/jeod_v5.4.0/models/dynamics/body_action/)
//! family of `BodyAction` subclasses. The variants currently ported are
//! `MassBodyInit`, `DynBodyInitTransState`, `DynBodyInitRotState`,
//! `DynBodyInitOrbit`, `DynBodyInitLvlhTransState`, and
//! `DynBodyInitNedTransState`. JEOD's `DynBodyInitLvlhRotState`
//! (LVLH-relative rotational-state init) is not yet ported.
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
//! The Bevy wiring lives in `bevy_jeod::body_action` (a `Commands`
//! extension that queues actions and a system that drains the queue
//! each tick before [`PipelineStage::Environment`](crate::PipelineStage)).

use glam::{DMat3, DVec3};

use jeod_dynamics::body_init::{
    init_from_lvlh, init_from_mean_anomaly, init_from_ned, init_from_orbital_elements,
    init_from_time_periapsis,
};
use jeod_dynamics::{MassProperties, RotationalState, TranslationalState};
use jeod_math::{GeodeticState, JeodQuat, OrbitalElements};
use jeod_quantities::frame::SelfPlanet;

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
#[derive(Debug, Clone)]
pub enum BodyAction {
    /// Replace the subject's mass properties.
    ///
    /// JEOD analog: `MassBodyInit::apply` (mass / position / inertia
    /// install with the configured `inertia_spec`). We accept a
    /// pre-resolved `MassProperties` because the JEOD spec
    /// transformations (`Body`, `StructCG`, `Struct`, `Spec`,
    /// `SpecCG`) are already factored into
    /// [`jeod_dynamics::MassProperties`] constructors and the
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

    /// Replace the subject's translational state from Keplerian
    /// orbital elements.
    ///
    /// JEOD analog: `DynBodyInitOrbit`. Delegates to
    /// [`jeod_dynamics::body_init::init_from_orbital_elements`] /
    /// [`jeod_dynamics::body_init::init_from_mean_anomaly`] /
    /// [`jeod_dynamics::body_init::init_from_time_periapsis`]
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
            BodyAction::InitMass { .. } | BodyAction::InitRot { .. } => None,
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
            BodyAction::InitMass { .. }
            | BodyAction::InitTrans { .. }
            | BodyAction::InitTransOrbital { .. }
            | BodyAction::InitTransLvlh { .. }
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
            | BodyAction::InitTransOrbital { .. }
            | BodyAction::InitTransLvlh { .. }
            | BodyAction::InitTransNed { .. } => None,
        }
    }
}

#[cfg(test)]
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
