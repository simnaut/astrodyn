//! Typestate vehicle builder for the typed pipeline.
//!
//! [`VehicleBuilder`] gates configuration via four phantom states —
//! [`NeedsState`] → [`NeedsMass`] → [`HasIntegrator`] → [`Ready`] — so
//! that "forgot to set translational state" or "forgot to choose an
//! integrator" become **compile errors** instead of runtime panics.
//!
//! The output of [`VehicleBuilder::build`] is a [`TypedVehicleConfig`],
//! a typed companion of `jeod_runner::VehicleConfig`. Phase 6 of #101
//! will add the conversion from [`TypedVehicleConfig`] into the
//! `jeod_runner` simulation pipeline; until then the typestate builder
//! exists alongside the existing runtime `jeod_runner::VehicleBuilder`,
//! which remains the way to construct a `jeod_runner::Simulation`
//! (no rustdoc intra-link — `jeod_sim` does not depend on
//! `jeod_runner`, so the path can't be resolved here).
//!
//! # Compile-time gating
//!
//! ```compile_fail
//! use jeod_sim::vehicle_builder::VehicleBuilder;
//! // `.rk4()` is only available on `VehicleBuilder<HasIntegrator>` —
//! // calling it before `.with_translational()` and `.three_dof_point_mass()`
//! // is a compile error, not a runtime panic.
//! let _ = VehicleBuilder::new().rk4();
//! ```
//!
//! # Happy path
//!
//! ```ignore
//! use jeod_sim::vehicle_builder::VehicleBuilder;
//! use jeod_sim::EARTH;
//! use jeod_quantities::ext::F64Ext;
//!
//! let cfg = VehicleBuilder::new()
//!     .from_orbital_elements(elems, EARTH.mu_typed())
//!     .three_dof_point_mass(420_000.0.kg())
//!     .rk4()
//!     .build();
//! ```

use core::marker::PhantomData;

use jeod_dynamics::body_init::init_from_orbital_elements_typed;
use jeod_dynamics::state::TranslationalStateTyped;
use jeod_dynamics::{
    GaussJacksonConfig, IntegratorType, MassProperties, RotationalState, TranslationalState,
};
use jeod_gravity::{GravityControl, GravityControls};
use jeod_interactions::DragConfigTyped;
use jeod_math::OrbitalElements;
use jeod_quantities::dims::GravParam;
use jeod_quantities::frame::Inertial;
use uom::si::f64::{Angle, Length, Mass};

use crate::interactions::FlatPlateState;

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for the four states of the typestate
/// [`VehicleBuilder`]. Sealed downstream — implementors are limited to
/// [`NeedsState`], [`NeedsMass`], [`HasIntegrator`], [`Ready`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid `VehicleBuilder` state. Use \
        `NeedsState`, `NeedsMass`, `HasIntegrator`, or `Ready`.",
    label = "not a `BuildState`"
)]
pub trait BuildState: sealed::Sealed {}

/// Stage 0: nothing configured yet. Call
/// [`VehicleBuilder::with_translational`] or
/// [`VehicleBuilder::from_orbital_elements`] to advance.
pub struct NeedsState;
/// Stage 1: translational state set. Call
/// [`VehicleBuilder::three_dof_point_mass`] or
/// [`VehicleBuilder::sixdof`] to advance.
pub struct NeedsMass;
/// Stage 2: state and mass set. Choose an integrator with
/// [`VehicleBuilder::rk4`], [`VehicleBuilder::rkf45`],
/// [`VehicleBuilder::gauss_jackson`], or
/// [`VehicleBuilder::with_integrator`].
pub struct HasIntegrator;
/// Stage 3: fully configured. Optional features (gravity, drag, SRP)
/// can be added; [`VehicleBuilder::build`] is available.
pub struct Ready;

impl sealed::Sealed for NeedsState {}
impl sealed::Sealed for NeedsMass {}
impl sealed::Sealed for HasIntegrator {}
impl sealed::Sealed for Ready {}
impl BuildState for NeedsState {}
impl BuildState for NeedsMass {}
impl BuildState for HasIntegrator {}
impl BuildState for Ready {}

/// Typed companion of `jeod_runner::VehicleConfig` produced by
/// [`VehicleBuilder::build`].
///
/// Phase 5 deliverable. Phase 6 (#108) will provide the conversion
/// into the existing `jeod_runner` pipeline. The fields below cover
/// only the typestate-tracked configuration; mission features that
/// stayed in `jeod_runner` (shadow body, frame switches, mass tree,
/// derived states, earth lighting, …) are still configured through
/// `jeod_runner::VehicleBuilder`.
#[derive(Clone, Debug)]
pub struct TypedVehicleConfig {
    /// Translational state in the inertial frame.
    pub trans: TranslationalStateTyped<Inertial>,
    /// Optional rotational state (6-DOF when present, `None` for 3-DOF
    /// point-mass bodies).
    pub rot: Option<RotationalState>,
    /// Mass properties. Always populated — the typestate path through
    /// either [`VehicleBuilder::three_dof_point_mass`] or
    /// [`VehicleBuilder::sixdof`] sets it before
    /// [`VehicleBuilder::build`] is reachable.
    pub mass: MassProperties,
    /// Selected integrator (chosen explicitly via the typestate
    /// transition through [`HasIntegrator`]).
    pub integrator: IntegratorType,
    /// Per-body gravity controls.
    pub gravity_controls: GravityControls<usize>,
    /// Optional aerodynamic drag configuration (typed).
    pub drag: Option<DragConfigTyped>,
    /// Optional flat-plate solar-radiation-pressure state.
    pub flat_plate_srp: Option<FlatPlateState>,
}

/// Typestate vehicle builder. The `S: BuildState` parameter advances
/// through [`NeedsState`] → [`NeedsMass`] → [`HasIntegrator`] →
/// [`Ready`] as required configuration is supplied. Methods that
/// require a particular state are only in-scope for that state's
/// `impl` block, so missing-step calls are compile errors.
///
/// The order is `with_translational`/`from_orbital_elements` →
/// `three_dof_point_mass`/`sixdof` →
/// `rk4`/`rkf45`/`gauss_jackson`/`with_integrator` → `build`. See
/// module-level docs for examples.
pub struct VehicleBuilder<S: BuildState = NeedsState> {
    trans: Option<TranslationalStateTyped<Inertial>>,
    rot: Option<RotationalState>,
    mass: Option<MassProperties>,
    integrator: Option<IntegratorType>,
    gravity_controls: GravityControls<usize>,
    drag: Option<DragConfigTyped>,
    flat_plate_srp: Option<FlatPlateState>,
    _state: PhantomData<S>,
}

impl Default for VehicleBuilder<NeedsState> {
    fn default() -> Self {
        Self::new()
    }
}

impl VehicleBuilder<NeedsState> {
    /// Create a fresh builder. The compiler will require
    /// `.with_translational(...)` or `.from_orbital_elements(...)`
    /// before any mass / integrator method becomes available.
    pub fn new() -> Self {
        Self {
            trans: None,
            rot: None,
            mass: None,
            integrator: None,
            gravity_controls: GravityControls::default(),
            drag: None,
            flat_plate_srp: None,
            _state: PhantomData,
        }
    }

    /// Set the initial translational state (typed).
    pub fn with_translational(
        self,
        s: TranslationalStateTyped<Inertial>,
    ) -> VehicleBuilder<NeedsMass> {
        VehicleBuilder {
            trans: Some(s),
            rot: self.rot,
            mass: self.mass,
            integrator: self.integrator,
            gravity_controls: self.gravity_controls,
            drag: self.drag,
            flat_plate_srp: self.flat_plate_srp,
            _state: PhantomData,
        }
    }

    /// Set the initial translational state from Keplerian orbital
    /// elements and the central-body gravitational parameter.
    ///
    /// Delegates to
    /// [`init_from_orbital_elements_typed`](jeod_dynamics::body_init::init_from_orbital_elements_typed).
    /// The angles in `oe` are interpreted in radians, the semi-major
    /// axis in meters, and `mu` carries its `GravParam` dimension.
    pub fn from_orbital_elements(
        self,
        oe: OrbitalElements,
        mu: GravParam,
    ) -> VehicleBuilder<NeedsMass> {
        use uom::si::angle::radian;
        use uom::si::length::meter;
        let trans = init_from_orbital_elements_typed(
            Length::new::<meter>(oe.semi_major_axis),
            oe.e_mag,
            Angle::new::<radian>(oe.inclination),
            Angle::new::<radian>(oe.long_asc_node),
            Angle::new::<radian>(oe.arg_periapsis),
            Angle::new::<radian>(oe.true_anom),
            mu,
        );
        self.with_translational(trans)
    }
}

impl VehicleBuilder<NeedsMass> {
    /// Configure as 3-DoF point mass with the given total mass. No
    /// rotational state, no inertia tensor — the most common
    /// translational-only orbital case.
    pub fn three_dof_point_mass(self, mass: Mass) -> VehicleBuilder<HasIntegrator> {
        use uom::si::mass::kilogram;
        let mass_props = MassProperties::new(mass.get::<kilogram>());
        VehicleBuilder {
            trans: self.trans,
            rot: self.rot,
            mass: Some(mass_props),
            integrator: self.integrator,
            gravity_controls: self.gravity_controls,
            drag: self.drag,
            flat_plate_srp: self.flat_plate_srp,
            _state: PhantomData,
        }
    }

    /// Configure as full 6-DoF body with the given rotational state and
    /// mass properties (including inertia tensor).
    pub fn sixdof(
        self,
        rot: RotationalState,
        mass: MassProperties,
    ) -> VehicleBuilder<HasIntegrator> {
        VehicleBuilder {
            trans: self.trans,
            rot: Some(rot),
            mass: Some(mass),
            integrator: self.integrator,
            gravity_controls: self.gravity_controls,
            drag: self.drag,
            flat_plate_srp: self.flat_plate_srp,
            _state: PhantomData,
        }
    }
}

impl VehicleBuilder<HasIntegrator> {
    /// Use the standard 4-stage Runge-Kutta integrator.
    pub fn rk4(self) -> VehicleBuilder<Ready> {
        self.with_integrator(IntegratorType::Rk4)
    }

    /// Use the Runge-Kutta-Fehlberg 4(5) adaptive integrator.
    pub fn rkf45(self) -> VehicleBuilder<Ready> {
        self.with_integrator(IntegratorType::Rkf45)
    }

    /// Use the Gauss-Jackson predictor-corrector integrator.
    pub fn gauss_jackson(self, cfg: GaussJacksonConfig) -> VehicleBuilder<Ready> {
        self.with_integrator(IntegratorType::GaussJackson(cfg))
    }

    /// Use a caller-supplied integrator (Adams-Bashforth-Moulton,
    /// Gauss-Jackson with custom config, etc.).
    pub fn with_integrator(self, integrator: IntegratorType) -> VehicleBuilder<Ready> {
        VehicleBuilder {
            trans: self.trans,
            rot: self.rot,
            mass: self.mass,
            integrator: Some(integrator),
            gravity_controls: self.gravity_controls,
            drag: self.drag,
            flat_plate_srp: self.flat_plate_srp,
            _state: PhantomData,
        }
    }
}

impl VehicleBuilder<Ready> {
    /// Append a gravity control. May be called multiple times to add
    /// additional sources (point-mass third bodies, spherical-harmonics
    /// central body, …).
    pub fn gravity(mut self, control: GravityControl<usize>) -> Self {
        self.gravity_controls.controls.push(control);
        self
    }

    /// Configure aerodynamic drag (typed: Cd, area, optional density override).
    pub fn drag(mut self, cfg: DragConfigTyped) -> Self {
        self.drag = Some(cfg);
        self
    }

    /// Configure flat-plate solar radiation pressure with the given
    /// per-plate state (geometry, optical, thermal).
    pub fn flat_plate_srp(mut self, state: FlatPlateState) -> Self {
        self.flat_plate_srp = Some(state);
        self
    }

    /// Build the typed vehicle configuration. The unwraps below are
    /// safe because every required field was set during a state
    /// transition; if any required field is `None` the typestate
    /// would not have advanced to [`Ready`].
    pub fn build(self) -> TypedVehicleConfig {
        TypedVehicleConfig {
            trans: self
                .trans
                .expect("typestate guarantees translational state"),
            rot: self.rot,
            mass: self.mass.expect("typestate guarantees mass"),
            integrator: self.integrator.expect("typestate guarantees integrator"),
            gravity_controls: self.gravity_controls,
            drag: self.drag,
            flat_plate_srp: self.flat_plate_srp,
        }
    }
}

impl TypedVehicleConfig {
    /// Drop the frame phantoms and emit an untyped
    /// [`TranslationalState`]. Convenience for callers that need to
    /// hand the typed config to APIs not yet migrated to typed
    /// states. **The caller asserts** the translational state is
    /// expressed in `Inertial`.
    pub fn trans_untyped(&self) -> TranslationalState {
        self.trans.to_untyped()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jeod_quantities::ext::F64Ext;

    /// Happy path: 3-DoF point mass advances through every stage and
    /// `.build()` returns a populated [`TypedVehicleConfig`].
    #[test]
    fn three_dof_happy_path() {
        let trans =
            TranslationalStateTyped::<Inertial>::from_untyped_unchecked(&TranslationalState {
                position: glam::DVec3::new(7_000_000.0, 0.0, 0.0),
                velocity: glam::DVec3::new(0.0, 7_500.0, 0.0),
            });
        let cfg = VehicleBuilder::new()
            .with_translational(trans)
            .three_dof_point_mass(420_000.0.kg())
            .rk4()
            .build();
        assert_eq!(cfg.integrator, IntegratorType::Rk4);
        assert_eq!(cfg.mass.mass, 420_000.0);
        assert!(cfg.rot.is_none());
    }

    /// 6-DoF path produces a config with both rotational state and
    /// mass populated.
    #[test]
    fn six_dof_happy_path() {
        use jeod_math::JeodQuat;
        let trans =
            TranslationalStateTyped::<Inertial>::from_untyped_unchecked(&TranslationalState {
                position: glam::DVec3::new(7_000_000.0, 0.0, 0.0),
                velocity: glam::DVec3::new(0.0, 7_500.0, 0.0),
            });
        let rot = RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: glam::DVec3::ZERO,
        };
        let inertia = glam::DMat3::IDENTITY * 100.0;
        let mass = MassProperties::with_inertia(420_000.0, inertia, glam::DVec3::ZERO);
        let cfg = VehicleBuilder::new()
            .with_translational(trans)
            .sixdof(rot, mass)
            .rk4()
            .build();
        assert!(cfg.rot.is_some());
        assert_eq!(cfg.mass.mass, 420_000.0);
    }
}
