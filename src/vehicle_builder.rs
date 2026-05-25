// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Typestate vehicle builder for the typed pipeline.
//!
//! [`VehicleBuilder`] gates configuration via four phantom states —
//! [`NeedsState`] → [`NeedsMass`] → [`HasIntegrator`] → [`Ready`] — so
//! that "forgot to set translational state" or "forgot to choose an
//! integrator" become **compile errors** instead of runtime panics.
//!
//! The output of [`VehicleBuilder::build`] is a [`VehicleConfig`] —
//! the same descriptor that `SimulationBuilder::add_body` accepts and
//! that `Simulation::body` carries through the runner's step loop.
//! Phase 6 of #101 consolidated the runner-side runtime fluent
//! `astrodyn_runner::VehicleBuilder` into this typestate builder, so every
//! method that builder offered is available here in the `Ready` impl.
//!
//! # Compile-time gating
//!
//! ```compile_fail
//! use astrodyn::vehicle_builder::VehicleBuilder;
//! // `.rk4()` is only available on `VehicleBuilder<HasIntegrator>` —
//! // calling it before `.with_translational()` and `.three_dof_point_mass()`
//! // is a compile error, not a runtime panic.
//! let _ = VehicleBuilder::new().rk4();
//! ```
//!
//! # Happy path
//!
//! ```
//! use astrodyn::vehicle_builder::VehicleBuilder;
//! use astrodyn::recipes::{constants, orbital_elements};
//! use astrodyn_quantities::ext::F64Ext;
//!
//! let cfg = VehicleBuilder::new()
//!     // `orbital_elements::iss()` returns `OrbitalElements<Earth>` and
//!     // `constants::mu_ggm05c()` returns `GravParam<Earth>` — sharing
//!     // the planet phantom is what `from_orbital_elements` requires.
//!     .from_orbital_elements(orbital_elements::iss(), constants::mu_ggm05c())
//!     .three_dof_point_mass(420_000.0.kg())
//!     .rk4()
//!     .build();
//! assert!(cfg.mass.is_some());
//! ```

use core::marker::PhantomData;

use glam::DMat3;

use astrodyn_dynamics::body_init::init_from_orbital_elements_typed;
use astrodyn_dynamics::state::TranslationalStateTyped;
use astrodyn_dynamics::{
    MassProperties, MassPropertiesTyped, RotationalState, RotationalStateTyped,
};

use crate::integrator::{GaussJacksonConfig, IntegratorType};
use astrodyn_gravity::{GravityControl, GravityControls};
use astrodyn_interactions::DragConfig;
use astrodyn_math::{EulerSequence, OrbitalElements};
use astrodyn_quantities::aliases::{AngularVelocity, InertiaTensor, Position};
use astrodyn_quantities::body_attitude::BodyAttitude;
use astrodyn_quantities::dims::GravParam;
use astrodyn_quantities::frame::{BodyFrame, RootInertial, StructuralFrame};
use uom::si::f64::{Angle, Length, Mass};
use uom::si::mass::kilogram;

use astrodyn_quantities::frame::SelfRef;

use crate::interactions::FlatPlateState;
use crate::planet_config::PlanetConfig;
use crate::source_handle::SourceHandle;
use crate::vehicle_config::{
    DerivedStateConfig, EarthLightingConfig, FrameSwitchConfig, GeodeticConfig, ShadowBody,
    SrpModel, VehicleConfig,
};

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
///
/// Phase 6 of #101 consolidated this typestate builder with the runtime
/// fluent builder previously in `astrodyn_runner::builder`: every method that
/// builder offered is now in the `Ready` impl block here, plus the
/// compile-time gating from Phase 5. `build()` returns the full
/// [`VehicleConfig`] (the type that goes into `SimulationBuilder::add_body`
/// and through `Simulation`).
pub struct VehicleBuilder<S: BuildState = NeedsState> {
    trans: Option<TranslationalStateTyped<RootInertial>>,
    rot: Option<RotationalStateTyped<SelfRef>>,
    mass: Option<MassPropertiesTyped<SelfRef>>,
    integrator: Option<IntegratorType>,
    t_struct_body: DMat3,
    gravity_controls: GravityControls<usize>,
    compute_gravity_gradient: bool,
    drag: Option<DragConfig>,
    srp: Option<SrpModel>,
    shadow_body: Option<ShadowBody>,
    derived: DerivedStateConfig,
    external_force: astrodyn_quantities::aliases::Force<RootInertial>,
    external_torque: astrodyn_quantities::aliases::Torque<
        astrodyn_quantities::frame::BodyFrame<astrodyn_quantities::frame::SelfRef>,
    >,
    integ_source: Option<usize>,
    frame_switches: Vec<FrameSwitchConfig>,
    _state: PhantomData<S>,
}

impl Default for VehicleBuilder<NeedsState> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: BuildState> VehicleBuilder<S> {
    fn empty() -> VehicleBuilder<NeedsState> {
        VehicleBuilder {
            trans: None,
            rot: None,
            mass: None,
            integrator: None,
            t_struct_body: DMat3::IDENTITY,
            gravity_controls: GravityControls::default(),
            compute_gravity_gradient: false,
            drag: None,
            srp: None,
            shadow_body: None,
            derived: DerivedStateConfig::default(),
            external_force: astrodyn_quantities::aliases::Force::<RootInertial>::zero(),
            external_torque: astrodyn_quantities::aliases::Torque::<
                astrodyn_quantities::frame::BodyFrame<astrodyn_quantities::frame::SelfRef>,
            >::zero(),
            integ_source: None,
            frame_switches: Vec::new(),
            _state: PhantomData,
        }
    }

    fn transition<T: BuildState>(self) -> VehicleBuilder<T> {
        VehicleBuilder {
            trans: self.trans,
            rot: self.rot,
            mass: self.mass,
            integrator: self.integrator,
            t_struct_body: self.t_struct_body,
            gravity_controls: self.gravity_controls,
            compute_gravity_gradient: self.compute_gravity_gradient,
            drag: self.drag,
            srp: self.srp,
            shadow_body: self.shadow_body,
            derived: self.derived,
            external_force: self.external_force,
            external_torque: self.external_torque,
            integ_source: self.integ_source,
            frame_switches: self.frame_switches,
            _state: PhantomData,
        }
    }
}

impl VehicleBuilder<NeedsState> {
    /// Create a fresh builder. The compiler will require
    /// `.with_translational(...)` or `.from_orbital_elements(...)`
    /// before any mass / integrator method becomes available.
    pub fn new() -> Self {
        Self::empty()
    }

    /// Set the initial translational state (typed).
    pub fn with_translational(
        mut self,
        s: TranslationalStateTyped<RootInertial>,
    ) -> VehicleBuilder<NeedsMass> {
        self.trans = Some(s);
        self.transition()
    }

    /// Set the initial translational state from Keplerian orbital
    /// elements and the central-body gravitational parameter.
    ///
    /// Delegates to
    /// [`astrodyn_dynamics::body_init::init_from_orbital_elements_typed`].
    /// The angles in `oe` are interpreted in radians, the semi-major
    /// axis in meters, and `mu` carries its `GravParam<P>` dimension.
    /// Both inputs share the planet phantom `P`, so `mu_sun()` cannot
    /// be paired with `iss()` (Earth-tagged) — the compiler refuses.
    pub fn from_orbital_elements<P: astrodyn_quantities::frame::Planet>(
        self,
        oe: OrbitalElements<P>,
        mu: GravParam<P>,
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
    pub fn three_dof_point_mass(mut self, mass: Mass) -> VehicleBuilder<HasIntegrator> {
        self.mass = Some(MassPropertiesTyped::<SelfRef>::new(mass));
        self.transition()
    }

    /// Configure as full 6-DoF body with the given rotational state and
    /// mass properties (including inertia tensor). Untyped inputs are
    /// lifted into the typed siblings at this boundary via inline
    /// construction (the implicit `From<Untyped>` bypass and the
    /// `from_untyped_unchecked` named helpers were removed in #397).
    pub fn sixdof(
        mut self,
        rot: RotationalState,
        mass: MassProperties,
    ) -> VehicleBuilder<HasIntegrator> {
        // allowed: typed↔raw kernel-boundary lift at the public API
        // boundary into the typed builder state.
        self.rot = Some(RotationalStateTyped::<SelfRef>::new(
            BodyAttitude::from_jeod_quat(rot.quaternion),
            AngularVelocity::<BodyFrame<SelfRef>>::from_raw_si(rot.ang_vel_body), // allowed: typed↔raw kernel boundary
        ));
        let typed_mass = MassPropertiesTyped::<SelfRef>::with_inertia(
            Mass::new::<kilogram>(mass.mass),
            InertiaTensor::<BodyFrame<SelfRef>>::from_dmat3_unchecked(mass.inertia), // allowed: typed↔raw kernel boundary
            Position::<StructuralFrame<SelfRef>>::from_raw_si(mass.position), // allowed: typed↔raw kernel boundary
        )
        .with_t_parent_this(mass.t_parent_this);
        self.mass = Some(typed_mass);
        self.transition()
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
    pub fn with_integrator(mut self, integrator: IntegratorType) -> VehicleBuilder<Ready> {
        self.integrator = Some(integrator);
        self.transition()
    }
}

impl VehicleBuilder<Ready> {
    // ── Gravity ──

    /// Append a gravity control. May be called multiple times to add
    /// additional sources (point-mass third bodies, spherical-harmonics
    /// central body, …).
    pub fn gravity(mut self, control: GravityControl<usize>) -> Self {
        self.gravity_controls.controls.push(control);
        self
    }

    /// Enable gravity gradient computation (needed for gravity torque).
    pub fn gravity_gradient(mut self) -> Self {
        self.compute_gravity_gradient = true;
        self
    }

    // ── Interactions ──

    /// Configure aerodynamic drag (Cd, area, optional density override).
    pub fn drag(mut self, cfg: DragConfig) -> Self {
        self.drag = Some(cfg);
        self
    }

    /// Configure flat-plate solar radiation pressure with the given
    /// per-plate state (geometry, optical, thermal).
    ///
    /// `FlatPlateState<SelfRef>` is the runtime-resolved instantiation
    /// at the adapter boundary; the underlying
    /// [`astrodyn_interactions::FlatPlate<V>`] retains the
    /// `<V: Vehicle>` phantom for mission code that pins a concrete
    /// vehicle.
    pub fn flat_plate_srp(
        mut self,
        state: FlatPlateState<astrodyn_quantities::frame::SelfRef>,
    ) -> Self {
        self.srp = Some(SrpModel::FlatPlate(state));
        self
    }

    /// Configure cannonball solar radiation pressure.
    pub fn cannonball_srp(mut self, cx_area: f64, albedo: f64, diffuse: f64) -> Self {
        self.srp = Some(SrpModel::Cannonball {
            cx_area,
            albedo,
            diffuse,
        });
        self
    }

    /// Set the shadow-casting body for SRP eclipse computation. Uses
    /// [`PlanetConfig::shadow_radius`] for consistent radius.
    ///
    /// `source` is `impl Into<SourceHandle>` — bare `usize` callsites
    /// continue to work via the `From<usize> for SourceHandle` blanket;
    /// new mission code can prefer `SourceHandle::central()` /
    /// `SourceHandle::index(n)` for self-documenting intent.
    pub fn shadow(mut self, source: impl Into<SourceHandle>, planet: &PlanetConfig) -> Self {
        self.shadow_body = Some(ShadowBody {
            source_idx: source.into().into_raw(),
            radius: planet.shadow_radius,
        });
        self
    }

    /// Set the shadow-casting body with explicit radius. See
    /// [`Self::shadow`] for the `impl Into<SourceHandle>` rationale.
    pub fn shadow_with_radius(mut self, source: impl Into<SourceHandle>, radius: f64) -> Self {
        self.shadow_body = Some(ShadowBody {
            source_idx: source.into().into_raw(),
            radius,
        });
        self
    }

    // ── Frame transforms / external loads ──

    /// Set the structural-to-body frame rotation (default: identity).
    pub fn structural_transform(mut self, t: DMat3) -> Self {
        self.t_struct_body = t;
        self
    }

    /// Set initial external force (root-inertial frame, N).
    pub fn external_force(mut self, f: astrodyn_quantities::aliases::Force<RootInertial>) -> Self {
        self.external_force = f;
        self
    }

    /// Set initial external torque (body frame, N·m).
    pub fn external_torque(
        mut self,
        t: astrodyn_quantities::aliases::Torque<
            astrodyn_quantities::frame::BodyFrame<astrodyn_quantities::frame::SelfRef>,
        >,
    ) -> Self {
        self.external_torque = t;
        self
    }

    // ── Frame switching ──

    /// Set the initial integration source (default: simulation root /
    /// central body). `source` identifies the gravity-source-table
    /// entry returned by `SimulationBuilder::add_source()`.
    ///
    /// `source` is `impl Into<SourceHandle>` — bare `usize` callsites
    /// continue to work via the `From<usize> for SourceHandle` blanket;
    /// new mission code can prefer `SourceHandle::central()` /
    /// `SourceHandle::index(n)` for self-documenting intent.
    ///
    /// When set to a non-root source, the body's translational state
    /// lives in that source's planet-inertial frame rather than the
    /// root inertial frame. The Bevy adapter expresses this
    /// structurally: `TranslationalStateC<P>` carries the
    /// `PlanetInertial<P>` phantom, and the caller pins `P` to the
    /// integration source's planet via the `spawn_bevy::<P>(...)`
    /// turbofish; the runner threads the same planet identity through
    /// its per-body integration-frame plumbing. Shift-site consumers
    /// that require root-inertial coordinates — gravity, SRP, solar
    /// beta, earth lighting, relativistic corrections — lift the
    /// integ-origin offset at the call site through
    /// `FrameOrigin::shift_position` (`docs/JEOD_invariants.md` row
    /// `RF.10`); non-shift consumers (atmosphere, drag, LVLH,
    /// geodetic, orbital elements) stay in `PlanetInertial<P>`
    /// throughout. Mismatching `P` against the integration source's
    /// planet is a compile error, not a runtime convention.
    pub fn integ_source(mut self, source: impl Into<SourceHandle>) -> Self {
        self.integ_source = Some(source.into().into_raw());
        self
    }

    /// Set distance-based frame switch triggers. Each
    /// [`FrameSwitchConfig`]'s `target_source` is a `usize` source-table
    /// index at this storage boundary; mission code can construct it
    /// via `SourceHandle::index(n).into()` to keep the typed-handle
    /// vocabulary consistent across the builder surface.
    pub fn frame_switches(mut self, switches: Vec<FrameSwitchConfig>) -> Self {
        self.frame_switches = switches;
        self
    }

    // ── Derived states ──

    /// Compute orbital elements relative to the given gravity source.
    /// See [`Self::shadow`] for the `impl Into<SourceHandle>` rationale.
    pub fn orbital_elements(mut self, source: impl Into<SourceHandle>) -> Self {
        self.derived.orbital_elements_source = Some(source.into().into_raw());
        self
    }

    /// Compute Euler angles with the given decomposition sequence.
    pub fn euler_angles(mut self, sequence: EulerSequence) -> Self {
        self.derived.euler_sequence = Some(sequence);
        self
    }

    /// Compute LVLH frame each step.
    pub fn lvlh(mut self) -> Self {
        self.derived.lvlh = true;
        self
    }

    /// Compute geodetic state. Uses [`PlanetConfig`] for consistent radii.
    /// See [`Self::shadow`] for the `impl Into<SourceHandle>` rationale.
    pub fn geodetic(mut self, source: impl Into<SourceHandle>, planet: &PlanetConfig) -> Self {
        self.derived.geodetic = Some(GeodeticConfig {
            source_idx: source.into().into_raw(),
            r_eq: planet.shape.r_eq(),
            r_pol: planet.shape.r_pol(),
        });
        self
    }

    /// Compute solar beta angle. Requires `sun_source` on the simulation.
    pub fn solar_beta(mut self) -> Self {
        self.derived.solar_beta = true;
        self
    }

    /// Compute earth lighting. Uses [`PlanetConfig`] presets for radii.
    /// Requires `sun_source` and `moon_source` on the simulation.
    pub fn earth_lighting(
        mut self,
        earth: &PlanetConfig,
        moon: &PlanetConfig,
        sun: &PlanetConfig,
    ) -> Self {
        self.derived.earth_lighting = Some(EarthLightingConfig {
            earth_radius: earth.shape.r_eq(),
            moon_radius: moon.shape.r_eq(),
            sun_radius: sun.shape.r_eq(),
        });
        self
    }

    // ── Build ──

    /// Build the [`VehicleConfig`]. The unwraps below are safe because
    /// every required field was set during a state transition; if any
    /// required field were `None`, the typestate would not have
    /// advanced to [`Ready`].
    pub fn build(self) -> VehicleConfig {
        VehicleConfig {
            trans: self
                .trans
                .expect("typestate guarantees translational state"),
            rot: self.rot,
            // Mass is `Option<MassProperties>` in the storage type
            // (so direct struct-literal construction stays
            // ergonomic), but the typestate path through either
            // `three_dof_point_mass` or `sixdof` always populates it
            // before `Ready`. Unwrap-and-rewrap so an internal
            // typestate bug surfaces here rather than as a silent
            // `mass = None` that fails later in validation or
            // physics.
            mass: Some(self.mass.expect("typestate guarantees mass")),
            integrator: self.integrator.expect("typestate guarantees integrator"),
            t_struct_body: self.t_struct_body,
            gravity_controls: self.gravity_controls,
            compute_gravity_gradient: self.compute_gravity_gradient,
            drag: self.drag,
            srp: self.srp,
            shadow_body: self.shadow_body,
            derived: self.derived,
            external_force: self.external_force,
            external_torque: self.external_torque,
            integ_source: self.integ_source,
            frame_switches: self.frame_switches,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "VehicleBuilder typestate tests assert bit-exact recovery of literal-built mass values"
)]
mod tests {
    use super::*;
    use astrodyn_quantities::ext::F64Ext;
    use glam::DVec3;

    fn iss_trans() -> TranslationalStateTyped<RootInertial> {
        use astrodyn_quantities::aliases::{Position, Velocity};
        TranslationalStateTyped::<RootInertial> {
            position: Position::<RootInertial>::from_raw_si(DVec3::new(7_000_000.0, 0.0, 0.0)),
            velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, 7_500.0, 0.0)),
        }
    }

    /// Happy path: 3-DoF point mass advances through every stage and
    /// `.build()` returns a populated [`VehicleConfig`].
    #[test]
    fn three_dof_happy_path() {
        let cfg = VehicleBuilder::new()
            .with_translational(iss_trans())
            .three_dof_point_mass(420_000.0.kg())
            .rk4()
            .build();
        assert_eq!(cfg.integrator, IntegratorType::Rk4);
        assert_eq!(
            cfg.mass
                .expect("mass set by typestate")
                .mass
                .get::<uom::si::mass::kilogram>(),
            420_000.0
        );
        assert!(cfg.rot.is_none());
    }

    /// 6-DoF path produces a config with both rotational state and mass
    /// populated.
    #[test]
    fn six_dof_happy_path() {
        use astrodyn_math::JeodQuat;
        let rot = RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        };
        let inertia = DMat3::IDENTITY * 100.0;
        let mass = MassProperties::with_inertia(420_000.0, inertia, DVec3::ZERO);
        let cfg = VehicleBuilder::new()
            .with_translational(iss_trans())
            .sixdof(rot, mass)
            .rk4()
            .build();
        assert!(cfg.rot.is_some());
        assert_eq!(
            cfg.mass
                .expect("mass set by typestate")
                .mass
                .get::<uom::si::mass::kilogram>(),
            420_000.0
        );
    }

    /// Ready-state methods (drag, gravity gradient, derived states, etc.)
    /// are reachable after the typestate completes, mirroring the runtime
    /// fluent builder's surface absorbed in Phase 6.
    #[test]
    fn ready_state_full_surface() {
        use astrodyn_interactions::DragConfig;
        use astrodyn_quantities::aliases::{Position, Velocity};
        let cfg = VehicleBuilder::new()
            .with_translational(TranslationalStateTyped::<RootInertial> {
                position: Position::<RootInertial>::from_raw_si(DVec3::new(7_000_000.0, 0.0, 0.0)),
                velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, 7_500.0, 0.0)),
            })
            .three_dof_point_mass(1_000.0.kg())
            .rk4()
            .gravity_gradient()
            .drag(DragConfig {
                cd: 2.2,
                area: 1.0,
                constant_density: None,
                ..Default::default()
            })
            .lvlh()
            .solar_beta()
            .build();
        assert!(cfg.compute_gravity_gradient);
        assert!(cfg.drag.is_some());
        assert!(cfg.derived.lvlh);
        assert!(cfg.derived.solar_beta);
    }

    /// Builder methods that name a gravity source — `integ_source`,
    /// `geodetic`, `orbital_elements`, `shadow` — accept both
    /// `SourceHandle::central()` / `SourceHandle::index(n)` and bare
    /// `usize` indices in the same chain. Both shapes resolve to
    /// identical underlying state because `SourceHandle` is a thin
    /// newtype around the source-table index.
    #[test]
    fn builder_accepts_source_handle_and_bare_usize_interchangeably() {
        let typed = VehicleBuilder::new()
            .with_translational(iss_trans())
            .three_dof_point_mass(420_000.0.kg())
            .rk4()
            .integ_source(SourceHandle::central())
            .orbital_elements(SourceHandle::central())
            .shadow(SourceHandle::index(2), &crate::EARTH)
            .geodetic(SourceHandle::central(), &crate::EARTH)
            .build();
        let untyped = VehicleBuilder::new()
            .with_translational(iss_trans())
            .three_dof_point_mass(420_000.0.kg())
            .rk4()
            .integ_source(0_usize)
            .orbital_elements(0_usize)
            .shadow(2_usize, &crate::EARTH)
            .geodetic(0_usize, &crate::EARTH)
            .build();

        assert_eq!(typed.integ_source, untyped.integ_source);
        assert_eq!(typed.integ_source, Some(0));
        assert_eq!(
            typed.derived.orbital_elements_source,
            untyped.derived.orbital_elements_source
        );
        assert_eq!(typed.derived.orbital_elements_source, Some(0));
        assert_eq!(
            typed.shadow_body.expect("shadow set above").source_idx,
            untyped.shadow_body.expect("shadow set above").source_idx
        );
        assert_eq!(typed.shadow_body.expect("shadow set above").source_idx, 2);
        assert_eq!(
            typed
                .derived
                .geodetic
                .expect("geodetic set above")
                .source_idx,
            untyped
                .derived
                .geodetic
                .expect("geodetic set above")
                .source_idx
        );
    }
}
