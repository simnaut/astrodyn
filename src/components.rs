//! Bevy `Component` newtypes wrapping `jeod_sim` typed siblings (state,
//! mass, gravity controls, interactions, derived states). Each component
//! is `#[reflect(opaque, Component)]` so it appears in Bevy's type
//! registry as a leaf of its `jeod_*` inner type.

use bevy::prelude::*;
use glam::DVec3;
use jeod_sim::{
    Angle, AngularVelocity, BodyFrame, DragConfig, DragConfigTyped, DynamicsConfig,
    FrameDerivatives, FrameDerivativesTyped, FrameTransform, GravityAcceleration,
    GravityAccelerationTyped, GravityControls, GravitySource, MassProperties, MassPropertiesTyped,
    Planet, PlanetFixed, PlanetInertial, PlanetShape, Position, Ratio, RootInertial,
    RotationalState, RotationalStateTyped, SelfPlanet, SelfRef, StructuralFrame, Torque,
    TotalForce, TotalForceTyped, TranslationalState, TranslationalStateTyped, Velocity,
};

// ── Dynamics ──
//
// Spatial Components wrap the **typed siblings** from `jeod_dynamics`,
// not the raw untyped storage. The frame phantoms (`RootInertial`,
// `BodyFrame<SelfRef>`, `StructuralFrame<SelfRef>`) are baked into the
// component at the type level, so systems read typed values directly
// without the per-step `from_raw_si` lifts that the audit's #172 H1
// flagged as the load-bearing failure mode of the typed-quantity
// facade. Mission code that mutates `c.0.position` directly via raw
// `DVec3` is now a compile error — the typed accessor `Position<RootInertial>`
// surfaces the convention as a type, not just a comment.
//
// `From<Untyped>` impls are provided on every spatial Component so
// existing test/example code that constructs `TranslationalStateC(state)`
// from an untyped `TranslationalState` switches to
// `TranslationalStateC::from(state)` without other changes.

/// Translational state (position, velocity) for the body being
/// integrated. Wraps a typed
/// [`TranslationalStateTyped<PlanetInertial<SelfPlanet>>`](TranslationalStateTyped)
/// sibling so the frame phantom is enforced at the type level.
///
/// # Frame semantics: planet-inertial, not root-inertial
///
/// The frame phantom is [`PlanetInertial<SelfPlanet>`] (a particular
/// planet's inertial frame, with the wildcard `SelfPlanet` standing
/// in for "this entity's planet"). Two relabel categories apply at
/// consumer call sites, and they are independent — a consumer may
/// need one, both, or neither:
///
/// 1. **Integ-origin shift** (arithmetic — adds the integ-origin
///    offset and relabels the phantom to `RootInertial`). Required by
///    consumers that mix the body's state with root-inertial source
///    positions: gravity, relativistic, SRP, solar beta, earth
///    lighting — the "shift sites" per RF.10. The runner's
///    [`crate::frame_param::FrameOrigin`] SystemParam supplies the
///    offset and the gravity / integration / SRP systems perform the
///    shift at the call site.
/// 2. **Wildcard → concrete planet relabel** (phantom-only,
///    bit-identical, no arithmetic). Required whenever a typed kernel
///    is parameterized over a concrete `PlanetInertial<P>` (e.g.
///    `PlanetInertial<Earth>`) rather than the wildcard
///    `SelfPlanet`. Callers do `Position::<PlanetInertial<P>>::
///    from_raw_si(state.position.raw_si())` at the boundary; the
///    underlying SI coordinates are preserved exactly.
///
/// Atmosphere/drag, LVLH, geodetic, and orbital-elements consumers
/// do **not** apply the integ-origin shift (they live in
/// planet-inertial throughout). Drag, LVLH, and geodetic still
/// apply category 2 to satisfy a concrete-planet typed kernel
/// signature — for them, calling the path "no relabel" would be
/// misleading because the phantom-tag attachment at the call site
/// *is* the relabel. Orbital elements is the one exception in this
/// group: its kernel
/// [`jeod_sim::compute_orbital_elements`] is planet-erased and
/// returns `OrbitalElements<SelfPlanet>` directly, so the
/// `OrbitalElementsC` storage matches the `<SelfPlanet>` source
/// state without any relabel. This was the original "kernels generic
/// over `SelfPlanet` avoid both categories" exit, now realized by one
/// consumer; atmosphere also avoids both categories, but by feeding
/// the kernel raw `DVec3` via `state.position.raw_si()` rather than
/// by being `<SelfPlanet>`-typed end-to-end.
///
/// For root-integrated bodies (`IntegSourceC` is `None` or omitted)
/// the integ-origin shift is zero, so the planet-inertial coordinates
/// numerically equal root-inertial; the typed phantom stays distinct
/// so arithmetic mixing the body state with a
/// `Position<RootInertial>` gravity-source position still requires
/// the explicit shift the runner already performs.
///
/// # `<SelfPlanet>` wildcard
///
/// `SelfPlanet` is the planet-side analog of `SelfRef` — the
/// "wildcard" tag the Bevy adapter uses when a Component carries a
/// frame phantom whose planet identity is determined at runtime by
/// the entity itself. The Component is non-generic so that
/// `Query<&TranslationalStateC>` resolves to a single concrete type
/// (Bevy queries cannot range over a free generic parameter), and
/// every body's integration-frame planet is recovered at runtime via
/// the body frame entity's `ChildOf` chain plus
/// [`crate::frame_param::FrameOrigin`].
///
/// Mission code that needs compile-time planet identity (e.g. a
/// Mars-orbit chief plus an Earth-orbit deputy in the same `World`)
/// should use the typed sibling
/// [`TranslationalStateTyped<PlanetInertial<P>>`](TranslationalStateTyped)
/// directly off-component, or a small mission-side wrapper Component
/// pinning `P`. The `<SelfPlanet>` storage is a zero-cost relabel
/// from any `<PlanetInertial<P>>` typed value via
/// [`Self::from_planet_inertial`] (which preserves the dimensional
/// SI coordinates exactly).
// JEOD_INV: DB.24 — default integrated_frame is composite_body (we integrate composite_body state)
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct TranslationalStateC(pub TranslationalStateTyped<PlanetInertial<SelfPlanet>>);

impl TranslationalStateC {
    /// Wrap an untyped [`TranslationalState`] as the typed Bevy
    /// Component. The caller asserts the values are in some planet's
    /// inertial frame: for non-root-integrated bodies this is the
    /// body's `IntegSourceC` planet; for root-integrated bodies the
    /// integration frame is the simulation's [`RootInertial`] frame.
    ///
    /// `RootInertial` is kind-distinct from `PlanetInertial<P>` in
    /// `jeod_quantities` — they are not unifiable types even when
    /// numerically coincident. In Earth-centered setups the runner
    /// conventionally treats the root inertial frame as numerically
    /// coincident with the central body's `PlanetInertial<Earth>`
    /// frame, but the type system keeps the two markers distinct;
    /// crossing between them is via the explicit integ-origin shift
    /// at RF.10 shift sites, not type unification. The wildcard
    /// `<SelfPlanet>` tag on this Component sidesteps that distinction
    /// at storage time — every body's storage is tagged
    /// `PlanetInertial<SelfPlanet>` regardless of integration frame.
    /// Consumers then either apply the integ-origin shift to relabel
    /// the phantom to `RootInertial` (gravity, solar beta, SRP, earth
    /// lighting), or stay within the planet-inertial flavor and do a
    /// wildcard → concrete-planet phantom relabel at the typed-kernel
    /// boundary (drag, LVLH, geodetic). Atmosphere and orbital
    /// elements stay within planet-inertial *without* any relabel —
    /// atmosphere because it consumes raw `DVec3` via
    /// `state.position.raw_si()`, orbital elements because its kernel
    /// is planet-erased and produces `OrbitalElements<SelfPlanet>`
    /// directly. See the type-level docstring's "Frame semantics"
    /// section for the two-category breakdown.
    ///
    /// No runtime check is performed; the conversion is a zero-cost
    /// type-tag attachment via
    /// [`TranslationalStateTyped::from_untyped_unchecked`].
    #[inline]
    pub fn from_untyped(state: TranslationalState) -> Self {
        Self(TranslationalStateTyped::<PlanetInertial<SelfPlanet>>::from_untyped_unchecked(&state))
    }

    /// Witness-gated constructor: wrap an already-typed
    /// [`TranslationalStateTyped`] expressed in `P`'s planet-inertial
    /// frame as the wildcard-tagged Component. The witness is the
    /// caller's compile-time choice of `P` plus the typed phantoms on
    /// the input — there is no untyped escape hatch in this signature.
    /// The relabel from `<PlanetInertial<P>>` to
    /// `<PlanetInertial<SelfPlanet>>` is a zero-cost reinterpretation
    /// at the storage boundary; the underlying SI coordinates are
    /// preserved exactly.
    ///
    /// Mirrors the witness pattern used by
    /// [`BodyAttitude::from_jeod_quat_unchecked`](jeod_sim::BodyAttitude)
    /// and the typed `from_typed_*` siblings in `jeod_sim::recipes` —
    /// mission code that holds a typed `TranslationalStateTyped` in a
    /// specific planet's inertial frame can attach it to an entity
    /// without dropping the type information by way of the untyped
    /// `TranslationalState` storage form.
    #[inline]
    pub fn from_planet_inertial<P: Planet>(
        state: TranslationalStateTyped<PlanetInertial<P>>,
    ) -> Self {
        // Zero-cost relabel: both sides serialize to the same SI
        // base-unit `DVec3`s; only the phantom planet tag differs.
        Self(TranslationalStateTyped {
            position: Position::<PlanetInertial<SelfPlanet>>::from_raw_si(state.position.raw_si()),
            velocity: Velocity::<PlanetInertial<SelfPlanet>>::from_raw_si(state.velocity.raw_si()),
        })
    }

    /// Read this state typed in `P`'s planet-inertial frame.
    ///
    /// Reads the wildcard-tagged storage and re-tags it with the
    /// caller-chosen `P: Planet`. The witness is the caller's
    /// compile-time choice — the function trusts that the entity's
    /// runtime planet identity (its `IntegSourceC` source's planet)
    /// matches `P`. Mission code that has already set up the world
    /// with all bodies orbiting the same planet can pin the type
    /// here to recover compile-time planet checking on downstream
    /// arithmetic without dropping into raw `DVec3`. Bit-identical
    /// numerics: only the phantom planet tag changes.
    ///
    /// For `P = SelfPlanet` the call is a no-op return of the stored
    /// value.
    #[inline]
    pub fn as_planet_inertial<P: Planet>(&self) -> TranslationalStateTyped<PlanetInertial<P>> {
        TranslationalStateTyped {
            position: Position::<PlanetInertial<P>>::from_raw_si(self.0.position.raw_si()),
            velocity: Velocity::<PlanetInertial<P>>::from_raw_si(self.0.velocity.raw_si()),
        }
    }
}

impl From<TranslationalState> for TranslationalStateC {
    #[inline]
    fn from(state: TranslationalState) -> Self {
        Self::from_untyped(state)
    }
}

/// Rotational state (attitude quaternion + body-frame angular
/// velocity / acceleration) for the body being integrated.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct RotationalStateC(pub RotationalStateTyped<SelfRef>);

impl RotationalStateC {
    /// Wrap an untyped [`RotationalState`] as the typed Bevy Component.
    /// The vehicle phantom is `SelfRef` (the Bevy adapter's wildcard
    /// "this entity's vehicle" tag). Panics if the quaternion is not
    /// unit-norm within `NormalizedQuat::DEFAULT_TOLERANCE` (1e-12) —
    /// the typed `RotationalStateTyped` carries a `NormalizedQuat`
    /// witness, so callers must pass a normalized input. Use
    /// `JeodQuat::normalize()` (or construct via the orbital-init
    /// helpers, which guarantee unit-norm) before constructing.
    #[inline]
    pub fn from_untyped(state: RotationalState) -> Self {
        Self(RotationalStateTyped::<SelfRef>::from_untyped_unchecked(
            &state,
        ))
    }
}

impl From<RotationalState> for RotationalStateC {
    #[inline]
    fn from(state: RotationalState) -> Self {
        Self::from_untyped(state)
    }
}

/// Body mass, center of mass, and inertia tensor (with cached
/// inverses). Required on any entity that produces a force or torque
/// requiring acceleration conversion.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct MassPropertiesC(pub MassPropertiesTyped<SelfRef>);

impl MassPropertiesC {
    /// Wrap an untyped [`MassProperties`] as the typed Bevy Component.
    /// The caller asserts the inertia tensor is in `BodyFrame<SelfRef>`
    /// and the center-of-mass position in `StructuralFrame<SelfRef>`.
    /// No runtime check is performed; the conversion is a zero-cost
    /// type-tag attachment via `MassPropertiesTyped::from_untyped_unchecked`
    /// (and `InertiaTensor::from_dmat3_unchecked` internally).
    #[inline]
    pub fn from_untyped(mp: MassProperties) -> Self {
        Self(MassPropertiesTyped::<SelfRef>::from_untyped_unchecked(&mp))
    }
}

impl From<MassProperties> for MassPropertiesC {
    #[inline]
    fn from(mp: MassProperties) -> Self {
        Self::from_untyped(mp)
    }
}

/// Per-step gravitational acceleration accumulator, populated by
/// `gravity_computation_system` and consumed by
/// `force_collection_system`.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct GravityAccelerationC(pub GravityAccelerationTyped<RootInertial>);

impl From<GravityAcceleration> for GravityAccelerationC {
    #[inline]
    fn from(g: GravityAcceleration) -> Self {
        Self(GravityAccelerationTyped::<RootInertial>::from_untyped_unchecked(&g))
    }
}

/// Per-step accumulator of structure-frame forces / torques
/// resolved into the inertial frame; consumed by the integration
/// system.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct TotalForceC(pub TotalForceTyped<SelfRef, RootInertial>);

impl From<TotalForce> for TotalForceC {
    #[inline]
    fn from(t: TotalForce) -> Self {
        Self(TotalForceTyped::<SelfRef, RootInertial>::from_untyped_unchecked(&t))
    }
}

/// Linear and angular accelerations passed to the integrator each
/// stage. Populated by `force_collection_system`.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct FrameDerivativesC(pub FrameDerivativesTyped<RootInertial, SelfRef>);

impl From<FrameDerivatives> for FrameDerivativesC {
    #[inline]
    fn from(d: FrameDerivatives) -> Self {
        Self(FrameDerivativesTyped::<RootInertial, SelfRef>::from_untyped_unchecked(&d))
    }
}

/// Per-body dynamics flags (translational on, rotational on, three-DOF
/// override). Required on every dynamic body.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(FrameDerivativesC)]
pub struct DynamicsConfigC(pub DynamicsConfig);

/// Integration method for this body. Defaults to RK4 when absent.
///
/// When present on a dynamic body entity, the integration system dispatches
/// to the specified method. When absent, `IntegratorType::Rk4` is used.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct IntegratorTypeC(pub jeod_sim::IntegratorType);

/// Persistent Gauss-Jackson (Störmer-Cowell) integrator state.
///
/// Required on entities using `IntegratorType::GaussJackson`. Created once
/// with `GaussJacksonState::new(config)` and maintained across steps.
/// When absent, `integration_system` will panic if `IntegratorTypeC` is GJ.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct GaussJacksonStateC(pub jeod_sim::GaussJacksonState);

/// Persistent Adams-Bashforth-Moulton 4 integrator state.
///
/// Required on entities using `IntegratorType::Abm4`. Created once with
/// `Abm4State::new()` and maintained across steps. When absent,
/// `integration_system` will panic if `IntegratorTypeC` is `Abm4`.
#[derive(Component, Debug, Clone, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct Abm4StateC(pub jeod_sim::Abm4State);

/// Per-body list of gravity controls keyed by source [`Entity`]. Each
/// control selects the model (point-mass / spherical-harmonics) and
/// which body it represents (central, third, etc.).
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(opaque, Component)]
#[require(GravityAccelerationC, TotalForceC)]
pub struct GravityControlsC(pub GravityControls<Entity>);

/// Gravity source attached to a planet entity (mu plus optional
/// spherical-harmonics coefficients). Queried by gravity controls
/// targeting this entity.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct GravitySourceC(pub GravitySource);

/// RootInertial-frame position of a gravity source.
///
/// For the central body (e.g., Earth in an Earth-centered sim), this is
/// typically `Position::<RootInertial>::zero()`. For third bodies (Sun, Moon),
/// this value should be provided and maintained by the application's
/// ephemeris/update logic. Used by the gravity computation to apply
/// differential (third-body) acceleration corrections.
///
/// Required on all gravity source entities. The gravity systems will panic
/// if a source entity referenced by a `GravityControlsC` is missing this
/// component.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct SourceInertialPositionC(pub Position<RootInertial>);

/// RootInertial-frame velocity of a gravity source.
///
/// Optional component. For the central body (e.g., Earth in an Earth-centered
/// sim), this is typically `Velocity::<RootInertial>::zero()`. For third bodies
/// (Sun, Moon), attach this component alongside [`EphemerisBodyC`] and the
/// `ephemeris_update_system` will populate it each step. When absent,
/// relativistic corrections fall back to zero source velocity.
///
/// Used by the gravity and integration systems to provide source velocity to
/// the relativistic correction computation. Stored separately from
/// `TranslationalStateC` to avoid Bevy query conflicts (the body's
/// `TranslationalStateC` is already mutably queried by the integration system).
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct SourceInertialVelocityC(pub Velocity<RootInertial>);

/// Aerodynamic force and torque in the **structural** frame (N, N*m).
///
/// Written by `aero_drag_system`.
/// `force_collection_system` rotates force to inertial and torque to body
/// via `StructuralTransformC`.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct AerodynamicForceC {
    /// Force in the body structural frame (N).
    pub force: DVec3,
    /// Torque about the body structural origin (N·m).
    pub torque: DVec3,
}

/// Solar radiation pressure force and torque.
///
/// Force is always in the **inertial** frame (`flat_plate_srp_system` rotates
/// from structural to inertial before writing).
/// Torque is always in the **structural** frame.
/// Written by `flat_plate_srp_system`.
/// `force_collection_system` rotates torque to body via `StructuralTransformC`.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct RadiationForceC {
    /// Force in the body structural frame (N).
    pub force: DVec3,
    /// Torque about the body structural origin (N·m).
    pub torque: DVec3,
}

/// Gravity gradient torque in the body frame (N·m).
///
/// Written by the gravity torque system.
/// Read by `force_collection_system` as `Option<&GravityTorqueC>`.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct GravityTorqueC(pub Torque<BodyFrame<SelfRef>>);

// JEOD_INV: AT.01 — active flag gates computation (presence of AtmosphericStateC = active)
/// Atmospheric state at the vehicle's position.
///
/// Wraps a typed `AtmosphereState<SelfPlanet>` whose `wind` field is
/// `Velocity<PlanetInertial<SelfPlanet>>` — the planet identity for
/// the atmosphere is determined at runtime by the planet entity carried
/// in [`crate::AtmosphereModelR`]. Mission code that knows the planet
/// at compile time can read the typed sibling
/// `AtmosphereState<P>` directly off-component via the relabel boundary
/// in `aero_drag_system`. Written by the atmosphere system; read by the
/// aerodynamic drag system.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct AtmosphericStateC(pub jeod_sim::AtmosphereState<SelfPlanet>);

/// Typed structural→body rotation for a vehicle entity.
///
/// Stores the rotation that maps structural-frame vectors into body-frame
/// vectors (matches JEOD `mass.composite_properties.T_parent_this` where
/// parent=structure). The `FrameTransform`'s phantom `<StructuralFrame<SelfRef>,
/// BodyFrame<SelfRef>>` parameters encode the *direction* — `SelfRef` is the
/// wildcard `Vehicle` marker indicating "this entity's vehicle"; the actual
/// vehicle identity stays at the entity level via Bevy queries.
///
/// Default is identity (structural frame = body frame), which is correct for
/// single-body vehicles with `eigen_angle=0`.
///
/// Used by `force_collection_system` to:
/// - Compute `T_inertial_struct = T_struct_body^T * T_inertial_body`
/// - Rotate structural-frame torques to body frame
// JEOD_INV: DB.28 — forces collected in structural frame, rotated to inertial at root
// JEOD_INV: DB.29 — torques collected in structural frame, rotated to body at root
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct StructuralTransformC(pub FrameTransform<StructuralFrame<SelfRef>, BodyFrame<SelfRef>>);

impl Default for StructuralTransformC {
    fn default() -> Self {
        Self(FrameTransform::from_matrix(glam::DMat3::IDENTITY))
    }
}

/// Typed inertial→planet-fixed rotation for a gravity source entity.
///
/// Stores the rotation that maps inertial-frame vectors into the planet-fixed
/// frame of the source. The `FrameTransform`'s phantom `<RootInertial,
/// PlanetFixed<SelfPlanet>>` parameters encode the *direction* — `SelfPlanet`
/// is the wildcard `Planet` marker indicating "this entity's planet"; the
/// actual planet identity stays at the entity level via the existing
/// `PlanetC` discriminator.
///
/// When present on a gravity source entity, `gravity_computation_system` and
/// `integration_system` use this rotation instead of `DMat3::IDENTITY` to
/// rotate the spacecraft position into the body-fixed frame before evaluating
/// spherical-harmonic gravity.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct PlanetFixedRotationC(pub FrameTransform<RootInertial, PlanetFixed<SelfPlanet>>);

/// Sidereal rotation rate (rad/s) used by `planet_fixed_rotation_system`
/// to populate [`PlanetAngularVelocityC`] each step. Sourced from
/// [`jeod_sim::PlanetConfig::omega`] at insertion (e.g. from
/// [`PlanetBundle::from_config`](crate::PlanetBundle::from_config)).
///
/// Issue #71 item 1: without this, velocity composition through
/// planet-fixed frames silently uses zero angular velocity, producing
/// the wrong NED-relative or geodetic velocity.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct PlanetOmegaC(pub f64);

/// Optional initial integration-frame source for a body (issue #71
/// item 4). Mirrors [`jeod_sim::VehicleConfig::integ_source`]: when set
/// to `Some(planet_entity)`, the body integrates in that source's
/// inertial frame; when `None` (or the component is absent), the body
/// integrates in the root inertial frame (the Bevy default).
///
/// Consumed at body-frame registration by `register_body_frames_system`
/// to parent the body's frame entity under the source's frame entity.
/// After registration the live "current integration frame" lookup is
/// the body frame entity's `ChildOf` parent — `gravity_computation_system`
/// and `integration_system` walk that hierarchy via the
/// [`crate::frame_param::FrameOrigin`] SystemParam, and
/// `frame_switch_system` reparents the body's frame entity on switch.
/// `IntegSourceC` is the configuration-time intent only and is
/// intentionally not mutated by the switch.
///
/// **Non-root semantics.** When `Some(...)`, the body's
/// [`TranslationalStateC`] stores position/velocity in the source's
/// inertial frame; the Component's `<PlanetInertial<SelfPlanet>>`
/// phantom encodes the planet-inertial framing structurally, so
/// arithmetic that mixes the body state with a `Position<RootInertial>`
/// gravity-source position no longer compiles without an explicit
/// integration-origin shift (RF.10). Two relabel categories apply at
/// consumer call sites and are independent (see
/// [`TranslationalStateC`] for the full breakdown):
///
/// 1. **Integ-origin shift** (arithmetic — adds the origin offset and
///    relabels `<PlanetInertial<SelfPlanet>>` → `<RootInertial>`).
///    Required only by shift sites: `gravity_computation_system`, the
///    SRP / solar-beta / earth-lighting systems. They lift the typed
///    origin offset from
///    [`crate::frame_param::FrameOrigin::origin_in_root`] and relabel
///    via `from_raw_si` at the call site. Non-shift consumers
///    (atmosphere, drag, LVLH, geodetic, orbital elements) skip this
///    category — their physics stays in planet-inertial.
/// 2. **Wildcard `<SelfPlanet>` → concrete `<P>` phantom relabel**
///    (phantom-only, no arithmetic). Required wherever a typed kernel
///    is parameterized over a concrete `PlanetInertial<P>` rather than
///    the wildcard. Drag, LVLH, and geodetic still apply this relabel
///    via `from_raw_si` at the typed-kernel boundary even though they
///    skip category 1. Atmosphere and orbital elements skip it too:
///    atmosphere feeds the kernel raw `DVec3` via
///    `state.position.raw_si()`, and orbital elements drives the
///    planet-erased `compute_orbital_elements` kernel, which returns
///    `OrbitalElements<SelfPlanet>` matching the source state's tag
///    directly.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct IntegSourceC(pub Option<Entity>);

/// Distance-based integration-frame switches for a body (issue #71
/// items 3 + Phase C4).
///
/// Each entry triggers a reparent + gravity-controls flip when the body
/// crosses the configured distance. The Bevy adapter uses
/// `FrameSwitchConfig<Entity>` so `target_source` references a gravity
/// source by ECS entity rather than by registration index — matching
/// `GravityControlsC`'s `Entity`-keyed semantics. Read by
/// [`crate::frame_switch_system`], which evaluates the predicates
/// against [`crate::frame_param::RelativeFrameState`] and reparents
/// the body's frame entity directly via Bevy `ChildOf`.
#[derive(Component, Debug, Clone, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct FrameSwitchesC(pub Vec<jeod_sim::FrameSwitchConfig<Entity>>);

// ── Frames-as-entities components ──
//
// Live on **frame entities** (not body or source entities) and carry
// the per-frame state described in the [Frame-Tree-ECS-Native wiki
// page][1] (Section 13 sequencing). The ECS hierarchy is the single
// source of truth for all frame-tree state — gravity, integration,
// frame-switch, and mission code via [`crate::frame_param`] all read
// from the ECS hierarchy directly via `ChildOf` / `Children`.
//
// Component split rationale: the three pieces of `RefFrameState` are
// independently mutated in practice. `FrameTransC` is rewritten by
// integration / source-position updates; `FrameRotC` is rewritten by
// planet-fixed rotation updates; `FrameAngVelC` is rewritten alongside
// `FrameRotC` for pfix frames but stays at zero for inertial / body
// frames. Splitting lets change-detection fire on the right writers
// only.
//
// [1]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#13-migration-sequencing

/// Translational state (position + velocity) of a frame entity
/// relative to its parent frame entity. Mirrors
/// [`jeod_sim::RefFrameTrans`]. Stored raw (`DVec3`) at the additive-
/// infrastructure stage; later PRs in the sequence may carry typed
/// `Position<P>` / `Velocity<P>` keyed off the parent frame entity's
/// marker. Issue #277.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct FrameTransC {
    /// Position relative to parent frame, in parent-frame coordinates (m).
    pub position: DVec3,
    /// Velocity relative to parent frame, in parent-frame coordinates (m/s).
    pub velocity: DVec3,
}

/// Rotational state of a frame entity relative to its parent: the
/// left-transformation quaternion and the cached transformation matrix
/// `t_parent_this`. Mirrors the rotation portion of
/// [`jeod_sim::RefFrameRot`] minus `ang_vel_this`, which lives in
/// [`FrameAngVelC`] for change-detection granularity. Issue #277.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct FrameRotC {
    /// Left-transformation quaternion (parent → this).
    pub q_parent_this: jeod_sim::JeodQuat,
    /// Transformation matrix `t_parent_this` derived from the quaternion (cache).
    pub t_parent_this: glam::DMat3,
}

impl Default for FrameRotC {
    fn default() -> Self {
        Self {
            q_parent_this: jeod_sim::JeodQuat::identity(),
            t_parent_this: glam::DMat3::IDENTITY,
        }
    }
}

/// Angular velocity of a frame entity relative to its parent, expressed
/// in this-frame coordinates (rad/s). Mirrors
/// [`jeod_sim::RefFrameRot::ang_vel_this`]. Split from [`FrameRotC`] so
/// pfix-rotation systems that only rewrite angular velocity (or body
/// integration that only rewrites attitude) get fine-grained
/// change-detection. Issue #277.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct FrameAngVelC(pub DVec3);

// Marker components — mark the kind of frame an entity represents.
// Bevy idiom: query keying via `With<…>`. Replaces the runtime
// `RefFrameKind` enum on the arena side. Suffix `Marker` avoids
// colliding with `jeod_sim`'s phantom-frame types (`BodyFrame`,
// `PlanetFixed`, `IntegrationFrame`, `RootInertial`).

/// Marker: this entity is a root or planet inertial frame. Issue #277.
#[derive(Component, Debug, Default, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct InertialFrameMarker;

/// Marker: this entity is a planet-fixed (pfix) frame, child of an
/// inertial frame, rotating with the planet. Issue #277.
#[derive(Component, Debug, Default, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct PlanetFixedFrameMarker;

/// Marker: this entity is a body's body-frame (composite_body in JEOD).
/// Issue #277.
#[derive(Component, Debug, Default, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct BodyFrameMarker;

/// Marker: this entity has been registered as an integration frame for
/// at least one body. Inserted (idempotently) by
/// `register_body_frames_system` when a body is spawned with this
/// frame as its integration frame; never removed. The marker has
/// **sticky** semantics — `frame_switch_system` reparents a body's
/// frame entity (via `commands.entity(...).insert(ChildOf(...))`)
/// when the body switches frames, but does not touch this marker,
/// because (a) one integration frame entity can serve many bodies
/// and tracking a "currently in use" predicate would require
/// ref-counting, and (b) downstream SystemParam consumers in later
/// PRs of the [Section 13 sequence][1] only need to know whether a
/// frame entity *can* serve as an integration frame, which the
/// registration-time signal answers correctly. The authoritative
/// "this body's integration frame is X" lookup is the body frame
/// entity's `ChildOf` parent. A frame entity may carry both
/// `InertialFrameMarker` and `IntegrationFrameMarker` simultaneously
/// — they describe orthogonal properties of the frame. Issue #277.
///
/// [1]: https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#13-migration-sequencing
#[derive(Component, Debug, Default, Clone, Copy, Reflect)]
#[reflect(Component)]
pub struct IntegrationFrameMarker;

/// Bidirectional handle linking a body / source / planet entity to its
/// frame entity in the ECS hierarchy. Inserted by
/// `register_*_frames_system` for every entity that carries dynamics
/// state. Internal physics consumers (gravity, integration,
/// frame-switch) and mission code via [`crate::frame_param`] read this
/// handle and walk `Query<&ChildOf>` from the frame entity to recover
/// the body's integration frame, the source's child frames, etc. The
/// frame entity itself carries [`FrameTransC`] / [`FrameRotC`] /
/// [`FrameAngVelC`] (the per-node state).
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct FrameEntityC(pub Entity);

/// Frame entity for a source's planet-fixed (pfix) child frame.
///
/// Inserted by `register_pfix_frames_system` for every gravity source
/// that carries [`PlanetFixedRotationC`] and a non-`None`
/// [`RotationModelC`]. Removed when the rotation model toggles to
/// `None` (in which case the underlying ECS entity is retained as
/// [`RetiredPfixFrameEntityC`] for reuse on the next toggle back to a
/// rotating model — see that component's docs).
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct PfixFrameEntityC(pub Entity);

/// Hidden component that stashes a previously-spawned pfix *frame
/// entity* (the canonical [`PfixFrameEntityC`]) on a source whose
/// [`RotationModelC`] just toggled to
/// [`RotationModel::None`](jeod_sim::RotationModel::None). The
/// public [`PfixFrameEntityC`] is removed at the same time so any
/// reader branching on its presence correctly observes "no
/// planet-fixed frame", but the orphan ECS entity itself is kept
/// alive — its `Name` is renamed to a `.retired` sentinel and its
/// `FrameRotC`/`FrameAngVelC` are reset to identity — so the next
/// toggle back to a rotating model can reuse it instead of spawning
/// a fresh entity.
///
/// Without this, every `None → rotating → None → rotating …` toggle
/// cycle would leak a fresh `<name>.frame.pfix` entity per cycle,
/// since [`crate::systems::register_pfix_frames_system`] filters by
/// `Without<PfixFrameEntityC>` and unconditionally spawns a new
/// entity for any source missing the public component.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct RetiredPfixFrameEntityC(pub Entity);

/// Angular velocity of the planet-fixed frame relative to its inertial
/// parent, expressed in pfix coordinates. Computed each step by
/// `planet_fixed_rotation_system` as `[0, 0, omega]` matching JEOD's
/// `planet_rnp.cc`.
///
/// The `AngularVelocity<PlanetFixed<SelfPlanet>>` phantom indicates "in
/// the pfix frame of this entity's planet"; the planet identity stays
/// at the entity level via [`PlanetC`].
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct PlanetAngularVelocityC(pub AngularVelocity<PlanetFixed<SelfPlanet>>);

/// Declarative spec for a kinematically driven single-axis joint.
///
/// Place this component on a *frame entity* (one carrying the full
/// [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`] triplet) to have
/// [`crate::systems::joint_kinematics_system`] drive the entity's
/// rotation about its parent frame each tick. The rotation angle
/// follows `θ(t) = initial_angle_rad + rate_rad_per_s · t`, applied
/// about `axis_in_parent` (a unit vector in the parent frame).
///
/// Mirrors [`jeod_sim::JointKinematicsSpec`] one-to-one. The component
/// is the analog of [`PlanetFixedRotationC`] generalised to an
/// arbitrary user-declared axis: where pfix entities are spun by
/// [`crate::systems::planet_fixed_rotation_system`] under an Earth-/
/// Mars-/Moon-rotation kernel, joint entities are spun by
/// [`crate::systems::joint_kinematics_system`] under a constant-rate
/// kernel.
///
/// "Kinematic" here means: the angle (and therefore rotation and
/// angular velocity) is an *input* — there is no torque, inertia, or
/// momentum exchange. Joint dynamics (free-swinging joints,
/// constraint-derived joint forces, inverse dynamics) are explicitly
/// out of scope; see the deferred-dynamics meta.
///
/// # Frame-tree contract
///
/// Frame-tree consumers ([`crate::frame_param::RelativeFrameState`])
/// treat every frame entity as carrying the full
/// [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`] triplet — a node
/// missing any of the three would make a hierarchy walk that crosses
/// the joint observe an undefined translation, rotation, or angular
/// velocity. This component therefore auto-inserts all three via
/// `#[require]`. A single-axis joint is by definition a pure rotation
/// about a fixed axis at a fixed point in the parent frame, so the
/// default [`FrameTransC`] (zero offset, zero relative velocity) is
/// the physically correct value for an articulated joint frame and
/// callers do not need to spawn it explicitly.
///
/// # Example
///
/// ```ignore
/// // A solar-array joint that spins at 6 °/min about the +Y axis,
/// // starting at θ = 0. FrameTransC / FrameRotC / FrameAngVelC are
/// // auto-inserted via the #[require] attribute on JointKinematicsC.
/// commands.spawn((
///     JointKinematicsC(JointKinematicsSpec {
///         axis_in_parent: DVec3::Y,
///         rate_rad_per_s: 6.0_f64.to_radians() / 60.0,
///         initial_angle_rad: 0.0,
///     }),
///     ChildOf(parent_frame_entity),
/// ));
/// ```
///
/// Per the design doc Section 15.1, articulated sub-trees declare a
/// chain of joint frame entities under a body frame; each joint frame
/// carries this component and the resulting `FrameTransC` /
/// `FrameRotC` / `FrameAngVelC` flow into the same
/// [`crate::frame_param::RelativeFrameState`] consumers that read
/// planet-fixed rotations.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(FrameTransC, FrameRotC, FrameAngVelC)]
pub struct JointKinematicsC(pub jeod_sim::JointKinematicsSpec);

impl JointKinematicsC {
    /// Convenience constructor: build a joint spec from raw axis / rate
    /// / initial-angle values.
    #[inline]
    pub fn new(axis_in_parent: DVec3, rate_rad_per_s: f64, initial_angle_rad: f64) -> Self {
        Self(jeod_sim::JointKinematicsSpec {
            axis_in_parent,
            rate_rad_per_s,
            initial_angle_rad,
        })
    }
}

impl From<jeod_sim::JointKinematicsSpec> for JointKinematicsC {
    #[inline]
    fn from(spec: jeod_sim::JointKinematicsSpec) -> Self {
        Self(spec)
    }
}

/// Declarative spec for a kinematically driven single-axis joint whose
/// angle is a sinusoidal function of time
/// (`θ(t) = offset + amplitude · sin(ω · t + phase)`).
///
/// Sibling component to [`JointKinematicsC`] for the periodic-articulation
/// case — solar-array dither, antenna scan, gimbal sweep — that the
/// constant-rate spec cannot express. The driving system writes the
/// same [`FrameRotC`] / [`FrameAngVelC`] storage as
/// [`JointKinematicsC`], so a downstream consumer that walks the
/// frame tree sees a uniform rotation snapshot regardless of which
/// kinematic style drives the joint.
///
/// The `#[require]` triplet matches [`JointKinematicsC`] so spawning a
/// joint frame entity carrying this component automatically materializes
/// the [`FrameTransC`] / [`FrameRotC`] / [`FrameAngVelC`] frame-tree
/// triplet, so `RelativeFrameState` walks across the joint remain
/// well-defined.
///
/// Wraps [`jeod_sim::SinusoidalJointKinematicsSpec`] one-to-one. Mission
/// code that needs richer kinematic styles than constant-rate /
/// sinusoidal / closure (e.g., piecewise-linear angular splines) reaches
/// for a custom system; the kinematic-only spec catalogue exposed here
/// covers the periodic / loop-closing / multi-DOF cases.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(FrameTransC, FrameRotC, FrameAngVelC)]
pub struct SinusoidalJointKinematicsC(pub jeod_sim::SinusoidalJointKinematicsSpec);

impl From<jeod_sim::SinusoidalJointKinematicsSpec> for SinusoidalJointKinematicsC {
    #[inline]
    fn from(spec: jeod_sim::SinusoidalJointKinematicsSpec) -> Self {
        Self(spec)
    }
}

/// Declarative spec for a *closure* joint — one pinned to a fixed
/// rotation about a single axis with no time dependence.
///
/// The kinematic-only degenerate case useful for closing kinematic
/// loops where one joint's pose is constrained at declaration time
/// rather than driven through `θ(t)`. The system writes a constant
/// `FrameRotC` and zero `FrameAngVelC` every tick, so the joint
/// frame's contribution to a `RelativeFrameState` walk is the same
/// every step (cheap; the per-tick reassignment is the same value
/// each time).
///
/// Wraps [`jeod_sim::ClosureJointKinematicsSpec`] one-to-one and
/// auto-inserts the frame-tree triplet via `#[require]`, matching
/// [`JointKinematicsC`].
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(FrameTransC, FrameRotC, FrameAngVelC)]
pub struct ClosureJointKinematicsC(pub jeod_sim::ClosureJointKinematicsSpec);

impl From<jeod_sim::ClosureJointKinematicsSpec> for ClosureJointKinematicsC {
    #[inline]
    fn from(spec: jeod_sim::ClosureJointKinematicsSpec) -> Self {
        Self(spec)
    }
}

/// Declarative spec for a multi-DOF kinematic joint — up to
/// [`jeod_sim::MAX_MULTI_DOF_AXES`] single-axis stages composed into
/// one chain.
///
/// Each stage is a `SingleDofKinematics` variant
/// (`ConstantRate`/`Sinusoidal`/`Closure`) that rotates about its
/// declared axis in the *intermediate frame produced by the
/// preceding stages*. Stages must be a contiguous prefix of the
/// fixed-size axes array; the kernel asserts this.
///
/// The semantic equivalence is deliberate: a multi-DOF joint with N
/// stages on a single entity produces the same `(rotation, angular
/// velocity)` snapshot as a chain of N single-DOF joint entities
/// linked by `ChildOf`. Mission code picks whichever shape is more
/// ergonomic — a long arm benefits from N entities (each with its
/// own name + frame-tree slot for inspection); a tightly-coupled 2-3
/// DOF gimbal benefits from one entity.
///
/// Wraps [`jeod_sim::MultiDofJointKinematicsSpec`] one-to-one and
/// auto-inserts the frame-tree triplet via `#[require]`.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(FrameTransC, FrameRotC, FrameAngVelC)]
pub struct MultiDofJointKinematicsC(pub jeod_sim::MultiDofJointKinematicsSpec);

impl From<jeod_sim::MultiDofJointKinematicsSpec> for MultiDofJointKinematicsC {
    #[inline]
    fn from(spec: jeod_sim::MultiDofJointKinematicsSpec) -> Self {
        Self(spec)
    }
}

/// Tidal configuration for a gravity source entity.
///
/// When present on a gravity source entity alongside `PlanetFixedRotationC`,
/// the `tidal_update_system` computes ΔC20 each step and writes it to
/// `TidalDeltaC20C`. The application is responsible for updating
/// `tidal_bodies[].position_inertial` each step from ephemeris data.
///
/// Wraps [`jeod_sim::TidalConfigTyped`] (typed sibling of
/// [`jeod_sim::TidalConfig`]) so the untyped → typed conversion happens
/// **once at insertion**, not per tick in `tidal_update_system`. This
/// eliminates the per-frame `TidalConfigTyped::from_untyped` allocation
/// (Vec collect + per-body f64 → typed boxing). Convenience constructors
/// `from_untyped` / `From` impls are provided for callers building from
/// the untyped struct.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(TidalDeltaC20C)]
pub struct TidalConfigC(pub jeod_sim::TidalConfigTyped);

impl TidalConfigC {
    /// Wrap an untyped [`jeod_sim::TidalConfig`] as a typed Bevy component.
    ///
    /// The dimensional lift (`f64` → `Ratio`/`GravParam`/`Length`/`Position`)
    /// happens here at insertion. After that, the wrapped value is already
    /// typed for the lifetime of the component, eliminating per-tick
    /// `from_untyped` calls in `tidal_update_system`.
    #[inline]
    pub fn from_untyped(config: &jeod_sim::TidalConfig) -> Self {
        Self(jeod_sim::TidalConfigTyped::from_untyped(config))
    }
}

impl From<jeod_sim::TidalConfig> for TidalConfigC {
    fn from(config: jeod_sim::TidalConfig) -> Self {
        Self::from_untyped(&config)
    }
}

impl From<jeod_sim::TidalConfigTyped> for TidalConfigC {
    fn from(config: jeod_sim::TidalConfigTyped) -> Self {
        Self(config)
    }
}

/// Computed tidal ΔC20 for a gravity source entity.
///
/// Written by `tidal_update_system`. Read by gravity computation and
/// integration systems. Defaults to zero (no tidal effect).
///
/// Wrapped as a [`Ratio`] (dimensionless) so the value carries unit
/// metadata at the type level — matching `compute_delta_c20_typed`'s
/// return type.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct TidalDeltaC20C(pub Ratio);

// ── Interactions ──

/// Vehicle drag configuration (Cd, area).
///
/// Wraps [`DragConfigTyped`] (typed sibling of [`DragConfig`]) so the
/// untyped → typed conversion happens **once at insertion**, not per tick
/// in `aero_drag_system`. Convenience constructors `from_untyped` /
/// `new` are provided for callers building from raw `f64` fields.
///
/// Auto-inserts [`AtmosphericStateC`] and [`AerodynamicForceC`] when added.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(AtmosphericStateC, AerodynamicForceC)]
pub struct DragConfigC(pub DragConfigTyped);

impl DragConfigC {
    /// Wrap an untyped [`DragConfig`] as a typed Bevy component.
    ///
    /// The dimensional lift (`f64` → `Ratio`/`Area`/`MassDensity`) happens
    /// here at insertion. After that, the wrapped value is already typed
    /// for the lifetime of the component, eliminating per-tick
    /// per-tick unchecked conversions in `aero_drag_system`. Per #172 H1,
    /// this is the documented insertion-time boundary: DragConfig has no
    /// spatial fields (Cd / area / optional density override only), so
    /// the lift here is the JEOD-CSV-style boundary the audit carves out.
    /// The component then stores DragConfigTyped for the rest of its
    /// lifetime; no per-step re-minting occurs.
    #[inline]
    pub fn from_untyped(config: &DragConfig) -> Self {
        Self(DragConfigTyped::from_untyped_unchecked(config)) // allowed: #172 H1 insertion-time boundary, see docstring
    }
}

impl From<DragConfig> for DragConfigC {
    fn from(config: DragConfig) -> Self {
        Self::from_untyped(&config)
    }
}

impl From<DragConfigTyped> for DragConfigC {
    fn from(config: DragConfigTyped) -> Self {
        Self(config)
    }
}

/// Flat-plate SRP configuration with thermal state.
///
/// Wraps [`jeod_sim::FlatPlateState`] so the same type (and its
/// `integrate_temperatures` method) is shared with the `Simulation` runner.
///
/// Auto-inserts [`RadiationForceC`] when added.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
#[require(RadiationForceC)]
pub struct FlatPlateConfigC(pub jeod_sim::FlatPlateState);

/// Marker for an entity that casts shadows (e.g., Earth).
///
/// The shadow detection system queries all entities with this component
/// and computes the illumination factor for SRP. Place on any planet
/// entity along with `TranslationalStateC`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct ShadowBodyC {
    /// Body radius (m) for conical shadow computation.
    pub radius: f64,
}

/// Per-source rotation model dispatch.
///
/// When present on a gravity source entity alongside `PlanetFixedRotationC`,
/// the `planet_fixed_rotation_system` dispatches to the correct rotation
/// computation based on this value. When absent, `EarthRNP` is assumed
/// for backward compatibility.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct RotationModelC(pub jeod_sim::RotationModel);

/// Ephemeris body mapping for automatic position updates from DE4xx.
///
/// When present on a gravity source entity, the `ephemeris_update_system`
/// queries the `EphemerisR` resource each step to update the entity's
/// `SourceInertialPositionC` (and optionally `TranslationalStateC`).
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct EphemerisBodyC {
    /// The body this source represents (e.g., `EphemerisBody::Sun`).
    pub target: jeod_sim::EphemerisBody,
    /// The integration frame center (e.g., `EphemerisBody::Earth`).
    pub observer: jeod_sim::EphemerisBody,
}

/// Cannonball SRP configuration using JEOD's `RadiationDefaultSurface` formula.
///
/// Force = (flux/c) * cx_area * [1 + albedo*diffuse*(4/9)] * flux_hat * illum_factor.
/// Mutually exclusive with `FlatPlateConfigC` (use one or the other).
///
/// Requires `SunMarker` entity in the world. Optional `ShadowBodyC` for eclipse.
/// Writes to `RadiationForceC`.
///
/// Auto-inserts [`RadiationForceC`] when added.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(RadiationForceC)]
pub struct CannonballSrpC {
    /// Cross-section area * Cr (m²).
    pub cx_area: f64,
    /// Surface albedo (0–1).
    pub albedo: f64,
    /// Diffuse reflection fraction (0–1).
    pub diffuse: f64,
}

/// Marker component for the Sun entity (used by SRP system to find Sun position).
#[derive(Component, Reflect, Default, Clone, Copy, Debug)]
#[reflect(opaque, Component)]
pub struct SunMarker;

/// Marker component for the Moon entity (used by earth lighting system).
#[derive(Component, Reflect, Default, Clone, Copy, Debug)]
#[reflect(opaque, Component)]
pub struct MoonMarker;

/// Marks an entity as the simulation's designated central body.
///
/// [`crate::SourceMutator::set_source_position`] and
/// [`crate::SourceMutator::set_source_state`] panic if the target entity
/// carries this marker. Mission code attaches it to the gravity-source
/// entity it treats as the pinned origin (e.g. Earth in an Earth-centered
/// scenario), opting that entity into the same protection that
/// `jeod_runner::Simulation::set_source_*` enforces against the
/// root-mapped source via `assert_ne!(fid, root_frame_id, …)`.
///
/// At most one entity should carry this marker per simulation; multiple
/// markers are not currently rejected, but `SourceMutator` panics on
/// every call that targets a marked entity, so a well-behaved app
/// attaches it once.
#[derive(Component, Reflect, Default, Clone, Copy, Debug)]
#[reflect(opaque, Component)]
pub struct CentralSourceMarker;

// ── Planet ──

/// Bevy component wrapping `PlanetShape`.
#[derive(Component, Debug, Clone, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct PlanetC(pub PlanetShape);

// ── Derived State Configuration ──

/// Configuration for orbital elements computation.
///
/// The `gravity_source` entity is queried for `GravitySourceC` to obtain `mu`.
/// Presence of this component + `OrbitalElementsC` on an entity enables
/// per-step orbital elements computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(OrbitalElementsC)]
pub struct OrbitalElementsConfigC {
    /// Gravity source entity supplying `mu` for the conversion.
    pub gravity_source: Entity,
}

/// Configuration for Euler angle decomposition.
///
/// Presence of this component + `EulerAnglesC` on an entity enables
/// per-step Euler angle computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(EulerAnglesC)]
pub struct EulerAnglesConfigC {
    /// Euler-angle decomposition convention (e.g., 3-2-1 yaw/pitch/roll).
    pub sequence: jeod_sim::EulerSequence,
}

/// Configuration for geodetic state computation.
///
/// The `planet` entity is queried for `PlanetFixedRotationC` and `PlanetC`
/// to obtain the rotation matrix and ellipsoid radii.
/// Presence of this component + `GeodeticStateC` on an entity enables
/// per-step geodetic computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(GeodeticStateC)]
pub struct GeodeticConfigC {
    /// Planet entity supplying ellipsoid radii (`PlanetC`) and
    /// `T_inertial→pfix` (`PlanetFixedRotationC`).
    pub planet: Entity,
}

// ── Derived State Outputs ──

/// Orbital elements computed each step.
///
/// Written by `orbital_elements_system` for entities that also have
/// `OrbitalElementsConfigC`.
#[derive(Component, Debug, Clone, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct OrbitalElementsC(
    // Bevy components are runtime-keyed by source index, so the planet
    // identity isn't statically known here — `SelfPlanet` is the
    // boundary-only escape hatch for the registry-side path.
    pub jeod_sim::OrbitalElements<jeod_sim::SelfPlanet>,
);

/// Euler angles `[phi, theta, psi]` computed each step.
///
/// Written by `euler_angles_system` for entities that also have
/// `EulerAnglesConfigC`. Each component is a [`Angle`] (uom radian-backed
/// scalar) so consumers don't have to remember the radian convention.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct EulerAnglesC(pub [Angle; 3]);

/// LVLH (Local Vertical Local Horizontal) frame computed each step.
///
/// Presence of this component alone enables computation — no separate
/// config component needed (only requires translational state).
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct LvlhFrameC(pub jeod_sim::LvlhFrame);

/// Geodetic state (latitude, longitude, altitude) computed each step.
///
/// Written by `geodetic_system` for entities that also have `GeodeticConfigC`.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct GeodeticStateC(pub jeod_sim::GeodeticState);

/// Solar beta angle (radians) computed each step.
///
/// Presence of this component alone enables computation — requires a
/// `SunMarker` entity to exist in the world.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct SolarBetaC(pub f64);

/// Configuration for Earth lighting (eclipse/albedo) computation.
///
/// Requires `SunMarker` and `MoonMarker` entities to exist in the world.
/// Presence of this component + `EarthLightingStateC` on an entity enables
/// per-step earth lighting computation in `JeodSet::DerivedState`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
#[require(EarthLightingStateC)]
pub struct EarthLightingConfigC {
    /// Earth equatorial radius (m).
    pub earth_radius: f64,
    /// Moon mean radius (m).
    pub moon_radius: f64,
    /// Sun mean radius (m).
    pub sun_radius: f64,
}

/// Earth lighting state computed each step.
///
/// Written by `earth_lighting_system` for entities that also have
/// `EarthLightingConfigC`.
#[derive(Component, Debug, Clone, Default, Reflect)]
#[reflect(opaque, Component)]
pub struct EarthLightingStateC(pub jeod_sim::EarthLightingState);

// ── External Loads ──

/// External force in the **inertial** frame.
///
/// Added to `TotalForceC.force` each step after force collection.
/// Matches `SimBody.external_force` in `jeod_sim::Simulation`.
///
/// Mutate between steps to implement time-scheduled force injection.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct ExternalForceC(pub jeod_sim::Force<RootInertial>);

/// External torque in the **body** frame.
///
/// Added to `TotalForceC.torque` each step after force collection.
/// Matches `SimBody.external_torque` in `jeod_sim::Simulation`.
///
/// Mutate between steps to implement time-scheduled torque injection.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut, Reflect)]
#[reflect(opaque, Component)]
pub struct ExternalTorqueC(pub Torque<BodyFrame<SelfRef>>);

// ── Mass Tree (Staging) ──

/// Maps this entity to a node in the shared [`MassTreeR`](crate::MassTreeR) resource.
///
/// Entities with this component participate in the mass tree. After
/// attach/detach events are processed, the entity's [`MassPropertiesC`]
/// is synced from the tree's composite properties.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct MassBodyIdC(pub jeod_sim::MassBodyId);

/// ECS-native mass-tree relation: marks `Entity` carrying this component
/// as a sub-mass attached to the referenced parent entity in the **mass
/// tree** (deliberately distinct from Bevy's frame-tree `ChildOf`).
///
/// Mirrors JEOD's separation of `RefFrame` and `MassBody` trees (see
/// [Frame-Tree-ECS-Native § 15.2](https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#152-mass--inertia-composition)
/// and Appendix A.3): the kinematic frame tree and the inertial mass
/// tree evolve under independent attach/detach paths, coupled only by
/// the explicit [`MassPointRef`] back-pointer. Keeping the two
/// relations as separate `Component`s makes Bevy's hierarchy +
/// observer plumbing one-to-one with JEOD's "two trees + named
/// coupling" architecture.
///
/// The component carries the **parent reference** plus the
/// attach-edge geometry (`offset` + `t_parent_child`), matching the
/// arena's per-body `MassBody::structure_point` (`offset` is the
/// child's structural origin in the *parent's* structural frame;
/// `t_parent_child` is the rotation from the parent's structural
/// frame into this body's structural frame). Edge geometry lives on
/// the child because every child has exactly one parent — which
/// matches both JEOD's `MassBody` layout and the natural ECS
/// component-per-entity grain.
///
/// The carrier entity must also have [`MassPropertiesC`]; the
/// `composite_mass_system` walks `MassChildOf` edges bottom-up via
/// the [`jeod_sim::MassStorage`] trait and writes the recomputed
/// composite properties back into [`MassPropertiesC`] on every node
/// in the affected subtree.
///
/// # JEOD precedent
///
/// `MassBody` nodes form a tree via `MassBodyLinks` (see
/// `models/dynamics/mass/include/mass.hh`); `MassBody::structure_point`
/// (`MassPointState`) carries the per-attach offset + rotation that
/// `MassChildOf` mirrors here. `BodyRefFrame::mass_point`
/// (`models/dynamics/dyn_body/include/body_ref_frame.hh`) is the
/// frame-side back-pointer connecting a kinematic frame to its
/// mass-tree origin — see [`MassPointRef`] for the Bevy port.
// JEOD_INV: MA.08 — no cycle in mass tree (composite_mass_system asserts via post-order walk)
// JEOD_INV: MA.19 — no same-tree attachment (cycle prevention)
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct MassChildOf {
    /// Parent entity in the mass tree.
    pub parent: Entity,
    /// Child's structural origin expressed in the **parent's**
    /// structural frame (m). Default `[0, 0, 0]` means the child's
    /// struct origin is co-located with the parent's struct origin.
    pub offset: glam::DVec3,
    /// Rotation from the parent's structural frame into this body's
    /// structural frame. Default identity (no relative rotation).
    pub t_parent_child: glam::DMat3,
}

impl MassChildOf {
    /// Convenience constructor for an axis-aligned (identity rotation)
    /// attach at the given offset.
    pub fn new(parent: Entity, offset: glam::DVec3) -> Self {
        Self {
            parent,
            offset,
            t_parent_child: glam::DMat3::IDENTITY,
        }
    }

    /// Convenience constructor for a co-located attach (zero offset,
    /// identity rotation). The child's struct origin sits on the
    /// parent's struct origin — useful when the child's CoM offset
    /// is encoded in its own [`MassPropertiesC.center_of_mass`](MassPropertiesC).
    pub fn at_origin(parent: Entity) -> Self {
        Self {
            parent,
            offset: glam::DVec3::ZERO,
            t_parent_child: glam::DMat3::IDENTITY,
        }
    }

    /// Full constructor with explicit offset + rotation, mirroring
    /// `MassTree::attach(child, parent, offset, t_parent_child)`.
    pub fn with_rotation(parent: Entity, offset: glam::DVec3, t_parent_child: glam::DMat3) -> Self {
        Self {
            parent,
            offset,
            t_parent_child,
        }
    }
}

/// Frame-side back-pointer linking a body's frame entity to the
/// mass-tree node that supplies the body's **mass-point origin**
/// (CoM offset + struct→body rotation).
///
/// Mirrors JEOD's `BodyRefFrame::mass_point` (a `MassPoint *`) defined
/// in `models/dynamics/dyn_body/include/body_ref_frame.hh`. JEOD uses
/// this back-pointer to route kinematic state queries on a body frame
/// (which knows the mass-side point) without forcing the frame and
/// mass trees to share their hierarchy, which is the same separation
/// the Bevy adapter mirrors via [`MassChildOf`] vs Bevy's `ChildOf`.
///
/// **Optional by design.** Per [Frame-Tree-ECS-Native § 15.2](https://github.com/simnaut/bevy_jeod/wiki/Frame-Tree-ECS-Native#152-mass--inertia-composition)
/// the back-pointer is *absent for kinematic-only attaches* — i.e.
/// frame entities whose kinematics ride a parent without contributing
/// to that parent's mass (sensor mounts, station-keeping vehicles
/// attached only via `attach_to_frame`). Mission code attaches it
/// only when the frame entity also participates in the mass tree.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct MassPointRef(pub Entity);

/// Marker: this entity is a kinematic non-root node in a
/// [`MassChildOf`] chain and must NOT be advanced by
/// [`integration_system`](crate::systems::integration_system).
///
/// JEOD's composite-rigid-body model integrates only the root of every
/// mass tree (`dyn_body_collect.cc:138` — every `dyn_parent != nullptr`
/// branch transmits forces upstream and computes no per-child
/// accelerations). The Bevy port mirrors this by:
///
/// 1. [`wrench_aggregation_system`](crate::wrench::wrench_aggregation_system)
///    walks every `MassChildOf` chain and folds each non-root child's
///    `(force, torque)` (with parallel-axis arm) into the root's
///    `TotalForceC`, then zeroes the children's
///    `TotalForceC` / `FrameDerivativesC`.
/// 2. The same system inserts `KinematicChildC` on every non-root
///    node and removes it from any node that becomes a root (mass tree
///    rewired or torn down).
/// 3. [`integration_system`](crate::systems::integration_system) filters
///    its body query with `Without<KinematicChildC>` so the kinematic
///    children's translational / rotational state never advances under
///    gravity (or any other contributor `integration_system` reads
///    directly). Without the marker, zeroing `TotalForceC` is not
///    enough — `integration_system` recomputes gravity at every RK
///    sub-stage from `GravityControlsC` and would still drift the
///    child's state.
///
/// This marker is purely a **gating hint** for the integrator. The
/// kinematic propagation that derives child poses *from* the root
/// each step lives at
/// [`crate::kinematic_propagation::propagate_state_from_root_system`]
/// (design-doc Section 15.3) and runs earlier in
/// `JeodSet::ForceCollection` so the wrench walk reads live
/// attitudes; non-root children's `TranslationalStateC` /
/// `RotationalStateC` are overwritten each step with the derived
/// value.
///
/// Mission code MUST NOT manage this marker manually — the
/// wrench-aggregation system owns its lifecycle. Inserting it on a
/// root-level body would freeze that body's state; removing it from a
/// non-root body would let the integrator double-count the wrench
/// (once via the aggregated root total, once via per-stage gravity on
/// the now-self-integrated child).
// JEOD_INV: DB.17 — only the root's TotalForce/FrameDerivatives drive the
// integrator (children are kinematic, gated by this marker)
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(Component)]
pub struct KinematicChildC;

/// Message: attach a child body to a parent in the mass tree.
///
/// Both entities must have [`MassBodyIdC`]. Processed by `staging_system`
/// before integration each step.
///
/// `offset` carries `Position<StructuralFrame<SelfRef>>`: the child's
/// structural origin lives in the parent's structural frame, and the
/// `SelfRef` wildcard tag mirrors the per-entity adapter pattern (the
/// concrete vehicle identity of the parent is determined at runtime
/// via the entity hierarchy). The compile-time guard is the *frame
/// kind*: a caller that holds an inertial-frame position cannot
/// accidentally feed it as the structural-frame attach offset.
///
/// `t_parent_child` stays raw `glam::DMat3` for now — typing it as a
/// `FrameTransform<StructuralFrame<SelfRef>, StructuralFrame<SelfRef>>`
/// would yield `From == To`, which is the identity-direction case
/// covered only by `FrameTransform::identity`. A two-vehicle phantom
/// distinguishing parent vs child structural frames is the right
/// follow-up but requires the `<V>` Bevy-component genericity from
/// Section A of the audit (see #263 / sibling issues) — the `SelfRef`
/// wildcard cannot encode that distinction at the type level.
#[derive(Message, Debug, Clone)]
pub struct AttachEvent {
    /// Entity of the child body.
    pub child: Entity,
    /// Entity of the parent body.
    pub parent: Entity,
    /// Child structural origin in the parent's structural frame (m).
    pub offset: jeod_sim::Position<jeod_sim::StructuralFrame<jeod_sim::SelfRef>>,
    /// Rotation from parent structural frame to child structural frame.
    pub t_parent_child: glam::DMat3,
}

/// Message: detach a child body from its parent in the mass tree.
///
/// The entity must have [`MassBodyIdC`] and be attached to a parent.
/// Processed by `staging_system` before integration each step.
#[derive(Message, Debug, Clone)]
pub struct DetachEvent {
    /// Entity to detach from its parent.
    pub child: Entity,
}

/// Composite-body inertial state of a free-flying mass-tree subtree
/// that has been detached from its parent and is coasting
/// ballistically (no force, no torque) until [`AttachEvent`] re-attaches
/// it.
///
/// Inserted on the child entity by `staging_system`'s `DetachEvent`
/// branch; removed by the `AttachEvent` branch when the same entity
/// is re-attached. While present, [`step_detached_system`](crate::step_detached_system)
/// advances the contained state by the schedule's fixed `dt` each
/// tick — position drifts at `composite_velocity`, attitude rotates
/// under `composite_ang_vel_body`. Also synchronizes the entity's
/// own [`TranslationalStateC`] / [`RotationalStateC`] each tick so
/// downstream consumers (gravity, derived state, mission code)
/// continue to read the body's current inertial state from the
/// canonical components rather than having to special-case detached
/// vs integrated bodies.
///
/// Bevy mirror of `jeod_runner::Simulation::detached_subtrees`. Wraps
/// [`jeod_sim::DetachedSubtreeState`] (which owns the JEOD scalar-first
/// left-multiply attitude convention via
/// [`jeod_sim::BodyAttitude<jeod_sim::SelfRef>`](jeod_sim::BodyAttitude)).
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(opaque, Component)]
pub struct DetachedSubtreeStateC(pub jeod_sim::DetachedSubtreeState);
