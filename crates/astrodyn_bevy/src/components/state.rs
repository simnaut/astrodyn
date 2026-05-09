// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy `Component` newtypes for vehicle state — translational /
//! rotational / mass properties, the per-step force and acceleration
//! accumulators consumed by the integrator, the integrator state
//! components themselves, the structural→body rotation, and the
//! external-force / external-torque injection slots.

use astrodyn::{
    BodyFrame, DynamicsConfig, FrameDerivatives, FrameDerivativesTyped, FrameTransform,
    GravityAcceleration, GravityAccelerationTyped, MassPropertiesTyped, Planet, PlanetInertial,
    Position, RootInertial, RotationalStateTyped, SelfRef, StructuralFrame, Torque, TotalForce,
    TotalForceTyped, TranslationalState, TranslationalStateTyped, Velocity,
};
use bevy::prelude::*;

// ── Dynamics ──
//
// Spatial Components wrap the **typed siblings** from `astrodyn_dynamics`,
// not the raw untyped storage. The frame phantoms (`RootInertial`,
// `BodyFrame<SelfRef>`, `StructuralFrame<SelfRef>`) are baked into the
// component at the type level, so systems read typed values directly
// without the per-step `from_raw_si` lifts that the audit's #172 H1
// flagged as the load-bearing failure mode of the typed-quantity
// facade. Mission code that mutates `c.0.position` directly via raw
// `DVec3` is now a compile error — the typed accessor `Position<RootInertial>`
// surfaces the convention as a type, not just a comment.
//
// The `From<Untyped>` impls and `RotationalStateC::from_untyped` /
// `MassPropertiesC::from_untyped` named opt-ins were deleted in #397
// (delete-not-allow): mission code constructs the typed sibling directly
// (`RotationalStateTyped::<SelfRef>::new(...)`,
// `MassPropertiesTyped::<SelfRef>::with_inertia(...)`) and lifts to the
// Component via the typed-side `From<RotationalStateTyped<SelfRef>>` /
// `From<MassPropertiesTyped<SelfRef>>` impls (still present below).
// The lone exception is `TranslationalStateC::<P>::from_untyped(state)`,
// which remains as a named opt-in because the relabel from the gateway's
// `TranslationalStateTyped<RootInertial>` to the Component's
// `TranslationalStateTyped<PlanetInertial<P>>` storage requires the
// caller to assert `P` — there is no information-preserving typed-side
// `From` impl that can do that without a witness.

/// Translational state (position, velocity) for the body being
/// integrated. Wraps a typed
/// [`TranslationalStateTyped<PlanetInertial<P>>`](TranslationalStateTyped)
/// sibling so the frame phantom is enforced at the type level.
///
/// # Frame semantics: planet-inertial, not root-inertial
///
/// The frame phantom is [`PlanetInertial<P>`] for the planet `P` that
/// this body integrates around. Two relabel categories apply at
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
/// 2. **Same-planet relabel** (phantom-only, bit-identical, no
///    arithmetic). The component already carries the concrete `P`,
///    so consumers that wanted `Position<PlanetInertial<P>>` get it
///    by direct projection through `as_planet_inertial()` — the
///    underlying SI coordinates are preserved exactly.
///
/// Atmosphere/drag, LVLH, geodetic, and orbital-elements consumers
/// do **not** apply the integ-origin shift (they live in
/// planet-inertial throughout).
///
/// For root-integrated bodies the integ-origin shift is zero, so the
/// planet-inertial coordinates numerically equal root-inertial; the
/// typed phantom stays distinct so arithmetic mixing the body state
/// with a `Position<RootInertial>` gravity-source position still
/// requires the explicit shift the runner already performs.
///
/// # `<P: Planet>` parametrization
///
/// The component is generic over the planet marker `P`. Every call site
/// must pin `P` explicitly — there is no fallback. Mission code that needs
/// multiple planets in a single `World` (e.g. a Mars-orbit chief plus an
/// Earth-orbit deputy) instantiates `TranslationalStateC<Mars>` and
/// `TranslationalStateC<Earth>` as distinct component types — Bevy
/// queries discriminate them at the type level.
///
/// **Per-planet system instantiation.** The Bevy adapter systems that
/// read or write `TranslationalStateC<P>` (gravity, atmosphere, drag,
/// SRP, integration, frame-switch, derived states, mass-tree staging,
/// kinematic and frame-attached propagation, etc.) are themselves
/// generic over `<P: Planet>`. [`crate::AstrodynPlugin`] registers the
/// `<astrodyn::Earth>` instantiation at startup, preserving the
/// single-planet pipeline for missions that don't need multi-planet
/// integration. A multi-planet mission calls
/// [`crate::register_planet_systems::<P>`](crate::register_planet_systems)
/// once per *additional* planet to register the parallel system set
/// for `<P>`. Each instantiation only matches entities whose
/// Planet-flavored components carry the matching `<P>` tag, so the
/// Earth and Mars systems run in parallel over disjoint entity sets.
// JEOD_INV: DB.24 — default integrated_frame is composite_body (we integrate composite_body state)
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct TranslationalStateC<P: Planet>(pub TranslationalStateTyped<PlanetInertial<P>>);

impl<P: Planet> Default for TranslationalStateC<P> {
    #[inline]
    fn default() -> Self {
        Self(TranslationalStateTyped::default())
    }
}

impl<P: Planet> TranslationalStateC<P> {
    /// Wrap an untyped [`TranslationalState`] as the typed Bevy
    /// Component. The caller asserts the values are in `P`'s
    /// planet-inertial frame. Typed↔raw kernel-boundary helper —
    /// the matching `From<TranslationalState>` impl was removed in
    /// #397, so callers must opt in by this named method rather than
    /// `.into()`.
    #[inline]
    pub fn from_untyped(state: TranslationalState) -> Self {
        // allowed: typed↔raw kernel boundary
        Self(TranslationalStateTyped::<PlanetInertial<P>> {
            position: Position::<PlanetInertial<P>>::from_raw_si(state.position),
            velocity: Velocity::<PlanetInertial<P>>::from_raw_si(state.velocity),
        })
    }

    /// Witness-gated constructor: wrap an already-typed
    /// [`TranslationalStateTyped`] expressed in `P`'s planet-inertial
    /// frame as the typed Component. The witness is the caller's
    /// compile-time choice of `P` plus the typed phantoms on the
    /// input — there is no untyped escape hatch in this signature.
    ///
    /// Mirrors the witness pattern used by
    /// [`BodyAttitude::from_jeod_quat_unchecked`](astrodyn::BodyAttitude)
    /// and the typed `from_typed_*` siblings in `astrodyn::recipes`.
    #[inline]
    pub fn from_planet_inertial(state: TranslationalStateTyped<PlanetInertial<P>>) -> Self {
        Self(state)
    }

    /// Read this state typed in `P`'s planet-inertial frame.
    ///
    /// The component already carries the concrete planet identity, so
    /// this is a direct copy of the underlying typed value (no relabel
    /// needed — the type tag matches what the caller asks for).
    #[inline]
    pub fn as_planet_inertial(&self) -> TranslationalStateTyped<PlanetInertial<P>> {
        TranslationalStateTyped {
            position: Position::<PlanetInertial<P>>::from_raw_si(self.0.position.raw_si()),
            velocity: Velocity::<PlanetInertial<P>>::from_raw_si(self.0.velocity.raw_si()),
        }
    }
}

impl<P: Planet> From<TranslationalStateTyped<RootInertial>> for TranslationalStateC<P> {
    /// Insertion-time boundary from the gateway's
    /// `<RootInertial>`-typed `VehicleConfig.trans` into the Bevy
    /// component's `<PlanetInertial<P>>` storage. Pure phantom relabel
    /// (numerics bit-identical) — no `from_*_unchecked` bypass needed
    /// because the gateway-side phantom is already asserted.
    #[inline]
    fn from(state: TranslationalStateTyped<RootInertial>) -> Self {
        Self(state.relabel_to::<PlanetInertial<P>>())
    }
}

/// Rotational state (attitude quaternion + body-frame angular
/// velocity / acceleration) for the body being integrated.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct RotationalStateC(pub RotationalStateTyped<SelfRef>);

impl From<RotationalStateTyped<SelfRef>> for RotationalStateC {
    /// Wrap an already-typed `<SelfRef>` rotational state directly. The
    /// inner phantom matches the storage phantom — this is the
    /// production path from `VehicleConfig.rot`, which is typed
    /// end-to-end (issue #388 follow-up).
    #[inline]
    fn from(state: RotationalStateTyped<SelfRef>) -> Self {
        Self(state)
    }
}

/// Body mass, center of mass, and inertia tensor (with cached
/// inverses). Required on any entity that produces a force or torque
/// requiring acceleration conversion.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut)]
pub struct MassPropertiesC(pub MassPropertiesTyped<SelfRef>);

impl From<MassPropertiesTyped<SelfRef>> for MassPropertiesC {
    /// Wrap an already-typed `<SelfRef>` mass properties directly. The
    /// inner phantom matches the storage phantom — this is the
    /// production path from `VehicleConfig.mass`, which is typed
    /// end-to-end (issue #388 follow-up).
    #[inline]
    fn from(mp: MassPropertiesTyped<SelfRef>) -> Self {
        Self(mp)
    }
}

/// Per-step gravitational acceleration accumulator, populated by
/// `gravity_computation_system` and consumed by
/// `force_collection_system`.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default)]
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
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default)]
pub struct TotalForceC(pub TotalForceTyped<SelfRef, RootInertial>);

impl From<TotalForce> for TotalForceC {
    #[inline]
    fn from(t: TotalForce) -> Self {
        Self(TotalForceTyped::<SelfRef, RootInertial>::from_untyped_unchecked(&t))
    }
}

/// Linear and angular accelerations passed to the integrator each
/// stage. Populated by `force_collection_system`.
#[derive(Component, Debug, Clone, Copy, Deref, DerefMut, Default)]
pub struct FrameDerivativesC(pub FrameDerivativesTyped<RootInertial, SelfRef>);

impl From<FrameDerivatives> for FrameDerivativesC {
    #[inline]
    fn from(d: FrameDerivatives) -> Self {
        Self(FrameDerivativesTyped::<RootInertial, SelfRef>::from_untyped_unchecked(&d))
    }
}

/// Per-body dynamics flags (translational on, rotational on, three-DOF
/// override). Required on every dynamic body.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
#[require(FrameDerivativesC)]
pub struct DynamicsConfigC(pub DynamicsConfig);

/// Integration method for this body. Defaults to RK4 when absent.
///
/// When present on a dynamic body entity, the integration system dispatches
/// to the specified method. When absent, `IntegratorType::Rk4` is used.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct IntegratorTypeC(pub astrodyn::IntegratorType);

/// Persistent Gauss-Jackson (Störmer-Cowell) integrator state.
///
/// Required on entities using `IntegratorType::GaussJackson`. Created once
/// with `GaussJacksonState::new(config)` and maintained across steps.
/// When absent, `integration_system` will panic if `IntegratorTypeC` is GJ.
#[derive(Component, Debug, Clone, Deref, DerefMut)]
pub struct GaussJacksonStateC(pub astrodyn::GaussJacksonState);

/// Persistent Adams-Bashforth-Moulton 4 integrator state.
///
/// Required on entities using `IntegratorType::Abm4`. Created once with
/// `Abm4State::new()` and maintained across steps. When absent,
/// `integration_system` will panic if `IntegratorTypeC` is `Abm4`.
#[derive(Component, Debug, Clone, Default, Deref, DerefMut)]
pub struct Abm4StateC(pub astrodyn::Abm4State);

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
#[derive(Component, Debug, Clone, Copy)]
pub struct StructuralTransformC(pub FrameTransform<StructuralFrame<SelfRef>, BodyFrame<SelfRef>>);

impl Default for StructuralTransformC {
    fn default() -> Self {
        Self(FrameTransform::from_matrix(glam::DMat3::IDENTITY))
    }
}

// ── External Loads ──

/// External force in the **inertial** frame.
///
/// Added to `TotalForceC.force` each step after force collection.
/// Matches `SimBody.external_force` in `astrodyn::Simulation`.
///
/// Mutate between steps to implement time-scheduled force injection.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct ExternalForceC(pub astrodyn::Force<RootInertial>);

/// External torque in the **body** frame.
///
/// Added to `TotalForceC.torque` each step after force collection.
/// Matches `SimBody.external_torque` in `astrodyn::Simulation`.
///
/// Mutate between steps to implement time-scheduled torque injection.
#[derive(Component, Debug, Clone, Copy, Default, Deref, DerefMut)]
pub struct ExternalTorqueC(pub Torque<BodyFrame<SelfRef>>);
