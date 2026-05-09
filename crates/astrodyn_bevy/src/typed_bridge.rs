//! Crate-internal typed↔raw kernel-boundary helpers.
//!
//! The `from_untyped_unchecked` named opt-ins on
//! `TranslationalStateTyped`/`RotationalStateTyped`/`MassPropertiesTyped`,
//! and the `MassPropertiesC::from_untyped` / `RotationalStateC::from_untyped`
//! Bevy Component opt-ins, were deleted in #397. The kernel functions in
//! `astrodyn` still take raw `RotationalState` / `TranslationalState` /
//! `MassProperties` structs, so the Bevy adapter has to translate at the
//! boundary anyway. Centralizing those translations here keeps the per-system
//! call sites a single line and means there's exactly one home for the
//! `// allowed: typed↔raw kernel boundary` annotation per direction.

use astrodyn::{
    kilogram, AngularVelocity, BodyAttitude, BodyFrame, Frame, InertiaTensor, Mass, MassProperties,
    MassPropertiesTyped, Planet, PlanetInertial, Position, RootInertial, RotationalState,
    RotationalStateTyped, SelfRef, StructuralFrame, TranslationalState, TranslationalStateTyped,
    Vehicle, Velocity,
};

/// Convert a typed `MassPropertiesTyped<V>` into the raw struct the kernel
/// functions consume. `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn mass_typed_to_raw<V: Vehicle>(m: &MassPropertiesTyped<V>) -> MassProperties {
    MassProperties {
        mass: m.mass.get::<kilogram>(),
        inverse_mass: m.inverse_mass,
        inertia: m.inertia.as_dmat3(),
        inverse_inertia: m.inverse_inertia,
        position: m.center_of_mass.raw_si(),
        t_parent_this: m.t_parent_this,
        dirty: m.dirty,
    }
}

/// Lift a raw `MassProperties` struct emitted by a kernel back into the typed
/// sibling. `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn mass_raw_to_typed<V: Vehicle>(mp: &MassProperties) -> MassPropertiesTyped<V> {
    MassPropertiesTyped::<V>::with_inertia(
        Mass::new::<kilogram>(mp.mass),
        InertiaTensor::<BodyFrame<V>>::from_dmat3_unchecked(mp.inertia),
        Position::<StructuralFrame<V>>::from_raw_si(mp.position),
    )
    .with_t_parent_this(mp.t_parent_this)
}

/// Convert a typed `RotationalStateTyped<V>` into the raw struct.
/// `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn rot_typed_to_raw<V: Vehicle>(s: &RotationalStateTyped<V>) -> RotationalState {
    RotationalState {
        quaternion: s.q_inertial_body.to_jeod_quat(),
        ang_vel_body: s.ang_vel_body.raw_si(),
    }
}

/// Lift a raw `RotationalState` struct emitted by a kernel back into the typed
/// sibling. `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn rot_raw_to_typed<V: Vehicle>(s: &RotationalState) -> RotationalStateTyped<V> {
    RotationalStateTyped::<V>::new(
        BodyAttitude::from_jeod_quat(s.quaternion),
        AngularVelocity::<BodyFrame<V>>::from_raw_si(s.ang_vel_body),
    )
}

/// Convert a typed `TranslationalStateTyped<F>` into the raw struct.
/// `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn trans_typed_to_raw<F: Frame>(s: &TranslationalStateTyped<F>) -> TranslationalState {
    TranslationalState {
        position: s.position.raw_si(),
        velocity: s.velocity.raw_si(),
    }
}

/// Lift a raw `TranslationalState` struct emitted by a kernel back into the
/// typed sibling. `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn trans_raw_to_typed<F: Frame>(s: &TranslationalState) -> TranslationalStateTyped<F> {
    TranslationalStateTyped::<F> {
        position: Position::<F>::from_raw_si(s.position),
        velocity: Velocity::<F>::from_raw_si(s.velocity),
    }
}

/// Specialization of `trans_raw_to_typed` for `RootInertial`. Used at the
/// gateway entry sites where the body always ends up phantom-tagged with
/// `RootInertial`. `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn trans_raw_to_root(s: &TranslationalState) -> TranslationalStateTyped<RootInertial> {
    trans_raw_to_typed::<RootInertial>(s)
}

/// Specialization of `trans_raw_to_typed` for `PlanetInertial<P>` —
/// `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn trans_raw_to_planet<P: Planet>(
    s: &TranslationalState,
) -> TranslationalStateTyped<PlanetInertial<P>> {
    trans_raw_to_typed::<PlanetInertial<P>>(s)
}

/// Specialization of `rot_raw_to_typed` for `SelfRef`. Used by every
/// adapter site that writes back into a `RotationalStateC` Component.
/// `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn rot_raw_to_self_ref(s: &RotationalState) -> RotationalStateTyped<SelfRef> {
    rot_raw_to_typed::<SelfRef>(s)
}

/// Specialization of `mass_raw_to_typed` for `SelfRef`. Used by every
/// adapter site that writes back into a `MassPropertiesC` Component.
/// `// allowed: typed↔raw kernel boundary`.
#[inline]
pub fn mass_raw_to_self_ref(mp: &MassProperties) -> MassPropertiesTyped<SelfRef> {
    mass_raw_to_typed::<SelfRef>(mp)
}
