//! Typed↔raw kernel-boundary helpers for the verif_jeod test suite.
//!
//! The `from_untyped_unchecked` named opt-ins on
//! `TranslationalStateTyped`/`RotationalStateTyped`/`MassPropertiesTyped`
//! were deleted in #397. The Tier 3 tests in `tests/tier3_*.rs` and the
//! per-family scenario builders in `src/run_verification/sim_*.rs`
//! still need a one-line typed↔raw bridge, so this module provides
//! `pub fn` shims that mirror the deleted helpers without re-introducing
//! a method-on-the-typed-sibling surface. Each shim carries the
//! `// allowed: typed↔raw kernel boundary` annotation in its body so
//! reviewers can grep for the surface area.

use astrodyn::{
    kilogram, AngularVelocity, BodyAttitude, BodyFrame, InertiaTensor, Mass, MassProperties,
    MassPropertiesTyped, Position, RootInertial, RotationalState, RotationalStateTyped, SelfRef,
    StructuralFrame, TranslationalState, TranslationalStateTyped, Velocity,
};

/// Convert a typed `TranslationalStateTyped<F>` into the raw struct.
#[inline]
pub fn trans_typed_to_raw<F: astrodyn::Frame>(
    s: &TranslationalStateTyped<F>,
) -> TranslationalState {
    // allowed: typed↔raw kernel boundary
    TranslationalState {
        position: s.position.raw_si(),
        velocity: s.velocity.raw_si(),
    }
}

/// Lift a raw `TranslationalState` struct into the typed sibling.
#[inline]
pub fn trans_raw_to_typed<F: astrodyn::Frame>(
    s: &TranslationalState,
) -> TranslationalStateTyped<F> {
    // allowed: typed↔raw kernel boundary
    TranslationalStateTyped::<F> {
        position: Position::<F>::from_raw_si(s.position),
        velocity: Velocity::<F>::from_raw_si(s.velocity),
    }
}

/// Specialization for `RootInertial` — common entry tag.
#[inline]
pub fn trans_raw_to_root(s: &TranslationalState) -> TranslationalStateTyped<RootInertial> {
    trans_raw_to_typed::<RootInertial>(s)
}

/// Convert a typed `RotationalStateTyped<V>` into the raw struct.
#[inline]
pub fn rot_typed_to_raw<V: astrodyn::Vehicle>(s: &RotationalStateTyped<V>) -> RotationalState {
    // allowed: typed↔raw kernel boundary
    RotationalState {
        quaternion: s.q_inertial_body.to_jeod_quat(),
        ang_vel_body: s.ang_vel_body.raw_si(),
    }
}

/// Lift a raw `RotationalState` struct into the typed sibling.
#[inline]
pub fn rot_raw_to_typed<V: astrodyn::Vehicle>(s: &RotationalState) -> RotationalStateTyped<V> {
    // allowed: typed↔raw kernel boundary
    RotationalStateTyped::<V>::new(
        BodyAttitude::from_jeod_quat(s.quaternion),
        AngularVelocity::<BodyFrame<V>>::from_raw_si(s.ang_vel_body),
    )
}

/// Specialization for `SelfRef` — the storage-side wildcard tag.
#[inline]
pub fn rot_raw_to_self_ref(s: &RotationalState) -> RotationalStateTyped<SelfRef> {
    rot_raw_to_typed::<SelfRef>(s)
}

/// Convert a typed `MassPropertiesTyped<V>` into the raw struct.
#[inline]
pub fn mass_typed_to_raw<V: astrodyn::Vehicle>(m: &MassPropertiesTyped<V>) -> MassProperties {
    // allowed: typed↔raw kernel boundary
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

/// Lift a raw `MassProperties` struct into the typed sibling.
#[inline]
pub fn mass_raw_to_typed<V: astrodyn::Vehicle>(mp: &MassProperties) -> MassPropertiesTyped<V> {
    // allowed: typed↔raw kernel boundary
    MassPropertiesTyped::<V>::with_inertia(
        Mass::new::<kilogram>(mp.mass),
        InertiaTensor::<BodyFrame<V>>::from_dmat3_unchecked(mp.inertia),
        Position::<StructuralFrame<V>>::from_raw_si(mp.position),
    )
    .with_t_parent_this(mp.t_parent_this)
}

/// Specialization for `SelfRef` — the storage-side wildcard tag.
#[inline]
pub fn mass_raw_to_self_ref(mp: &MassProperties) -> MassPropertiesTyped<SelfRef> {
    mass_raw_to_typed::<SelfRef>(mp)
}
