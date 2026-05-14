//! Typed↔raw kernel-boundary helpers.
//!
//! The `from_untyped_unchecked` named opt-ins on
//! `TranslationalStateTyped`/`RotationalStateTyped`/`MassPropertiesTyped`,
//! and the `MassPropertiesC::from_untyped` / `RotationalStateC::from_untyped`
//! Bevy Component opt-ins, were deleted in #397. The kernel functions in
//! `astrodyn` still take raw `RotationalState` / `TranslationalState` /
//! `MassProperties` structs by design (typing the integrator interfaces
//! themselves was out of scope for #397), so adapters and verification
//! fixtures must translate at the boundary anyway.
//!
//! Centralizing those translations here keeps the per-system call sites a
//! single line and means there's exactly one home for the
//! `// allowed: typed↔raw kernel boundary` annotation per direction.
//! Consumers (the Bevy adapter, the verif crates, and the runner's own
//! kernel-boundary sites) all import from `astrodyn::typed_bridge::*` so
//! there is one canonical implementation, not N.

use astrodyn_dynamics::mass::MassPropertiesTyped;
use astrodyn_dynamics::rotational::RotationalStateTyped;
use astrodyn_dynamics::state::TranslationalStateTyped;
use astrodyn_dynamics::{MassProperties, RotationalState, TranslationalState};
use astrodyn_quantities::aliases::{AngularVelocity, InertiaTensor, Position, Velocity};
use astrodyn_quantities::body_attitude::BodyAttitude;
use astrodyn_quantities::frame::{
    BodyFrame, Frame, Planet, PlanetInertial, RootInertial, SelfRef, StructuralFrame, Vehicle,
};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

/// Convert a typed `MassPropertiesTyped<V>` into the raw struct the
/// kernel functions consume. Field-by-field copy preserving the
/// caller's `inverse_mass`, `inverse_inertia`, and `dirty` exactly —
/// no recomputation. Mirror of [`mass_raw_to_typed`].
#[inline]
pub fn mass_typed_to_raw<V: Vehicle>(m: &MassPropertiesTyped<V>) -> MassProperties {
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

/// Lift a raw `MassProperties` struct emitted by a kernel back into
/// the typed sibling.
///
/// **Note:** routes through [`MassPropertiesTyped::with_inertia`], which
/// **recomputes `inverse_mass = 1/mass` and `inverse_inertia = inertia⁻¹`
/// from the freshly-supplied inputs and resets `dirty = false`.** The
/// caller's `mp.inverse_mass` / `mp.inverse_inertia` / `mp.dirty` are
/// not propagated — the rebuilt typed sibling carries fresh, consistent
/// derived values regardless of the raw input's bookkeeping. Round-trips
/// through `mass_typed_to_raw` → `mass_raw_to_typed` therefore canonicalize
/// the dirty flag and re-derive the inverses; this is intentional (the
/// JEOD invariant `MA.04` requires `inertia · inverse_inertia ≈ I` and
/// `with_inertia` is the canonical re-derivation site).
#[inline]
pub fn mass_raw_to_typed<V: Vehicle>(mp: &MassProperties) -> MassPropertiesTyped<V> {
    MassPropertiesTyped::<V>::with_inertia(
        Mass::new::<kilogram>(mp.mass),
        InertiaTensor::<BodyFrame<V>>::from_dmat3_unchecked(mp.inertia), // allowed: typed↔raw kernel boundary
        Position::<StructuralFrame<V>>::from_raw_si(mp.position), // allowed: typed↔raw kernel boundary
    )
    .with_t_parent_this(mp.t_parent_this)
}

/// Convert a typed `RotationalStateTyped<V>` into the raw struct.
#[inline]
pub fn rot_typed_to_raw<V: Vehicle>(s: &RotationalStateTyped<V>) -> RotationalState {
    // allowed: typed↔raw kernel boundary
    RotationalState {
        quaternion: s.q_inertial_body.to_jeod_quat(),
        ang_vel_body: s.ang_vel_body.raw_si(),
    }
}

/// Lift a raw `RotationalState` struct emitted by a kernel back into
/// the typed sibling. Validates the quaternion's unit norm via
/// [`BodyAttitude::from_jeod_quat`] (panics on drift past
/// `NormalizedQuat::DEFAULT_TOLERANCE`).
#[inline]
pub fn rot_raw_to_typed<V: Vehicle>(s: &RotationalState) -> RotationalStateTyped<V> {
    RotationalStateTyped::<V>::new(
        BodyAttitude::from_jeod_quat(s.quaternion),
        AngularVelocity::<BodyFrame<V>>::from_raw_si(s.ang_vel_body), // allowed: typed↔raw kernel boundary
    )
}

/// Convert a typed `TranslationalStateTyped<F>` into the raw struct.
#[inline]
pub fn trans_typed_to_raw<F: Frame>(s: &TranslationalStateTyped<F>) -> TranslationalState {
    // allowed: typed↔raw kernel boundary
    TranslationalState {
        position: s.position.raw_si(),
        velocity: s.velocity.raw_si(),
    }
}

/// Lift a raw `TranslationalState` struct emitted by a kernel back into
/// the typed sibling.
#[inline]
pub fn trans_raw_to_typed<F: Frame>(s: &TranslationalState) -> TranslationalStateTyped<F> {
    TranslationalStateTyped::<F> {
        position: Position::<F>::from_raw_si(s.position), // allowed: typed↔raw kernel boundary
        velocity: Velocity::<F>::from_raw_si(s.velocity), // allowed: typed↔raw kernel boundary
    }
}

/// Specialization of [`trans_raw_to_typed`] for `RootInertial`. Used at
/// the gateway entry sites where the body always ends up phantom-tagged
/// with `RootInertial`.
#[inline]
pub fn trans_raw_to_root(s: &TranslationalState) -> TranslationalStateTyped<RootInertial> {
    trans_raw_to_typed::<RootInertial>(s)
}

/// Specialization of [`trans_raw_to_typed`] for `PlanetInertial<P>`.
#[inline]
pub fn trans_raw_to_planet<P: Planet>(
    s: &TranslationalState,
) -> TranslationalStateTyped<PlanetInertial<P>> {
    trans_raw_to_typed::<PlanetInertial<P>>(s)
}

/// Specialization of [`rot_raw_to_typed`] for `SelfRef`. Used by every
/// adapter site that writes back into a `RotationalStateC` Component.
#[inline]
pub fn rot_raw_to_self_ref(s: &RotationalState) -> RotationalStateTyped<SelfRef> {
    rot_raw_to_typed::<SelfRef>(s)
}

/// Specialization of [`mass_raw_to_typed`] for `SelfRef`. Used by every
/// adapter site that writes back into a `MassPropertiesC` Component.
/// Inherits the `with_inertia` recomputation behavior documented on
/// [`mass_raw_to_typed`].
#[inline]
pub fn mass_raw_to_self_ref(mp: &MassProperties) -> MassPropertiesTyped<SelfRef> {
    mass_raw_to_typed::<SelfRef>(mp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MassPropertiesTyped::<V>::new(mass)` and the raw→typed bridge
    /// (`MassProperties::new(mass)` → `mass_raw_to_self_ref`, which routes
    /// through `MassPropertiesTyped::with_inertia`) must produce
    /// byte-identical `inverse_inertia` for the same input mass. Both
    /// constructors now compute the inverse via glam's general 3×3
    /// inverse formula; the previous element-wise `IDENTITY / m` form
    /// in `new` differed by sub-ULP amounts on the diagonal and ~1e-25
    /// on the off-diagonals from adjugate cancellations, which amplified
    /// to ~91 km position error over a 7-day Clementine rotational-
    /// dynamics integration.
    #[test]
    fn point_mass_inverse_inertia_matches_across_construction_paths() {
        let a = MassPropertiesTyped::<SelfRef>::new(Mass::new::<kilogram>(424.0));
        let b = mass_raw_to_self_ref(&MassProperties::new(424.0));
        assert_eq!(a.inverse_inertia, b.inverse_inertia);
        // Sanity: the other derived caches and the inertia tensor also
        // agree, so the full struct round-trips bit-for-bit.
        assert_eq!(a.inverse_mass, b.inverse_mass);
        assert_eq!(a.inertia.as_dmat3(), b.inertia.as_dmat3());
    }
}
