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
    /// byte-identical structs for the same input mass. Both
    /// constructors now compute the inverse via glam's general 3×3
    /// inverse formula; the previous element-wise `IDENTITY / m` form
    /// in `new` differed by sub-ULP amounts on the diagonal and ~1e-25
    /// on the off-diagonals from adjugate cancellations, which amplified
    /// to ~91 km position error over a 7-day Clementine rotational-
    /// dynamics integration.
    ///
    /// Coverage scope: this test exercises the *default* configuration
    /// of `MassProperties::new(mass)`, where `center_of_mass = ZERO`
    /// and `t_parent_this = IDENTITY`. The field-by-field assertions
    /// would therefore not catch a regression that drops one of those
    /// trivially-zero / trivially-identity fields from the bridge —
    /// `assert_eq!(ZERO, ZERO)` and `assert_eq!(IDENTITY, IDENTITY)`
    /// trivially pass even if either side never read the field at all.
    /// The companion test
    /// `non_default_mass_props_round_trip_across_construction_paths`
    /// closes that gap with a non-zero CoM, a non-identity
    /// `t_parent_this`, and a non-diagonal inertia tensor.
    ///
    /// The assertions below are stated explicitly per-field rather than
    /// via a struct-level `PartialEq` comparison. `MassPropertiesTyped<V>`
    /// does derive `PartialEq`, but the derive synthesizes a
    /// `V: PartialEq` bound on the impl, and the `SelfRef` vehicle
    /// marker used here only derives `Debug + Clone + Copy` (`PartialEq`
    /// is intentionally omitted because the type is a zero-sized
    /// phantom tag with no distinguishing state). Direct
    /// `assert_eq!(a, b)` on `MassPropertiesTyped<SelfRef>` therefore
    /// does not compile; the field-by-field projection sidesteps that
    /// without losing any coverage — see the `to_untyped()` projection
    /// assertion below which exercises the untyped sibling's
    /// struct-level `PartialEq`.
    #[test]
    fn point_mass_inverse_inertia_matches_across_construction_paths() {
        let a = MassPropertiesTyped::<SelfRef>::new(Mass::new::<kilogram>(424.0));
        let b = mass_raw_to_self_ref(&MassProperties::new(424.0));
        // Cache fields — the primary regression class.
        assert_eq!(a.inverse_inertia, b.inverse_inertia);
        assert_eq!(a.inverse_mass, b.inverse_mass);
        // Stored inputs and derived storage fields.
        assert_eq!(a.mass, b.mass);
        assert_eq!(a.inertia.as_dmat3(), b.inertia.as_dmat3());
        assert_eq!(a.center_of_mass.raw_si(), b.center_of_mass.raw_si());
        assert_eq!(a.t_parent_this, b.t_parent_this);
        // Bookkeeping flag — constructors leave caches consistent, so
        // `dirty` is `false` on both sides.
        assert_eq!(a.dirty, b.dirty);
        assert!(!a.dirty);
        // Untyped projection equality closes the field coverage: any
        // future field added to `MassPropertiesTyped` that is also
        // exported into the untyped `MassProperties` will be compared
        // here verbatim, which catches the same dropped-field class as
        // the proptest round-trips in `crates/astrodyn_dynamics/src/mass.rs`.
        assert_eq!(
            MassPropertiesTyped::<SelfRef>::to_untyped(&a),
            MassPropertiesTyped::<SelfRef>::to_untyped(&b),
        );
    }

    /// Companion to
    /// `point_mass_inverse_inertia_matches_across_construction_paths`
    /// that exercises a **non-default** configuration: non-zero
    /// centre-of-mass offset, non-identity `t_parent_this`, and a
    /// non-diagonal inertia tensor. The point-mass test above only
    /// covers `center_of_mass = ZERO` and `t_parent_this = IDENTITY`,
    /// so dropping either of those fields from the raw→typed bridge
    /// would still let it pass (`assert_eq!(ZERO, ZERO)` and
    /// `assert_eq!(IDENTITY, IDENTITY)` are vacuous). This test
    /// constructs a raw `MassProperties` whose fields each carry a
    /// distinct, distinguishable value, then asserts the
    /// `mass_raw_to_self_ref` bridge propagates every one of them
    /// verbatim — the field-by-field assertions are the field-drop
    /// regression fence.
    #[test]
    fn non_default_mass_props_round_trip_across_construction_paths() {
        use glam::{DMat3, DVec3};

        let mass = 424.0_f64;
        // Non-diagonal, well-conditioned inertia: diag conjugated by a
        // small rotation so off-diagonal entries are non-zero but
        // det != 0 and the inverse is well-defined.
        let diag = DMat3::from_diagonal(DVec3::new(100.0, 200.0, 300.0));
        let rot = DMat3::from_axis_angle(DVec3::new(1.0, 2.0, 3.0).normalize(), 0.5_f64);
        let inertia = rot.transpose() * diag * rot;
        let com = DVec3::new(0.1, -0.2, 0.3);
        // Non-identity `t_parent_this` — Apollo regression class
        // (#393): a 180° rotation about Z, the same eigen-rotation
        // SIM_Apollo's modules declare.
        let t_parent_this = DMat3::from_axis_angle(DVec3::Z, std::f64::consts::PI);

        // `MassProperties::with_inertia` doesn't set `t_parent_this`
        // (it stays at `IDENTITY`), so populate the raw struct
        // directly — the bridge has to carry every field through
        // regardless of which constructor produced the raw form.
        let raw = MassProperties {
            mass,
            inverse_mass: 1.0 / mass,
            inertia,
            inverse_inertia: inertia.inverse(),
            position: com,
            t_parent_this,
            dirty: false,
        };

        let typed = mass_raw_to_self_ref(&raw);

        // Every non-trivial field is asserted distinctly so a dropped
        // field can't slide through with a default value.
        assert_eq!(typed.mass.get::<kilogram>(), raw.mass);
        assert_eq!(typed.inverse_mass, raw.inverse_mass);
        assert_eq!(typed.inertia.as_dmat3(), raw.inertia);
        assert_eq!(typed.inverse_inertia, raw.inverse_inertia);
        assert_eq!(typed.center_of_mass.raw_si(), raw.position);
        assert_eq!(typed.t_parent_this, raw.t_parent_this);
        assert_eq!(typed.dirty, raw.dirty);

        // Negative controls: confirm the values are actually
        // distinguishable from the defaults so the assertions above
        // can't pass vacuously.
        assert_ne!(raw.position, DVec3::ZERO);
        assert_ne!(raw.t_parent_this, DMat3::IDENTITY);

        // Round-trip via the untyped projection exercises the
        // struct-level `PartialEq` and catches any future field added
        // to one side but not the other.
        assert_eq!(MassPropertiesTyped::<SelfRef>::to_untyped(&typed), raw);
    }
}
