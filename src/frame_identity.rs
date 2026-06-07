//! Shared frame-identity minting conventions for hosts.
//!
//! The compile-time path mints identities via
//! [`FrameUid::of`](astrodyn_quantities::frame_descriptor::FrameUid::of)
//! in [`Namespace::LOCAL`] — exclusively type-derived. Bodies, however, are
//! an *open, instance-scoped* set: mission configurations legitimately
//! spawn them in loops, so their identity is a mission-supplied **value**
//! (a name), not a type. This module is the single home of that
//! convention, shared by every host (the runner today, the Bevy adapter's
//! `FrameUidC` in issue #664) so the mapping from a body name to its
//! identity is shared *code*, never a per-host convention that can drift.
//!
//! ## Namespace allocation
//!
//! | Namespace | Owner | Mint |
//! |---|---|---|
//! | `LOCAL` (0) | type-derived identities | `FrameUid::of::<F>()` only |
//! | [`MISSION_NAMED_NS`] (1) | mission-named bodies | [`named_body_frame_uid`] |
//! | ≥ 2 | host-allocated (imports, external producers) | `FrameUid::external` / `FrameTree::import_subtree` |
//!
//! Hosts importing foreign frame trees **must allocate a namespace ≥ 2**:
//! reusing namespace 1 would let an imported body impersonate a
//! mission-named body — the duplicate-identity check catches exact
//! collisions, but distinct foreign names would silently coexist as if
//! mission-named.

use astrodyn_quantities::frame_descriptor::{FrameClass, FrameRole, FrameUid, Namespace, Tag};

/// Namespace reserved for mission-supplied *named* bodies — bodies whose
/// identity is a configuration value (`VehicleConfig::named`,
/// `VehicleBuilder::vehicle_named`) rather than a compile-time `Vehicle`
/// marker. Type-derived identities live in [`Namespace::LOCAL`]; a named
/// `"iss"` body and a `BodyFrame<Iss>` body therefore never alias.
pub const MISSION_NAMED_NS: Namespace = Namespace(1);

/// Mint the runtime identity of a mission-named body's composite-body
/// frame. The single shared mint for [`MISSION_NAMED_NS`] — the runner and
/// the Bevy adapter both route named-body identity through here, so the
/// convention cannot drift between hosts.
pub fn named_body_frame_uid(name: &str) -> FrameUid {
    FrameUid::external(
        MISSION_NAMED_NS,
        FrameClass::Body,
        FrameRole::CompositeBody,
        Tag::Named(name.into()),
    )
}

/// The planet-fixed sibling of a gravity source's **inertial-frame**
/// identity: same namespace, role, and tag, with the class swapped to
/// [`FrameClass::PlanetFixed`]. The single shared derivation both hosts
/// use when a source's pfix frame is created from its carried inertial
/// identity (issue #664: the Bevy adapter's frame-registration systems
/// derive the pfix identity from the source entity's `FrameUidC`), so
/// the inertial↔pfix pairing cannot drift between hosts.
///
/// For type-derived identities this provably equals the typed mint:
/// `pfix_sibling_uid(&FrameUid::of::<PlanetInertial<P>>()) ==
/// FrameUid::of::<PlanetFixed<P>>()` for every sealed planet (both are
/// `{LOCAL, class, Primary, Named(P::NAME)}` differing only in class) —
/// pinned by this module's tests.
pub fn pfix_sibling_uid(inertial: &FrameUid) -> FrameUid {
    assert!(
        inertial.class == FrameClass::PlanetInertial,
        "pfix_sibling_uid: identity `{inertial}` has class {:?}, not \
         PlanetInertial — only a planet's inertial-frame identity has a \
         planet-fixed sibling.",
        inertial.class
    );
    FrameUid {
        namespace: inertial.namespace,
        class: FrameClass::PlanetFixed,
        role: inertial.role.clone(),
        tag: inertial.tag.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use astrodyn_quantities::frame::BodyFrame;
    use astrodyn_quantities::frame_descriptor::FrameUid;

    astrodyn_quantities::define_vehicle!(IdentityTestVehicle);

    #[test]
    fn named_identity_never_aliases_typed_identity() {
        // Even a named body whose name exactly matches a Vehicle marker's
        // NAME lives in a different namespace — impersonation of a
        // type-derived identity is impossible by construction.
        let named = named_body_frame_uid("IdentityTestVehicle");
        let typed = FrameUid::of::<BodyFrame<IdentityTestVehicle>>();
        assert_ne!(named, typed);
        assert_eq!(named.namespace, MISSION_NAMED_NS);
        assert!(!named.is::<BodyFrame<IdentityTestVehicle>>());
    }

    #[test]
    fn pfix_sibling_matches_typed_mint_for_every_sealed_planet() {
        use astrodyn_quantities::frame::{
            Earth, Jupiter, Mars, Moon, PlanetFixed, PlanetInertial, Saturn, Sun,
        };
        macro_rules! check {
            ($($p:ty),+) => {$(
                assert_eq!(
                    pfix_sibling_uid(&FrameUid::of::<PlanetInertial<$p>>()),
                    FrameUid::of::<PlanetFixed<$p>>(),
                    "pfix sibling derivation must equal the typed mint",
                );
            )+};
        }
        check!(Earth, Moon, Sun, Mars, Jupiter, Saturn);
    }

    #[test]
    #[should_panic(expected = "not PlanetInertial")]
    fn pfix_sibling_rejects_non_inertial_input() {
        let _ = pfix_sibling_uid(&named_body_frame_uid("iss"));
    }

    #[test]
    fn named_identities_are_name_keyed() {
        assert_eq!(named_body_frame_uid("iss"), named_body_frame_uid("iss"));
        assert_ne!(named_body_frame_uid("iss"), named_body_frame_uid("soyuz"));
        assert_eq!(
            named_body_frame_uid("iss").to_string(),
            "ns1:iss.composite_body"
        );
    }
}
