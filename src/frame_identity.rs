//! Shared frame-identity minting conventions for hosts.
//!
//! The compile-time path mints identities via
//! [`FrameUid::of`](astrodyn_quantities::frame_descriptor::FrameUid::of)
//! in [`Namespace::LOCAL`] — exclusively type-derived. Some frames, however,
//! belong to *open, instance-scoped* sets: mission configurations
//! legitimately spawn bodies in loops and populate ground-site catalogs from
//! data, so their identity is a mission-supplied **value** (a body name, a
//! site key), not a type. This module is the single home of that convention,
//! shared by every host (the runner today, the Bevy adapter's `FrameUidC` in
//! issue #664) so the mapping from a value to its identity is shared *code*,
//! never a per-host convention that can drift.
//!
//! ## Namespace allocation
//!
//! | Namespace | Owner | Mint |
//! |---|---|---|
//! | `LOCAL` (0) | type-derived identities | `FrameUid::of::<F>()` only |
//! | [`MISSION_NAMED_NS`] (1) | mission-supplied **value** identities (named bodies, topocentric sites) | [`named_body_frame_uid`], [`topocentric_site_frame_uid`] |
//! | ≥ 2 | host-allocated (imports, external producers) | `FrameUid::external` / `FrameTree::import_subtree` |
//!
//! Within [`MISSION_NAMED_NS`] the `(class, role)` pair keeps the value kinds
//! disjoint — named bodies mint `(Body, CompositeBody)`, topocentric sites
//! mint `(Topocentric, Primary)` — so a body name and a site key can never
//! alias even when both are `Tag::Named`. Two values *of the same kind* are
//! distinguished by their tag (RF.14).
//!
//! Hosts importing foreign frame trees **must allocate a namespace ≥ 2**:
//! reusing namespace 1 would let an imported frame impersonate a
//! mission-supplied identity — the duplicate-identity check catches exact
//! collisions, but distinct foreign values would silently coexist as if
//! mission-supplied.

use astrodyn_quantities::frame_descriptor::{FrameClass, FrameRole, FrameUid, Namespace, Tag};

/// Namespace reserved for mission-supplied **value** identities — frames
/// whose identity is a configuration value (a body name, a site key) rather
/// than a compile-time marker type. Named bodies (`VehicleConfig::named`,
/// `VehicleBuilder::vehicle_named` → [`named_body_frame_uid`]) and
/// topocentric sites ([`topocentric_site_frame_uid`]) both live here, kept
/// disjoint by their `(class, role)` pair. Type-derived identities live in
/// [`Namespace::LOCAL`]; a named `"iss"` body and a `BodyFrame<Iss>` body
/// therefore never alias.
pub const MISSION_NAMED_NS: Namespace = Namespace(1);

/// Mint the runtime identity of a mission-named body's composite-body
/// frame. The single shared mint for *named bodies* in [`MISSION_NAMED_NS`]
/// (its sibling [`topocentric_site_frame_uid`] mints sites in the same
/// namespace) — the runner and the Bevy adapter both route named-body
/// identity through here, so the convention cannot drift between hosts.
pub fn named_body_frame_uid(name: &str) -> FrameUid {
    FrameUid::external(
        MISSION_NAMED_NS,
        FrameClass::Body,
        FrameRole::CompositeBody,
        Tag::Named(name.into()),
    )
}

/// Mint the runtime identity of a topocentric (ENU) frame anchored at a fixed
/// geodetic site on `planet`. The single shared mint for site identities in
/// [`MISSION_NAMED_NS`]: every host — independent producer and consumer —
/// routes through here, so a `(planet, site_key)` pair maps to the **same**
/// [`FrameUid`] across processes without coordination. That convergence is
/// the property a scene store / visualizer relies on to recognize a
/// producer's frames (see [#688]).
///
/// Sites are an *open, config-driven* set (launch / landing / station
/// catalogs populated from data), so — like mission-named bodies
/// ([`named_body_frame_uid`]) and unlike compile-time frame types — their
/// identity is a **value**, not a type: there is deliberately no
/// [`FrameUid::of`] / `is::<…>` path for a specific site.
///
/// # Canonical tag composition (stable cross-process contract)
///
/// The instance tag is `Tag::Named("{planet}/site:{site_key}")`. The planet
/// name qualifies the key so the same `site_key` on two planets cannot
/// collide, and the `site:` segment follows the [`Tag::Named`] site
/// convention. Producer- and consumer-minted uids are byte-identical *only
/// because both call this one function*, so the exact string is a contract:
/// it is pinned by `topocentric_site_uid_format_is_stable` and must not
/// drift. The tag is an **opaque identity key, not a parseable structure** —
/// recovering `(planet, site_key)` from it is out of scope (that is the
/// deferred structured-`Tag::Site` option). A future typed `Topocentric<P,
/// S>` sugar, if ever added, must lower through this same composition so the
/// typed and value spellings of a site converge on one identity.
///
/// Pass the planet's marker `NAME` verbatim (e.g.
/// [`Planet::NAME`](astrodyn_quantities::frame::Planet::NAME)); the mint is
/// case-sensitive, matching [`sealed_planet_inertial_uid`]. The site
/// `site_key` is a **label**: nothing here validates it against the geodetic
/// anchor carried by the `FrameTransform` the site builder returns — the
/// anchor is authoritative ([#688] req 6; the transform is orientation-only,
/// [#689]).
///
/// [#688]: https://github.com/simnaut/astrodyn/issues/688
/// [#689]: https://github.com/simnaut/astrodyn/issues/689
pub fn topocentric_site_frame_uid(planet: &str, site_key: &str) -> FrameUid {
    FrameUid::external(
        MISSION_NAMED_NS,
        FrameClass::Topocentric,
        FrameRole::Primary,
        Tag::Named(format!("{planet}/site:{site_key}").into()),
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

/// Resolve a sealed built-in planet's **inertial-frame** identity from
/// its name, or `None` for any other name. The single string-dispatch
/// table shared by every host's name-keyed source registration (the
/// runner's `add_source`, the Bevy adapter's `spawn_source`) — a
/// per-host copy of this match could drift and mint *different
/// identities for the same source name across backends*, the exact
/// failure class value identity exists to kill.
///
/// Unknown names return `None` so each host can fail loudly with its
/// own registration-path guidance (`add_source_typed::<P>` vs
/// `SimulationBuilder::add_source_typed`). The pfix sibling, where the
/// source rotates, is [`pfix_sibling_uid`] of the returned identity —
/// pinned equal to the typed mint for all six planets.
pub fn sealed_planet_inertial_uid(name: &str) -> Option<FrameUid> {
    use astrodyn_quantities::frame::{Earth, Jupiter, Mars, Moon, PlanetInertial, Saturn, Sun};
    Some(match name {
        "Earth" => FrameUid::of::<PlanetInertial<Earth>>(),
        "Moon" => FrameUid::of::<PlanetInertial<Moon>>(),
        "Sun" => FrameUid::of::<PlanetInertial<Sun>>(),
        "Mars" => FrameUid::of::<PlanetInertial<Mars>>(),
        "Jupiter" => FrameUid::of::<PlanetInertial<Jupiter>>(),
        "Saturn" => FrameUid::of::<PlanetInertial<Saturn>>(),
        _ => return None,
    })
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
    fn sealed_planet_dispatch_matches_typed_mint() {
        use astrodyn_quantities::frame::{Earth, Jupiter, Mars, Moon, PlanetInertial, Saturn, Sun};
        macro_rules! check {
            ($($p:ty),+) => {$(
                assert_eq!(
                    sealed_planet_inertial_uid(<$p as astrodyn_quantities::frame::Planet>::NAME),
                    Some(FrameUid::of::<PlanetInertial<$p>>()),
                    "name dispatch must equal the typed mint",
                );
            )+};
        }
        check!(Earth, Moon, Sun, Mars, Jupiter, Saturn);
        assert_eq!(sealed_planet_inertial_uid("Pluto"), None);
        assert_eq!(sealed_planet_inertial_uid("earth"), None, "case-sensitive");
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

    #[test]
    fn topocentric_site_identities_are_key_keyed() {
        // Same (planet, site) → same identity across calls: the cross-process
        // convergence contract a producer and a consumer rely on.
        assert_eq!(
            topocentric_site_frame_uid("Earth", "KSC-LC39A"),
            topocentric_site_frame_uid("Earth", "KSC-LC39A"),
        );
        // Distinct sites on the same planet → distinct identities (RF.14) —
        // the gap #688 was filed to close.
        assert_ne!(
            topocentric_site_frame_uid("Earth", "KSC-LC39A"),
            topocentric_site_frame_uid("Earth", "Baikonur-1-5"),
        );
        // Same key on different planets → distinct identities (planet-qualified).
        assert_ne!(
            topocentric_site_frame_uid("Earth", "Base-Alpha"),
            topocentric_site_frame_uid("Moon", "Base-Alpha"),
        );
    }

    #[test]
    fn topocentric_site_uid_format_is_stable() {
        // The canonical composition is a cross-process contract: changing any
        // field or the tag string breaks convergence with already-recorded
        // documents and with independent producers.
        let uid = topocentric_site_frame_uid("Earth", "KSC-LC39A");
        assert_eq!(uid.namespace, MISSION_NAMED_NS);
        assert_eq!(uid.class, FrameClass::Topocentric);
        assert_eq!(uid.role, FrameRole::Primary);
        assert_eq!(uid.tag, Tag::Named("Earth/site:KSC-LC39A".into()));
        assert_eq!(uid.to_string(), "ns1:Earth/site:KSC-LC39A.topo");
    }

    #[test]
    fn topocentric_site_never_aliases_named_body() {
        // Both kinds live in MISSION_NAMED_NS; the (class, role) pair keeps
        // them disjoint even if a body name and a site key were spelled
        // identically — the invariant that lets the two value kinds share one
        // namespace.
        let site = topocentric_site_frame_uid("Earth", "iss");
        let body = named_body_frame_uid("iss");
        assert_ne!(site, body);
        assert_eq!(site.namespace, body.namespace, "same namespace…");
        assert_ne!(site.class, body.class, "…disjoint by class");
        assert_ne!(site.role, body.role, "…and by role");
    }
}
