// JEOD_INV: TS.01 — this inventory test enumerates the storage-boundary
// wildcard tags (SelfRef / SelfPlanet / MassNode) by token to assert the
// Wildcard mint set is exact; see `docs/JEOD_invariants.md` row TS.01 and
// the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Inventory test for the frame-descriptor vocabulary (issue #660).
//!
//! Asserts the three load-bearing properties of `Frame::DESCRIPTOR` across
//! representative instantiations of every sealed frame impl:
//!
//! 1. **Injectivity** — every `Stable` instantiation mints a pairwise
//!    distinct `FrameUid` (checked recovery in PR-2 relies on this).
//! 2. **Wildcard exactness** — `is_wildcard()` is true for exactly the
//!    wildcard-tagged frames and `MassNode`, nothing else.
//! 3. **Per-body exactness** — `MintPolicy::PerBodyIntegration` is carried
//!    by exactly `IntegrationFrame`.
//!
//! Local `Vehicle`/`Planet` tags are minted with the public macros so the
//! test also exercises the downstream extension path.

use astrodyn_quantities::frame::{
    BodyFrame, Earth, Ecef, Frame, IntegrationFrame, Lvlh, MassNode, Moon, Ned, PlanetFixed,
    PlanetInertial, RootInertial, SelfPlanet, SelfRef, StructuralFrame, Topocentric,
};
use astrodyn_quantities::{FrameUid, MintPolicy};

/// Macro-minted local tags live in a private module so the generated
/// `pub struct`s are not test-crate-public (workspace `missing_docs` deny).
mod tags {
    astrodyn_quantities::define_planet!(InvPlanet);
    astrodyn_quantities::define_vehicle!(InvVehicle);
}
use tags::{InvPlanet, InvVehicle};

/// Every representative instantiation in one place: `(label, mint, mintable
/// uid if Stable)`. Extending the sealed frame set should extend this list.
fn inventory() -> Vec<(&'static str, MintPolicy, Option<FrameUid>)> {
    fn entry<F: Frame>(label: &'static str) -> (&'static str, MintPolicy, Option<FrameUid>) {
        let mint = F::DESCRIPTOR.mint;
        let uid = match mint {
            MintPolicy::Stable => Some(FrameUid::of::<F>()),
            _ => None,
        };
        (label, mint, uid)
    }
    vec![
        // Unparameterized.
        entry::<RootInertial>("RootInertial"),
        entry::<Ecef>("Ecef"),
        entry::<IntegrationFrame>("IntegrationFrame"),
        entry::<MassNode>("MassNode"),
        // Planet-parameterized, concrete tags.
        entry::<PlanetInertial<Earth>>("PlanetInertial<Earth>"),
        entry::<PlanetInertial<Moon>>("PlanetInertial<Moon>"),
        entry::<PlanetFixed<Earth>>("PlanetFixed<Earth>"),
        entry::<PlanetFixed<Moon>>("PlanetFixed<Moon>"),
        entry::<PlanetFixed<InvPlanet>>("PlanetFixed<InvPlanet>"),
        entry::<Topocentric<Earth>>("Topocentric<Earth>"),
        // Vehicle-parameterized, concrete tags (macro-minted).
        entry::<BodyFrame<InvVehicle>>("BodyFrame<InvVehicle>"),
        entry::<StructuralFrame<InvVehicle>>("StructuralFrame<InvVehicle>"),
        entry::<Lvlh<InvVehicle>>("Lvlh<InvVehicle>"),
        entry::<Ned<InvVehicle>>("Ned<InvVehicle>"),
        // Wildcard-tagged instantiations (TS.01 storage-boundary set).
        entry::<PlanetInertial<SelfPlanet>>("PlanetInertial<SelfPlanet>"),
        entry::<PlanetFixed<SelfPlanet>>("PlanetFixed<SelfPlanet>"),
        entry::<Topocentric<SelfPlanet>>("Topocentric<SelfPlanet>"),
        entry::<BodyFrame<SelfRef>>("BodyFrame<SelfRef>"),
        entry::<StructuralFrame<SelfRef>>("StructuralFrame<SelfRef>"),
        entry::<Lvlh<SelfRef>>("Lvlh<SelfRef>"),
        entry::<Ned<SelfRef>>("Ned<SelfRef>"),
    ]
}

#[test]
fn stable_descriptors_are_pairwise_distinct() {
    let stable: Vec<(&str, FrameUid)> = inventory()
        .into_iter()
        .filter_map(|(label, _, uid)| uid.map(|u| (label, u)))
        .collect();
    assert!(
        stable.len() >= 12,
        "expected at least 12 Stable instantiations in the inventory, got {}",
        stable.len()
    );
    for (i, (label_a, uid_a)) in stable.iter().enumerate() {
        for (label_b, uid_b) in stable.iter().skip(i + 1) {
            assert!(
                uid_a != uid_b,
                "FrameUid collision between `{label_a}` and `{label_b}` (both mint \
                 {uid_a}). Descriptor injectivity is load-bearing for checked typed \
                 recovery: give one of them a distinct class/role/tag."
            );
        }
    }
}

#[test]
fn wildcard_set_is_exactly_self_ref_self_planet_massnode() {
    let expected_wildcards = [
        "MassNode",
        "PlanetInertial<SelfPlanet>",
        "PlanetFixed<SelfPlanet>",
        "Topocentric<SelfPlanet>",
        "BodyFrame<SelfRef>",
        "StructuralFrame<SelfRef>",
        "Lvlh<SelfRef>",
        "Ned<SelfRef>",
    ];
    for (label, mint, _) in inventory() {
        let should_be_wildcard = expected_wildcards.contains(&label);
        let is_wildcard = mint == MintPolicy::Wildcard;
        assert_eq!(
            is_wildcard, should_be_wildcard,
            "`{label}` has mint {mint:?}, but the TS.01 wildcard set says \
             is_wildcard should be {should_be_wildcard}. The wildcard set must stay \
             exactly the SelfRef/SelfPlanet-tagged instantiations plus MassNode."
        );
    }
}

#[test]
fn per_body_integration_is_exactly_integration_frame() {
    for (label, mint, _) in inventory() {
        let should_be_per_body = label == "IntegrationFrame";
        let is_per_body = mint == MintPolicy::PerBodyIntegration;
        assert_eq!(
            is_per_body, should_be_per_body,
            "`{label}` has mint {mint:?}; MintPolicy::PerBodyIntegration must be \
             carried by exactly IntegrationFrame (RF.10)."
        );
    }
}

#[test]
fn display_matches_dotted_convention() {
    assert_eq!(
        FrameUid::of::<PlanetInertial<Earth>>().to_string(),
        "Earth.inertial"
    );
    assert_eq!(
        FrameUid::of::<PlanetFixed<Earth>>().to_string(),
        "Earth.pfix"
    );
    assert_eq!(FrameUid::of::<Ecef>().to_string(), "Earth.pfix.alt");
    assert_eq!(FrameUid::of::<RootInertial>().to_string(), "inertial");
    assert_eq!(
        FrameUid::of::<BodyFrame<InvVehicle>>().to_string(),
        "InvVehicle.composite_body"
    );
    assert_eq!(
        FrameUid::of::<StructuralFrame<InvVehicle>>().to_string(),
        "InvVehicle.structural"
    );
    assert_eq!(
        FrameUid::of::<Lvlh<InvVehicle>>().to_string(),
        "InvVehicle.lvlh"
    );
    assert_eq!(
        FrameUid::of::<Ned<InvVehicle>>().to_string(),
        "InvVehicle.ned"
    );
    assert_eq!(
        FrameUid::of::<Topocentric<Earth>>().to_string(),
        "Earth.topo"
    );
}

#[test]
fn display_is_role_faithful_for_external_combinations() {
    use astrodyn_quantities::{FrameClass, FrameRole, Namespace, Tag};
    // FrameUid::external can construct any class/role combination; the
    // rendered suffix must reflect the actual role, never a collapsed
    // class-conventional guess.
    let core = FrameUid::external(
        Namespace(1),
        FrameClass::Body,
        FrameRole::CoreBody,
        Tag::Named("probe".into()),
    );
    assert_eq!(core.to_string(), "ns1:probe.core_body");
    let orbit_primary = FrameUid::external(
        Namespace(1),
        FrameClass::OrbitRelative,
        FrameRole::Primary,
        Tag::Named("probe".into()),
    );
    assert_eq!(orbit_primary.to_string(), "ns1:probe.orbit_relative");
}

#[test]
fn is_predicate_agrees_with_minted_equality() {
    // The zero-allocation structural path in `FrameUid::is` must agree
    // with the mint-then-compare definition for every Stable entry.
    for (label, mint, uid) in inventory() {
        if mint != MintPolicy::Stable {
            continue;
        }
        let uid = uid.expect("Stable entries carry a minted uid");
        assert!(
            uid.is::<RootInertial>() == (uid == FrameUid::of::<RootInertial>()),
            "`is` and minted equality disagree for `{label}` vs RootInertial"
        );
    }
    // And positively: each minted uid matches its own type via `is`.
    assert!(FrameUid::of::<PlanetFixed<Earth>>().is::<PlanetFixed<Earth>>());
    assert!(!FrameUid::of::<PlanetFixed<Earth>>().is::<PlanetFixed<Moon>>());
    assert!(!FrameUid::of::<PlanetFixed<Earth>>().is::<Ecef>());
}
