// JEOD_INV: TS.01 — these negative tests exercise the wildcard tags
// (SelfRef / SelfPlanet / MassNode) at the FrameUid::of refusal boundary;
// see `docs/JEOD_invariants.md` row TS.01 and the lint at
// `tests/self_ref_self_planet_discipline.rs`.
//! Fail-loudly tests for the frame-identity mint boundary (issue #660):
//! every refusal path panics with a diagnostic naming the broken invariant,
//! and the non-panicking `is` predicate stays total.

use astrodyn_quantities::frame::{
    BodyFrame, Ecef, IntegrationFrame, MassNode, PlanetInertial, RootInertial, SelfPlanet, SelfRef,
};
use astrodyn_quantities::{FrameClass, FrameRole, FrameUid, Namespace, Tag};

#[test]
#[should_panic(expected = "JEOD_INV TS.01")]
fn of_refuses_self_ref_tagged_body_frame() {
    let _ = FrameUid::of::<BodyFrame<SelfRef>>();
}

#[test]
#[should_panic(expected = "JEOD_INV TS.01")]
fn of_refuses_self_planet_tagged_planet_inertial() {
    let _ = FrameUid::of::<PlanetInertial<SelfPlanet>>();
}

#[test]
#[should_panic(expected = "JEOD_INV TS.01")]
fn of_refuses_mass_node() {
    let _ = FrameUid::of::<MassNode>();
}

#[test]
#[should_panic(expected = "RF.10")]
fn of_refuses_integration_frame() {
    let _ = FrameUid::of::<IntegrationFrame>();
}

#[test]
#[should_panic(expected = "reserved for type-derived identity")]
fn external_refuses_local_namespace() {
    let _ = FrameUid::external(
        Namespace::LOCAL,
        FrameClass::External,
        FrameRole::Custom("imposter".into()),
        Tag::Named("Earth".into()),
    );
}

#[test]
fn external_accepts_non_local_namespace() {
    let uid = FrameUid::external(
        Namespace(7),
        FrameClass::External,
        FrameRole::Custom("sensor_boresight".into()),
        Tag::Named("ext-probe-42".into()),
    );
    assert_eq!(uid.namespace, Namespace(7));
    assert_eq!(uid.to_string(), "ns7:ext-probe-42.sensor_boresight");
}

#[test]
fn is_predicate_never_panics_and_answers_honestly() {
    let root = FrameUid::of::<RootInertial>();
    // Matching type: true.
    assert!(root.is::<RootInertial>());
    // Different concrete type: false.
    assert!(!root.is::<Ecef>());
    // Non-mintable types: false WITHOUT panicking — a wildcard or
    // per-body frame is never equal to a concrete identity.
    assert!(!root.is::<MassNode>());
    assert!(!root.is::<IntegrationFrame>());
    assert!(!root.is::<BodyFrame<SelfRef>>());
}

#[test]
fn external_uid_never_matches_type_derived_identity() {
    // Even a field-for-field imitation of a type-derived identity cannot
    // claim it from a non-LOCAL namespace: `is` compares the namespace too.
    let imposter = FrameUid::of::<RootInertial>().with_namespace(Namespace(3));
    assert!(!imposter.is::<RootInertial>());
}
