//! Positive-path tests for the Act-5 phantom-wrapped types' vehicle
//! witness methods (`assert_vehicle`, `assert_pair`, `assert_reference`,
//! `assert_chief`).
//!
//! Each method is a zero-cost type-level no-op whose `where` bound
//! resolves only when the caller's vehicle phantoms match the value's.
//! When they mismatch the
//! `astrodyn_quantities::diagnostics::CompatibleVehicles` /
//! `CompatibleVehiclePair` `#[diagnostic::on_unimplemented]` attributes
//! surface a physics-language diagnostic instead of a `PhantomData<…>`
//! type-mismatch wall. The negative branch is exercised by the
//! `compile_fail` doctest on each method; this file confirms the
//! positive branch round-trips at runtime.
//!
//! The five touch sites (per #353):
//!
//! - `FlatPlate<V>` — `assert_vehicle::<W>()`
//! - `AttachEvent<VParent, VChild>` — `assert_pair::<P, C>()`
//! - `RelativeState<Subject, Reference>` — `assert_pair::<S, R>()`
//! - `RelativeTranslation<Reference>` — `assert_reference::<R>()`
//! - `LvlhRelativeState<Chief>` — `assert_chief::<C>()`

use astrodyn::{
    compute_lvlh_relative_state, compute_relative_state, define_vehicle, FlatPlate, FrameTransform,
    LvlhRelativeState, RelativeState, RelativeTranslation, RotationalState, StructuralFrame,
    TranslationalState, Vec3Ext,
};
use astrodyn_bevy::AttachEvent;
use bevy::prelude::Entity;
use glam::{DMat3, DVec3};

define_vehicle!(Iss);
define_vehicle!(Soyuz);

#[test]
fn flat_plate_assert_vehicle_round_trips() {
    let plate: FlatPlate<Iss> = FlatPlate {
        area: 10.0,
        normal: DVec3::X,
        position: DVec3::ZERO.m_at::<StructuralFrame<Iss>>(),
    };
    let plate = plate.assert_vehicle::<Iss>();
    assert_eq!(plate.area, 10.0);
}

#[test]
fn attach_event_assert_pair_round_trips() {
    let evt: AttachEvent<Iss, Soyuz> = AttachEvent {
        child: Entity::PLACEHOLDER,
        parent: Entity::PLACEHOLDER,
        offset: DVec3::ZERO.m_at::<StructuralFrame<Iss>>(),
        // Cross-vehicle `FrameTransform::identity()` doesn't typecheck
        // (it's defined only for `<F, F>`); use `from_matrix` to mint
        // an identity-rotation transform across the two phantoms.
        t_parent_child: FrameTransform::<StructuralFrame<Iss>, StructuralFrame<Soyuz>>::from_matrix(
            DMat3::IDENTITY,
        ),
    };
    let evt = evt.assert_pair::<Iss, Soyuz>();
    assert_eq!(evt.parent, Entity::PLACEHOLDER);
}

#[test]
fn relative_state_assert_pair_round_trips() {
    let trans = TranslationalState {
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
    };
    let rel: RelativeState<Iss, Soyuz> =
        compute_relative_state::<Iss, Soyuz>(&trans, None, &trans, None);
    // assert_pair consumes self; round-trip and confirm the trans
    // variant survived the no-op.
    let rel = rel.assert_pair::<Iss, Soyuz>();
    assert!(matches!(rel.trans, RelativeTranslation::Inertial { .. }));
}

#[test]
fn relative_translation_assert_reference_round_trips() {
    let trans_a = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    let trans_b = TranslationalState {
        position: DVec3::new(6_778_237.0, 100.0, -50.0),
        velocity: DVec3::new(0.01, 7668.55, 0.005),
    };
    let rot = RotationalState {
        quaternion: astrodyn::JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    let rel = compute_relative_state::<Iss, Soyuz>(&trans_a, Some(&rot), &trans_b, Some(&rot));
    // Reference phantom is `Soyuz`; asserting it through round-trips.
    let _trans = rel.trans.assert_reference::<Soyuz>();
}

#[test]
fn lvlh_relative_state_assert_chief_round_trips() {
    let lvlh: LvlhRelativeState<Iss> = compute_lvlh_relative_state::<Iss>(
        DVec3::new(6.778e6, 0.0, 0.0),
        DVec3::new(0.0, 7.668e3, 0.0),
        DVec3::new(6.778e6 + 50.0, 0.0, 0.0),
        DVec3::new(0.0, 7.668e3, 0.0),
    );
    let _lvlh = lvlh.assert_chief::<Iss>();
}
