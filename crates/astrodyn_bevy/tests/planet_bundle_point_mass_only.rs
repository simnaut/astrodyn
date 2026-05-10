//! Verifies [`PlanetBundle::point_mass_only`] inserts exactly the four
//! components every "gravity source only" spawn site needs and nothing else.
//!
//! The omitted components ([`PlanetFixedRotationC`], [`PlanetOmegaC`],
//! [`PlanetAngularVelocityC`], [`RotationModelC`], [`PlanetC`]) are what
//! gate the planet-fixed-rotation, atmosphere, and geodetic systems each
//! step. If a future edit silently expands the bundle's component set, a
//! parity test that today inserts only the four components would start
//! activating those systems and re-baseline its trajectory — this test
//! pins the contract so that drift is caught at compile/test time on the
//! bundle itself rather than buried in a parity tolerance miss.

use astrodyn::{GravityModel, GravitySource, EARTH};
use astrodyn_bevy::{
    GravitySourceC, PlanetAngularVelocityC, PlanetBundle, PlanetC, PlanetFixedRotationC,
    PlanetOmegaC, RotationModelC, SourceInertialPositionC, TranslationalStateC,
};
use bevy::prelude::*;

fn earth_point_mass_source() -> GravitySource {
    GravitySource {
        mu: EARTH.shape.mu,
        model: GravityModel::PointMass,
    }
}

#[test]
fn point_mass_only_inserts_exactly_the_four_required_components() {
    let mut world = World::new();
    let entity = world
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass_only(
            "Earth",
            earth_point_mass_source(),
        ))
        .id();
    let e = world.entity(entity);

    // The four components every gravity-source spawn site needs.
    assert!(e.contains::<Name>(), "Name missing from point_mass_only");
    assert!(
        e.contains::<GravitySourceC>(),
        "GravitySourceC missing from point_mass_only",
    );
    assert!(
        e.contains::<SourceInertialPositionC>(),
        "SourceInertialPositionC missing from point_mass_only",
    );
    assert!(
        e.contains::<TranslationalStateC<astrodyn::Earth>>(),
        "TranslationalStateC<Earth> missing from point_mass_only",
    );
}

#[test]
fn point_mass_only_omits_rotation_shape_and_rnp_components() {
    let mut world = World::new();
    let entity = world
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass_only(
            "Earth",
            earth_point_mass_source(),
        ))
        .id();
    let e = world.entity(entity);

    // None of the rotation/shape/RNP components the full PlanetBundle
    // inserts may leak through point_mass_only; their presence would
    // activate planet-fixed-rotation, atmosphere, and geodetic systems
    // each step.
    assert!(
        !e.contains::<PlanetFixedRotationC<astrodyn::Earth>>(),
        "PlanetFixedRotationC leaked into point_mass_only",
    );
    assert!(
        !e.contains::<PlanetOmegaC>(),
        "PlanetOmegaC leaked into point_mass_only",
    );
    assert!(
        !e.contains::<PlanetAngularVelocityC<astrodyn::Earth>>(),
        "PlanetAngularVelocityC leaked into point_mass_only",
    );
    assert!(
        !e.contains::<RotationModelC>(),
        "RotationModelC leaked into point_mass_only",
    );
    assert!(
        !e.contains::<PlanetC>(),
        "PlanetC leaked into point_mass_only",
    );
}

#[test]
fn point_mass_only_name_round_trips_through_into_name() {
    // The `impl Into<Name>` parameter accepts both `&str` and `String`;
    // verify the resulting Name reads back the caller-supplied string.
    let mut world = World::new();
    let entity = world
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass_only(
            String::from("Mars"),
            earth_point_mass_source(),
        ))
        .id();
    let name = world
        .entity(entity)
        .get::<Name>()
        .expect("point_mass_only must insert Name");
    assert_eq!(name.as_str(), "Mars");
}
