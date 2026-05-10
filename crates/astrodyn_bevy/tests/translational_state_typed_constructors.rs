//! Round-trip tests for the typed entry points on
//! [`TranslationalStateC`].
//!
//! Mission and parity-test code that already has typed planet-inertial
//! inputs should reach the Bevy component without crossing the
//! typed/raw kernel boundary. Two entry points provide that path:
//!
//! - [`TranslationalStateC::<P>::point_mass(position, velocity)`] —
//!   takes the typed pair directly.
//! - `TranslationalStateC::<P>::from(state)` for an already-bundled
//!   [`TranslationalStateTyped<PlanetInertial<P>>`].
//!
//! Both must produce a component bit-identical to the `from_untyped`
//! kernel-boundary helper given the same underlying SI numerics, since
//! they all wrap the same `TranslationalStateTyped<PlanetInertial<P>>`
//! storage. This test fences that invariant so a future refactor of
//! either entry point can't silently drift from the others.
//!
//! The test uses non-trivial values (a 400 km ISS-like circular orbit
//! state, not zeros or unit axes) so accidental field swaps surface
//! immediately.

use astrodyn::{Position, TranslationalState, TranslationalStateTyped, Velocity};
use astrodyn_bevy::prelude::*;
use glam::DVec3;

/// Position and velocity for a circular ISS-like orbit at 400 km
/// altitude in the equatorial plane. Values are non-trivial enough
/// that a swapped `position`/`velocity` field would surface as a
/// failed bit-equality check.
fn iss_like_state_raw() -> (DVec3, DVec3) {
    let r = DVec3::new(6_778_137.0, 0.0, 0.0); // m
    let v = DVec3::new(0.0, 7668.56, 0.0); // m/s
    (r, v)
}

#[test]
fn point_mass_constructor_round_trip() {
    let (r_raw, v_raw) = iss_like_state_raw();
    let r: Position<PlanetInertial<Earth>> = r_raw.m_at::<PlanetInertial<Earth>>();
    let v: Velocity<PlanetInertial<Earth>> = v_raw.m_per_s_at::<PlanetInertial<Earth>>();

    let state = TranslationalStateC::<Earth>::point_mass(r, v);

    // The component's underlying typed SI raw values must match the
    // raw inputs bit-for-bit — `point_mass` is a no-op wrap.
    assert_eq!(state.0.position.raw_si(), r_raw);
    assert_eq!(state.0.velocity.raw_si(), v_raw);
}

#[test]
fn from_planet_inertial_typed_state_round_trip() {
    let (r_raw, v_raw) = iss_like_state_raw();
    let typed = TranslationalStateTyped::<PlanetInertial<Earth>> {
        position: r_raw.m_at::<PlanetInertial<Earth>>(),
        velocity: v_raw.m_per_s_at::<PlanetInertial<Earth>>(),
    };

    let state = TranslationalStateC::<Earth>::from(typed);

    assert_eq!(state.0.position.raw_si(), r_raw);
    assert_eq!(state.0.velocity.raw_si(), v_raw);
}

#[test]
fn point_mass_and_from_typed_produce_same_component() {
    // Construct the same logical state through both typed entry points
    // and confirm the component is field-equal. Pins the no-op-wrap
    // invariant — neither path may diverge in framing or unit handling.
    let (r_raw, v_raw) = iss_like_state_raw();
    let r: Position<PlanetInertial<Earth>> = r_raw.m_at::<PlanetInertial<Earth>>();
    let v: Velocity<PlanetInertial<Earth>> = v_raw.m_per_s_at::<PlanetInertial<Earth>>();

    let via_point_mass = TranslationalStateC::<Earth>::point_mass(r, v);
    let typed = TranslationalStateTyped::<PlanetInertial<Earth>> {
        position: r,
        velocity: v,
    };
    let via_from = TranslationalStateC::<Earth>::from(typed);

    assert_eq!(via_point_mass.0, via_from.0);
}

#[test]
fn point_mass_matches_from_untyped_numerics() {
    // The new typed entry point and the `from_untyped` kernel-boundary
    // helper must produce the same numeric storage when the inputs
    // describe the same physical state — they only differ in which
    // side of the typed/raw boundary the caller lives on.
    let (r_raw, v_raw) = iss_like_state_raw();
    let r: Position<PlanetInertial<Earth>> = r_raw.m_at::<PlanetInertial<Earth>>();
    let v: Velocity<PlanetInertial<Earth>> = v_raw.m_per_s_at::<PlanetInertial<Earth>>();

    let via_point_mass = TranslationalStateC::<Earth>::point_mass(r, v);
    let via_from_untyped = TranslationalStateC::<Earth>::from_untyped(TranslationalState {
        position: r_raw,
        velocity: v_raw,
    });

    assert_eq!(
        via_point_mass.0.position.raw_si(),
        via_from_untyped.0.position.raw_si()
    );
    assert_eq!(
        via_point_mass.0.velocity.raw_si(),
        via_from_untyped.0.velocity.raw_si()
    );
}
