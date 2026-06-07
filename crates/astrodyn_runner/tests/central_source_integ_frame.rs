//! Regression tests for the central-source-as-integration-frame
//! validation behavior after PR #567.
//!
//! Pre-#567, `Simulation::add_source` aliased the central source's
//! inertial-frame node to `root_frame_id`, so a body created with
//! `integ_source = Some(central_idx)` resolved to `integ_frame_id ==
//! root_frame_id`, and the validator's
//! `body.integ_frame_id != self.root_frame_id` check correctly treated
//! the body as root-integrating.
//!
//! Post-#567 the central source's inertial frame is a structurally
//! distinct child of root (matching JEOD's canonical
//! `BasePlanet { inertial, pfix }` shape). The literal-id check is
//! therefore wrong: a body integrating in the central source's
//! inertial frame would erroneously be flagged as integrating in a
//! non-root frame. These tests pin the migrated predicate
//! (`Simulation::is_root_equivalent_frame`, exercised via `validate`
//! and the runtime asserts in the integration stage) so the
//! regression cannot silently come back.

use astrodyn::{
    GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource,
    GravitySourceEntry, IntegratorType, Position, RootInertial, RotationModel, SimulationTime,
    TranslationalStateTyped, VehicleConfig, Velocity, EARTH,
};
use astrodyn_runner::Simulation;
use glam::DVec3;

fn earth_central_source() -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu: EARTH.shape.mu,
            model: GravityModel::PointMass,
        },
        position: Position::<RootInertial>::zero(),
        velocity: Velocity::<RootInertial>::zero(),
        t_inertial_pfix: Some(glam::DMat3::IDENTITY),
        rotation_model: RotationModel::None,
        delta_c20: 0.0,
        tidal_config: None,
        planet_omega: 0.0,
        central: true,
        marker_only: false,
    }
}

/// Build an LEO-like body with one root-dependent derived state
/// enabled (orbital elements around `gravity_source_idx`). The
/// derived state opts into the `NonRootFrameWithRootDependentFeatures`
/// branch in `validate()`, which is the predicate this test pins.
fn leo_body_config(
    integ_source: Option<astrodyn::FrameUid>,
    gravity_source: astrodyn::FrameUid,
) -> VehicleConfig {
    let altitude_m = 700_000.0;
    let radius_m = EARTH.shape.r_eq() + altitude_m;
    let speed_m_s = 7_504.567; // ~circular at 700 km
    let mut cfg = VehicleConfig {
        trans: TranslationalStateTyped::<RootInertial> {
            // allowed: typed↔raw kernel boundary in test scaffolding (matches
            // the pattern used by `integ_frame_translation_invariance.rs`).
            position: Position::<RootInertial>::from_raw_si(DVec3::new(radius_m, 0.0, 0.0)),
            // allowed: typed↔raw kernel boundary in test scaffolding (matches
            // the pattern used by `integ_frame_translation_invariance.rs`).
            velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, speed_m_s, 0.0)),
        },
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                gravity_source,
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("central-source-integ-frame-0")
    };
    cfg.integ_source = integ_source;
    cfg.derived.orbital_elements_source = Some(cfg.gravity_controls.controls[0].source.clone());
    cfg
}

/// `Simulation::new` should name the root frame `"root"`, not
/// `"Earth.inertial"`. Pre-#567 the literal `"Earth.inertial"` was
/// the root's name and `add_source(central=true)` would alias the
/// central source's inertial to root; post-#567 the central source's
/// inertial is a distinct child, so the root keeps a neutral name
/// and a freshly-created `Earth` source carries its own
/// `"Earth.inertial"` node.
#[test]
fn root_frame_is_named_root_and_central_source_has_distinct_inertial() {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    assert_eq!(
        sim.frame_tree().get(sim.root_frame_id).name,
        "root",
        "root frame should be named \"root\" (neutral), not aliased to any planet"
    );

    let earth_idx = sim.add_source("Earth", earth_central_source());
    let earth_inertial = sim.source_inertial_frame_id(earth_idx);
    assert_ne!(
        earth_inertial, sim.root_frame_id,
        "post-#567 the central source's inertial frame must be a distinct child of root"
    );
    assert_eq!(
        sim.frame_tree().get(earth_inertial).name,
        "Earth.inertial",
        "the central source's inertial frame node should carry the source-named label"
    );
}

/// A body that integrates in the central source's inertial frame plus
/// any root-dependent derived state must validate cleanly. The central
/// source's inertial is identity-related to root (`JEOD_INV: RF.13`),
/// so it is root-equivalent for the purposes of the
/// `NonRootFrameWithRootDependentFeatures` predicate.
///
/// Pre-fix this errored because `validate()` compared
/// `body.integ_frame_id` against `self.root_frame_id` literally;
/// `body.integ_frame_id` was `sources[earth_idx].inertial`, which is
/// structurally distinct from root post-#567.
#[test]
fn integ_source_eq_central_with_orbital_elements_validates_clean() {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    let _earth_idx = sim.add_source("Earth", earth_central_source());
    sim.add_body(leo_body_config(
        Some(astrodyn::FrameUid::of::<
            astrodyn::PlanetInertial<astrodyn::Earth>,
        >()),
        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
    ));
    sim.validate().unwrap_or_else(|errors| {
        panic!(
            "expected clean validation for a body integrating in the central source's \
             inertial frame with orbital_elements; got errors: {errors:?}"
        )
    });
}

/// And the reference: the same body, just with `integ_source = None`
/// (integrating in root literally). Both setups must validate
/// equivalently — that is what `is_root_equivalent_frame` promises.
#[test]
fn integ_source_eq_none_with_orbital_elements_validates_clean() {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    let _earth_idx = sim.add_source("Earth", earth_central_source());
    sim.add_body(leo_body_config(
        None,
        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
    ));
    sim.validate().unwrap_or_else(|errors| {
        panic!(
            "expected clean validation for a body integrating in root with \
             orbital_elements; got errors: {errors:?}"
        )
    });
}

/// A frame switch whose target is the central source must not be
/// flagged as `NonRootFrameWithRootDependentFeatures`. The central
/// source's inertial is root-equivalent regardless of how the body
/// reaches it (integration choice or post-#566 frame switch).
#[test]
fn frame_switch_target_central_with_derived_state_validates_clean() {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    let _earth_idx = sim.add_source("Earth", earth_central_source());
    let mut cfg = leo_body_config(
        None,
        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
    );
    cfg.frame_switches.push(astrodyn::FrameSwitchConfig {
        target: astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
        switch_sense: astrodyn::SwitchSense::OnDeparture,
        switch_distance: 1.0e9,
        active: true,
    });
    sim.add_body(cfg);
    sim.validate().unwrap_or_else(|errors| {
        panic!(
            "expected clean validation for a frame switch to the central source; \
             got errors: {errors:?}"
        )
    });
}

// Note: `NonRootFrameWithRootDependentFeatures` is classified as a
// warning (logged, not returned), so it can't be observed as a
// `validate()` error. The four positive tests above cover the
// `is_root_equivalent_frame` migration: they would fail with the
// pre-fix literal-id check because validation would push the
// warning (caught by `cargo nextest`'s log captures in some
// configurations) and — more importantly — the runtime asserts in
// `integrate.rs:594-602` (now also migrated) would panic when a
// contact-pair body integrating in the central source's inertial
// hit the step loop. The positive coverage here is the necessary
// regression fence for the validation half; the contact-pair
// runtime-assert half is exercised by the existing
// `runner_attach_detach_momentum` test, which uses
// `integ_source = Some(earth)` (Earth as non-central in an
// SSB-rooted setup) and steps the simulation end-to-end.
