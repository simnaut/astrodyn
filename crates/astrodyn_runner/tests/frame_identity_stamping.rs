//! Positive tests for issue #662: the runner stamps real frame
//! identities on every production frame node, and those identities are
//! stable handles — a body's `FrameUid` resolves to the same `FrameId`
//! across structural changes (frame switches), and the resolved
//! integration-frame identity is published per body.

use astrodyn::{
    named_body_frame_uid, FrameUid, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, GravitySourceEntry, IntegratorType, PlanetInertial, Position, RootInertial,
    RotationModel, SimulationTime, TranslationalStateTyped, VehicleConfig, Velocity, EARTH,
};
use astrodyn_runner::Simulation;
use glam::DVec3;

fn point_mass_entry(
    mu: f64,
    position: Position<RootInertial>,
    central: bool,
) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position,
        velocity: Velocity::<RootInertial>::zero(),
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::None,
        tidal_config: None,
        planet_omega: 0.0,
        central,
        marker_only: false,
    }
}

fn leo_body(name: &str, gravity_source: astrodyn::FrameUid) -> VehicleConfig {
    VehicleConfig {
        trans: TranslationalStateTyped::<RootInertial> {
            // allowed: typed↔raw kernel-boundary helpers used in test scaffolding
            position: Position::<RootInertial>::from_raw_si(DVec3::new(7.0e6, 0.0, 0.0)),
            // allowed: typed↔raw kernel-boundary helpers used in test scaffolding
            velocity: Velocity::<RootInertial>::from_raw_si(DVec3::new(0.0, 7.5e3, 0.0)),
        },
        integrator: IntegratorType::Rk4,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(
                gravity_source,
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named(name)
    }
}

/// `body_integ_frame_uid` publishes the *resolved* integration-frame
/// identity (RF.10: the `IntegrationFrame` marker resolves per body):
/// `RootInertial` for `integ_source = None`, the source's
/// `PlanetInertial<P>` identity for `integ_source = Some(idx)`.
#[test]
fn body_integ_frame_uid_publishes_resolved_identity() {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    let _earth_idx = sim.add_source(
        "Earth",
        point_mass_entry(EARTH.shape.mu, Position::<RootInertial>::zero(), true),
    );

    let root_integrating = sim.add_body(leo_body(
        "root-integrating",
        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
    ));
    let mut cfg = leo_body(
        "earth-integrating",
        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
    );
    cfg.integ_source = Some(astrodyn::FrameUid::of::<
        astrodyn::PlanetInertial<astrodyn::Earth>,
    >());
    let earth_integrating = sim.add_body(cfg);

    assert_eq!(
        sim.body_integ_frame_uid(root_integrating),
        &FrameUid::of::<RootInertial>(),
        "integ_source = None resolves to the root frame's identity"
    );
    assert_eq!(
        sim.body_integ_frame_uid(earth_integrating),
        &FrameUid::of::<PlanetInertial<astrodyn::Earth>>(),
        "integ_source = Some(earth) resolves to Earth's inertial identity"
    );
}

/// Acceptance item from #662: a body's `FrameUid` is a stable handle —
/// after a frame switch reparents the body's frame node, `find(&uid)`
/// resolves to the **same** `FrameId`, and the published
/// integration-frame identity flips to the switch target's.
#[test]
fn frame_switch_preserves_uid_to_frame_id_resolution() {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 1.0);
    let _earth_idx = sim.add_source(
        "Earth",
        point_mass_entry(EARTH.shape.mu, Position::<RootInertial>::zero(), true),
    );
    let _moon_idx = sim.add_source(
        "Moon",
        point_mass_entry(
            4.9028e12,
            // allowed: typed↔raw kernel-boundary helpers used in test scaffolding
            Position::<RootInertial>::from_raw_si(DVec3::new(3.84e8, 0.0, 0.0)),
            false,
        ),
    );

    let mut cfg = leo_body(
        "switcher",
        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
    );
    cfg.frame_switches.push(astrodyn::FrameSwitchConfig {
        target: astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>(),
        switch_sense: astrodyn::SwitchSense::OnApproach,
        // Far larger than the ~3.8e8 m body-to-Moon distance, so the
        // switch triggers on the first step.
        switch_distance: 1.0e12,
        active: true,
    });
    let body_idx = sim.add_body(cfg);

    let uid = named_body_frame_uid("switcher");
    let fid_before = sim
        .frame_tree()
        .find(&uid)
        .expect("body frame is stamped with the mission-supplied identity at add_body");
    assert_eq!(
        sim.body_integ_frame_uid(body_idx),
        &FrameUid::of::<RootInertial>(),
        "before the switch the body integrates in root"
    );

    sim.step().expect("step with a pending frame switch");

    let fid_after = sim
        .frame_tree()
        .find(&uid)
        .expect("the body's identity survives the reparent");
    assert_eq!(
        fid_before, fid_after,
        "a frame switch reparents the body's node; its FrameUid must keep \
         resolving to the same FrameId"
    );
    assert_eq!(
        sim.body_integ_frame_uid(body_idx),
        &FrameUid::of::<PlanetInertial<astrodyn::Moon>>(),
        "after the switch the published integration-frame identity is the \
         switch target's"
    );
}
