//! Tier 3: Bevy frame-switch parity (issue #71 item 3).
//!
//! Verifies that distance-based integration-frame switching is
//! bit-identical between Bevy `FrameSwitchesC` + `frame_switch_system`
//! and `jeod_runner::Simulation` with `VehicleConfig::frame_switches`.
//!
//! Scenario: a body initially integrates in Earth.inertial; a single
//! `FrameSwitchConfig` triggers when the body comes within `R` of the
//! Moon (a third-body source positioned via `SourceMutator` in Bevy /
//! `set_source_position` in jeod_runner). Once the switch fires, the
//! body's frame is reparented under Moon.inertial in the frame tree,
//! its translational state is rewritten in Moon-centered coordinates,
//! and its gravity controls flip so Moon becomes central
//! (`differential = false`) and Earth becomes the third body.

use std::time::Duration;

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, FrameSwitchesC, FrameTreeR, GravityControlsC, IntegFrameIdC, JeodPlugin,
    MassPropertiesC, PlanetBundle, RotationalStateC, SourceFrameIdC, SourceInertialVelocityC,
    SourceMutator, TranslationalStateC,
};
use glam::DVec3;
use jeod_runner::Simulation;
use jeod_sim::{
    DynamicsConfig, FrameSwitchConfig, GravityControl, GravityControls, GravitySourceEntry,
    JeodQuat, MassProperties, RotationalState, SwitchSense, TranslationalState, VehicleConfig,
    EARTH, MOON,
};

const DT: f64 = 60.0;
const NUM_STEPS: usize = 80;
// Place the Moon close enough that a body launched from Earth-relative
// coordinates approaches within `SWITCH_RADIUS` over `NUM_STEPS * DT`.
const MOON_OFFSET: DVec3 = DVec3::new(2.0e7, 0.0, 0.0);
const SWITCH_RADIUS: f64 = 1.5e7;

fn initial_trans() -> TranslationalState {
    // Body launched from Earth on a trajectory pointing at the Moon.
    TranslationalState {
        position: DVec3::new(7_000_000.0, 0.0, 0.0),
        velocity: DVec3::new(7000.0, 0.0, 0.0),
    }
}

fn initial_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    }
}

fn vehicle_mass() -> MassProperties {
    MassProperties::with_inertia(
        1_000.0,
        glam::DMat3::from_diagonal(DVec3::new(100.0, 100.0, 100.0)),
        DVec3::ZERO,
    )
}

fn assert_bits_eq(label: &str, component: &str, a: f64, b: f64) {
    assert!(
        a.to_bits() == b.to_bits(),
        "{label} {component} not bit-identical:\n  \
         A: {a} (bits={:#018x})\n  \
         B: {b} (bits={:#018x})",
        a.to_bits(),
        b.to_bits(),
    );
}

#[test]
fn tier3_bevy_frame_switch_earth_to_moon_matches_simulation() {
    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(JeodPlugin);

    let earth = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::point_mass("Moon", &MOON))
        .insert(SourceInertialVelocityC::default())
        .id();

    // Phase C4: `FrameSwitchConfig<Entity>` — the Bevy adapter references
    // gravity sources by their ECS entity, no usize bridge.
    let switches: Vec<FrameSwitchConfig<Entity>> = vec![FrameSwitchConfig {
        target_source: moon,
        switch_sense: SwitchSense::OnApproach,
        switch_distance: SWITCH_RADIUS,
        active: true,
    }];

    let vehicle = app
        .world_mut()
        .spawn((
            Name::new("EarthToMoon"),
            TranslationalStateC::from(initial_trans()),
            RotationalStateC::from(initial_rot()),
            MassPropertiesC::from(vehicle_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(earth, false), {
                    let mut c = GravityControl::new_spherical(moon, false);
                    c.differential = true;
                    c
                }],
            }),
            FrameSwitchesC(switches.clone()),
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    // Position Moon at non-zero offset.
    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator| {
            m.set_source_position(moon, MOON_OFFSET);
        });
    app.world_mut().run_system(sys).unwrap();

    let _earth_fid = app.world().get::<SourceFrameIdC>(earth).unwrap().0;
    let moon_fid = app.world().get::<SourceFrameIdC>(moon).unwrap().0;
    let initial_integ_fid = app.world().get::<IntegFrameIdC>(vehicle).unwrap().0;
    let root_fid = app.world().resource::<bevy_jeod::RootFrameIdR>().0;
    // Body has no IntegSourceC, so it defaults to root inertial — which
    // for an Earth-central scenario carries the same numeric pos/vel as
    // Earth.inertial (the central source's offset from root is zero by
    // convention in Bevy and root-by-construction in jeod_runner).
    assert_eq!(
        initial_integ_fid, root_fid,
        "before switch, body integrates in root inertial"
    );

    for _ in 0..NUM_STEPS {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    let bevy_trans = app
        .world()
        .get::<TranslationalStateC>(vehicle)
        .unwrap()
        .0
        .to_untyped();
    let bevy_integ_fid = app.world().get::<IntegFrameIdC>(vehicle).unwrap().0;
    let bevy_controls = app
        .world()
        .get::<GravityControlsC>(vehicle)
        .unwrap()
        .0
        .clone();
    let frame_tree_pos = app
        .world()
        .resource::<FrameTreeR>()
        .0
        .get(
            app.world()
                .get::<bevy_jeod::components::BodyFrameIdC>(vehicle)
                .unwrap()
                .0,
        )
        .state
        .trans
        .position;

    assert_eq!(
        bevy_integ_fid, moon_fid,
        "post-switch, body integrates in Moon.inertial"
    );
    // Earth control should now be differential, Moon non-differential.
    assert!(
        bevy_controls.controls[0].differential,
        "Earth becomes differential after switch"
    );
    assert!(
        !bevy_controls.controls[1].differential,
        "Moon becomes non-differential after switch"
    );

    // ── jeod_runner ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let _earth_idx = sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
    let moon_idx = sim.add_source("Moon", GravitySourceEntry::third_body(&MOON, MOON_OFFSET));

    sim.add_body(VehicleConfig {
        trans: initial_trans(),
        rot: Some(initial_rot()),
        mass: Some(vehicle_mass()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(0_usize, false), {
                let mut c = GravityControl::new_spherical(moon_idx, false);
                c.differential = true;
                c
            }],
        },
        frame_switches: vec![FrameSwitchConfig {
            target_source: moon_idx,
            switch_sense: SwitchSense::OnApproach,
            switch_distance: SWITCH_RADIUS,
            active: true,
        }],
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);

    // Body trans state in post-switch (Moon-centered) coords must match.
    for i in 0..3 {
        assert_bits_eq(
            "Bevy vs Sim post-switch position",
            &format!("[{i}]"),
            bevy_trans.position[i],
            sim_body.trans.position[i],
        );
        assert_bits_eq(
            "Bevy vs Sim post-switch velocity",
            &format!("[{i}]"),
            bevy_trans.velocity[i],
            sim_body.trans.velocity[i],
        );
    }
    // Frame tree's body node should also reflect the same (it's the
    // source-of-truth for the lifted helper's reparent).
    for i in 0..3 {
        assert_bits_eq(
            "Bevy FrameTreeR body vs Sim",
            &format!("pos[{i}]"),
            frame_tree_pos[i],
            sim_body.trans.position[i],
        );
    }
}
