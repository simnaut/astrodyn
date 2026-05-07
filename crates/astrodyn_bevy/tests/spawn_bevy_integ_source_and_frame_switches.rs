//! Regression tests for `VehicleConfig::spawn_bevy`'s `integ_source`
//! and `frame_switches` translation.
//!
//! `spawn_bevy` (lib.rs) accepts a `VehicleConfig` whose
//! `integ_source: Option<usize>` and `frame_switches: Vec<FrameSwitchConfig<usize>>`
//! reference gravity sources by index and translates each `usize` to the
//! caller-supplied [`Entity`] when inserting [`IntegSourceC`] /
//! [`FrameSwitchesC`]. The existing `mission_crate_sanity` and
//! `validation_added_trigger` tests only cover the legacy gravity-controls
//! path — a bug in the `usize -> Entity` translation here would slip
//! through even though the manual-spawn parity tests in
//! `bevy_parity_integ_source.rs` / `bevy_parity_frame_switch.rs` stay
//! green (they bypass `spawn_bevy` and assemble entities by hand).
//!
//! Coverage:
//!
//! 1. `spawn_bevy` translates `integ_source: Some(idx)` to
//!    `IntegSourceC(Some(source_entities[idx]))`.
//! 2. `spawn_bevy` translates each `FrameSwitchConfig<usize>` to
//!    `FrameSwitchConfig<Entity>` with `target_source` retagged from the
//!    `source_entities` table; field-by-field (sense, distance, active)
//!    is preserved.
//! 3. `spawn_bevy` skips `IntegSourceC` insertion when `integ_source` is
//!    `None` (preserves the implicit-root default).
//! 4. `spawn_bevy` skips `FrameSwitchesC` insertion when the switch list
//!    is empty (consumers may rely on `Without<FrameSwitchesC>` to fast-
//!    path no-switch vehicles).
//! 5. Out-of-bounds `integ_source` panics with the "Spawn all gravity
//!    sources before calling spawn_bevy" diagnostic.
//! 6. Out-of-bounds `FrameSwitchConfig::target_source` panics with the
//!    same diagnostic.
//! 7. End-to-end behavior parity: a vehicle wired via `spawn_bevy` with
//!    `integ_source` + `frame_switches` propagates bit-identically to
//!    `astrodyn_runner::Simulation` configured with the same `VehicleConfig`.

use std::time::Duration;

use astrodyn::{
    FrameSwitchConfig, GravityControl, GravitySourceEntry, JeodQuat, MassProperties,
    RotationalState, SixDofState, SwitchSense, TranslationalState, VehicleBuilder, VehicleConfig,
    EARTH, MOON,
};
use astrodyn_bevy::{
    AstrodynPlugin, FrameEntityC, FrameSwitchesC, IntegSourceC, PlanetBundle, RotationalStateC,
    SourceInertialVelocityC, SourceMutator, TranslationalStateC, VehicleConfigBevyExt,
};
use astrodyn_runner::Simulation;
use bevy::prelude::*;
use glam::DVec3;

const DT: f64 = 60.0;
const NUM_STEPS: usize = 80;
const MOON_OFFSET: DVec3 = DVec3::new(2.0e7, 0.0, 0.0);
const SWITCH_RADIUS: f64 = 1.5e7;

fn initial_trans() -> TranslationalState {
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

fn assert_sixdof_bit_identical(label: &str, a: &SixDofState, b: &SixDofState) {
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("pos[{i}]"),
            a.trans.position[i],
            b.trans.position[i],
        );
        assert_bits_eq(
            label,
            &format!("vel[{i}]"),
            a.trans.velocity[i],
            b.trans.velocity[i],
        );
    }
}

/// Build a `VehicleConfig` with `integ_source = Some(0)` (Earth-centered
/// initial frame) and a switch targeting source 1 (Moon). Exercises both
/// `usize -> Entity` translation paths in one call. Earth is the central
/// body for the initial integration; Moon is the differential third
/// body. After the switch fires, the body integrates in Moon and Earth
/// becomes the differential source — `frame_switch_system` flips the
/// `differential` flags on the gravity controls to match.
fn earth_then_moon_config() -> VehicleConfig {
    VehicleBuilder::new()
        .with_translational(astrodyn::TranslationalStateTyped::<
            astrodyn_quantities::frame::RootInertial,
        >::from_untyped_unchecked(&initial_trans()))
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .gravity({
            let mut c = GravityControl::new_spherical(1_usize, false);
            c.differential = true;
            c
        })
        .integ_source(0)
        .frame_switches(vec![FrameSwitchConfig {
            target_source: 1,
            switch_sense: SwitchSense::OnApproach,
            switch_distance: SWITCH_RADIUS,
            active: true,
        }])
        .build()
}

#[test]
fn spawn_bevy_translates_integ_source_index_to_entity() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Moon", &MOON))
        .id();

    // Build a config that integrates in Moon (source index 1).
    let cfg = VehicleBuilder::new()
        .with_translational(astrodyn::TranslationalStateTyped::<
            astrodyn_quantities::frame::RootInertial,
        >::from_untyped_unchecked(&initial_trans()))
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .gravity(GravityControl::new_spherical(1_usize, false))
        .integ_source(1)
        .build();

    let vehicle = {
        let mut commands_queue = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth, moon])
    };
    // Apply queued commands so the components land on the entity.
    app.world_mut().flush();

    let integ = app
        .world()
        .get::<IntegSourceC>(vehicle)
        .expect("spawn_bevy must insert IntegSourceC when integ_source is Some");
    assert_eq!(
        integ.0,
        Some(moon),
        "integ_source index 1 must translate to the Moon entity"
    );
}

#[test]
fn spawn_bevy_translates_frame_switch_target_source_to_entity() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Moon", &MOON))
        .id();

    let cfg = earth_then_moon_config();
    let vehicle = {
        let mut commands_queue = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth, moon])
    };
    app.world_mut().flush();

    let switches = app
        .world()
        .get::<FrameSwitchesC>(vehicle)
        .expect("spawn_bevy must insert FrameSwitchesC when frame_switches is non-empty");
    assert_eq!(switches.0.len(), 1, "switch count must be preserved");
    let sw = &switches.0[0];
    assert_eq!(
        sw.target_source, moon,
        "FrameSwitchConfig::target_source index 1 must translate to the Moon entity"
    );
    assert!(matches!(sw.switch_sense, SwitchSense::OnApproach));
    assert_eq!(sw.switch_distance, SWITCH_RADIUS);
    assert!(sw.active);
}

#[test]
fn spawn_bevy_omits_integ_source_component_when_default() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();

    // No `.integ_source(...)` call -> `integ_source: None` in the config.
    // `spawn_bevy` must skip the insert so consumers branching on the
    // *presence* of `IntegSourceC` (or relying on `Without<IntegSourceC>`)
    // see the same shape as a manually-spawned root-integrated vehicle.
    let cfg = VehicleBuilder::new()
        .with_translational(astrodyn::TranslationalStateTyped::<
            astrodyn_quantities::frame::RootInertial,
        >::from_untyped_unchecked(&initial_trans()))
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    let vehicle = {
        let mut commands_queue = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth])
    };
    app.world_mut().flush();

    assert!(
        app.world().get::<IntegSourceC>(vehicle).is_none(),
        "spawn_bevy must NOT insert IntegSourceC when integ_source is None — \
         the implicit-root default must be expressed by the component's absence \
         to match the legacy manual-spawn pattern"
    );
}

#[test]
fn spawn_bevy_omits_frame_switches_component_when_empty() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();

    let cfg = VehicleBuilder::new()
        .with_translational(astrodyn::TranslationalStateTyped::<
            astrodyn_quantities::frame::RootInertial,
        >::from_untyped_unchecked(&initial_trans()))
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .build();

    let vehicle = {
        let mut commands_queue = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth])
    };
    app.world_mut().flush();

    assert!(
        app.world().get::<FrameSwitchesC>(vehicle).is_none(),
        "spawn_bevy must NOT insert FrameSwitchesC when frame_switches is empty"
    );
}

#[test]
#[should_panic(expected = "integ_source references source index")]
fn spawn_bevy_panics_on_out_of_bounds_integ_source() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();

    let cfg = VehicleBuilder::new()
        .with_translational(astrodyn::TranslationalStateTyped::<
            astrodyn_quantities::frame::RootInertial,
        >::from_untyped_unchecked(&initial_trans()))
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .integ_source(7) // out of range — only Earth was spawned
        .build();

    let mut commands_queue = app.world_mut().commands();
    cfg.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth]);
}

#[test]
#[should_panic(expected = "FrameSwitchConfig::target_source references source index")]
fn spawn_bevy_panics_on_out_of_bounds_frame_switch_target() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();

    let cfg = VehicleBuilder::new()
        .with_translational(astrodyn::TranslationalStateTyped::<
            astrodyn_quantities::frame::RootInertial,
        >::from_untyped_unchecked(&initial_trans()))
        .sixdof(initial_rot(), vehicle_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, false))
        .frame_switches(vec![FrameSwitchConfig {
            target_source: 7, // out of range
            switch_sense: SwitchSense::OnApproach,
            switch_distance: SWITCH_RADIUS,
            active: true,
        }])
        .build();

    let mut commands_queue = app.world_mut().commands();
    cfg.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth]);
}

/// End-to-end parity: vehicle wired through `spawn_bevy` with
/// `integ_source` + `frame_switches` set must propagate bit-identically
/// to `astrodyn_runner::Simulation` consuming the same `VehicleConfig`. A bug
/// in the `usize` -> `Entity` translation would surface as a divergent
/// trajectory (wrong integ frame or unfired switch), even if the unit
/// translation tests above happen to stay green.
#[test]
fn tier3_spawn_bevy_integ_source_plus_frame_switch_matches_simulation() {
    // ── Bevy via spawn_bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.add_plugins(AstrodynPlugin);

    let earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Moon", &MOON))
        .insert(SourceInertialVelocityC::default())
        .id();

    let cfg = earth_then_moon_config();
    let vehicle = {
        let mut commands_queue = app.world_mut().commands();
        cfg.spawn_bevy::<astrodyn::Earth>(&mut commands_queue, &[earth, moon])
    };
    app.world_mut().flush();

    // Sanity: the two components under test made it onto the entity.
    let integ = app
        .world()
        .get::<IntegSourceC>(vehicle)
        .expect("IntegSourceC must be present on the spawn_bevy vehicle");
    assert_eq!(
        integ.0,
        Some(earth),
        "integ_source index 0 must translate to the Earth entity"
    );
    let switches = app
        .world()
        .get::<FrameSwitchesC>(vehicle)
        .expect("FrameSwitchesC must be present on the spawn_bevy vehicle");
    assert_eq!(switches.0[0].target_source, moon);

    // Optional sanity-check: the additional Bevy components inserted by
    // the registration systems show up after the first tick.
    app.world_mut().run_schedule(Startup);

    // Position the Moon at MOON_OFFSET so the body crosses SWITCH_RADIUS
    // during propagation.
    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator<astrodyn::Earth>| {
            m.set_source_position(moon, MOON_OFFSET);
        });
    app.world_mut().run_system(sys).unwrap();

    for _ in 0..NUM_STEPS {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    // After the switch fires, the body's frame entity must be reparented
    // under the Moon's frame entity (the load-bearing ECS-side
    // reparent); this is a cheap shape check that confirms
    // `frame_switch_system` saw the entity-tagged switch list.
    let body_frame_entity = app
        .world()
        .get::<FrameEntityC>(vehicle)
        .expect("body should carry FrameEntityC after registration");
    let body_integ_frame_entity = app
        .world()
        .get::<bevy::prelude::ChildOf>(body_frame_entity.0)
        .expect("body's frame entity must be parented under its integration frame")
        .parent();
    let moon_frame_entity = app
        .world()
        .get::<FrameEntityC>(moon)
        .expect("Moon should carry FrameEntityC after registration");
    assert_eq!(
        body_integ_frame_entity, moon_frame_entity.0,
        "frame_switch_system must reparent the body's frame entity under \
         Moon.inertial after the OnApproach switch fires"
    );

    let bevy_state = SixDofState {
        trans: app
            .world()
            .get::<TranslationalStateC<astrodyn::Earth>>(vehicle)
            .unwrap()
            .0
            .to_untyped(),
        rot: app
            .world()
            .get::<RotationalStateC>(vehicle)
            .unwrap()
            .0
            .to_untyped(),
    };

    // ── astrodyn_runner reference ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let _earth_idx = sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::third_body(
            &MOON,
            astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(MOON_OFFSET),
        ),
    );

    let mut sim_cfg = earth_then_moon_config();
    sim_cfg.frame_switches[0].target_source = moon_idx;
    sim.add_body(sim_cfg);
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");
    let body = sim.body(0);
    let sim_state = SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    };

    assert_sixdof_bit_identical(
        "spawn_bevy(integ_source + frame_switches) vs Sim",
        &bevy_state,
        &sim_state,
    );
}
