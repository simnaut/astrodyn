// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: Bevy frame-switch parity.
//!
//! Verifies that distance-based integration-frame switching is
//! bit-identical between Bevy `FrameSwitchesC` + `frame_switch_system`
//! and `astrodyn_runner::Simulation` with `VehicleConfig::frame_switches`.
//!
//! Scenario: a body initially integrates in Earth.inertial; a single
//! `FrameSwitchConfig` triggers when the body comes within `R` of the
//! Moon (a third-body source positioned via `SourceMutator` in Bevy /
//! `set_source_position` in astrodyn_runner). Once the switch fires, the
//! body's frame is reparented under Moon.inertial in the frame tree,
//! its translational state is rewritten in Moon-centered coordinates,
//! and its gravity controls flip so Moon becomes central
//! (`differential = false`) and Earth becomes the third body.

use std::time::Duration;

use astrodyn::{
    AngularVelocity, BodyAttitude, BodyFrame, DynamicsConfig, FrameSwitchConfig, GravityControl,
    GravityControls, GravityGradient, GravitySourceEntry, InertiaTensor, JeodQuat,
    MassPropertiesTyped, Position, RotationalStateTyped, SelfRef, StructuralFrame, SwitchSense,
    TranslationalState, VehicleConfig, EARTH, MOON,
};
use astrodyn_bevy::frame_param::RelativeFrameState;
use astrodyn_bevy::{
    AstrodynPlugin, DynamicsConfigC, FrameEntityC, FrameSwitchesC, GravityControlsC,
    IntegrationDtR, MassPropertiesC, PlanetBundle, RotationalStateC, SourceInertialVelocityC,
    SourceMutator, TranslationalStateC,
};
use astrodyn_runner::Simulation;
use bevy::prelude::*;
use glam::DVec3;
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

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

fn initial_rot() -> RotationalStateTyped<SelfRef> {
    RotationalStateTyped::<SelfRef>::new(
        BodyAttitude::<SelfRef>::from_jeod_quat(JeodQuat::identity()),
        AngularVelocity::<BodyFrame<SelfRef>>::from_raw_si(DVec3::ZERO),
    )
}

fn vehicle_mass() -> MassPropertiesTyped<SelfRef> {
    MassPropertiesTyped::<SelfRef>::with_inertia(
        Mass::new::<kilogram>(1_000.0),
        InertiaTensor::<BodyFrame<SelfRef>>::from_dmat3_unchecked(glam::DMat3::from_diagonal(
            DVec3::new(100.0, 100.0, 100.0),
        )),
        Position::<StructuralFrame<SelfRef>>::zero(),
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
fn bevy_parity_frame_switch_earth_to_moon_matches_simulation() {
    // ── Bevy ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(AstrodynPlugin);

    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth> {
            // Identity = the source's own planet (issue #664); the
            // bundle's <Earth> only tags component storage (the sim's
            // central-planet convention, see SunBundle).
            uid: astrodyn_bevy::FrameUidC(astrodyn::FrameUid::of::<
                astrodyn::PlanetInertial<astrodyn::Moon>,
            >()),
            ..PlanetBundle::<astrodyn::Earth>::point_mass("Moon", &MOON)
        })
        .insert(SourceInertialVelocityC::default())
        .id();

    // `FrameSwitchConfig` — the Bevy adapter references
    // gravity sources by their ECS entity, no usize bridge.
    let switches: Vec<FrameSwitchConfig> = vec![FrameSwitchConfig {
        target: astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>(),
        switch_sense: SwitchSense::OnApproach,
        switch_distance: SWITCH_RADIUS,
        active: true,
    }];

    let vehicle = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(&format!(
                "bevy-parity-frame-switch-b1-{}",
                NEXT_BODY_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ))),
            Name::new("EarthToMoon"),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(initial_trans()),
            RotationalStateC::from(initial_rot()),
            MassPropertiesC::from(vehicle_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![
                    GravityControl::new_spherical(
                        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                        GravityGradient::Skip,
                    ),
                    {
                        let mut c = GravityControl::new_spherical(
                            astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>(),
                            GravityGradient::Skip,
                        );
                        c.differential = true;
                        c
                    },
                ],
            }),
            FrameSwitchesC(switches.clone()),
        ))
        .id();

    app.world_mut().run_schedule(Startup);

    // Position Moon at non-zero offset.
    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator<astrodyn::Earth>| {
            m.set_source_position(moon, MOON_OFFSET);
        });
    app.world_mut().run_system(sys).unwrap();

    // The body's current integration frame is the parent of its frame
    // entity in the ECS hierarchy.
    let body_frame_entity = app.world().get::<FrameEntityC>(vehicle).unwrap().0;
    let initial_integ_frame_entity = app
        .world()
        .get::<bevy::prelude::ChildOf>(body_frame_entity)
        .unwrap()
        .parent();
    let root_frame_entity = app.world().resource::<astrodyn_bevy::RootFrameEntityR>().0;
    // Body has no IntegSourceC, so it defaults to root inertial — which
    // for an Earth-central scenario carries the same numeric pos/vel as
    // Earth.inertial (the central source's offset from root is zero by
    // convention in Bevy and root-by-construction in astrodyn_runner).
    assert_eq!(
        initial_integ_frame_entity, root_frame_entity,
        "before switch, body integrates in root inertial"
    );

    for _ in 0..NUM_STEPS {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    let bevy_trans = astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .get::<TranslationalStateC<astrodyn::Earth>>(vehicle)
            .unwrap()
            .0,
    );
    // Post-switch: the body frame entity's `ChildOf` parent must be
    // the Moon's frame entity (the load-bearing ECS reparent).
    let bevy_integ_frame_entity = app
        .world()
        .get::<bevy::prelude::ChildOf>(body_frame_entity)
        .unwrap()
        .parent();
    let moon_frame_entity = app.world().get::<FrameEntityC>(moon).unwrap().0;
    let bevy_controls = app
        .world()
        .get::<GravityControlsC>(vehicle)
        .unwrap()
        .0
        .clone();
    // Body's position in its (post-switch) integration frame's
    // coordinates, read via `RelativeFrameState`. This is the same
    // value the arena's `frame_tree.get(body_fid).state.trans.position`
    // returned before the arena was removed.
    let frame_relative_pos = app
        .world_mut()
        .run_system_cached_with(
            |In((from, to)): In<(Entity, Entity)>, rel: RelativeFrameState| -> glam::DVec3 {
                rel.relative_state(from, to).trans.position
            },
            (moon_frame_entity, body_frame_entity),
        )
        .expect("RelativeFrameState run_system_cached_with");

    assert_eq!(
        bevy_integ_frame_entity, moon_frame_entity,
        "post-switch, body frame entity must be ChildOf Moon's frame entity"
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

    // ── astrodyn_runner ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let _earth_idx = sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
    let _moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::third_body(
            &MOON,
            astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(MOON_OFFSET),
        ),
    );

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&initial_trans()),
        rot: Some(initial_rot()),
        mass: Some(vehicle_mass()),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(
                    astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                    GravityGradient::Skip,
                ),
                {
                    let mut c = GravityControl::new_spherical(
                        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>(),
                        GravityGradient::Skip,
                    );
                    c.differential = true;
                    c
                },
            ],
        },
        frame_switches: vec![FrameSwitchConfig {
            target: astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>(),
            switch_sense: SwitchSense::OnApproach,
            switch_distance: SWITCH_RADIUS,
            active: true,
        }],
        ..VehicleConfig::named("bevy-parity-frame-switch-1")
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_body = sim.body(0);
    let sim_pos = sim_body.trans.position.raw_si();
    let sim_vel = sim_body.trans.velocity.raw_si();

    // Body trans state in post-switch (Moon-centered) coords must match.
    for i in 0..3 {
        assert_bits_eq(
            "Bevy vs Sim post-switch position",
            &format!("[{i}]"),
            bevy_trans.position[i],
            sim_pos[i],
        );
        assert_bits_eq(
            "Bevy vs Sim post-switch velocity",
            &format!("[{i}]"),
            bevy_trans.velocity[i],
            sim_vel[i],
        );
    }
    // Body frame entity's relative-state-in-Moon position should
    // equal the body's TranslationalStateC (which carries the body's
    // post-switch state in the new integration frame's coordinates).
    // Both are written by `frame_switch_system`, so a discrepancy
    // would indicate the FrameTransC sync diverged from the body
    // state on the switch tick.
    for i in 0..3 {
        assert_bits_eq(
            "Bevy frame-relative pos vs Sim",
            &format!("pos[{i}]"),
            frame_relative_pos[i],
            sim_pos[i],
        );
    }
}

#[test]
fn bevy_parity_frame_switch_on_departure_matches_simulation() {
    // Cover the `OnDeparture` predicate in the shared generic helper.
    // The `OnApproach` test above triggers a
    // switch when the body comes within `SWITCH_RADIUS` of the Moon;
    // this test triggers when the body departs beyond a threshold from
    // its *current* integration frame's origin (the body's
    // `trans.position` magnitude exceeds `switch_distance`). Without
    // this case, a regression in the OnDeparture predicate or its
    // reparent/apply path inside `evaluate_and_apply_frame_switch`
    // would still pass the suite.
    //
    // Scenario: same Earth-orbit launch toward Moon; the switch
    // threshold is the body's distance from Earth (current integ
    // frame). Once the body's position magnitude exceeds the threshold,
    // the switch triggers and reparents to Moon — bit-identical
    // outcome between Bevy and `astrodyn_runner`.
    let departure_threshold = 1.0e7;

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(AstrodynPlugin);

    let _earth = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth>::point_mass("Earth", &EARTH))
        .id();
    let moon = app
        .world_mut()
        .spawn(PlanetBundle::<astrodyn::Earth> {
            // Identity = the source's own planet (issue #664); the
            // bundle's <Earth> only tags component storage (the sim's
            // central-planet convention, see SunBundle).
            uid: astrodyn_bevy::FrameUidC(astrodyn::FrameUid::of::<
                astrodyn::PlanetInertial<astrodyn::Moon>,
            >()),
            ..PlanetBundle::<astrodyn::Earth>::point_mass("Moon", &MOON)
        })
        .id();

    let switches: Vec<FrameSwitchConfig> = vec![FrameSwitchConfig {
        target: astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>(),
        switch_sense: SwitchSense::OnDeparture,
        switch_distance: departure_threshold,
        active: true,
    }];

    let vehicle = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(&format!(
                "bevy-parity-frame-switch-b2-{}",
                NEXT_BODY_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ))),
            Name::new("EarthDeparture"),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(initial_trans()),
            RotationalStateC::from(initial_rot()),
            MassPropertiesC::from(vehicle_mass()),
            DynamicsConfigC(DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: true,
                three_dof: false,
            }),
            GravityControlsC(GravityControls {
                controls: vec![
                    GravityControl::new_spherical(
                        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                        GravityGradient::Skip,
                    ),
                    {
                        let mut c = GravityControl::new_spherical(
                            astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>(),
                            GravityGradient::Skip,
                        );
                        c.differential = true;
                        c
                    },
                ],
            }),
            FrameSwitchesC(switches),
        ))
        .id();
    app.world_mut().run_schedule(Startup);
    let sys = app
        .world_mut()
        .register_system(move |mut m: SourceMutator<astrodyn::Earth>| {
            m.set_source_position(moon, MOON_OFFSET);
        });
    app.world_mut().run_system(sys).unwrap();

    let moon_frame_entity = app.world().get::<FrameEntityC>(moon).unwrap().0;
    let body_frame_entity = app.world().get::<FrameEntityC>(vehicle).unwrap().0;

    for _ in 0..NUM_STEPS {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);
    }

    let bevy_trans = astrodyn::typed_bridge::trans_typed_to_raw(
        &app.world()
            .get::<TranslationalStateC<astrodyn::Earth>>(vehicle)
            .unwrap()
            .0,
    );
    let bevy_integ_frame_entity = app
        .world()
        .get::<bevy::prelude::ChildOf>(body_frame_entity)
        .unwrap()
        .parent();
    assert_eq!(
        bevy_integ_frame_entity, moon_frame_entity,
        "OnDeparture switch should reparent body frame entity under Moon.inertial"
    );

    // ── astrodyn_runner ──
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    let _earth_idx = sim.add_source("Earth", GravitySourceEntry::central_body(&EARTH));
    let _moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::third_body(
            &MOON,
            astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(MOON_OFFSET),
        ),
    );
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&initial_trans()),
        rot: Some(initial_rot()),
        mass: Some(vehicle_mass()),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(
                    astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>(),
                    GravityGradient::Skip,
                ),
                {
                    let mut c = GravityControl::new_spherical(
                        astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>(),
                        GravityGradient::Skip,
                    );
                    c.differential = true;
                    c
                },
            ],
        },
        frame_switches: vec![FrameSwitchConfig {
            target: astrodyn::FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>(),
            switch_sense: SwitchSense::OnDeparture,
            switch_distance: departure_threshold,
            active: true,
        }],
        ..VehicleConfig::named("bevy-parity-frame-switch-0")
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");
    let sim_body = sim.body(0);
    let sim_pos = sim_body.trans.position.raw_si();
    let sim_vel = sim_body.trans.velocity.raw_si();

    for i in 0..3 {
        assert_bits_eq(
            "Bevy vs Sim OnDeparture position",
            &format!("[{i}]"),
            bevy_trans.position[i],
            sim_pos[i],
        );
        assert_bits_eq(
            "Bevy vs Sim OnDeparture velocity",
            &format!("[{i}]"),
            bevy_trans.velocity[i],
            sim_vel[i],
        );
    }
}

/// Per-call unique suffix for swept test-body identities (#664): helpers
/// spawning multiple bodies per App must mint distinct identities.
static NEXT_BODY_UID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
