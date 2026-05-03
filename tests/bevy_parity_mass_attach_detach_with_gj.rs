//! Bevy adapter wiring for IG.37: mass-tree attach/detach must reset GJ /
//! ABM4 integrator state on the affected bodies.
//!
//! Mirrors JEOD's `dyn_body_attach.cc::reset_integrators()` (lines 860,
//! 871) and `dyn_body_detach.cc:271-273` precedent. Verifies that
//! `staging_system` calls `jeod_sim::reset_integrators` on the
//! `GaussJacksonStateC` of every body whose composite mass changed.
//!
//! The runner-side equivalent lives in
//! `crates/jeod_runner/src/simulation/mass_tree.rs::tests`. Together they
//! prove the same JEOD invariant on both consumers of `jeod_sim`.

use bevy::prelude::*;
use bevy_jeod::{
    AttachEvent, DetachEvent, DynamicsConfigC, GaussJacksonStateC, GravityControlsC,
    GravitySourceC, IntegratorTypeC, JeodPlugin, MassBodyIdC, MassPropertiesC, MassTreeR,
    SourceInertialPositionC, TranslationalStateC,
};
use glam::{DMat3, DVec3};
use jeod_sim::{
    GaussJacksonConfig, GaussJacksonState, GravityControl, GravityControls, GravityModel,
    GravitySource, IntegratorType, MassProperties, MassTree, TranslationalState,
};
use std::time::Duration;

/// Non-standard μ matching `tests/bevy_parity_gj.rs` so the GJ trajectory
/// is identical to a known good baseline.
const MU_GJ_TEST: f64 = 5.76e14;

fn step_bevy(app: &mut App, n: usize, dt: f64) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

/// Build a Bevy app with two GJ-integrated bodies registered in a shared
/// `MassTreeR`. Returns the app, both vehicle entities, and the mass-tree
/// node ids so the test can poke at internal state.
fn build_two_body_app(
    sim_dt: f64,
) -> (
    App,
    Entity,
    Entity,
    jeod_sim::MassBodyId,
    jeod_sim::MassBodyId,
) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(sim_dt));
    app.add_plugins(JeodPlugin);

    // Mass tree resource — required for `staging_system` to be a no-op-free
    // path through the attach/detach handler. Both bodies share this tree.
    let mut tree = MassTree::new();
    let id_a = tree.add_body("BodyA".into(), MassProperties::new(1000.0));
    let id_b = tree.add_body("BodyB".into(), MassProperties::new(500.0));
    app.insert_resource(MassTreeR(tree));

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Planet"),
            GravitySourceC(GravitySource {
                mu: MU_GJ_TEST,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::default(),
        ))
        .id();

    let trans_a = TranslationalState {
        position: DVec3::new(9e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 8000.0, 0.0),
    };
    let trans_b = TranslationalState {
        position: DVec3::new(9e6, 1.0, 0.0),
        velocity: DVec3::new(0.0, 7900.0, 0.0),
    };

    let gj_cfg = GaussJacksonConfig::with_order(8);
    let body_a = app
        .world_mut()
        .spawn((
            Name::new("VehicleA"),
            DynamicsConfigC::default(),
            TranslationalStateC::from(trans_a),
            MassPropertiesC::from(MassProperties::new(1000.0)),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            IntegratorTypeC(IntegratorType::GaussJackson(gj_cfg)),
            GaussJacksonStateC(GaussJacksonState::new(gj_cfg)),
            MassBodyIdC(id_a),
        ))
        .id();
    let body_b = app
        .world_mut()
        .spawn((
            Name::new("VehicleB"),
            DynamicsConfigC::default(),
            TranslationalStateC::from(trans_b),
            MassPropertiesC::from(MassProperties::new(500.0)),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            IntegratorTypeC(IntegratorType::GaussJackson(gj_cfg)),
            GaussJacksonStateC(GaussJacksonState::new(gj_cfg)),
            MassBodyIdC(id_b),
        ))
        .id();

    (app, body_a, body_b, id_a, id_b)
}

fn read_gj_priming(world: &World, entity: Entity) -> bool {
    world
        .get::<GaussJacksonStateC>(entity)
        .expect("entity missing GaussJacksonStateC")
        .0
        .is_priming()
}

fn read_gj_topology_dirty(world: &World, entity: Entity) -> bool {
    world
        .get::<GaussJacksonStateC>(entity)
        .expect("entity missing GaussJacksonStateC")
        .0
        .is_topology_dirty()
}

/// Driver for the full IG.37 flow on the Bevy adapter:
///   1. Step long enough to leave GJ priming.
///   2. Send `AttachEvent` — `staging_system` mutates the tree and
///      resets both bodies' GJ state.
///   3. Verify both GJ states are back in priming and topology-clean.
///   4. Step once more without panicking — proving the IG.37 assertion
///      in `GaussJacksonState::integrate` does not fire.
#[test]
fn bevy_parity_mass_attach_with_gj_resets_integrator() {
    let sim_dt = 1.0_f64;
    let (mut app, body_a, body_b, id_a, id_b) = build_two_body_app(sim_dt);

    // ── Step 200 sim steps to leave GJ priming on both bodies. ──
    step_bevy(&mut app, 200, sim_dt);
    assert!(
        !read_gj_priming(app.world(), body_a),
        "test setup expected body A GJ past priming after 200 steps"
    );
    assert!(
        !read_gj_priming(app.world(), body_b),
        "test setup expected body B GJ past priming after 200 steps"
    );
    assert!(!read_gj_topology_dirty(app.world(), body_a));
    assert!(!read_gj_topology_dirty(app.world(), body_b));

    // ── Send AttachEvent — staging_system mutates tree + resets state. ──
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: body_b,
            parent: body_a,
            offset: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
        });

    // Run one more step so staging_system processes the event before
    // integration. (staging_system is scheduled in JeodSet::Staging,
    // before integration in the same FixedUpdate.)
    step_bevy(&mut app, 1, sim_dt);

    assert!(
        read_gj_priming(app.world(), body_a),
        "body A GJ must be back in priming after AttachEvent (IG.37)"
    );
    assert!(
        read_gj_priming(app.world(), body_b),
        "body B GJ must be back in priming after AttachEvent (IG.37)"
    );
    assert!(!read_gj_topology_dirty(app.world(), body_a));
    assert!(!read_gj_topology_dirty(app.world(), body_b));

    // Sanity: the mass tree actually changed.
    let tree = &app.world().resource::<MassTreeR>().0;
    assert_eq!(tree.parent(id_b), Some(id_a));

    // ── Step several more times — must not trip IG.37 assertion. ──
    step_bevy(&mut app, 5, sim_dt);
}

/// Mirror of the attach test for `DetachEvent`.
#[test]
fn bevy_parity_mass_detach_with_gj_resets_integrator() {
    let sim_dt = 1.0_f64;
    let (mut app, body_a, body_b, id_a, id_b) = build_two_body_app(sim_dt);

    // Pre-attach so detach has something to undo.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent>>()
        .write(AttachEvent {
            child: body_b,
            parent: body_a,
            offset: DVec3::ZERO,
            t_parent_child: DMat3::IDENTITY,
        });
    step_bevy(&mut app, 200, sim_dt);
    assert!(!read_gj_priming(app.world(), body_a));
    assert!(!read_gj_priming(app.world(), body_b));

    // Detach.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent { child: body_b });
    step_bevy(&mut app, 1, sim_dt);

    assert!(
        read_gj_priming(app.world(), body_a),
        "parent's GJ must reset on DetachEvent (IG.37)"
    );
    assert!(
        read_gj_priming(app.world(), body_b),
        "child's GJ must reset on DetachEvent (IG.37)"
    );
    assert!(!read_gj_topology_dirty(app.world(), body_a));
    assert!(!read_gj_topology_dirty(app.world(), body_b));

    // Mass tree updated.
    let tree = &app.world().resource::<MassTreeR>().0;
    assert_eq!(tree.parent(id_b), None);
    let _ = id_a;

    // No IG.37 panic on subsequent steps.
    step_bevy(&mut app, 5, sim_dt);
}
