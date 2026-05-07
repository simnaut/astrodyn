// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy adapter wiring for IG.37 — ABM4 sibling of
//! `bevy_parity_mass_attach_detach_with_gj.rs`.
//!
//! `staging_system` resets ABM4 history on the affected bodies as well
//! as Gauss-Jackson; the GJ-only test left the ABM4 arm uncovered, so
//! a regression in `Abm4StateC` reset wiring would slip through. Also
//! exercises the ancestor-chain coverage fix from PR #282 review
//! threads `PRRT_kwDORtae6c5_J-qF` / `PRRT_kwDORtae6c5_J-qI`.

use astrodyn::{
    Abm4State, GravityControl, GravityControls, GravityModel, GravitySource, IntegratorType,
    MassProperties, MassTree, TranslationalState,
};
use astrodyn_bevy::{
    Abm4StateC, AttachEvent, DetachEvent, DynamicsConfigC, GravityControlsC, GravitySourceC,
    IntegratorTypeC, JeodPlugin, MassBodyIdC, MassPropertiesC, MassTreeR, SourceInertialPositionC,
    TranslationalStateC,
};
use bevy::prelude::*;
use glam::DVec3;
use std::time::Duration;

const MU: f64 = 5.76e14;

fn step_bevy(app: &mut App, n: usize, dt: f64) {
    for _ in 0..n {
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(dt));
        app.world_mut().run_schedule(FixedUpdate);
    }
}

fn read_abm4_priming(world: &World, entity: Entity) -> bool {
    world
        .get::<Abm4StateC>(entity)
        .expect("entity missing Abm4StateC")
        .0
        .is_priming()
}

fn read_abm4_topology_dirty(world: &World, entity: Entity) -> bool {
    world
        .get::<Abm4StateC>(entity)
        .expect("entity missing Abm4StateC")
        .0
        .is_topology_dirty()
}

/// Build a Bevy app with two ABM4-integrated bodies registered in a
/// shared `MassTreeR`.
fn build_two_body_app(
    sim_dt: f64,
) -> (
    App,
    Entity,
    Entity,
    astrodyn::MassBodyId,
    astrodyn::MassBodyId,
) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(sim_dt));
    app.add_plugins(JeodPlugin);

    let mut tree = MassTree::new();
    let id_a = tree.add_body("BodyA".into(), MassProperties::new(1000.0));
    let id_b = tree.add_body("BodyB".into(), MassProperties::new(500.0));
    app.insert_resource(MassTreeR(tree));

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Planet"),
            GravitySourceC(GravitySource {
                mu: MU,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
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

    let body_a = app
        .world_mut()
        .spawn((
            Name::new("VehicleA"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from(trans_a),
            MassPropertiesC::from(MassProperties::new(1000.0)),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            IntegratorTypeC(IntegratorType::Abm4),
            Abm4StateC(Abm4State::new()),
            MassBodyIdC(id_a),
        ))
        .id();
    let body_b = app
        .world_mut()
        .spawn((
            Name::new("VehicleB"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from(trans_b),
            MassPropertiesC::from(MassProperties::new(500.0)),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, false)],
            }),
            IntegratorTypeC(IntegratorType::Abm4),
            Abm4StateC(Abm4State::new()),
            MassBodyIdC(id_b),
        ))
        .id();

    (app, body_a, body_b, id_a, id_b)
}

#[test]
fn bevy_parity_mass_attach_with_abm4_resets_integrator() {
    let sim_dt = 1.0_f64;
    let (mut app, body_a, body_b, id_a, id_b) = build_two_body_app(sim_dt);

    // ABM4 primes after `HIST_LEN - 1 = 3` steps; 5 puts it comfortably
    // into operational mode on both bodies.
    step_bevy(&mut app, 5, sim_dt);
    assert!(!read_abm4_priming(app.world(), body_a));
    assert!(!read_abm4_priming(app.world(), body_b));
    assert!(!read_abm4_topology_dirty(app.world(), body_a));
    assert!(!read_abm4_topology_dirty(app.world(), body_b));

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: body_b,
            parent: body_a,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step_bevy(&mut app, 1, sim_dt);

    assert!(
        read_abm4_priming(app.world(), body_a),
        "body A ABM4 must be back in priming after AttachEvent (IG.37)"
    );
    assert!(
        read_abm4_priming(app.world(), body_b),
        "body B ABM4 must be back in priming after AttachEvent (IG.37)"
    );
    assert!(!read_abm4_topology_dirty(app.world(), body_a));
    assert!(!read_abm4_topology_dirty(app.world(), body_b));

    let tree = &app.world().resource::<MassTreeR>().0;
    assert_eq!(tree.parent(id_b), Some(id_a));
    let _ = id_a;

    // No IG.37 panic on subsequent steps.
    step_bevy(&mut app, 5, sim_dt);
}

#[test]
fn bevy_parity_mass_detach_with_abm4_resets_integrator() {
    let sim_dt = 1.0_f64;
    let (mut app, body_a, body_b, id_a, id_b) = build_two_body_app(sim_dt);

    // Pre-attach so detach has something to undo.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: body_b,
            parent: body_a,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step_bevy(&mut app, 5, sim_dt);
    assert!(!read_abm4_priming(app.world(), body_a));
    assert!(!read_abm4_priming(app.world(), body_b));

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent { child: body_b });
    step_bevy(&mut app, 1, sim_dt);

    assert!(
        read_abm4_priming(app.world(), body_a),
        "parent ABM4 must reset on DetachEvent (IG.37)"
    );
    assert!(
        read_abm4_priming(app.world(), body_b),
        "child ABM4 must reset on DetachEvent (IG.37)"
    );
    assert!(!read_abm4_topology_dirty(app.world(), body_a));
    assert!(!read_abm4_topology_dirty(app.world(), body_b));

    let tree = &app.world().resource::<MassTreeR>().0;
    assert_eq!(tree.parent(id_b), None);
    let _ = id_a;

    step_bevy(&mut app, 5, sim_dt);
}

/// `staging_system` must reset integrators on the **full ancestor
/// chain**, not just the directly-named bodies. Builds a 3-body chain
/// `top → middle → leaf`, then attaches a fourth body underneath
/// `middle` and verifies that `top`'s ABM4 state is reset (in addition
/// to `middle` and the new attachee). Mirrors PR #282 review threads
/// `PRRT_kwDORtae6c5_J-qF` (attach) and `PRRT_kwDORtae6c5_J-qI`
/// (detach).
#[test]
fn bevy_parity_mass_attach_resets_full_ancestor_chain() {
    let sim_dt = 1.0_f64;
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(sim_dt));
    app.add_plugins(JeodPlugin);

    let mut tree = MassTree::new();
    let id_top = tree.add_body("Top".into(), MassProperties::new(1000.0));
    let id_middle = tree.add_body("Middle".into(), MassProperties::new(500.0));
    let id_leaf = tree.add_body("Leaf".into(), MassProperties::new(100.0));
    let id_new = tree.add_body("NewAttachee".into(), MassProperties::new(50.0));
    app.insert_resource(MassTreeR(tree));

    let planet = app
        .world_mut()
        .spawn((
            Name::new("Planet"),
            GravitySourceC(GravitySource {
                mu: MU,
                model: GravityModel::PointMass,
            }),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
        ))
        .id();

    let trans = TranslationalState {
        position: DVec3::new(9e6, 0.0, 0.0),
        velocity: DVec3::new(0.0, 8000.0, 0.0),
    };

    let mk_body = |app: &mut App, id: astrodyn::MassBodyId, mass: f64, name: &str| -> Entity {
        app.world_mut()
            .spawn((
                Name::new(name.to_string()),
                DynamicsConfigC::default(),
                TranslationalStateC::<astrodyn::Earth>::from(trans),
                MassPropertiesC::from(MassProperties::new(mass)),
                GravityControlsC(GravityControls {
                    controls: vec![GravityControl::new_spherical(planet, false)],
                }),
                IntegratorTypeC(IntegratorType::Abm4),
                Abm4StateC(Abm4State::new()),
                MassBodyIdC(id),
            ))
            .id()
    };
    let e_top = mk_body(&mut app, id_top, 1000.0, "Top");
    let e_middle = mk_body(&mut app, id_middle, 500.0, "Middle");
    let e_leaf = mk_body(&mut app, id_leaf, 100.0, "Leaf");
    let e_new = mk_body(&mut app, id_new, 50.0, "NewAttachee");

    // Build the chain: middle → top, leaf → middle.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: e_middle,
            parent: e_top,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: e_leaf,
            parent: e_middle,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    // One step processes both attach events. Then prime ABM4.
    step_bevy(&mut app, 5, sim_dt);
    assert!(
        !read_abm4_priming(app.world(), e_top),
        "test setup: top ABM4 must be past priming"
    );

    // ── Attach `e_new` under `e_middle`. This recomputes middle's
    //    AND top's composite properties, so top's ABM4 must reset. ──
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: e_new,
            parent: e_middle,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });
    step_bevy(&mut app, 1, sim_dt);

    assert!(
        read_abm4_priming(app.world(), e_top),
        "ancestor `top`'s ABM4 must be reset when a body is attached \
         under its descendant `middle` (IG.37 ancestor coverage)"
    );
    assert!(!read_abm4_topology_dirty(app.world(), e_top));

    // Prime past again, then detach to verify ancestor coverage on
    // the detach branch too.
    step_bevy(&mut app, 5, sim_dt);
    assert!(!read_abm4_priming(app.world(), e_top));

    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent { child: e_new });
    step_bevy(&mut app, 1, sim_dt);

    assert!(
        read_abm4_priming(app.world(), e_top),
        "ancestor `top`'s ABM4 must be reset when a descendant of \
         `middle` is detached (IG.37 ancestor coverage)"
    );
    assert!(!read_abm4_topology_dirty(app.world(), e_top));
    let _ = e_leaf;
}
