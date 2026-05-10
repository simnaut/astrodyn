// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Bevy adapter wiring for IG.37: mass-tree attach/detach must reset GJ /
//! ABM4 integrator state on the affected bodies.
//!
//! Mirrors JEOD's `dyn_body_attach.cc::reset_integrators()` (lines 860,
//! 871) and `dyn_body_detach.cc:271-273` precedent. Verifies that
//! `staging_system` calls `astrodyn::reset_integrators` on the
//! `GaussJacksonStateC` of every body whose composite mass changed.
//!
//! The runner-side equivalent lives in
//! `crates/astrodyn_runner/src/simulation/mass_tree.rs::tests`. Together they
//! prove the same JEOD invariant on both consumers of `astrodyn`.

use astrodyn::{
    GaussJacksonConfig, GaussJacksonState, GravityControl, GravityControls, GravityGradient,
    GravityModel, GravitySource, IntegratorType, MassProperties, MassTree, TranslationalState,
};
use astrodyn_bevy::{
    AstrodynPlugin, AttachEvent, DetachEvent, DynamicsConfigC, GaussJacksonStateC,
    GravityControlsC, GravitySourceC, IntegratorTypeC, MassBodyIdC, MassPropertiesC, MassTreeR,
    SourceInertialPositionC, TranslationalStateC,
};
use bevy::prelude::*;
use glam::DVec3;
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
    astrodyn::MassBodyId,
    astrodyn::MassBodyId,
) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(sim_dt));
    app.add_plugins(AstrodynPlugin);

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

    let gj_cfg = GaussJacksonConfig::with_order(8);
    let body_a = app
        .world_mut()
        .spawn((
            Name::new("VehicleA"),
            DynamicsConfigC::default(),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(trans_a),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                &(MassProperties::new(1000.0)),
            )),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, GravityGradient::Skip)],
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
            TranslationalStateC::<astrodyn::Earth>::from_untyped(trans_b),
            MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                &(MassProperties::new(500.0)),
            )),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_spherical(planet, GravityGradient::Skip)],
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
fn bevy_parity_mass_attach_detach_with_gj_mass_attach_with_gj_resets_integrator() {
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
        .resource_mut::<bevy::ecs::message::Messages<AttachEvent<astrodyn::SelfRef, astrodyn::SelfRef>>>()
        .write(AttachEvent {
            child: body_b,
            parent: body_a,
            offset: astrodyn::Vec3Ext::m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(
                DVec3::ZERO,
            ),
            t_parent_child: astrodyn::FrameTransform::identity(),
        });

    // Run one more step so staging_system processes the event before
    // integration. `staging_system` is registered with
    // `.after(AstrodynSet::Environment).before(AstrodynSet::Interaction)` in
    // `AstrodynPlugin::build` (`src/lib.rs`) — there is no dedicated
    // `AstrodynSet::Staging` variant; staging is wedged between Environment
    // and Interaction so mass-tree changes affect the current step's
    // interactions and integration.
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

/// `staging_system` must reset GJ state on the **full ancestor
/// chain**, not just the directly-named bodies. Builds a 3-body chain
/// `top → middle → leaf`, then attaches a fourth body underneath
/// `middle` and verifies that `top`'s GJ state is reset (in addition
/// to `middle` and the new attachee). Mirrors PR #282 review thread
/// `PRRT_kwDORtae6c5_J-qF` (attach branch).
#[test]
fn bevy_parity_mass_attach_detach_with_gj_mass_attach_with_gj_resets_full_ancestor_chain() {
    let sim_dt = 1.0_f64;
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(sim_dt));
    app.add_plugins(AstrodynPlugin);

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
                mu: MU_GJ_TEST,
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
    let gj_cfg = GaussJacksonConfig::with_order(8);
    let mk_body = |app: &mut App, id: astrodyn::MassBodyId, mass: f64, name: &str| -> Entity {
        app.world_mut()
            .spawn((
                Name::new(name.to_string()),
                DynamicsConfigC::default(),
                TranslationalStateC::<astrodyn::Earth>::from_untyped(trans),
                MassPropertiesC::from(astrodyn::typed_bridge::mass_raw_to_self_ref(
                    &(MassProperties::new(mass)),
                )),
                GravityControlsC(GravityControls {
                    controls: vec![GravityControl::new_spherical(planet, GravityGradient::Skip)],
                }),
                IntegratorTypeC(IntegratorType::GaussJackson(gj_cfg)),
                GaussJacksonStateC(GaussJacksonState::new(gj_cfg)),
                MassBodyIdC(id),
            ))
            .id()
    };
    let e_top = mk_body(&mut app, id_top, 1000.0, "Top");
    let e_middle = mk_body(&mut app, id_middle, 500.0, "Middle");
    let e_leaf = mk_body(&mut app, id_leaf, 100.0, "Leaf");
    let e_new = mk_body(&mut app, id_new, 50.0, "NewAttachee");

    // Chain: middle → top, leaf → middle.
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
    step_bevy(&mut app, 200, sim_dt);
    assert!(
        !read_gj_priming(app.world(), e_top),
        "test setup: top GJ must be past priming"
    );

    // Attach e_new under e_middle — recomputes middle's AND top's
    // composites, so top's GJ must reset.
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
        read_gj_priming(app.world(), e_top),
        "ancestor `top`'s GJ must be reset when a body is attached \
         under its descendant `middle` (IG.37 ancestor coverage)"
    );
    assert!(!read_gj_topology_dirty(app.world(), e_top));

    // Prime past again, then detach to verify ancestor coverage on
    // the detach branch too.
    step_bevy(&mut app, 200, sim_dt);
    assert!(!read_gj_priming(app.world(), e_top));
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<DetachEvent>>()
        .write(DetachEvent { child: e_new });
    step_bevy(&mut app, 1, sim_dt);
    assert!(
        read_gj_priming(app.world(), e_top),
        "ancestor `top`'s GJ must be reset when a descendant of \
         `middle` is detached (IG.37 ancestor coverage)"
    );
    assert!(!read_gj_topology_dirty(app.world(), e_top));
    let _ = e_leaf;
}

/// Mirror of the attach test for `DetachEvent`.
#[test]
fn bevy_parity_mass_attach_detach_with_gj_mass_detach_with_gj_resets_integrator() {
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
