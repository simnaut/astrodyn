//! Tier 3: Bevy App vs jeod_runner::Simulation bit-identical parity tests
//! for relative dynamics and LVLH-relative state computations.
//!
//! Each test sets up identical initial conditions in both a Bevy App
//! and a jeod_runner::Simulation, steps both the same number of times,
//! then computes relative state post-step and asserts `f64::to_bits()`
//! equality.

use bevy::prelude::*;
use bevy_jeod::{
    DynamicsConfigC, GravityControlsC, JeodPlugin, MassPropertiesC, RotationalStateC,
    TranslationalStateC,
};
use glam::DVec3;
use jeod_runner::{Simulation, VehicleConfig};
use jeod_sim::{
    DynamicsConfig, GravityControls, JeodQuat, MassProperties, RotationalState, SixDofState,
    TranslationalState,
};

const DT: f64 = 10.0;
const NUM_STEPS: usize = 100;

fn new_bevy_app(dt: f64) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(dt));
    app.add_plugins(JeodPlugin);
    app
}

fn step_bevy(app: &mut App, n: usize) {
    for _ in 0..n {
        let dt = app.world().resource::<Time<Fixed>>().timestep();
        app.world_mut().resource_mut::<Time<Fixed>>().advance_by(dt);
        app.world_mut().run_schedule(FixedUpdate);
    }
}

fn read_sixdof(world: &World, entity: Entity) -> SixDofState {
    SixDofState {
        trans: world
            .get::<TranslationalStateC>(entity)
            .unwrap()
            .0
            .to_untyped(),
        rot: world
            .get::<RotationalStateC>(entity)
            .unwrap()
            .0
            .to_untyped(),
    }
}

fn read_trans(world: &World, entity: Entity) -> TranslationalState {
    world
        .get::<TranslationalStateC>(entity)
        .unwrap()
        .0
        .to_untyped()
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

fn assert_sixdof_eq(label: &str, a: &SixDofState, b: &SixDofState) {
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("position[{i}]"),
            a.trans.position[i],
            b.trans.position[i],
        );
        assert_bits_eq(
            label,
            &format!("velocity[{i}]"),
            a.trans.velocity[i],
            b.trans.velocity[i],
        );
        assert_bits_eq(
            label,
            &format!("ang_vel[{i}]"),
            a.rot.ang_vel_body[i],
            b.rot.ang_vel_body[i],
        );
    }
    for i in 0..4 {
        assert_bits_eq(
            label,
            &format!("quat[{i}]"),
            a.rot.quaternion.data[i],
            b.rot.quaternion.data[i],
        );
    }
    println!("  {label}: bit-identical (all 13 components)");
}

fn assert_trans_eq(label: &str, a: &TranslationalState, b: &TranslationalState) {
    for i in 0..3 {
        assert_bits_eq(
            label,
            &format!("position[{i}]"),
            a.position[i],
            b.position[i],
        );
        assert_bits_eq(
            label,
            &format!("velocity[{i}]"),
            a.velocity[i],
            b.velocity[i],
        );
    }
    println!("  {label}: bit-identical (all 6 components)");
}

// ── Relative Dynamics Parity Tests ──
// Two-body force-free 6-DOF: step both runners, compute relative state
// post-step, assert bit-identical.

fn run_relative_parity(
    label: &str,
    trans_a: TranslationalState,
    rot_a: RotationalState,
    trans_b: TranslationalState,
    rot_b: RotationalState,
) {
    let dummy_mass = MassProperties::new(1.0);
    let config_6dof = DynamicsConfig {
        translational_dynamics: true,
        rotational_dynamics: true,
        three_dof: false,
    };

    // ── Bevy ──
    let mut app = new_bevy_app(DT);

    let veh_a = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(trans_a),
            RotationalStateC::from(rot_a),
            MassPropertiesC::from(dummy_mass),
            DynamicsConfigC(config_6dof),
            GravityControlsC(GravityControls::<Entity> { controls: vec![] }),
        ))
        .id();

    let veh_b = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(trans_b),
            RotationalStateC::from(rot_b),
            MassPropertiesC::from(dummy_mass),
            DynamicsConfigC(config_6dof),
            GravityControlsC(GravityControls::<Entity> { controls: vec![] }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_a = read_sixdof(app.world(), veh_a);
    let bevy_b = read_sixdof(app.world(), veh_b);
    let bevy_rel = jeod_sim::compute_relative_state(
        &bevy_a.trans,
        Some(&bevy_a.rot),
        &bevy_b.trans,
        Some(&bevy_b.rot),
    );

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    sim.add_body(VehicleConfig {
        trans: trans_a,
        rot: Some(rot_a),
        mass: Some(dummy_mass),
        ..Default::default()
    });
    sim.add_body(VehicleConfig {
        trans: trans_b,
        rot: Some(rot_b),
        mass: Some(dummy_mass),
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_a_state = SixDofState {
        trans: sim.body(0).trans,
        rot: sim.body(0).rot.unwrap(),
    };
    let sim_b_state = SixDofState {
        trans: sim.body(1).trans,
        rot: sim.body(1).rot.unwrap(),
    };
    let sim_rel = jeod_sim::compute_relative_state(
        &sim_a_state.trans,
        Some(&sim_a_state.rot),
        &sim_b_state.trans,
        Some(&sim_b_state.rot),
    );

    // Assert body states are bit-identical
    assert_sixdof_eq(&format!("Bevy vs Sim A ({label})"), &bevy_a, &sim_a_state);
    assert_sixdof_eq(&format!("Bevy vs Sim B ({label})"), &bevy_b, &sim_b_state);

    // Assert relative state is bit-identical
    for i in 0..3 {
        assert_bits_eq(
            &format!("Bevy vs Sim rel ({label})"),
            &format!("rel_pos[{i}]"),
            bevy_rel.position[i],
            sim_rel.position[i],
        );
        assert_bits_eq(
            &format!("Bevy vs Sim rel ({label})"),
            &format!("rel_vel[{i}]"),
            bevy_rel.velocity[i],
            sim_rel.velocity[i],
        );
    }
    println!("  Bevy vs Sim relative state ({label}): bit-identical");
}

#[test]
fn tier3_bevy_relative_ab_rot_ab_trans() {
    let trans_a = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    let rot_a = RotationalState {
        quaternion: {
            let mut q = JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
            q.normalize();
            q
        },
        ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
    };
    let trans_b = TranslationalState {
        position: DVec3::new(6_778_237.0, 100.0, -50.0),
        velocity: DVec3::new(0.01, 7668.55, 0.005),
    };
    let rot_b = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::new(0.0, 0.0, 0.001),
    };
    run_relative_parity("ab_rot_ab_trans", trans_a, rot_a, trans_b, rot_b);
}

#[test]
fn tier3_bevy_relative_no_rot_ab_trans() {
    let trans_a = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    let rot_zero = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    let trans_b = TranslationalState {
        position: DVec3::new(6_778_237.0, 100.0, -50.0),
        velocity: DVec3::new(0.01, 7668.55, 0.005),
    };
    run_relative_parity("no_rot_ab_trans", trans_a, rot_zero, trans_b, rot_zero);
}

#[test]
fn tier3_bevy_relative_a_rot_no_trans() {
    let trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    let rot_a = RotationalState {
        quaternion: {
            let mut q = JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
            q.normalize();
            q
        },
        ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
    };
    let rot_b = RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    };
    // Same translational ICs for both — only rotation differs
    run_relative_parity("a_rot_no_trans", trans, rot_a, trans, rot_b);
}

// ── LVLH-Relative Parity Tests ──
// Two-body force-free 3-DOF: step both runners, compute LVLH-relative
// state post-step, assert bit-identical.

fn run_lvlhrel_parity(label: &str, ref_trans: TranslationalState, subj_trans: TranslationalState) {
    // ── Bevy ──
    let mut app = new_bevy_app(DT);

    let ref_veh = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(ref_trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls::<Entity> { controls: vec![] }),
        ))
        .id();

    let subj_veh = app
        .world_mut()
        .spawn((
            TranslationalStateC::from(subj_trans),
            DynamicsConfigC::default(),
            GravityControlsC(GravityControls::<Entity> { controls: vec![] }),
        ))
        .id();

    step_bevy(&mut app, NUM_STEPS);
    let bevy_ref = read_trans(app.world(), ref_veh);
    let bevy_subj = read_trans(app.world(), subj_veh);
    let bevy_rel = jeod_sim::compute_lvlh_relative_state(
        bevy_ref.position,
        bevy_ref.velocity,
        bevy_subj.position,
        bevy_subj.velocity,
    );

    // ── Simulation ──
    let time = jeod_sim::SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    sim.add_body(VehicleConfig {
        trans: ref_trans,
        ..Default::default()
    });
    sim.add_body(VehicleConfig {
        trans: subj_trans,
        ..Default::default()
    });
    sim.validate().unwrap();
    sim.step_n(NUM_STEPS).expect("step_n failed");

    let sim_ref = sim.body(0).trans;
    let sim_subj = sim.body(1).trans;
    let sim_rel = jeod_sim::compute_lvlh_relative_state(
        sim_ref.position,
        sim_ref.velocity,
        sim_subj.position,
        sim_subj.velocity,
    );

    // Assert body states are bit-identical
    assert_trans_eq(&format!("Bevy vs Sim ref ({label})"), &bevy_ref, &sim_ref);
    assert_trans_eq(
        &format!("Bevy vs Sim subj ({label})"),
        &bevy_subj,
        &sim_subj,
    );

    // Assert LVLH-relative state is bit-identical
    for i in 0..3 {
        assert_bits_eq(
            &format!("Bevy vs Sim lvlhrel ({label})"),
            &format!("pos[{i}]"),
            bevy_rel.position[i],
            sim_rel.position[i],
        );
        assert_bits_eq(
            &format!("Bevy vs Sim lvlhrel ({label})"),
            &format!("vel[{i}]"),
            bevy_rel.velocity[i],
            sim_rel.velocity[i],
        );
    }
    println!("  Bevy vs Sim LVLH-relative ({label}): bit-identical");
}

#[test]
fn tier3_bevy_lvlhrel_test0() {
    let ref_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    let subj_trans = TranslationalState {
        position: DVec3::new(6_778_237.0, 100.0, -50.0),
        velocity: DVec3::new(0.01, 7668.55, 0.005),
    };
    run_lvlhrel_parity("test0", ref_trans, subj_trans);
}

#[test]
fn tier3_bevy_lvlhrel_test1() {
    // Coplanar formation with larger separation
    let ref_trans = TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    };
    let subj_trans = TranslationalState {
        position: DVec3::new(6_779_137.0, 0.0, 0.0), // 1 km ahead
        velocity: DVec3::new(0.0, 7667.56, 0.0),     // slightly slower
    };
    run_lvlhrel_parity("test1", ref_trans, subj_trans);
}
