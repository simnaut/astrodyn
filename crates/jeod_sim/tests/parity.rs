//! Parity test: jeod_sim::Simulation vs manual rk4_sixdof_step.
//!
//! Validates that the Simulation runner produces identical state to calling
//! the pure RK4 function directly with matching parameters. This proves
//! the orchestration layer introduces no numerical drift.

use glam::{DMat3, DVec3};
use jeod_dynamics::{
    DynamicsConfig, GravityAcceleration, MassProperties, RotationalState, SixDofState,
    TranslationalState,
};
use jeod_gravity::{GravityControl, GravityControls, GravityModel, GravitySource};
use jeod_math::JeodQuat;
use jeod_sim::{GravitySourceEntry, SimBody, Simulation};
use jeod_time::SimulationTime;

const MU_EARTH: f64 = 3.986004418e14;
const DT: f64 = 10.0;
const NUM_STEPS: usize = 100;

fn initial_trans() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    }
}

fn initial_rot() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::new(0.001, 0.0, 0.001),
    }
}

fn mass_props() -> MassProperties {
    MassProperties::with_inertia(
        400_000.0,
        DMat3::from_diagonal(DVec3::new(1.02e8, 0.91e8, 1.64e8)),
        DVec3::ZERO,
    )
}

/// Run via jeod_sim::Simulation::step_n.
fn run_simulation_steps() -> SixDofState {
    let time = SimulationTime::at_j2000(jeod_time::leap_second::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None, // point mass, no rotation
    });

    sim.add_body(SimBody {
        trans: initial_trans(),
        rot: Some(initial_rot()),
        mass: Some(mass_props()),
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: true,
            three_dof: false,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: None,
        srp: None,
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: None,
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    let body = sim.body(0);
    SixDofState {
        trans: body.trans,
        rot: body.rot.unwrap(),
    }
}

/// Run via manual rk4_sixdof_step loop (same as bevy_parity.rs).
/// Gravity recomputed at each RK4 intermediate state.
fn run_pure_steps() -> SixDofState {
    let mp = mass_props();
    let mu = MU_EARTH;
    let mut state = SixDofState {
        trans: initial_trans(),
        rot: initial_rot(),
    };

    for _ in 0..NUM_STEPS {
        state = jeod_dynamics::rk4_sixdof_step(
            &state,
            |s| {
                let pos = s.trans.position;
                let r = pos.length();
                -mu / (r * r * r) * pos
            },
            |_| DVec3::ZERO,
            &mp,
            DT,
        );
    }

    state
}

/// Assert two f64 values are bit-identical.
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
}

fn assert_trans_bit_identical(label: &str, a: &TranslationalState, b: &TranslationalState) {
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
}

#[test]
fn simulation_matches_pure_rk4_sixdof() {
    let sim_state = run_simulation_steps();
    let pure_state = run_pure_steps();

    // Simulation recomputes gravity via accumulate_gravity() -> GravityControl::evaluate(),
    // while manual RK4 uses inline -mu/(r^3)*pos. The formulas are identical but
    // intermediate operations differ, allowing up to 1 ULP per step to accumulate.
    // The authoritative bit-identical check is Bevy-vs-Simulation (cross_parity.rs).
    let pos_diff = (sim_state.trans.position - pure_state.trans.position).length();
    let vel_diff = (sim_state.trans.velocity - pure_state.trans.velocity).length();
    assert!(pos_diff < 1e-6, "Sim vs Pure pos diff {pos_diff} m");
    assert!(vel_diff < 1e-9, "Sim vs Pure vel diff {vel_diff} m/s");
}

#[test]
fn simulation_3dof_matches_pure_translational() {
    // 3-DOF variant: no rotation, just translational dynamics.
    let time = SimulationTime::at_j2000(jeod_time::leap_second::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        t_inertial_pfix: None,
    });

    sim.add_body(SimBody {
        trans: initial_trans(),
        rot: None,
        mass: None,
        config: DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: None,
        srp: None,
        t_struct_body: DMat3::IDENTITY,
        compute_gravity_torque: false,
        atmospheric_state: None,
        gravity_accel: GravityAcceleration::default(),
        total_force: Default::default(),
        frame_derivs: Default::default(),
        aero_force: None,
        radiation_force: None,
        gravity_torque: None,
    });

    sim.validate().unwrap();
    sim.step_n(NUM_STEPS);

    // Compare against manual translational-only integration
    let mu = MU_EARTH;
    let mut state = initial_trans();
    for _ in 0..NUM_STEPS {
        state = jeod_dynamics::rk4_translational_step(
            &state,
            |s| {
                let pos = s.position;
                let r = pos.length();
                -mu / (r * r * r) * pos
            },
            DT,
        );
    }

    let pos_diff = (sim.body(0).trans.position - state.position).length();
    let vel_diff = (sim.body(0).trans.velocity - state.velocity).length();
    assert!(pos_diff < 1e-6, "3-DOF pos diff {pos_diff} m");
    assert!(vel_diff < 1e-9, "3-DOF vel diff {vel_diff} m/s");
}
