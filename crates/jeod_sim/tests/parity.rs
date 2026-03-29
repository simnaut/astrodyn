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
/// Gravity computed once per step, held constant across RK4 stages.
fn run_pure_steps() -> SixDofState {
    let mp = mass_props();
    let mu = MU_EARTH;
    let mut state = SixDofState {
        trans: initial_trans(),
        rot: initial_rot(),
    };

    for _ in 0..NUM_STEPS {
        let pos = state.trans.position;
        let r = pos.length();
        let grav_accel = -mu / (r * r * r) * pos;

        state = jeod_dynamics::rk4_sixdof_step(
            &state,
            |_s| grav_accel, // constant across RK4 stages
            |_| DVec3::ZERO, // no external torque
            &mp,
            DT,
        );
    }

    state
}

#[test]
fn simulation_matches_pure_rk4_sixdof() {
    let sim_state = run_simulation_steps();
    let pure_state = run_pure_steps();

    // Position parity: should be identical to machine precision.
    let pos_diff = (sim_state.trans.position - pure_state.trans.position).length();
    assert!(
        pos_diff < 1e-8,
        "Position difference between Simulation and pure RK4: {} m (exceeds 1e-8 m)\n\
         Sim:   {:?}\n\
         Pure:  {:?}",
        pos_diff,
        sim_state.trans.position,
        pure_state.trans.position,
    );

    // Velocity parity.
    let vel_diff = (sim_state.trans.velocity - pure_state.trans.velocity).length();
    assert!(
        vel_diff < 1e-11,
        "Velocity difference between Simulation and pure RK4: {} m/s (exceeds 1e-11 m/s)\n\
         Sim:   {:?}\n\
         Pure:  {:?}",
        vel_diff,
        sim_state.trans.velocity,
        pure_state.trans.velocity,
    );

    // Quaternion parity.
    let q_sim = sim_state.rot.quaternion.data;
    let q_pure = pure_state.rot.quaternion.data;
    let q_diff: f64 = (0..4)
        .map(|i| (q_sim[i] - q_pure[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        q_diff < 1e-14,
        "Quaternion difference: {} (exceeds 1e-14)\n\
         Sim:   {:?}\n\
         Pure:  {:?}",
        q_diff,
        q_sim,
        q_pure,
    );

    // Angular velocity parity.
    let omega_diff = (sim_state.rot.ang_vel_body - pure_state.rot.ang_vel_body).length();
    assert!(
        omega_diff < 1e-14,
        "Angular velocity difference: {} rad/s (exceeds 1e-14)\n\
         Sim:   {:?}\n\
         Pure:  {:?}",
        omega_diff,
        sim_state.rot.ang_vel_body,
        pure_state.rot.ang_vel_body,
    );
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
        let pos = state.position;
        let r = pos.length();
        let grav_accel = -mu / (r * r * r) * pos;
        state = jeod_dynamics::rk4_translational_step(&state, |_s| grav_accel, DT);
    }

    let pos_diff = (sim.body(0).trans.position - state.position).length();
    assert!(
        pos_diff < 1e-8,
        "3-DOF position difference: {} m (exceeds 1e-8 m)",
        pos_diff,
    );

    let vel_diff = (sim.body(0).trans.velocity - state.velocity).length();
    assert!(
        vel_diff < 1e-11,
        "3-DOF velocity difference: {} m/s (exceeds 1e-11 m/s)",
        vel_diff,
    );
}
