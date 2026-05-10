// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: SIM_dyncomp RUN_9A/9C/9D — External force/torque via Simulation pipeline
//!
//! Uses `Simulation::step()` with `external_force` / `external_torque` fields
//! for time-scheduled force injection. All parameters loaded from JEOD sources.

use astrodyn::{
    GravityControl, GravityControls, GravityModel, GravityRole, GravitySource, JeodQuat,
    MassProperties, RotationalState, SimulationTime, TranslationalState,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_runner::{RotationModel, Simulation};
use astrodyn_verif_jeod::crossval::{CrossvalReport, StateLog};
use astrodyn_verif_jeod::mass_data::MassInitData;
use astrodyn_verif_jeod::tier3_csv::{dyncomp_to_state_log_6dof, load_dyncomp_csv, test_data_path};
use glam::{DMat3, DVec3};

/// Build [`MassProperties`] from parsed JEOD mass-init data (test-only helper).
///
/// Inlined here because `MassProperties` lives in `astrodyn_dynamics` (a real
/// dep of `astrodyn_verif_jeod` would create a cycle), and the helper is used
/// only by Tier 3 mass-fixture tests.
fn mass_props_from_init(init: &MassInitData) -> MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(init.inertia[0][0], init.inertia[1][0], init.inertia[2][0]),
        DVec3::new(init.inertia[0][1], init.inertia[1][1], init.inertia[2][1]),
        DVec3::new(init.inertia[0][2], init.inertia[1][2], init.inertia[2][2]),
    );
    MassProperties::with_inertia(init.mass, inertia, DVec3::from_slice(&init.position))
}

/// Torque window: [1000, 2000) seconds.
const TORQUE_START: f64 = 1000.0;
const TORQUE_END: f64 = 2000.0;

/// Check whether `t` is in the external force/torque window.
/// Uses half-dt margin to match JEOD's Trick scheduling boundary.
fn in_torque_window(t: f64, dt: f64) -> bool {
    t + dt * 0.5 >= TORQUE_START && t + dt * 0.5 < TORQUE_END
}

/// Shared setup for RUN_9 variants: load params from JEOD, create Simulation.
fn setup_run9(
    csv_name: &str,
    init_ang_vel: DVec3,
) -> (
    Simulation,
    Vec<astrodyn_verif_jeod::dyncomp_csv::DyncompRecord>,
    f64,
) {
    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    // Dynamics timestep: 0.03125 s (32 Hz) per
    // verif/SIM_dyncomp/S_define `#define DYNAMICS`.
    let dt = 0.03125_f64;
    let mu_earth = astrodyn::gravity_fixtures::load_ggm05c().mu;

    // ISS mass properties from
    // verif/SIM_dyncomp/Modified_data/mass.py `def set_mass_iss()`.
    let mass_init = MassInitData {
        mass: 400_000.0,
        position: [-3.0, -1.5, 4.0],
        inertia: [
            [1.02e+8, -6.96e+6, -5.48e+6],
            [-6.96e+6, 0.91e+8, 5.90e+5],
            [-5.48e+6, 5.90e+5, 1.64e+8],
        ],
    };
    let mass_props = mass_props_from_init(&mass_init);

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() >= 100);
    let init = &trajectory[0];

    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
                ang_vel_body: init_ang_vel,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityRole::Central)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    (sim, trajectory, dt)
}

/// Step simulation to `target_time`, updating external force/torque at each dt.
/// `force_torque_fn` returns (force_inertial, torque_body) given the current
/// simulation time and body quaternion.
fn step_with_external<F>(sim: &mut Simulation, target_time: f64, dt: f64, force_torque_fn: F)
where
    F: Fn(f64, &JeodQuat) -> (DVec3, DVec3),
{
    while sim.elapsed() + dt <= target_time + 0.001 {
        let quat = sim
            .body(0)
            .rot
            .as_ref()
            .unwrap()
            .q_inertial_body
            .to_jeod_quat();
        let (force, torque) = force_torque_fn(sim.elapsed(), &quat);
        sim.set_body_external_force(0, force);
        sim.set_body_external_torque(0, torque);
        sim.step().expect("step failed");
    }
    // Fractional remainder
    let remainder = target_time - sim.elapsed();
    if remainder > 0.001 {
        let quat = sim
            .body(0)
            .rot
            .as_ref()
            .unwrap()
            .q_inertial_body
            .to_jeod_quat();
        let (force, torque) = force_torque_fn(sim.elapsed(), &quat);
        sim.set_body_external_force(0, force);
        sim.set_body_external_torque(0, torque);
        sim.set_dt(remainder);
        sim.step().expect("step failed");
        sim.set_dt(dt);
    }
}

// ── RUN_9A: External torque only, zero inertial rate ──

#[test]
fn tier3_simulation_run9a_torque() {
    let (mut sim, trajectory, dt) = setup_run9("dyncomp_run9a_state.csv", DVec3::ZERO);

    println!(
        "Tier 3 (Simulation): RUN_9A torque 6-DOF, {} points",
        trajectory.len()
    );

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        step_with_external(&mut sim, record.time, dt, |t, _quat| {
            // Torque [10,0,0] N·m in body frame during [1000, 2000)s
            let torque = if in_torque_window(t, dt) {
                DVec3::new(10.0, 0.0, 0.0)
            } else {
                DVec3::ZERO
            };
            (DVec3::ZERO, torque)
        });

        let body = sim.body(0);
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position.raw_si()),
            velocity: Some(body.trans.velocity.raw_si()),
            acceleration: Some(body.trans_accel),
            quaternion: Some(
                body.rot
                    .as_ref()
                    .unwrap()
                    .q_inertial_body
                    .as_witness()
                    .inner()
                    .to_glam(),
            ),
            ang_vel: Some(body.rot.as_ref().unwrap().ang_vel_body.raw_si()),
            ang_accel: body.rot_accel,
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(dyncomp_to_state_log_6dof)
        .collect();

    let report = CrossvalReport::compute("tier3_simulation_run9a_torque", &our_states, &ref_states);
    report.write();

    println!(
        "  Max position error:  {:.6e} m",
        report.max_position_component()
    );
    println!(
        "  Max velocity error:  {:.6e} m/s",
        report.max_velocity_component()
    );
    println!(
        "  Max quaternion error: {:.6e} rad",
        report.max_quat_angle()
    );
    println!(
        "  Max omega error:     {:.6e} rad/s",
        report.max_ang_vel_component()
    );

    report.assert_position([1.370e-6, 2.154e-6, 1.826e-6]);
    report.assert_velocity([1.446e-9, 2.389e-9, 1.814e-9]);
    report.assert_quat_angle(4.426e-8);
    report.assert_ang_vel([3.558e-20, 4.447e-21, 7.116e-21]);
}

// ── RUN_9C: External force + torque, zero inertial rate ──

#[test]
fn tier3_simulation_run9c_force_torque() {
    let (mut sim, trajectory, dt) = setup_run9("dyncomp_run9c_state.csv", DVec3::ZERO);

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        step_with_external(&mut sim, record.time, dt, |t, quat| {
            if in_torque_window(t, dt) {
                // Force [10,0,0] N in structural frame → rotate to inertial.
                // t_struct_body = IDENTITY, so structural = body frame.
                let t_inertial_body = quat.left_quat_to_transformation();
                let force_inertial = t_inertial_body.transpose() * DVec3::new(10.0, 0.0, 0.0);
                let torque = DVec3::new(10.0, 0.0, 0.0);
                (force_inertial, torque)
            } else {
                (DVec3::ZERO, DVec3::ZERO)
            }
        });

        let body = sim.body(0);
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position.raw_si()),
            velocity: Some(body.trans.velocity.raw_si()),
            acceleration: Some(body.trans_accel),
            quaternion: Some(
                body.rot
                    .as_ref()
                    .unwrap()
                    .q_inertial_body
                    .as_witness()
                    .inner()
                    .to_glam(),
            ),
            ang_vel: Some(body.rot.as_ref().unwrap().ang_vel_body.raw_si()),
            ang_accel: body.rot_accel,
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(dyncomp_to_state_log_6dof)
        .collect();

    let report = CrossvalReport::compute(
        "tier3_simulation_run9c_force_torque",
        &our_states,
        &ref_states,
    );
    report.write();

    println!(
        "RUN_9C: max pos={:.4} m  vel={:.6} m/s  quat={:.6e} rad  omega={:.6e} rad/s",
        report.max_position_component(),
        report.max_velocity_component(),
        report.max_quat_angle(),
        report.max_ang_vel_component(),
    );

    report.assert_position([7.679e-5, 1.2e-4, 8.628e-5]);
    report.assert_velocity([8.526e-8, 1.269e-7, 1.062e-7]);
    report.assert_quat_angle(4.426e-8);
    report.assert_ang_vel([3.558e-20, 4.447e-21, 7.116e-21]);
}

// ── RUN_9D: External force + torque, with orbit rate ──

#[test]
fn tier3_simulation_run9d_force_torque_rate() {
    // RUN_9D has initial angular velocity from the JEOD reference.
    // Pre-load to extract initial ang_vel before setup_run9 consumes it.
    let pre_csv = test_data_path("dyncomp_run9d_state.csv");
    assert!(
        pre_csv.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        pre_csv.display()
    );
    let pre_traj = load_dyncomp_csv(&pre_csv);
    let init_ang_vel = pre_traj[0].composite_body.ang_vel;

    let (mut sim, trajectory, dt) = setup_run9("dyncomp_run9d_state.csv", init_ang_vel);

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    for record in &trajectory[1..] {
        step_with_external(&mut sim, record.time, dt, |t, quat| {
            if in_torque_window(t, dt) {
                let t_inertial_body = quat.left_quat_to_transformation();
                let force_inertial = t_inertial_body.transpose() * DVec3::new(10.0, 0.0, 0.0);
                let torque = DVec3::new(10.0, 0.0, 0.0);
                (force_inertial, torque)
            } else {
                (DVec3::ZERO, DVec3::ZERO)
            }
        });

        let body = sim.body(0);
        our_states.push(StateLog {
            time: record.time,
            position: Some(body.trans.position.raw_si()),
            velocity: Some(body.trans.velocity.raw_si()),
            acceleration: Some(body.trans_accel),
            quaternion: Some(
                body.rot
                    .as_ref()
                    .unwrap()
                    .q_inertial_body
                    .as_witness()
                    .inner()
                    .to_glam(),
            ),
            ang_vel: Some(body.rot.as_ref().unwrap().ang_vel_body.raw_si()),
            ang_accel: body.rot_accel,
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(dyncomp_to_state_log_6dof)
        .collect();

    let report = CrossvalReport::compute(
        "tier3_simulation_run9d_force_torque_rate",
        &our_states,
        &ref_states,
    );
    report.write();

    println!(
        "RUN_9D: max pos={:.4} m  vel={:.6} m/s  quat={:.6e} rad  omega={:.6e} rad/s",
        report.max_position_component(),
        report.max_velocity_component(),
        report.max_quat_angle(),
        report.max_ang_vel_component(),
    );

    report.assert_position([5.278e-3, 8.255e-3, 6.635e-3]);
    report.assert_velocity([5.911e-6, 9.056e-6, 7.276e-6]);
    report.assert_quat_angle(4.426e-8);
    report.assert_ang_vel([1.651e-18, 1.367e-18, 6.262e-19]);
}
