//! Tier 3: SIM_dyncomp RUN_9A/9C/9D — External force/torque via Simulation pipeline
//!
//! Uses `Simulation::step()` with `external_force` / `external_torque` fields
//! for time-scheduled force injection. All parameters loaded from JEOD sources.

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, RotationModel, Simulation, VehicleConfig};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, JeodQuat, MassProperties,
    RotationalState, SimulationTime, TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use jeod_test_data::mass_data::MassInitData;
use jeod_test_data::tier3_csv::{load_dyncomp_csv, test_data_path};

/// Build [`MassProperties`] from parsed JEOD mass-init data (test-only helper).
///
/// Inlined here because `MassProperties` lives in `jeod_dynamics` (a real
/// dep of `jeod_test_data` would create a cycle), and the helper is used
/// only by Tier 3 mass-fixture tests.
fn mass_props_from_init(init: &MassInitData) -> MassProperties {
    let inertia = DMat3::from_cols(
        DVec3::new(init.inertia[0][0], init.inertia[1][0], init.inertia[2][0]),
        DVec3::new(init.inertia[0][1], init.inertia[1][1], init.inertia[2][1]),
        DVec3::new(init.inertia[0][2], init.inertia[1][2], init.inertia[2][2]),
    );
    MassProperties::with_inertia(init.mass, inertia, DVec3::from_slice(&init.position))
}

/// SIM_dyncomp root directory (relative to JEOD_HOME).
const SIM_DYNCOMP: &str = "verif/SIM_dyncomp";

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
    Vec<jeod_test_data::dyncomp_csv::DyncompRecord>,
    f64,
) {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let csv_path = test_data_path(csv_name);
    assert!(
        csv_path.exists(),
        "JEOD reference not found at {}.\n\
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
         -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro jeod-trick",
        csv_path.display()
    );

    let sim_dir = jeod_root.join(SIM_DYNCOMP);
    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");

    let dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));
    let mu_earth =
        jeod_sim::coefficients::load_mu_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
            .expect("load Earth mu");

    let mass_init = jeod_test_data::mass_data::load_mass_from_file(
        &sim_dir.join("Modified_data/mass.py"),
        Some("set_mass_iss"),
    );
    let mass_props = mass_props_from_init(&mass_init);

    let trajectory = load_dyncomp_csv(&csv_path);
    assert!(trajectory.len() >= 100);
    let init = &trajectory[0];

    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.composite_body.position,
            velocity: init.composite_body.velocity,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::from_glam(init.composite_body.quaternion),
            ang_vel_body: init_ang_vel,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
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
        let quat = sim.body(0).rot.as_ref().unwrap().quaternion;
        let (force, torque) = force_torque_fn(sim.elapsed(), &quat);
        sim.set_body_external_force(0, force);
        sim.set_body_external_torque(0, torque);
        sim.step();
    }
    // Fractional remainder
    let remainder = target_time - sim.elapsed();
    if remainder > 0.001 {
        let quat = sim.body(0).rot.as_ref().unwrap().quaternion;
        let (force, torque) = force_torque_fn(sim.elapsed(), &quat);
        sim.set_body_external_force(0, force);
        sim.set_body_external_torque(0, torque);
        sim.set_dt(remainder);
        sim.step();
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
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            quaternion: Some(body.rot.as_ref().unwrap().quaternion.to_glam()),
            ang_vel: Some(body.rot.as_ref().unwrap().ang_vel_body),
            ..Default::default()
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(|r| StateLog {
            time: r.time,
            position: Some(r.composite_body.position),
            velocity: Some(r.composite_body.velocity),
            acceleration: r.derivs.as_ref().map(|d| d.trans_accel),
            quaternion: Some(r.composite_body.quaternion),
            ang_vel: Some(r.composite_body.ang_vel),
            ang_accel: r.derivs.as_ref().map(|d| d.rot_accel),
        })
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
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            quaternion: Some(body.rot.as_ref().unwrap().quaternion.to_glam()),
            ang_vel: Some(body.rot.as_ref().unwrap().ang_vel_body),
            ..Default::default()
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(|r| StateLog {
            time: r.time,
            position: Some(r.composite_body.position),
            velocity: Some(r.composite_body.velocity),
            acceleration: r.derivs.as_ref().map(|d| d.trans_accel),
            quaternion: Some(r.composite_body.quaternion),
            ang_vel: Some(r.composite_body.ang_vel),
            ang_accel: r.derivs.as_ref().map(|d| d.rot_accel),
        })
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
         Generate with: docker run --rm -v $(pwd)/test_data:/output \
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
            position: Some(body.trans.position),
            velocity: Some(body.trans.velocity),
            quaternion: Some(body.rot.as_ref().unwrap().quaternion.to_glam()),
            ang_vel: Some(body.rot.as_ref().unwrap().ang_vel_body),
            ..Default::default()
        });
    }

    let ref_states: Vec<StateLog> = trajectory[1..]
        .iter()
        .map(|r| StateLog {
            time: r.time,
            position: Some(r.composite_body.position),
            velocity: Some(r.composite_body.velocity),
            acceleration: r.derivs.as_ref().map(|d| d.trans_accel),
            quaternion: Some(r.composite_body.quaternion),
            ang_vel: Some(r.composite_body.ang_vel),
            ang_accel: r.derivs.as_ref().map(|d| d.rot_accel),
        })
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
