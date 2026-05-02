//! Tier 3: JEOD `SIM_contact` — spring-damper contact between two bodies.
//!
//! Propagates two free-floating vehicles through `Simulation::step()` with
//! contact pairs registered via `Simulation::register_contact_pair`. Contact
//! forces are evaluated at each RK4 stage inside the coupled integration
//! loop, matching JEOD's `check_contact()` derivative-class scheduling.
//! Compares positions against JEOD's reference CSVs from
//! `verif/SIM_contact/SET_test/RUN_*`.
//!
//! Matches JEOD configuration from `SIM_contact/SET_test/RUN_*/input.py`:
//! - RK4 integrator, dt = 0.01 s, sim time = 10 s
//! - Empty-space (no gravity, no atmosphere)
//! - Spring material: k = 20 lbf/in = 3502.5006 N/m, c = 0.4 lbf·s/in
//!   = 70.050012 N·s/m, mu = 0.05
//! - Point facet: sphere radius 1 m at origin in structural frame
//! - Line facet: cylinder (capsule) length 2 m along body x-axis, radius 1 m
//! - veh1 at (0,0,0), at rest; veh2 at (12,0,0) with v=(-2,0,0)
//! - Point scenario: 100 kg sphere each, inertia = diag(40)
//! - Line scenario: 200 kg cylinder each, inertia = diag(100, 116.67, 116.67)
//!
//! Issue #88 / #205 lands the ground-contact pipeline (Terrain trait,
//! GroundFacet, evaluator, runner registration, RK4 wiring) and exercises
//! it via `tier3_contact_ground_smoke`. The full Tier 3 trajectory
//! cross-validation (`tier3_contact_ground`) is deferred — see that
//! function's docstring for the JEOD frame-convention gap that needs
//! follow-up.
//!
//! Tests **must panic** (not skip) when reference CSVs are absent, per
//! `CLAUDE.md`. The panic message includes the exact Docker command.

use jeod_test_data::tier3_csv::test_data_path;

use glam::{DMat3, DVec3};
use jeod_interactions::{ContactFacet, ContactMaterial};
use jeod_runner::{GroundFacet, RotationModel, Simulation, SphericalTerrain};
use jeod_sim::{
    evaluate_contact_pair, GravityControl, GravityControls, GravityModel, GravitySource, JeodQuat,
    MassProperties, RotationalState, SimulationTime, TranslationalState,
};
use jeod_sim::{GravitySourceEntry, VehicleConfig};
use std::path::Path;
use std::sync::Arc;

// ── Shared JEOD material constants ──────────────────────────────────

/// Spring stiffness: 20 lbf/in = 20 × 4.4482216152605 / 0.0254 N/m = 3502.5 N/m
const JEOD_SPRING_K: f64 = 3502.500484488583;
/// Damping: 0.4 lbf·s/in = 0.4 × 4.4482216152605 / 0.0254 N·s/m
const JEOD_DAMPING_B: f64 = 70.05000968977167;
/// Friction coefficient
const JEOD_MU: f64 = 0.05;

/// Integration step from SIM_contact S_define: `DYNAMICS 0.01` (50 Hz).
const DT: f64 = 0.01;

/// Log cycle (for matching checkpoints): 0.05 s (from input.py LOG_CYCLE).
#[allow(dead_code)] // Retained for documentation; checkpoints come from CSV rows.
const LOG_CYCLE: f64 = 0.05;

/// Simulation duration (from input.py `exec_set_terminate_time(10)`).
const SIM_DURATION: f64 = 10.0;

// ── CSV loading ─────────────────────────────────────────────────────

/// One row of a SIM_contact ASCII log.
///
/// Force and torque columns are retained in the CSV loader for future
/// pipeline-integrated regression tests, even though the current
/// trajectory-only assertions don't consume them. Marked `allow(dead_code)`
/// so the unused-field lint stays clean.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ContactRecord {
    time: f64,
    veh1_pos: DVec3,
    veh1_vel: DVec3,
    veh1_force: DVec3,
    veh1_torque: DVec3,
    veh2_pos: DVec3,
    veh2_vel: DVec3,
    veh2_force: DVec3,
    veh2_torque: DVec3,
    veh1_mass: f64,
    veh2_mass: f64,
}

fn load_contact_csv(path: &Path) -> Vec<ContactRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_contact CSV from {}: {e}\n\
             Generate with:\n  \
             docker build -f trick/Dockerfile -t jeod-trick ..\n  \
             docker run --rm -v $(pwd)/test_data:/output \\\n    \
               -v $(pwd)/trick/generate_references.sh:/generate_references.sh:ro \\\n    \
               jeod-trick",
            path.display()
        )
    });
    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(
            f.len() >= 27,
            "line {}: expected >=27 columns for SIM_contact CSV, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(ContactRecord {
            time: p(0),
            veh1_pos: DVec3::new(p(1), p(2), p(3)),
            veh1_vel: DVec3::new(p(4), p(5), p(6)),
            veh1_force: DVec3::new(p(7), p(8), p(9)),
            veh1_torque: DVec3::new(p(10), p(11), p(12)),
            veh2_pos: DVec3::new(p(13), p(14), p(15)),
            veh2_vel: DVec3::new(p(16), p(17), p(18)),
            veh2_force: DVec3::new(p(19), p(20), p(21)),
            veh2_torque: DVec3::new(p(22), p(23), p(24)),
            veh1_mass: p(25),
            veh2_mass: p(26),
        });
    }
    records
}

// ── Simulation harness ──────────────────────────────────────────────

/// Add an inertial-only gravity "source" with mu=0 so the Simulation has a
/// root frame. Matches JEOD's `EphemerisMode_EmptySpace` which provides the
/// Space.inertial root frame with no gravitational body.
fn add_empty_space_root(sim: &mut Simulation) {
    sim.add_source(
        "Space",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );
}

/// Build a simulation with two free-floating bodies of the given mass /
/// inertia, and the standard SIM_contact initial state:
///   veh1 at rest at origin, veh2 at (12,0,0) moving at (-2,0,0) m/s.
fn make_two_body_sim(mass: f64, inertia_diag: DVec3) -> Simulation {
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    add_empty_space_root(&mut sim);

    let inertia = DMat3::from_cols(
        DVec3::new(inertia_diag.x, 0.0, 0.0),
        DVec3::new(0.0, inertia_diag.y, 0.0),
        DVec3::new(0.0, 0.0, inertia_diag.z),
    );
    let mass_props = MassProperties::with_inertia(mass, inertia, DVec3::ZERO);

    let id1 = sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });

    let id2 = sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::new(12.0, 0.0, 0.0),
            velocity: DVec3::new(-2.0, 0.0, 0.0),
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });
    assert_eq!(id1, 0);
    assert_eq!(id2, 1);

    sim.validate().unwrap();
    sim
}

/// JEOD steel material from `Contact_Modified_data/contact/pair_interaction.py`.
fn jeod_steel() -> ContactMaterial {
    ContactMaterial::jeod_spring(JEOD_SPRING_K, JEOD_DAMPING_B, JEOD_MU)
}

/// Mass/inertia for the 100 kg point scenarios (`veh_mass_point.py`).
fn point_mass_props() -> MassProperties {
    MassProperties::with_inertia(
        100.0,
        DMat3::from_cols(
            DVec3::new(40.0, 0.0, 0.0),
            DVec3::new(0.0, 40.0, 0.0),
            DVec3::new(0.0, 0.0, 40.0),
        ),
        DVec3::ZERO,
    )
}

/// Mass/inertia for the 200 kg line/capsule scenarios (`veh_mass_line.py`).
fn line_mass_props() -> MassProperties {
    MassProperties::with_inertia(
        200.0,
        DMat3::from_cols(
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 116.6667, 0.0),
            DVec3::new(0.0, 0.0, 116.6667),
        ),
        DVec3::ZERO,
    )
}

// Tolerances for contact-force/torque comparisons. The evaluator is
// re-run at logged (end-of-step) states — not at the RK4 stages JEOD
// used during integration — so a small difference is expected. Values
// are set at 5% above the observed maximum per CLAUDE.md policy.
//
// Head-on scenarios agree with JEOD essentially exactly (< 30 mN, < 1e-12 N*m);
// the off-center oblique case has known state drift (~cm-scale) after
// contact, so the tolerance for that scenario is larger (~5 N / ~5 N*m).
const CONTACT_FORCE_TOL: f64 = 0.028; // N (head-on cases; observed max 26 mN)
const CONTACT_TORQUE_TOL: f64 = 2.0e-13; // N*m (head-on; observed ~1.2e-13 machine noise)
const POINT_OFF_CENTER_FORCE_TOL: f64 = 5.4; // N (observed 5.07 N due to ~5% oblique drift)
const POINT_OFF_CENTER_TORQUE_TOL: f64 = 5.2; // N*m (observed 4.82 N*m due to ~5% oblique drift)

/// Body state snapshot at a single checkpoint. Carries the full 6-DOF
/// state (position, velocity, attitude, angular velocity) for each of
/// the two bodies so tests can both compare trajectories AND re-evaluate
/// contact force/torque at the logged state.
#[derive(Debug, Clone, Copy)]
struct CheckpointBodies {
    veh1_trans: TranslationalState,
    veh1_rot: RotationalState,
    veh2_trans: TranslationalState,
    veh2_rot: RotationalState,
}

/// Propagate two bodies with contact forces evaluated at each RK4 stage
/// inside `Simulation::step()`. Returns per-step full-6-DOF snapshots at
/// `LOG_CYCLE` intervals.
///
/// `facet_a` and `facet_b` define the two contact facets — the shape
/// positions are relative to each body's structural frame origin, which in
/// SIM_contact coincides with the body's CoM and inertial position.
fn propagate_with_contact(
    sim: &mut Simulation,
    facet_a: ContactFacet,
    facet_b: ContactFacet,
    checkpoints: &[f64],
) -> Vec<CheckpointBodies> {
    // Register the contact pair so forces are computed at every RK4 stage
    // (matching JEOD's check_contact derivative-class job).
    sim.register_contact_pair(0, facet_a, 1, facet_b);

    let mut out = Vec::with_capacity(checkpoints.len());
    let mut cp_iter = checkpoints.iter().copied().peekable();

    let steps_total = (SIM_DURATION / DT).round() as usize;
    for step in 0..=steps_total {
        let b_a = sim.body(0);
        let b_b = sim.body(1);

        // Record output at checkpoints (±0.5·dt tolerance on time)
        let t = step as f64 * DT;
        if let Some(&cp) = cp_iter.peek() {
            if (t - cp).abs() <= 0.5 * DT {
                out.push(CheckpointBodies {
                    veh1_trans: b_a.trans,
                    veh1_rot: b_a.rot.expect("6-DOF required for SIM_contact"),
                    veh2_trans: b_b.trans,
                    veh2_rot: b_b.rot.expect("6-DOF required for SIM_contact"),
                });
                cp_iter.next();
            }
        }

        if step == steps_total {
            break;
        }

        sim.step_n(1).expect("step_n failed");
    }
    out
}

/// Per-checkpoint assertion on contact force and torque against the JEOD
/// CSV. JEOD logs `contact_surface.contact_force` in each body's
/// *structural* frame; our [`evaluate_contact_pair`] returns
/// `force_on_a` in the inertial frame and `torque_*_body` in each body's
/// *body* frame. For all SIM_contact scenarios, `t_struct_body = I`
/// (structure frame coincides with body frame), so we only need the
/// body attitude to transform the inertial force back to the structural
/// frame, and the logged torque can be compared to `torque_*_body`
/// directly.
///
/// The contact evaluator is re-run at the logged (end-of-step) states,
/// not the stage states JEOD used during integration, so a small
/// mismatch is expected; tolerances are per-test and set 5 % above
/// observed maxima per CLAUDE.md's cross-validation policy.
#[allow(clippy::too_many_arguments)]
fn assert_contact_force_torque(
    label: &str,
    facet_a: ContactFacet,
    facet_b: ContactFacet,
    mass_a: &MassProperties,
    mass_b: &MassProperties,
    ours: &[CheckpointBodies],
    records: &[ContactRecord],
    force_tol: f64,
    torque_tol: f64,
) {
    assert_eq!(ours.len(), records.len());
    let mut max_force_err_1 = 0.0_f64;
    let mut max_force_err_2 = 0.0_f64;
    let mut max_torque_err_1 = 0.0_f64;
    let mut max_torque_err_2 = 0.0_f64;
    let mut any_contact = false;
    for (b, rec) in ours.iter().zip(records.iter()) {
        let eval = evaluate_contact_pair(
            &facet_a,
            &facet_b,
            &b.veh1_trans,
            &b.veh2_trans,
            Some(&b.veh1_rot),
            Some(&b.veh2_rot),
            DMat3::IDENTITY, // t_struct_body_a (SIM_contact: struct == body)
            DMat3::IDENTITY, // t_struct_body_b
            Some(mass_a),
            Some(mass_b),
        );
        let (force_on_a_struct, torque_a_struct, force_on_b_struct, torque_b_struct) = match eval {
            Some(ev) => {
                any_contact = true;
                // t_struct_body = I ⇒ t_inertial_struct = t_inertial_body.
                let t_inertial_body_1 = b.veh1_rot.quaternion.left_quat_to_transformation();
                let t_inertial_body_2 = b.veh2_rot.quaternion.left_quat_to_transformation();
                let force_on_a_struct = t_inertial_body_1 * ev.force_on_a;
                let force_on_b_struct = t_inertial_body_2 * (-ev.force_on_a);
                (
                    force_on_a_struct,
                    ev.torque_a_body,
                    force_on_b_struct,
                    ev.torque_b_body,
                )
            }
            None => (DVec3::ZERO, DVec3::ZERO, DVec3::ZERO, DVec3::ZERO),
        };
        max_force_err_1 = max_force_err_1.max((force_on_a_struct - rec.veh1_force).length());
        max_force_err_2 = max_force_err_2.max((force_on_b_struct - rec.veh2_force).length());
        max_torque_err_1 = max_torque_err_1.max((torque_a_struct - rec.veh1_torque).length());
        max_torque_err_2 = max_torque_err_2.max((torque_b_struct - rec.veh2_torque).length());
    }

    println!(
        "{label}: contact force err max = ({max_force_err_1:.3e}, {max_force_err_2:.3e}) N; \
         torque err max = ({max_torque_err_1:.3e}, {max_torque_err_2:.3e}) N*m"
    );

    assert!(
        any_contact,
        "{label}: evaluate_contact_pair never reported contact — scenario never in contact?"
    );
    assert!(
        max_force_err_1 < force_tol,
        "{label}: veh1 force error {max_force_err_1:.3e} > tol {force_tol:.3e}"
    );
    assert!(
        max_force_err_2 < force_tol,
        "{label}: veh2 force error {max_force_err_2:.3e} > tol {force_tol:.3e}"
    );
    assert!(
        max_torque_err_1 < torque_tol,
        "{label}: veh1 torque error {max_torque_err_1:.3e} > tol {torque_tol:.3e}"
    );
    assert!(
        max_torque_err_2 < torque_tol,
        "{label}: veh2 torque error {max_torque_err_2:.3e} > tol {torque_tol:.3e}"
    );
}

// ── Tier 3 tests ────────────────────────────────────────────────────

// non-recipe: all 6 contact tests run SIM_contact with 1 m / 100 kg test
// spheres, lines, and ground geometries with bespoke contact pairs and
// initial velocities. The geometries themselves are the test content; no
// recipe vehicle preset matches.
/// RUN_point: two 1 m radius spheres, 100 kg each. veh2 at (12,0,0) with
/// v=(-2,0,0). Contact starts when the centers are 2 m apart (t ≈ 5 s).
#[test]
fn tier3_contact_point_pair() {
    let csv_path = test_data_path("contact_point_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(
        records.len() > 50,
        "Expected >50 log rows for 10 s at 20 Hz, got {}",
        records.len()
    );

    // Point facet: sphere radius 1 m centered at body origin.
    let facet = ContactFacet::point(DVec3::ZERO, 1.0, jeod_steel());
    let mass = point_mass_props();

    let mut sim = make_two_body_sim(100.0, DVec3::new(40.0, 40.0, 40.0));
    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(&mut sim, facet, facet, &checkpoints);
    assert_eq!(ours.len(), records.len());

    let mut max_pos_err_1 = 0.0_f64;
    let mut max_pos_err_2 = 0.0_f64;
    let mut max_vel_err_1 = 0.0_f64;
    let mut max_vel_err_2 = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err_1 = max_pos_err_1.max((our.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err_2 = max_pos_err_2.max((our.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err_1 = max_vel_err_1.max((our.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err_2 = max_vel_err_2.max((our.veh2_trans.velocity - rec.veh2_vel).length());
    }

    println!("SIM_contact RUN_point:");
    println!("  veh1 max pos error: {max_pos_err_1:.3e} m");
    println!("  veh2 max pos error: {max_pos_err_2:.3e} m");
    println!("  veh1 max vel error: {max_vel_err_1:.3e} m/s");
    println!("  veh2 max vel error: {max_vel_err_2:.3e} m/s");

    // Head-on sphere-sphere symmetric contact: pipeline-coupled RK4 matches
    // JEOD to ~14 μm over 10 s (observed max). Tolerances set at 5% above
    // observed max per CLAUDE.md cross-validation policy.
    assert!(
        max_pos_err_1 < 1.5e-5,
        "veh1 position error {max_pos_err_1:.3e} > 15 μm"
    );
    assert!(
        max_pos_err_2 < 1.5e-5,
        "veh2 position error {max_pos_err_2:.3e} > 15 μm"
    );
    assert!(
        max_vel_err_1 < 8.0e-6,
        "veh1 velocity error {max_vel_err_1:.3e} > 8 μm/s"
    );
    assert!(
        max_vel_err_2 < 8.0e-6,
        "veh2 velocity error {max_vel_err_2:.3e} > 8 μm/s"
    );

    assert_contact_force_torque(
        "SIM_contact RUN_point",
        facet,
        facet,
        &mass,
        &mass,
        &ours,
        &records,
        CONTACT_FORCE_TOL,
        CONTACT_TORQUE_TOL,
    );
}

/// RUN_line: two capsules (length 2 m, radius 1 m) aligned along x.
/// 200 kg each, approaching head-on. Identical trajectory to RUN_point
/// because the lines are collinear with the approach direction (the end
/// caps act as spheres).
#[test]
fn tier3_contact_line_pair() {
    let csv_path = test_data_path("contact_line_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(records.len() > 50);

    // Line facet: capsule along body x-axis, length 2 m, radius 1 m.
    let facet = ContactFacet::line(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        1.0,
        jeod_steel(),
    );

    let mass = line_mass_props();
    let mut sim = make_two_body_sim(200.0, DVec3::new(100.0, 116.6667, 116.6667));
    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(&mut sim, facet, facet, &checkpoints);
    assert_eq!(ours.len(), records.len());

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err = max_pos_err.max((our.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err = max_pos_err.max((our.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err = max_vel_err.max((our.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err = max_vel_err.max((our.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!("SIM_contact RUN_line: max pos={max_pos_err:.3e} m, max vel={max_vel_err:.3e} m/s");

    // Head-on capsule-capsule (line-line) head-on contact: observed max
    // ~11 μm. Tolerances set at 5% above observed per CLAUDE.md policy.
    assert!(max_pos_err < 1.2e-5, "position error {max_pos_err:.3e}");
    assert!(max_vel_err < 9.0e-6, "velocity error {max_vel_err:.3e}");

    assert_contact_force_torque(
        "SIM_contact RUN_line",
        facet,
        facet,
        &mass,
        &mass,
        &ours,
        &records,
        CONTACT_FORCE_TOL,
        CONTACT_TORQUE_TOL,
    );
}

/// RUN_line_point: capsule (veh1) meets sphere (veh2) head-on. Same mass /
/// inertia as RUN_line for both vehicles since the default mass file is
/// `veh_mass_line.py` (cylinder mass properties).
#[test]
fn tier3_contact_line_point() {
    let csv_path = test_data_path("contact_line_point_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(records.len() > 50);

    let line_facet = ContactFacet::line(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        1.0,
        jeod_steel(),
    );
    let point_facet = ContactFacet::point(DVec3::ZERO, 1.0, jeod_steel());
    let mass = line_mass_props();

    let mut sim = make_two_body_sim(200.0, DVec3::new(100.0, 116.6667, 116.6667));
    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(&mut sim, line_facet, point_facet, &checkpoints);
    assert_eq!(ours.len(), records.len());

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err = max_pos_err.max((our.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err = max_pos_err.max((our.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err = max_vel_err.max((our.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err = max_vel_err.max((our.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!(
        "SIM_contact RUN_line_point: max pos={max_pos_err:.3e} m, max vel={max_vel_err:.3e} m/s"
    );

    // Head-on line-point (capsule end-cap vs sphere) contact: observed
    // max ~10 μm. Tolerances at 5% above observed per CLAUDE.md policy.
    assert!(max_pos_err < 1.0e-5, "pos err {max_pos_err:.3e}");
    assert!(max_vel_err < 9.0e-6, "vel err {max_vel_err:.3e}");

    assert_contact_force_torque(
        "SIM_contact RUN_line_point",
        line_facet,
        point_facet,
        &mass,
        &mass,
        &ours,
        &records,
        CONTACT_FORCE_TOL,
        CONTACT_TORQUE_TOL,
    );
}

/// RUN_line_side_to_side: two capsules rotated 90° relative to each other
/// so their cylindrical sides meet. JEOD's `input.py` sets:
///   veh1 euler Yaw_Pitch_Roll = (0, 90°, 0) — pitched up 90°
///   veh2 euler Yaw_Pitch_Roll = (0, 0, 90°) — rolled right 90°
/// Facets are along body-x in structural coords (same as RUN_line).
#[test]
fn tier3_contact_line_side_to_side() {
    let csv_path = test_data_path("contact_line_side_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(records.len() > 50);

    // Both facets are along body-x in structural coords. Body attitude
    // rotates the shape into the world frame at each integration stage.
    let facet = ContactFacet::line(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        1.0,
        jeod_steel(),
    );

    // JEOD Yaw_Pitch_Roll convention: apply yaw (about z), then pitch
    // (about new y), then roll (about new x). veh1 pitch=90° ⇒ +90° about y,
    // veh2 roll=90° ⇒ +90° about x.
    //
    // Must use `left_quat_from_eigen_rotation` rather than glam's
    // `DQuat::from_axis_angle` + field copy: JEOD's left-quat convention
    // stores the vector part as `-sin(θ/2)·axis` (note the minus sign),
    // which `left_quat_from_eigen_rotation` applies. Using glam's
    // positive-sine quaternion directly would store the opposite-sign
    // attitude — invisible to pos/vel (line geometry is symmetric under
    // the rotation flip) but it inverts forces in the structural frame.
    let jeod_veh1 = JeodQuat::left_quat_from_eigen_rotation(90.0_f64.to_radians(), DVec3::Y);
    let jeod_veh2 = JeodQuat::left_quat_from_eigen_rotation(90.0_f64.to_radians(), DVec3::X);

    // Build sim with non-identity initial rotations.
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    add_empty_space_root(&mut sim);

    let mass_props = MassProperties::with_inertia(
        200.0,
        DMat3::from_cols(
            DVec3::new(100.0, 0.0, 0.0),
            DVec3::new(0.0, 116.6667, 0.0),
            DVec3::new(0.0, 0.0, 116.6667),
        ),
        DVec3::ZERO,
    );

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        },
        rot: Some(RotationalState {
            quaternion: jeod_veh1,
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });
    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::new(12.0, 0.0, 0.0),
            velocity: DVec3::new(-2.0, 0.0, 0.0),
        },
        rot: Some(RotationalState {
            quaternion: jeod_veh2,
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });
    sim.validate().unwrap();

    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(&mut sim, facet, facet, &checkpoints);
    assert_eq!(ours.len(), records.len());

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err = max_pos_err.max((our.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err = max_pos_err.max((our.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err = max_vel_err.max((our.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err = max_vel_err.max((our.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!(
        "SIM_contact RUN_line_side: max pos={max_pos_err:.3e} m, max vel={max_vel_err:.3e} m/s"
    );

    // Perpendicular-capsule side-to-side contact — the capsules meet at
    // their midpoints so the contact is effectively sphere-sphere along
    // the inter-body x-axis. Rotated geometry is exercised here but the
    // collision remains symmetric. Observed max ~10 μm. Tolerances at 5%
    // above observed per CLAUDE.md policy.
    assert!(max_pos_err < 1.0e-5, "pos err {max_pos_err:.3e}");
    assert!(max_vel_err < 9.0e-6, "vel err {max_vel_err:.3e}");

    assert_contact_force_torque(
        "SIM_contact RUN_line_side",
        facet,
        facet,
        &mass_props,
        &mass_props,
        &ours,
        &records,
        CONTACT_FORCE_TOL,
        CONTACT_TORQUE_TOL,
    );
}

/// RUN_point_off_center: same spheres as RUN_point but veh2 starts with a
/// transverse offset so the collision is oblique. Uses identical mass
/// properties (100 kg sphere, 40 kg·m² inertia).
///
/// The exact offset is read from the CSV's t=0 row rather than hardcoded;
/// that row is JEOD source data (initial conditions), not mid-sim output.
#[test]
fn tier3_contact_point_off_center() {
    let csv_path = test_data_path("contact_point_off_center_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(records.len() > 50);

    let init = &records[0];

    let facet = ContactFacet::point(DVec3::ZERO, 1.0, jeod_steel());

    // Reconstruct the sim from the t=0 row (initial conditions from CSV are
    // allowed per CLAUDE.md; only mid-sim CSV data is forbidden as input).
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    add_empty_space_root(&mut sim);

    let mass_props = MassProperties::with_inertia(
        100.0,
        DMat3::from_cols(
            DVec3::new(40.0, 0.0, 0.0),
            DVec3::new(0.0, 40.0, 0.0),
            DVec3::new(0.0, 0.0, 40.0),
        ),
        DVec3::ZERO,
    );

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.veh1_pos,
            velocity: init.veh1_vel,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });
    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.veh2_pos,
            velocity: init.veh2_vel,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });
    sim.validate().unwrap();

    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let ours = propagate_with_contact(&mut sim, facet, facet, &checkpoints);

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for (our, rec) in ours.iter().zip(records.iter()) {
        max_pos_err = max_pos_err.max((our.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err = max_pos_err.max((our.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err = max_vel_err.max((our.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err = max_vel_err.max((our.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!(
        "SIM_contact RUN_point_off_center: max pos={max_pos_err:.3e} m, max vel={max_vel_err:.3e} m/s"
    );

    // Oblique collision: unlike the head-on tests (which agree with JEOD to
    // 14 μm), RUN_point_off_center drifts from JEOD by ~2.7 cm in position
    // and ~5.7 mm/s in velocity after the stiff 0.5 s contact event, which
    // then accumulates over the 4.5 s free-flight tail. `evaluate_contact_pair`
    // is a faithful port of JEOD's `point_contact_pair.cc:83-84` rel_vel
    // formula (single-cross (ω_B − ω_A) × r_A_contact), and both bodies
    // develop equal, same-direction ω from Newton's-third-law torques on
    // equal-mass, equal-inertia spheres ⇒ ω_rel = 0, so the rel_vel term is
    // identically zero — the formula is not the source of the drift.
    //
    // The remaining gap likely stems from JEOD's Trick integrator sub-
    // stepping the stiff contact event differently from our fixed
    // four-stage RK4. See issue #117 for the investigation plan.
    assert!(
        max_pos_err < 3.0e-2,
        "veh{{1,2}} position error {max_pos_err:.3e} > 3 cm"
    );
    assert!(
        max_vel_err < 6.0e-3,
        "veh{{1,2}} velocity error {max_vel_err:.3e} > 6 mm/s"
    );

    // Oblique friction differs from JEOD (~5 % in tangential momentum), so
    // state drift during and after the contact event means our
    // `evaluate_contact_pair` at a logged state is *not* at the same body
    // configuration as JEOD's. Use looser tolerances that reflect the
    // documented drift — the assertion still catches gross regressions
    // (bad frame transforms, missing terms) without getting confused by
    // the well-characterized tangential discrepancy.
    assert_contact_force_torque(
        "SIM_contact RUN_point_off_center",
        facet,
        facet,
        &mass_props,
        &mass_props,
        &ours,
        &records,
        POINT_OFF_CENTER_FORCE_TOL,
        POINT_OFF_CENTER_TORQUE_TOL,
    );
}

/// RUN_contact_ground: SIM_ground_contact.
///
/// Two vehicles (veh1 = line cylinder, veh2 = point sphere, 200 kg each)
/// initialized at Earth's surface in `Earth.inertial`. Spherical Earth
/// gravity pulls them toward the planet center; a ground-contact spring
/// (k = 1751.25 N/m, c = 35.025 N·s/m, μ = 0.5) pushes back. The vehicles
/// start interpenetrating the ground at t=0, producing an impulsive
/// ~2.2 × 10¹⁰ N force that launches them outward at ~93 km/s within
/// 50 ms; the rest of the 10-second run is ballistic coast under
/// spherical gravity.
///
/// JEOD source: `verif/SIM_ground_contact/SET_test/RUN_contact_ground/input.py`
/// + `Modified_data/{ground/{ground_facet,pair_interaction},vehicle/sv_earth}.py`.
fn make_ground_contact_sim() -> Simulation {
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    // Earth as a spherical (point-mass) central body. JEOD's
    // SIM_ground_contact configures the gravity controls with degree=0,
    // order=0, spherical=true so the SH model collapses to point mass.
    // We use the same effective physics via a PointMass GravitySource.
    let earth_mu = jeod_sim::EARTH.shape.mu;
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: earth_mu,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            // SphericalTerrain does not consult pfix rotation, so we omit
            // the rotation model here — keeps the test self-contained.
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    let mass_props = line_mass_props(); // 200 kg, diag(100, 116.667, 116.667)
    let earth_radius = jeod_sim::EARTH.shape.r_eq;

    let earth_grav = GravityControls {
        controls: vec![GravityControl::new_spherical(earth_idx, false)],
    };

    // veh1 — line cylinder along structural x-axis.
    let veh1 = sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::new(earth_radius, 0.0, 0.0),
            velocity: DVec3::ZERO,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: earth_grav.clone(),
        compute_gravity_gradient: false,
        ..Default::default()
    });

    // veh2 — point sphere 10 m radially outward from veh1.
    let veh2 = sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::new(earth_radius + 10.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        },
        rot: Some(RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        }),
        mass: Some(mass_props),
        gravity_controls: earth_grav,
        compute_gravity_gradient: false,
        ..Default::default()
    });
    assert_eq!(veh1, 0);
    assert_eq!(veh2, 1);

    sim.validate().unwrap();
    sim
}

/// Ground-contact material: JEOD `Modified_data/ground/pair_interaction.py`
/// — spring_k=10 lbf/in (≈1751.25 N/m), damping_b=0.2 lbf·s/in
/// (≈35.025 N·s/m), mu=0.5. Note that JEOD looks up "steel↔dirt" via
/// `(params_1, params_2)` name pair on a single `SpringPairInteraction`;
/// we collapse to a single `ContactMaterial` because the values are
/// pair-unique.
const GROUND_SPRING_K: f64 = 1751.250242244291; // 10 lbf/in
const GROUND_DAMPING_B: f64 = 35.025004844885835; // 0.2 lbf*s/in
const GROUND_MU: f64 = 0.5;

fn ground_steel() -> ContactMaterial {
    ContactMaterial::jeod_spring(GROUND_SPRING_K, GROUND_DAMPING_B, GROUND_MU)
}

/// Smoke test: verify that the ground-contact pipeline (Terrain →
/// GroundFacet → register_ground_contact_pair → coupled-RK4 evaluation)
/// is wired through `Simulation::step()` end-to-end. Issue #88 / #205
/// landed the GroundFacet port and pipeline integration; bit-for-bit
/// trajectory cross-validation against `SIM_ground_contact/RUN_contact_ground`
/// is tracked separately because reproducing JEOD's exact first-step
/// impulse depends on subtleties of JEOD's `state.trans.position` /
/// frame-tree conventions for surface-model-created `BodyRefFrame`s
/// that the documentation doesn't pin down.
///
/// What this test verifies:
///   1. The `GroundFacet` + `SphericalTerrain` constructors compose with
///      `Simulation::register_ground_contact_pair` without panic.
///   2. The simulation steps for 10 s under spherical Earth gravity +
///      ground contact without producing NaN or panicking.
///   3. The contact-coupled RK4 path activates when only ground-contact
///      pairs are present (the empty `contact_pairs` branch in
///      `step/integrate.rs` no longer short-circuits the coupled path).
#[test]
fn tier3_contact_ground_smoke() {
    let csv_path = test_data_path("contact_ground_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(!records.is_empty(), "expected at least one CSV record");

    let mut sim = make_ground_contact_sim();
    let earth_radius = jeod_sim::EARTH.shape.r_eq;
    let mat = ground_steel();
    let terrain = Arc::new(SphericalTerrain::new(earth_radius));
    let ground = GroundFacet::new(terrain, 0.0, mat);

    let veh1_facet = ContactFacet::line(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        1.0,
        mat,
    );
    let veh2_facet = ContactFacet::point(DVec3::ZERO, 1.0, mat);
    sim.register_ground_contact_pair(0, veh1_facet, ground.clone(), 0);
    sim.register_ground_contact_pair(1, veh2_facet, ground, 0);

    assert_eq!(sim.num_ground_contact_pairs(), 2);

    // Step through ~10 s. We don't assert against the JEOD CSV
    // trajectory here (see the docstring), but we DO assert the
    // simulation never produces NaN and the bodies stay on the +x axis
    // (the test geometry has y/z symmetry).
    let steps_total = (SIM_DURATION / DT).round() as usize;
    for _ in 0..steps_total {
        sim.step_n(1).expect("step_n failed");
        let b = sim.body(0);
        assert!(
            b.trans.position.is_finite(),
            "veh1 position became non-finite"
        );
        assert!(
            b.trans.velocity.is_finite(),
            "veh1 velocity became non-finite"
        );
        assert!(b.trans.position.y.abs() < 1e-6);
        assert!(b.trans.position.z.abs() < 1e-6);
    }
}

/// Tier 3 cross-validation against `SIM_ground_contact/RUN_contact_ground` CSV.
///
/// **DEFERRED**: tracked under follow-up work. The GroundFacet algebra
/// matches JEOD's source line-for-line, but a second-order frame-
/// convention detail (the relationship between
/// `BodyRefFrame::state.trans.position` for surface-model-created
/// vehicle points and the body's structure / integration-frame
/// position) governs whether `subject_mag < ground_mag` ever fires for
/// a vehicle at the planet surface, and this is sensitive to JEOD
/// runtime initialization order we can't reproduce statically.
///
/// Concretely:
/// - The CSV t=0 row shows ~2.2 × 10¹⁰ N contact force per vehicle, but
///   any direct trace of `point_ground_interaction.cc::in_contact()`
///   from the documented initial state produces either "no contact"
///   (with `state.trans.position` in the inertial frame, where
///   `subject_mag` ≈ R+1 > R = `ground_mag`) or "always contact" (with
///   `state.trans.position` in the structural frame, where
///   `subject_mag` ≈ 1 << R = `ground_mag` regardless of altitude).
/// - JEOD's CSV trajectory (v ≈ 93 km/s at t=0.05) consistent with
///   ~1/6 of a single-stage RK4 contact-spring impulse, suggesting JEOD
///   detects contact at exactly stage 1 of step 1 and then loses contact
///   at stages 2–4. Our port doesn't reproduce that stage-dependent
///   behavior because both interpretations of
///   `state.trans.position` give a position-only contact decision that
///   is stage-invariant.
///
/// The unit-test coverage in `jeod_interactions::contact::tests::*ground*`
/// validates the algebra at the documented initial state. Re-enabling
/// this test should be coupled with running JEOD itself to capture the
/// per-stage `subject_mag` / `ground_mag` values during the first step
/// of `RUN_contact_ground` so we can identify which JEOD-runtime detail
/// produces the observed trajectory.
#[test]
#[ignore = "tier3 trajectory cross-validation deferred — see docstring"]
fn tier3_contact_ground() {
    let _records = load_contact_csv(&test_data_path("contact_ground_contact_state.csv"));
    panic!(
        "tier3_contact_ground deferred. GroundFacet pipeline is wired and \
         unit-tested; trajectory cross-validation against SIM_ground_contact \
         RUN_contact_ground CSV awaits follow-up. See docstring."
    );
}
