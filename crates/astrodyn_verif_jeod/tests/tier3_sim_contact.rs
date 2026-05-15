// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: JEOD `SIM_contact` — spring-damper contact between two bodies.

#![allow(
    clippy::float_cmp,
    reason = "Tier 3 tests assert bit-exact recovery of literal-built / analytic state values"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Tier 3 step counts and indices fit exactly in f64 mantissa and usize"
)]
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
//! GroundFacet, evaluator, runner registration, RK4 wiring) and the
//! `tier3_contact_ground` cross-validation against
//! `SIM_ground_contact/RUN_contact_ground` CSV — see that test's
//! docstring for the JEOD initialization-state semantics our port
//! mirrors via `Phase::Initialization` / `Phase::SteadyState`.
//!
//! Tests **must panic** (not skip) when reference CSVs are absent, per
//! `CLAUDE.md`. The panic message includes the exact Docker command.

use astrodyn_verif_jeod::tier3_csv::test_data_path;

use astrodyn::{
    evaluate_contact_pair, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, JeodQuat, MassProperties, RotationalState, SimulationTime, TranslationalState,
};
use astrodyn::{ContactFacet, ContactMaterial};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_runner::{GroundFacet, RotationModel, Simulation, SphericalTerrain};
use glam::{DMat3, DVec3};
use std::path::Path;
use std::sync::Arc;

// ── Shared JEOD material constants ──────────────────────────────────
//
// Match `Trick::attach_units("lbf/in", 20.0)` and `attach_units("lbf*s/in", 0.4)`
// in JEOD's `Contact_Modified_data/contact/pair_interaction.py`. Trick's
// internal SI conversion uses NIST CODATA exact values
// `1 lbf = 4.4482216152605 N` and `1 in = 0.0254 m`, yielding
// `20 lbf/in = 3502.53670492952642 N/m` and
// `0.4 lbf·s/in = 70.0507340985905387 N·s/m`.
//
// Issue #117: prior values (3502.500484488583 / 70.05000968977167) used an
// incorrect lbf conversion factor (4.4481756 instead of 4.4482216152605),
// producing a ~1e-5 relative error in spring stiffness and damping. The
// error compounded through friction and angular dynamics during the oblique
// `RUN_point_off_center` contact event, accumulating to ~2.7 cm trajectory
// drift over 10 s. Head-on tests showed only ~14 μm drift from the same bug
// because head-on contact has no torque and no compounding friction loop.
// Diagnosis confirmed by capturing JEOD's reported `spring_k` / `damping_b`
// from a `FORCE_COMPONENT_TRACE` patch on `spring_pair_interaction.cc` and
// directly invoking `evaluate_contact_pair` at JEOD-reported state — same
// formula, same state, force agreed to ~5e-13 relative once the constants
// matched.
/// Spring stiffness: 20 lbf/in (NIST exact conversion, matching Trick's
/// `attach_units("lbf/in", 20.0)`). Truncated to f64 precision; the
/// trailing digits beyond ~16 sig figs in `3502.53670492952642` are
/// not representable.
const JEOD_SPRING_K: f64 = 3_502.536_704_929_526_4;
/// Damping: 0.4 lbf·s/in (NIST exact conversion, matching Trick's
/// `attach_units("lbf*s/in", 0.4)`). Truncated to f64 precision.
const JEOD_DAMPING_B: f64 = 70.050_734_098_590_54;
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
             docker run --rm -v $(pwd)/crates/astrodyn_verif_jeod/test_data:/output \\\n    \
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
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );
}

/// Build a simulation with two free-floating bodies of the given mass /
/// inertia, and the standard SIM_contact initial state:
///   veh1 at rest at origin, veh2 at (12,0,0) moving at (-2,0,0) m/s.
fn make_two_body_sim(mass: f64, inertia_diag: DVec3) -> Simulation {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);
    add_empty_space_root(&mut sim);

    let inertia = DMat3::from_cols(
        DVec3::new(inertia_diag.x, 0.0, 0.0),
        DVec3::new(0.0, inertia_diag.y, 0.0),
        DVec3::new(0.0, 0.0, inertia_diag.z),
    );
    let mass_props = MassProperties::with_inertia(mass, inertia, DVec3::ZERO);

    let id1 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });

    let id2 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(12.0, 0.0, 0.0),
            velocity: DVec3::new(-2.0, 0.0, 0.0),
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
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
// follow CLAUDE.md's "5% above observed maximum" tolerance policy,
// with one explicitly-noted exception for `CONTACT_TORQUE_TOL` where
// the observed max sits at the machine-precision noise floor.
//
// Issue #117 closed two bugs that previously inflated these tolerances:
// (1) the spring/damping unit-conversion constants used a slightly off
// `lbf` factor, producing a 1e-5 relative error in `K` and `c`; and
// (2) the relative-velocity formula in `evaluate_contact_pair` omitted
// the rotating-frame contribution that JEOD includes for sphere-sphere
// contact. After both fixes the head-on scenarios match JEOD to
// machine precision (~1e-15 m position over 10 s); the off-center
// oblique case drops from ~2.7 cm trajectory drift to ~2.5 mm — an
// ω²-scaled per-stage residual of ~120 μN per RK4 stage remains,
// likely from JEOD's per-stage `Q_parent_this.normalize_integ` +
// `compute_transformation` recomputation that our coupled-RK4 kernel
// doesn't currently mirror. The remaining off-center drift is 12
// orders of magnitude better than head-on (machine precision) but
// not at parity; tracked for future tightening.
const CONTACT_FORCE_TOL: f64 = 0.034; // N — observed max 32 mN; literal is 1.05× observed (policy).

// `CONTACT_TORQUE_TOL` is the documented noise-floor exception: the
// observed max (~1.2e-13 N·m) is f64 round-off on torque arms of order
// 1 m crossed with forces of order 1 N. A strict 1.05× literal
// (~1.26e-13) would false-fail on platforms whose FP rounding paths
// differ by a few ULPs. `2.0e-13` sits ~1.7× above the observed
// noise floor — large enough to absorb cross-platform FP variance,
// still ~12 orders of magnitude tighter than the pre-issue-#117
// envelope, so any real torque regression trips it.
const CONTACT_TORQUE_TOL: f64 = 2.0e-13;

const POINT_OFF_CENTER_FORCE_TOL: f64 = 0.63; // N — observed 0.60 N; literal is 1.05× observed (policy).
const POINT_OFF_CENTER_TORQUE_TOL: f64 = 0.61; // N·m — observed 0.57 N·m; literal is ~1.07× observed (policy).

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
                    veh1_trans: astrodyn::typed_bridge::trans_typed_to_raw(&b_a.trans),
                    veh1_rot: astrodyn::typed_bridge::rot_typed_to_raw(
                        &b_a.rot.expect("6-DOF required for SIM_contact"),
                    ),
                    veh2_trans: astrodyn::typed_bridge::trans_typed_to_raw(&b_b.trans),
                    veh2_rot: astrodyn::typed_bridge::rot_typed_to_raw(
                        &b_b.rot.expect("6-DOF required for SIM_contact"),
                    ),
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

    // Head-on sphere-sphere symmetric contact: after the issue-#117 fixes
    // (unit-conversion constants + rotating-frame rel-vel term), our
    // pipeline-coupled RK4 matches JEOD to ~1e-15 m / 1e-15 m·s⁻¹ over
    // 10 s — i.e. the machine-precision floor for f64 arithmetic on this
    // trajectory length.
    //
    // The literal `1.0e-13` is a deliberate ~100× exception to the
    // CLAUDE.md "5% above observed" tolerance policy: the observed
    // max is f64 round-off noise (a few ULPs of ~1 m positions), which
    // is platform-, microarchitecture-, and codegen-dependent within
    // a small constant factor. Setting the literal to `observed * 1.05`
    // (~1e-15) would produce false failures on x86_64 hosts whose
    // FMA / x87-80-bit rounding differs from the host that observed
    // the current value. `1.0e-13` is the noise-floor budget — large
    // enough to absorb cross-platform FP variance, but still 12
    // orders of magnitude below the pre-issue-#117 head-on drift
    // (~14 μm), so any genuine regression in the contact pipeline
    // will trip it immediately.
    assert!(
        max_pos_err_1 < 1.0e-13,
        "veh1 position error {max_pos_err_1:.3e} > 100 fm"
    );
    assert!(
        max_pos_err_2 < 1.0e-13,
        "veh2 position error {max_pos_err_2:.3e} > 100 fm"
    );
    assert!(
        max_vel_err_1 < 1.0e-13,
        "veh1 velocity error {max_vel_err_1:.3e} > 1e-13 m/s"
    );
    assert!(
        max_vel_err_2 < 1.0e-13,
        "veh2 velocity error {max_vel_err_2:.3e} > 1e-13 m/s"
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

    // Head-on capsule-capsule. After issue #117 fixes, matches JEOD to
    // machine precision over 10 s.
    assert!(max_pos_err < 1.0e-13, "position error {max_pos_err:.3e}");
    assert!(max_vel_err < 1.0e-13, "velocity error {max_vel_err:.3e}");

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

    // Head-on line-point. After issue #117 fixes, matches JEOD to
    // machine precision over 10 s.
    assert!(max_pos_err < 1.0e-13, "pos err {max_pos_err:.3e}");
    assert!(max_vel_err < 1.0e-13, "vel err {max_vel_err:.3e}");

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
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
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
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: jeod_veh1,
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(12.0, 0.0, 0.0),
            velocity: DVec3::new(-2.0, 0.0, 0.0),
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: jeod_veh2,
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
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
    // collision remains symmetric. After issue #117 fixes, matches JEOD
    // to machine precision over 10 s.
    assert!(max_pos_err < 1.0e-13, "pos err {max_pos_err:.3e}");
    assert!(max_vel_err < 1.0e-13, "vel err {max_vel_err:.3e}");

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
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
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
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init.veh1_pos,
            velocity: init.veh1_vel,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls { controls: vec![] },
        compute_gravity_gradient: false,
        ..Default::default()
    });
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init.veh2_pos,
            velocity: init.veh2_vel,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
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

    // Oblique collision. Issue #117 closed the two principal bugs that
    // previously held this test at ~2.7 cm trajectory drift:
    //
    // 1. Spring/damping unit-conversion (`JEOD_SPRING_K`, `JEOD_DAMPING_B`):
    //    used a slightly off `lbf` factor (`4.4481756` vs NIST CODATA
    //    `4.4482216152605`), producing a 1e-5 relative error in `K` and
    //    `c`. Affected both head-on and oblique tests, but compounded
    //    far more in oblique through the tangential-friction loop.
    //
    // 2. Relative-velocity formula in `evaluate_contact_pair`: the prior
    //    one-cross form `(ω_b − ω_a) × arm_a` (PR #87) was identically
    //    zero for equal-ω cases (sphere-sphere with symmetric
    //    Newton's-third-law torques) and missed the textbook
    //    `(ω_a + ω_b) × arm_a` rotating-frame term. Replaced with the
    //    full two-body kinematic formula
    //    `(v_a − v_b) + ω_a × arm_a − ω_b × arm_b`, which matches JEOD's
    //    `(ω_target − ω_subject) × r_subject_contact − v_target_in_subject_frame`
    //    formulation for sphere-sphere contact (`src/interactions.rs::evaluate_contact_pair`).
    //
    // After both fixes, oblique trajectory error drops from ~2.7 cm to
    // ~2.5 mm (and head-on tests reach machine precision). The residual
    // ~120 μN per-RK4-stage force divergence scales with ω², attributable
    // to JEOD's per-stage `Q_parent_this.normalize_integ()` +
    // `compute_transformation()` recomputation
    // (`dyn_body_integration.cc:380-383`) that our coupled-RK4 kernel
    // does not currently mirror. Tracked for follow-up.
    assert!(
        max_pos_err < 2.7e-3,
        "veh{{1,2}} position error {max_pos_err:.3e} > 2.7 mm"
    );
    assert!(
        max_vel_err < 5.7e-4,
        "veh{{1,2}} velocity error {max_vel_err:.3e} > 0.57 mm/s"
    );

    // Oblique force/torque tolerances reflect the residual ~120 μN
    // per-stage divergence accumulated over the contact event.
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
fn make_ground_contact_sim() -> (Simulation, usize) {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, DT);

    // Earth as a spherical (point-mass) central body. JEOD's
    // SIM_ground_contact configures the gravity controls with degree=0,
    // order=0, spherical=true so the SH model collapses to point mass.
    // We use the same effective physics via a PointMass GravitySource.
    let earth_mu = astrodyn::EARTH.shape.mu;
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: earth_mu,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            // SphericalTerrain does not consult pfix rotation, so we omit
            // the rotation model here — keeps the test self-contained.
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );

    let mass_props = line_mass_props(); // 200 kg, diag(100, 116.667, 116.667)
    let earth_radius = astrodyn::EARTH.shape.r_eq;

    let earth_grav = GravityControls {
        controls: vec![GravityControl::new_spherical(
            earth_idx,
            GravityGradient::Skip,
        )],
    };

    // veh1 — line cylinder along structural x-axis.
    let veh1 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(earth_radius, 0.0, 0.0),
            velocity: DVec3::ZERO,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: earth_grav.clone(),
        compute_gravity_gradient: false,
        ..Default::default()
    });

    // veh2 — point sphere 10 m radially outward from veh1.
    let veh2 = sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: DVec3::new(earth_radius + 10.0, 0.0, 0.0),
            velocity: DVec3::ZERO,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: earth_grav,
        compute_gravity_gradient: false,
        ..Default::default()
    });
    assert_eq!(veh1, 0);
    assert_eq!(veh2, 1);

    sim.validate().unwrap();
    (sim, earth_idx)
}

/// Ground-contact material: JEOD `Modified_data/ground/pair_interaction.py`
/// — spring_k=10 lbf/in, damping_b=0.2 lbf·s/in, mu=0.5. JEOD's CSV
/// trajectory is generated using Trick's `attach_units("lbf/in", X)`
/// conversion, which uses the NIST-exact 1 lbf = 4.4482216152605 N (vs
/// the Older `4.448222 N` mantissa baked into JEOD_SPRING_K above for
/// SIM_contact). The resulting 10 lbf/in = 1751.2683...; using
/// 1751.2502 (half of the SIM_contact constant) gives a constant
/// ~0.96 m/s velocity offset because Δv = (1/6) k R dt / m amplifies
/// the 0.001% conversion gap into 1 m/s on a 93 km/s impulse. We use
/// the NIST-exact value here.
const GROUND_SPRING_K: f64 = 10.0 * 4.4482216152605 / 0.0254; // 1751.2683… N/m
const GROUND_DAMPING_B: f64 = 0.2 * 4.4482216152605 / 0.0254; // 35.0254… N·s/m
const GROUND_MU: f64 = 0.5;

fn ground_steel() -> ContactMaterial {
    ContactMaterial::jeod_spring(GROUND_SPRING_K, GROUND_DAMPING_B, GROUND_MU)
}

/// Tier 3 cross-validation against `SIM_ground_contact/RUN_contact_ground` CSV.
///
/// JEOD's CSV trajectory is produced by an initialization-state artifact
/// in `ContactGround::initialize_ground`: a pre-propagation
/// `in_contact()` call writes an impulsive force onto `subject->force`,
/// which the integrator consumes at stage 1 of step 1 (RK4 weight 1/6)
/// and `ContactSurface::collect_forces_torques` zeroes thereafter. Our
/// port models this explicitly via [`Phase::Initialization`] —
/// `Simulation::register_ground_contact_pair` evaluates the
/// pre-propagation force at registration time and stores it on the pair
/// as `pending_initial_impulse`. The coupled-RK4 stage closure consumes
/// it on the first invocation and clears it for stages 2-4 and all
/// subsequent steps. After consumption, the steady-state path
/// ([`Phase::SteadyState`]) reports no contact for vehicles above the
/// surface — physically correct, matching JEOD.
///
/// ## Root cause
///
/// JEOD's `BodyRefFrame::state.trans.position` for a surface-model-created
/// `vehicle_point` (the C++ frame backing each `ContactFacet`) is
/// **default-constructed to (0, 0, 0)** when the frame is created, and
/// only later populated to its true inertial position by
/// `DynBody::compute_vehicle_point_states` (see
/// `dyn_body_propagate_state.cc::compute_derived_state_forward`).
/// `ContactGround::initialize_ground` runs **before** that propagation
/// (the `P_DYN("initialization")` job in
/// `verif/SIM_ground_contact/S_modules/contact.sm:70`), and inside
/// `GroundInteraction::initialize` calls `in_contact()` once with
/// `vehicle_point.state.trans.position == (0, 0, 0)`. Tracing
/// `point_ground_interaction.cc::in_contact` from that state:
///
/// - `vec = structure.pos + vp.pos = (R, 0, 0) + (0, 0, 0) = (R, 0, 0)`
///   (interpreted as the vehicle's inertial position).
/// - Ground point in body frame ≈ `(R, 0, 0)`; sphere/cylinder
///   `contact_point` ≈ `(1, 0, 0)`.
/// - `facet_pos = T_parent_this * vp.pos = identity * (0, 0, 0) = (0, 0, 0)`.
/// - `rel_state = contact_point + facet_pos = (1, 0, 0)` →
///   `subject_mag = 1 << R = ground_mag` → **contact triggers** with
///   penetration ≈ R, force ≈ k·R = 1.117 × 10¹⁰ N per vehicle. This
///   value is what JEOD writes into `subject->force` and what eventually
///   surfaces as the `~2.2 × 10¹⁰ N` first-row CSV value (a factor of 2
///   suggesting init runs `in_contact` twice, once per ground facet
///   pairing — to be confirmed by a JEOD live-run trace).
///
/// At the **first integration step**, before any RK4 stage runs,
/// `compute_vehicle_point_states` has propagated `vp.state.trans.position`
/// to its true inertial value `(R, 0, 0)`. The same algorithm now gives:
///
/// - `vec = (R, 0, 0) + (R, 0, 0) = (2R, 0, 0)` (this is the JEOD code's
///   apparent doubled-position symptom — only consistent because the
///   init-time vp.pos was (0, 0, 0)).
/// - `facet_pos = identity * (R, 0, 0) = (R, 0, 0)`.
/// - `rel_state = (R+1, 0, 0)` → `subject_mag = R+1 > R = ground_mag` →
///   **no contact** at any altitude.
///
/// Net JEOD behaviour: an impulsive force of 1.117 × 10¹⁰ N on
/// `subject->force` from initialization is consumed at stage 1 of step 1
/// (RK4 weight 1/6), and stages 2–4 plus all subsequent steps see zero
/// contact force. RK4 then yields
/// `Δv ≈ (1/6) × F × dt / m = 93 081 m/s`, exactly matching the t=0.05 CSV
/// velocity.
///
/// Tolerances per CLAUDE.md "5% above observed max" policy. Observed:
/// position 2.79 nm, velocity 4.4e-11 m/s — essentially bit-for-bit
/// agreement with JEOD's CSV (residual is f64 roundoff in the
/// gravity-coupled RK4). The constants below are 1.05× observed maxima.
const GROUND_POS_TOL: f64 = 3.0e-9; // m (observed 2.794 nm)
const GROUND_VEL_TOL: f64 = 5.0e-11; // m/s (observed 4.366e-11 m/s)

#[test]
fn tier3_contact_ground() {
    let csv_path = test_data_path("contact_ground_contact_state.csv");
    let records = load_contact_csv(&csv_path);
    assert!(!records.is_empty(), "expected at least one CSV record");

    let (mut sim, earth_idx) = make_ground_contact_sim();
    let earth_radius = astrodyn::EARTH.shape.r_eq;
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
    sim.register_ground_contact_pair(0, veh1_facet, ground.clone(), earth_idx);
    sim.register_ground_contact_pair(1, veh2_facet, ground, earth_idx);

    // Step at the SIM_contact native rate (DT = 0.01 s) and snapshot at
    // each CSV checkpoint (LOG_CYCLE = 0.05 s).
    let checkpoints: Vec<f64> = records.iter().map(|r| r.time).collect();
    let mut cp_iter = checkpoints.iter().copied().peekable();

    let mut snapshots: Vec<CheckpointBodies> = Vec::with_capacity(records.len());
    let steps_total = (SIM_DURATION / DT).round() as usize;
    for step in 0..=steps_total {
        let b1 = sim.body(0);
        let b2 = sim.body(1);

        let t = step as f64 * DT;
        if let Some(&cp) = cp_iter.peek() {
            if (t - cp).abs() <= 0.5 * DT {
                snapshots.push(CheckpointBodies {
                    veh1_trans: astrodyn::TranslationalState {
                        position: b1.trans.position.raw_si(),
                        velocity: b1.trans.velocity.raw_si(),
                    },
                    veh1_rot: {
                        let _r = b1.rot.expect("6-DOF required");
                        astrodyn::RotationalState {
                            quaternion: _r.q_inertial_body.to_jeod_quat(),
                            ang_vel_body: _r.ang_vel_body.raw_si(),
                        }
                    },
                    veh2_trans: astrodyn::TranslationalState {
                        position: b2.trans.position.raw_si(),
                        velocity: b2.trans.velocity.raw_si(),
                    },
                    veh2_rot: {
                        let _r = b2.rot.expect("6-DOF required");
                        astrodyn::RotationalState {
                            quaternion: _r.q_inertial_body.to_jeod_quat(),
                            ang_vel_body: _r.ang_vel_body.raw_si(),
                        }
                    },
                });
                cp_iter.next();
            }
        }
        if step == steps_total {
            break;
        }
        sim.step_n(1).expect("step_n failed");
    }

    assert_eq!(
        snapshots.len(),
        records.len(),
        "snapshot/CSV checkpoint count mismatch ({} vs {})",
        snapshots.len(),
        records.len()
    );

    let mut max_pos_err_1 = 0.0_f64;
    let mut max_pos_err_2 = 0.0_f64;
    let mut max_vel_err_1 = 0.0_f64;
    let mut max_vel_err_2 = 0.0_f64;
    for (snap, rec) in snapshots.iter().zip(records.iter()) {
        max_pos_err_1 = max_pos_err_1.max((snap.veh1_trans.position - rec.veh1_pos).length());
        max_pos_err_2 = max_pos_err_2.max((snap.veh2_trans.position - rec.veh2_pos).length());
        max_vel_err_1 = max_vel_err_1.max((snap.veh1_trans.velocity - rec.veh1_vel).length());
        max_vel_err_2 = max_vel_err_2.max((snap.veh2_trans.velocity - rec.veh2_vel).length());
    }
    println!(
        "tier3_contact_ground: pos err max = ({max_pos_err_1:.3e}, {max_pos_err_2:.3e}) m; \
         vel err max = ({max_vel_err_1:.3e}, {max_vel_err_2:.3e}) m/s"
    );

    assert!(
        max_pos_err_1 < GROUND_POS_TOL,
        "veh1 position error {max_pos_err_1} m > tol {GROUND_POS_TOL}"
    );
    assert!(
        max_pos_err_2 < GROUND_POS_TOL,
        "veh2 position error {max_pos_err_2} m > tol {GROUND_POS_TOL}"
    );
    assert!(
        max_vel_err_1 < GROUND_VEL_TOL,
        "veh1 velocity error {max_vel_err_1} m/s > tol {GROUND_VEL_TOL}"
    );
    assert!(
        max_vel_err_2 < GROUND_VEL_TOL,
        "veh2 velocity error {max_vel_err_2} m/s > tol {GROUND_VEL_TOL}"
    );
}
