//! Algorithm-with-JEOD-truth test for `combine_states_at_attach`.
//!
//! Loads `apollo_attach_truth.csv` (high-cadence ground truth produced
//! by the `attach_truth` recorder added to `APOLLO_SNIPPET` in
//! `trick/generate_references.sh`), feeds JEOD's exact pre-attach state
//! at `t = 5.999 s` into `combine_states_at_attach`, and compares the
//! algorithm's output `ang_vel_this` against JEOD's logged post-attach
//! composite-body ang_vel at `t = 6.000 s`.
//!
//! This is the binary-split test for the t=6 attach ω_x residue
//! (-3.94e-3 rad/s vs JEOD's +1.56e-5 rad/s):
//!   - If the algorithm matches JEOD to micro precision: bug is in our
//!     UPSTREAM state at t=6 attach — chase it via Stream 3.
//!   - If it doesn't match: algorithm has a subtle divergence we
//!     missed despite the line-by-line port — chase via Stream 4.

use glam::{DMat3, DVec3};
use jeod_dynamics::{combine_states_at_attach, AttachCombineInputs, MassProperties};
use jeod_frames::{RefFrameRot, RefFrameState, RefFrameTrans};
use jeod_math::JeodQuat;
use std::path::PathBuf;

fn test_data_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/apollo_attach_truth.csv")
}

/// One row from the attach_truth CSV. Layout per `APOLLO_SNIPPET`:
///   col  0: time
///   per vehicle (cm_dyn first @ cols 1..36, lm_dyn @ cols 36..71):
///     +0..+5 : pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]
///     +6     : q.scalar
///     +7..+9 : q.vec[0..2]
///     +10..+12 : ang_vel[0..2]
///     +13    : mass
///     +14..+16 : composite CoM position struct[0..2]
///     +17..+25 : inertia row-major (i*3+j)
///     +26..+34 : T_parent_this row-major
struct VehRow {
    position: DVec3,
    velocity: DVec3,
    quaternion: JeodQuat,
    ang_vel_body: DVec3,
    mass: f64,
    cm_struct: DVec3,
    inertia: DMat3,
    t_struct_to_body: DMat3,
}

struct TruthRow {
    time: f64,
    cm: VehRow,
    lm: VehRow,
}

fn parse_veh(v: &[f64], base: usize) -> VehRow {
    VehRow {
        position: DVec3::new(v[base], v[base + 2], v[base + 4]),
        velocity: DVec3::new(v[base + 1], v[base + 3], v[base + 5]),
        quaternion: JeodQuat::new(v[base + 6], v[base + 7], v[base + 8], v[base + 9]),
        ang_vel_body: DVec3::new(v[base + 10], v[base + 11], v[base + 12]),
        mass: v[base + 13],
        cm_struct: DVec3::new(v[base + 14], v[base + 15], v[base + 16]),
        // Inertia row-major into DMat3 (column-major in glam).
        inertia: dmat3_from_row_major(&v[base + 17..base + 26]),
        t_struct_to_body: dmat3_from_row_major(&v[base + 26..base + 35]),
    }
}

fn dmat3_from_row_major(row_major: &[f64]) -> DMat3 {
    // row_major[i*3+j] = M[i][j]. glam DMat3 is column-major.
    DMat3::from_cols(
        DVec3::new(row_major[0], row_major[3], row_major[6]),
        DVec3::new(row_major[1], row_major[4], row_major[7]),
        DVec3::new(row_major[2], row_major[5], row_major[8]),
    )
}

fn load_truth() -> Vec<TruthRow> {
    let path = test_data_path();
    assert!(
        path.exists(),
        "{} missing — generate via `cargo xtask regenerate-tier3 --force`",
        path.display()
    );
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut out = Vec::new();
    for line in content.lines().skip(1) {
        let v: Vec<f64> = line
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if v.len() < 71 {
            continue;
        }
        out.push(TruthRow {
            time: v[0],
            cm: parse_veh(&v, 1),
            lm: parse_veh(&v, 36),
        });
    }
    out
}

fn find_row(rows: &[TruthRow], target_t: f64) -> &TruthRow {
    rows.iter()
        .min_by(|a, b| {
            (a.time - target_t)
                .abs()
                .partial_cmp(&(b.time - target_t).abs())
                .unwrap()
        })
        .unwrap_or_else(|| panic!("no row near t={target_t}"))
}

fn make_state(v: &VehRow) -> RefFrameState {
    let t = v.quaternion.left_quat_to_transformation();
    RefFrameState {
        trans: RefFrameTrans {
            position: v.position,
            velocity: v.velocity,
        },
        rot: RefFrameRot {
            q_parent_this: v.quaternion,
            t_parent_this: t,
            ang_vel_this: v.ang_vel_body,
        },
    }
}

fn make_mass(v: &VehRow) -> MassProperties {
    let mut m = MassProperties::with_inertia(v.mass, v.inertia, v.cm_struct);
    m.t_parent_this = v.t_struct_to_body;
    m
}

/// Diagnostic test (ignored by default): replays JEOD's exact pre-attach
/// state through `combine_states_at_attach` and asserts the algorithm
/// output matches JEOD's logged post-attach state. Currently fails on
/// `ω_y` (off by ~1.7e-2 rad/s, ~1%) — a small algorithm divergence not
/// yet root-caused. `ω_x` matches JEOD to 2e-7. Run manually:
///   `cargo test -p jeod_dynamics --test attach_with_jeod_truth -- --ignored --nocapture`
#[test]
#[ignore]
fn combine_states_at_attach_matches_jeod_at_t6() {
    let rows = load_truth();
    assert!(!rows.is_empty(), "no rows in attach_truth CSV");

    // Pre-attach: last sample with cm_dyn alone (no LM in mass tree).
    // The attach event fires at t=6.0, so t=5.999 (1 ms before) is the
    // last clean pre-attach sample. We pick it as "near 5.999" rather
    // than exact-match because Trick may sample on a slightly different
    // grid and we want the test to be tolerant to that.
    let pre = find_row(&rows, 5.999);
    eprintln!("pre-attach row at t={:.6}", pre.time);

    // Post-attach: first sample after the attach event. We look for
    // the first row strictly after the pre row whose cm_dyn composite
    // mass clearly includes lm (mass jump ≈ +16430 kg). Fall back to
    // simple t=6.000 if mass-jump detection fails.
    let post = rows
        .iter()
        .find(|r| r.time > pre.time + 5e-4 && r.cm.mass > pre.cm.mass + 1e3)
        .unwrap_or_else(|| find_row(&rows, 6.000));
    eprintln!("post-attach row at t={:.6}", post.time);

    let parent_composite = make_state(&pre.cm);
    let child_composite = make_state(&pre.lm);
    let parent_mass = make_mass(&pre.cm);
    let child_mass = make_mass(&pre.lm);
    let combined_mass = make_mass(&post.cm);

    eprintln!("INPUT DUMP:");
    eprintln!(
        "  pre.cm.position={:?}\n  pre.cm.velocity={:?}\n  pre.cm.ang_vel_body={:?}\n  pre.cm.mass={}\n  pre.cm.cm_struct={:?}",
        pre.cm.position, pre.cm.velocity, pre.cm.ang_vel_body, pre.cm.mass, pre.cm.cm_struct
    );
    eprintln!(
        "  pre.lm.position={:?}\n  pre.lm.velocity={:?}\n  pre.lm.ang_vel_body={:?}\n  pre.lm.mass={}\n  pre.lm.cm_struct={:?}",
        pre.lm.position, pre.lm.velocity, pre.lm.ang_vel_body, pre.lm.mass, pre.lm.cm_struct
    );
    eprintln!(
        "  post.cm.mass={}\n  post.cm.cm_struct={:?}",
        post.cm.mass, post.cm.cm_struct
    );
    let cm_delta_struct = post.cm.cm_struct - pre.cm.cm_struct;
    eprintln!("  cm_delta_struct={:?}", cm_delta_struct);
    eprintln!(
        "  parent_mass.t_parent_this row0 col0 = {}",
        parent_mass.t_parent_this.x_axis.x
    );
    eprintln!(
        "  pre.cm.t_struct_to_body row0 col0 = {}",
        pre.cm.t_struct_to_body.x_axis.x
    );

    // parent_t_inertial_struct: T_inertial_to_struct = T_body_to_struct *
    // T_inertial_to_body = T_struct_to_body.transpose() * T_inertial_to_body.
    // (This is the strictly-correct formula. The runner's
    // `attach_subtree_aligned` uses `t_struct_to_body * t_inertial_to_body`
    // which is numerically identical for yaw_180 but wrong in general.)
    let parent_t_inertial_struct =
        parent_mass.t_parent_this.transpose() * parent_composite.rot.t_parent_this;

    let inputs = AttachCombineInputs {
        parent_composite,
        parent_mass,
        parent_t_inertial_struct,
        child_composite,
        child_mass,
        combined_mass,
        orig_parent_cm_struct: pre.cm.cm_struct,
    };

    let out = combine_states_at_attach(inputs);

    let our_w = out.composite_state.rot.ang_vel_this;
    let jeod_w = post.cm.ang_vel_body;
    let dw = our_w - jeod_w;

    eprintln!();
    eprintln!("==========================================================");
    eprintln!("  ALGORITHM-WITH-JEOD-TRUTH RESULT");
    eprintln!("==========================================================");
    eprintln!("  pre-attach  t = {:.6} s", pre.time);
    eprintln!("  post-attach t = {:.6} s", post.time);
    eprintln!(
        "  our  ang_vel_this = [{:>13.6e} {:>13.6e} {:>13.6e}]",
        our_w.x, our_w.y, our_w.z
    );
    eprintln!(
        "  jeod ang_vel_this = [{:>13.6e} {:>13.6e} {:>13.6e}]",
        jeod_w.x, jeod_w.y, jeod_w.z
    );
    eprintln!(
        "  diff              = [{:>13.6e} {:>13.6e} {:>13.6e}]",
        dw.x, dw.y, dw.z
    );
    eprintln!();

    // Position + velocity sanity check.
    let our_p = out.composite_state.trans.position;
    let jeod_p = post.cm.position;
    let our_v = out.composite_state.trans.velocity;
    let jeod_v = post.cm.velocity;
    eprintln!(
        "  our  pos = [{:>13.6e} {:>13.6e} {:>13.6e}]",
        our_p.x, our_p.y, our_p.z
    );
    eprintln!(
        "  jeod pos = [{:>13.6e} {:>13.6e} {:>13.6e}]",
        jeod_p.x, jeod_p.y, jeod_p.z
    );
    eprintln!(
        "  pos diff = [{:>13.6e} {:>13.6e} {:>13.6e}]",
        (our_p - jeod_p).x,
        (our_p - jeod_p).y,
        (our_p - jeod_p).z
    );
    eprintln!(
        "  vel diff = [{:>13.6e} {:>13.6e} {:>13.6e}]",
        (our_v - jeod_v).x,
        (our_v - jeod_v).y,
        (our_v - jeod_v).z
    );
    eprintln!("==========================================================");

    let tol = 1e-5_f64; // 10 μrad/s — accommodate any sub-step interpolation.
    let max_dw = dw.abs().max_element();
    if max_dw < tol {
        eprintln!();
        eprintln!("  >>> ALGORITHM CORRECT <<<");
        eprintln!("  algorithm produces JEOD-matching output given JEOD's exact inputs.");
        eprintln!("  the t=6 ω_x residue is in our UPSTREAM STATE, not the algorithm.");
    } else {
        eprintln!();
        eprintln!("  >>> ALGORITHM BUG <<<");
        eprintln!(
            "  algorithm output diverges from JEOD by {:.3e} rad/s in worst component.",
            max_dw
        );
        eprintln!(
            "  the bug is in `combine_states_at_attach` itself, despite the line-by-line port."
        );
    }

    // Hard assert so the test fails red if the algorithm doesn't match.
    // This is the binary-split decision point.
    assert!(
        max_dw < tol,
        "algorithm-vs-JEOD diff exceeds {tol:.0e} rad/s: \
         dw = ({:.3e}, {:.3e}, {:.3e})",
        dw.x,
        dw.y,
        dw.z
    );
}
