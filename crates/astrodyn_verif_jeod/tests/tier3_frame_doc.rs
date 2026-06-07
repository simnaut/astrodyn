// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: frame-document serialize → reload → continue, bit-exact
//! (issue #663; spec trace RFS-601/602/603).
//!
//! Two scenarios, both comparing an **uninterrupted** run A against a
//! run B that steps to a snapshot instant, exports a
//! [`astrodyn::frame_doc::FrameDocument`], rebuilds a fresh simulation
//! from the same configuration, applies the document, and continues —
//! asserting `f64::to_bits` equality of the full frame tree and the
//! authoritative body store at the end:
//!
//! 1. [`tier3_frame_doc_leo_rnp_continue`] — LEO 6-DOF over GGM05C
//!    spherical-harmonics Earth with the `EarthRNP` rotation model. This
//!    covers **both rotation-canonicity regimes** in one trajectory: the
//!    pfix node is matrix-canonical (`sync_pfix_rotation` stores the RNP
//!    matrix verbatim) and the body path is quaternion-canonical
//!    (RF.04), and the SH gravity consumes the pfix rotation, so a
//!    single-ULP restore error in either representation diverges the
//!    trajectory.
//! 2. [`tier3_frame_doc_frame_switch_continue`] — the Apollo 8
//!    trans-lunar frame-switch scenario (`SIM_verif_frame_switch` ICs).
//!    A [`FrameSeriesRecorder`] records across the switch and must
//!    produce **exactly two segments** with the boundary at the reparent
//!    step (replay v1: segment-per-topology-change); the snapshot is
//!    taken after the switch, so run B rebuilds the **post-switch**
//!    configuration and `apply` verifies each record's declared parent
//!    against the rebuilt topology.
//!
//! Scope limits (documented in `Simulation::apply_frame_document`):
//! single-step integrators only (RK4 here) and the default Earth-RNP
//! refresh cadence (0).

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Tier 3 step counts and indices fit exactly in f64 mantissa and usize"
)]
#![allow(
    clippy::excessive_precision,
    reason = "Apollo 8 IC constants are copied digit-for-digit from JEOD's vehicle.py"
)]

use astrodyn::frame_doc::{CanonicalRotation, FrameDocument, Origin};
use astrodyn::typed_bridge::{mass_raw_to_self_ref, rot_raw_to_self_ref, trans_raw_to_root};
use astrodyn::{
    FrameSwitchConfig, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, GravitySourceEntry, JeodQuat, MassProperties, RotationalState, SimulationTime,
    SwitchSense, TranslationalState, VehicleConfig,
};
use astrodyn_runner::{FrameSeriesRecorder, Simulation};
use glam::{DMat3, DVec3};

/// Every f64 of the simulation's frame layer and body store, as bits.
/// Bit-equality of this vector is the RFS-601 acceptance criterion: the
/// restored-and-continued run must be indistinguishable from the
/// uninterrupted one.
fn state_bits(sim: &Simulation, body_idx: usize) -> Vec<u64> {
    let mut bits = Vec::new();
    let body = sim.body(body_idx);
    bits.extend(body.trans.position.raw_si().to_array().map(f64::to_bits));
    bits.extend(body.trans.velocity.raw_si().to_array().map(f64::to_bits));
    if let Some(rot) = &body.rot {
        let raw = astrodyn::typed_bridge::rot_typed_to_raw(rot);
        bits.extend(raw.quaternion.data.map(f64::to_bits));
        bits.extend(raw.ang_vel_body.to_array().map(f64::to_bits));
    }
    let tree = sim.frame_tree();
    for id in 0..tree.len() {
        let node = tree.get(id);
        bits.extend(node.state.trans.position.to_array().map(f64::to_bits));
        bits.extend(node.state.trans.velocity.to_array().map(f64::to_bits));
        bits.extend(node.state.rot.q_parent_this.data.map(f64::to_bits));
        bits.extend(
            node.state
                .rot
                .t_parent_this
                .to_cols_array()
                .map(f64::to_bits),
        );
        bits.extend(node.state.rot.ang_vel_this.to_array().map(f64::to_bits));
        bits.push(
            node.epoch
                .map(|e| e.as_seconds().to_bits())
                .unwrap_or(u64::MAX),
        );
    }
    bits.push(sim.time.simtime.to_bits());
    bits.push(sim.time.tdb_seconds.to_bits());
    bits
}

/// Round-trip a document through an on-disk JSON file — the acceptance
/// test exercises the wire form, not just the in-memory structs.
fn through_json_file(doc: &FrameDocument, label: &str) -> FrameDocument {
    let path = std::env::temp_dir().join(format!(
        "astrodyn_tier3_frame_doc_{label}_{}.json",
        std::process::id()
    ));
    std::fs::write(&path, doc.to_json_string()).expect("write document JSON");
    let json = std::fs::read_to_string(&path).expect("read document JSON");
    std::fs::remove_file(&path).ok();
    FrameDocument::from_json_str(&json).expect("parse document JSON")
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 1: LEO 6-DOF over GGM05C + EarthRNP (both canonicity regimes)
// ─────────────────────────────────────────────────────────────────────────

const LEO_DT: f64 = 1.0;
/// Steps before the snapshot…
const LEO_N: usize = 60;
/// …and steps after it.
const LEO_M: usize = 60;

fn build_leo_rnp_sim() -> (Simulation, usize) {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, LEO_DT);
    // GGM05C spherical harmonics + EarthRNP rotation model: the per-step
    // RNP evaluation writes the pfix node matrix-canonically AND feeds
    // the SH gravity evaluation, so the body trajectory depends on the
    // restored pfix rotation bits.
    let earth = sim.add_source("Earth", astrodyn::recipes::earth::ggm05c());

    let body = sim.add_body(VehicleConfig {
        trans: trans_raw_to_root(&TranslationalState {
            position: DVec3::new(6_778_137.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 6_700.0, 3_700.0),
        }),
        // Non-trivial attitude (not identity, not a 90° axis) and a slow
        // tumble, per the quaternion-convention test rule.
        rot: Some(rot_raw_to_self_ref(&RotationalState {
            quaternion: JeodQuat::left_quat_from_eigen_rotation(
                0.7,
                DVec3::new(1.0, 2.0, 3.0).normalize(),
            ),
            ang_vel_body: DVec3::new(0.01, -0.02, 0.005),
        })),
        // allowed: typed↔raw kernel-boundary lift on scenario mass
        // construction (named-method opt-in; see #397).
        mass: Some(mass_raw_to_self_ref(&MassProperties::with_inertia(
            1_000.0,
            DMat3::from_diagonal(DVec3::new(100.0, 80.0, 120.0)),
            DVec3::ZERO,
        ))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(
                earth,
                8,
                8,
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named("frame-doc-leo")
    });
    sim.validate().expect("validation failed");
    (sim, body)
}

#[test]
fn tier3_frame_doc_leo_rnp_continue() {
    // Run A: uninterrupted.
    let (mut sim_a, body_a) = build_leo_rnp_sim();
    for _ in 0..(LEO_N + LEO_M) {
        sim_a.step().expect("run A step");
    }

    // Run B: step to the snapshot, export, rebuild, apply, continue.
    let (mut sim_b1, _) = build_leo_rnp_sim();
    for _ in 0..LEO_N {
        sim_b1.step().expect("run B pre-snapshot step");
    }
    let doc = sim_b1.export_frame_document();

    // The document must carry both canonicity regimes and the origin
    // taxonomy this scenario produces.
    let pfix = doc
        .records
        .iter()
        .find(|r| r.name == "Earth.pfix")
        .expect("pfix record");
    assert!(
        matches!(pfix.rotation, CanonicalRotation::Matrix(_)),
        "pfix node is matrix-canonical (sync_pfix_rotation)"
    );
    assert!(
        matches!(&pfix.origin, Origin::Derived { model } if model == "EarthRNP"),
        "pfix state is a rotation-model evaluation"
    );
    let body_rec = doc
        .records
        .iter()
        .find(|r| r.name.starts_with("body_"))
        .expect("body record");
    assert!(
        matches!(body_rec.rotation, CanonicalRotation::Quat(_)),
        "body node is quaternion-canonical (RF.04)"
    );
    assert!(
        matches!(
            body_rec.origin,
            Origin::Integrated {
                attitude_quat: Some(_),
                ang_vel_body: Some(_)
            }
        ),
        "body state projects the authoritative store (6-DOF payload)"
    );

    let doc = through_json_file(&doc, "leo_rnp");
    let (mut sim_b2, body_b) = build_leo_rnp_sim();
    sim_b2.apply_frame_document(&doc);
    for _ in 0..LEO_M {
        sim_b2.step().expect("run B post-restore step");
    }

    assert_eq!(
        state_bits(&sim_a, body_a),
        state_bits(&sim_b2, body_b),
        "restored-and-continued run must be bit-identical to the uninterrupted run \
         (RFS-601; covers the matrix-canonical pfix and quat-canonical body regimes)"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 2: Apollo 8 frame switch (segment boundary + post-switch restore)
// ─────────────────────────────────────────────────────────────────────────
// Initial conditions from `SIM_verif_frame_switch` / `Modified_data/vehicle.py`
// (Apollo 8 trans-lunar coast, Dec 23 1968 19:38 UTC), matching
// `tier3_apollo8_frame_switch.rs`.

const A8_DT: f64 = 0.5;
const A8_TOTAL_STEPS: usize = 200; // 100 s
const A8_POS_ECI: DVec3 = DVec3::new(
    302_274_887.753_810_17,
    -119_023_818.108_825_01,
    -56_915_743.953_866_437,
);
const A8_VEL_ECI: DVec3 = DVec3::new(
    942.182_494_673_019_85,
    -189.920_638_006_114_07,
    -292.959_665_506_469_89,
);
const A8_MU_MOON: f64 = 4.902_801_076e12;
const A8_SWITCH_DISTANCE: f64 = 66.1e6;

/// Build the Apollo 8 scenario. `post_switch` rebuilds the configuration
/// **as of after the frame switch fired**: the body integrates in the
/// Moon's inertial frame, the switch is spent (absent), and the
/// gravity-control `differential` flags carry the flip
/// `evaluate_and_apply_frame_switch` applied in the producing run.
fn build_apollo8_sim(post_switch: bool) -> (Simulation, usize, usize) {
    let bsp_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/de405.bsp");
    assert!(
        bsp_path.exists(),
        "DE405 ephemeris not found at {}",
        bsp_path.display()
    );

    // Dec 23, 1968, 19:38:00 UTC (TJT = JD − 2440000.5).
    let utc_tjt = 2_440_214.318_055_555_5 - 2_440_000.5;
    let leap_table = astrodyn::default_leap_second_table();
    let tai_tjt = leap_table.utc_to_tai_tjt(utc_tjt);
    let time = SimulationTime::new(tai_tjt, leap_table);

    let mut sim = Simulation::new(time, A8_DT);
    sim.ephemeris =
        Some(astrodyn::Ephemeris::from_bsp(&bsp_path).expect("Failed to load DE405 ephemeris"));

    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: astrodyn::SUN.shape.mu,
                model: GravityModel::PointMass,
            },
            astrodyn::Position::<astrodyn::RootInertial>::zero(),
            None,
        ),
    );
    sim.set_source_ephemeris(
        sun,
        astrodyn::EphemerisBody::Sun,
        astrodyn::EphemerisBody::Earth,
    );

    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: astrodyn::EARTH.shape.mu,
            model: GravityModel::PointMass,
        },
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth = sim.add_source("Earth", earth_entry);

    let moon = sim.add_source(
        "Moon",
        GravitySourceEntry::new(
            GravitySource {
                mu: A8_MU_MOON,
                model: GravityModel::PointMass,
            },
            astrodyn::Position::<astrodyn::RootInertial>::zero(),
            None,
        ),
    );
    sim.set_source_ephemeris(
        moon,
        astrodyn::EphemerisBody::Moon,
        astrodyn::EphemerisBody::Earth,
    );

    // The producing run's controls, then (for the post-switch rebuild)
    // the exact flip `evaluate_and_apply_frame_switch` performs: the
    // switch target becomes non-differential, everything else
    // differential.
    let mut controls = vec![
        GravityControl::new_spherical(earth, GravityGradient::Skip),
        GravityControl::new_third_body(sun),
        GravityControl::new_third_body(moon),
    ];
    if post_switch {
        for ctrl in &mut controls {
            ctrl.differential = ctrl.source_name != moon;
        }
    }

    let body = sim.add_body(VehicleConfig {
        // Run-B placeholder semantics: `apply_frame_document` overwrites
        // both the node and the store, so the configured state only needs
        // to validate. The uninterrupted run uses it as the real IC.
        trans: trans_raw_to_root(&TranslationalState {
            position: A8_POS_ECI,
            velocity: A8_VEL_ECI,
        }),
        rot: Some(rot_raw_to_self_ref(&RotationalState {
            quaternion: JeodQuat::identity(),
            ang_vel_body: DVec3::ZERO,
        })),
        // allowed: typed↔raw kernel-boundary lift on scenario mass
        // construction (named-method opt-in; see #397).
        mass: Some(mass_raw_to_self_ref(&MassProperties::with_inertia(
            91_589.71,
            DMat3::from_diagonal(DVec3::new(135_581.79, 271_163.59, 542_327.18)),
            DVec3::ZERO,
        ))),
        gravity_controls: GravityControls { controls },
        integ_source: if post_switch { Some(moon) } else { None },
        frame_switches: if post_switch {
            vec![]
        } else {
            vec![FrameSwitchConfig {
                target_source: moon,
                switch_sense: SwitchSense::OnApproach,
                switch_distance: A8_SWITCH_DISTANCE,
                active: true,
            }]
        },
        ..VehicleConfig::named("frame-doc-apollo8")
    });
    sim.validate().expect("validation failed");
    (sim, body, moon)
}

#[test]
fn tier3_frame_doc_frame_switch_continue() {
    // Run A: uninterrupted, with the series recorder watching the
    // topology across the switch.
    let (mut sim_a, body_a, _moon) = build_apollo8_sim(false);
    let mut recorder = FrameSeriesRecorder::new(&sim_a);
    recorder.record(&sim_a); // t₀ keyframe
    let mut switch_step = None;
    for step in 0..A8_TOTAL_STEPS {
        sim_a.step().expect("run A step");
        recorder.record(&sim_a);
        if switch_step.is_none() && sim_a.body(body_a).integ_frame_id != sim_a.root_frame_id {
            switch_step = Some(step + 1);
        }
    }
    let switch_step = switch_step.expect("the frame switch must fire within the run");
    let series = recorder.finish();

    // Replay v1: the reparent closes the segment — exactly two segments,
    // with the boundary row at the switch step (a seek keyframe).
    assert_eq!(
        series.segments.len(),
        2,
        "one topology change (the frame-switch reparent) must produce exactly two segments"
    );
    assert_eq!(
        series.segments[0].epochs.len(),
        switch_step, // t₀ keyframe + steps 1..switch_step-1
        "first segment spans the pre-switch epochs"
    );
    assert_eq!(
        series.segments[1].start_simtime.to_bits(),
        (switch_step as f64 * A8_DT).to_bits(),
        "segment boundary sits at the reparent step"
    );
    series.validate().expect("recorded series validates");

    // Run B: step a second uninterrupted sim past the switch, snapshot,
    // rebuild the post-switch configuration, apply, continue.
    let snapshot_step = switch_step + 10;
    assert!(
        snapshot_step < A8_TOTAL_STEPS,
        "snapshot must precede run end"
    );
    let (mut sim_b1, _, _) = build_apollo8_sim(false);
    for _ in 0..snapshot_step {
        sim_b1.step().expect("run B pre-snapshot step");
    }
    let doc = through_json_file(&sim_b1.export_frame_document(), "frame_switch");

    let (mut sim_b2, body_b, _) = build_apollo8_sim(true);
    sim_b2.apply_frame_document(&doc);
    for _ in 0..(A8_TOTAL_STEPS - snapshot_step) {
        sim_b2.step().expect("run B post-restore step");
    }

    assert_eq!(
        state_bits(&sim_a, body_a),
        state_bits(&sim_b2, body_b),
        "post-switch restore must continue bit-identically to the uninterrupted run \
         (apply verified each record's declared parent against the rebuilt topology)"
    );
}
