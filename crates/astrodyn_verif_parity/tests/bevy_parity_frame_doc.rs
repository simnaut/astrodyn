//! Bevy-vs-Simulation parity for the frame-document layer (issues
//! #663/#664).
//!
//! Both backends carry the document IO halves (the runner since #663,
//! the ECS adapter since #664), and both stores stamp the same
//! identities — so the document is the interchange contract these tests
//! pin, four ways:
//!
//! 1. **Restore is invisible to parity**: a runner that serializes
//!    mid-run, rebuilds, applies, and continues lands bit-identical to
//!    the uninterrupted Bevy run.
//! 2. **Export equality**: at the same instant, the two backends export
//!    documents that are equal **uid-keyed and bit-for-bit** (same
//!    identity set; per identity: parent identity, epoch, state,
//!    canonical rotation, origin). Names are excluded — they are
//!    diagnostics-only and deliberately differ between backends.
//! 3. **Bevy restore continuity**: a Bevy App that exports mid-run,
//!    rebuilds, applies, and continues lands bit-identical to the
//!    uninterrupted Bevy run.
//! 4. **Cross-restore (the interchangeability statement)**: a document
//!    produced by either backend restores into the other and continues
//!    bit-identically.
//!
//! Scenario mirrors `bevy_parity_highfidelity_sh4x4_rnp` (GGM02C 4×4
//! spherical harmonics + EarthRNP), whose gravity consumes the
//! matrix-canonical pfix rotation each step — a single-ULP error in any
//! restored representation diverges the trajectory. A second export-eq
//! scenario (Earth + DE421 Moon) pins the ephemeris-driven half of the
//! origin/epoch mirror that the SH+RNP scenario cannot reach.

mod common;

use std::collections::BTreeMap;

use astrodyn::frame_doc::{FrameDocument, FrameRecord, Origin};
use astrodyn::{
    Ephemeris, EphemerisBody, FrameUid, GravityControl, GravityControls, GravityGradient,
    GravityModel, GravitySource, GravitySourceEntry, TranslationalState, VehicleConfig,
};
use astrodyn_bevy::{
    DynamicsConfigC, EphemerisBodyC, GravityControlsC, GravitySourceC, IntegrationDtR, MoonMarker,
    PlanetFixedRotationC, SourceInertialPositionC, TranslationalStateC,
};
use astrodyn_runner::RotationModel;
use bevy::prelude::*;
use glam::{DMat3, DVec3};

use common::*;

/// Steps before the snapshot; the remainder of [`NUM_STEPS`] runs after
/// the restore.
const SNAPSHOT_STEP: usize = 50;
/// The body's mission-supplied identity — the SAME value on both
/// backends (that is the point of the chain).
const BODY_LABEL: &str = "bevy-parity-frame-doc";

fn earth_sh_source() -> GravitySource {
    GravitySource {
        mu: astrodyn::gravity_fixtures::load_ggm02c().mu,
        model: GravityModel::SphericalHarmonics(
            Box::new(astrodyn::gravity_fixtures::load_ggm02c()),
        ),
    }
}

fn build_runner_sim() -> astrodyn_runner::Simulation {
    let time = astrodyn::SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = astrodyn_runner::Simulation::new(time, DT);
    let earth_idx = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_sh_source(),
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: astrodyn::planet_config::EARTH.omega,
            central: true,
            marker_only: false,
        },
    );
    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_nonspherical(
                earth_idx,
                4,
                4,
                GravityGradient::Skip,
            )],
        },
        ..VehicleConfig::named(BODY_LABEL)
    });
    sim.validate().unwrap();
    sim
}

/// Build the matching Bevy App; returns (app, vehicle entity).
fn build_bevy_app() -> (App, Entity) {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(astrodyn_bevy::AstrodynPlugin);

    let planet = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Earth>>()),
            Name::new("Earth"),
            GravitySourceC(earth_sh_source()),
            SourceInertialPositionC::default(),
            TranslationalStateC::<astrodyn::Earth>::default(),
            PlanetFixedRotationC::<astrodyn::Earth>(astrodyn::FrameTransform::from_matrix(
                DMat3::IDENTITY,
            )),
        ))
        .id();
    let vehicle = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(BODY_LABEL)),
            TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
            DynamicsConfigC(astrodyn::DynamicsConfig {
                translational_dynamics: true,
                rotational_dynamics: false,
                three_dof: true,
            }),
            GravityControlsC(GravityControls {
                controls: vec![GravityControl::new_nonspherical(
                    planet,
                    4,
                    4,
                    GravityGradient::Skip,
                )],
            }),
        ))
        .id();
    (app, vehicle)
}

/// Uid-keyed, name-free document view for bit-exact cross-backend
/// comparison: identity → (parent identity, record). Names are
/// diagnostics-only and deliberately differ between backends.
fn by_uid(doc: &FrameDocument) -> BTreeMap<String, (Option<String>, &FrameRecord)> {
    doc.records
        .iter()
        .map(|r| {
            let uid = doc.uids[r.uid_index as usize].to_string();
            let parent = r.parent.map(|p| doc.uids[p as usize].to_string());
            (uid, (parent, r))
        })
        .collect()
}

/// Assert two documents are equal uid-keyed and bit-for-bit (excluding
/// diagnostic names and record order).
fn assert_docs_eq(label: &str, a: &FrameDocument, b: &FrameDocument) {
    assert_eq!(
        a.header.simtime.to_bits(),
        b.header.simtime.to_bits(),
        "{label}: header simtime"
    );
    assert_eq!(
        a.header.tai_tjt_at_epoch.to_bits(),
        b.header.tai_tjt_at_epoch.to_bits(),
        "{label}: header epoch"
    );
    let (ma, mb) = (by_uid(a), by_uid(b));
    let keys_a: Vec<_> = ma.keys().collect();
    let keys_b: Vec<_> = mb.keys().collect();
    assert_eq!(keys_a, keys_b, "{label}: identity sets differ");
    for (uid, (parent_a, ra)) in &ma {
        let (parent_b, rb) = &mb[uid];
        assert_eq!(parent_a, parent_b, "{label}: parent of `{uid}`");
        assert_eq!(
            ra.epoch.map(f64::to_bits),
            rb.epoch.map(f64::to_bits),
            "{label}: epoch of `{uid}`"
        );
        assert_eq!(
            ra.trans.position.map(f64::to_bits),
            rb.trans.position.map(f64::to_bits),
            "{label}: position of `{uid}`"
        );
        assert_eq!(
            ra.trans.velocity.map(f64::to_bits),
            rb.trans.velocity.map(f64::to_bits),
            "{label}: velocity of `{uid}`"
        );
        assert_eq!(
            ra.ang_vel_this.map(f64::to_bits),
            rb.ang_vel_this.map(f64::to_bits),
            "{label}: ang_vel of `{uid}`"
        );
        // Canonical rotation: same variant, same bits.
        use astrodyn::frame_doc::CanonicalRotation as CR;
        match (&ra.rotation, &rb.rotation) {
            (CR::Quat(qa), CR::Quat(qb)) => {
                assert_eq!(
                    qa.map(f64::to_bits),
                    qb.map(f64::to_bits),
                    "{label}: quat of `{uid}`"
                );
            }
            (CR::Matrix(ta), CR::Matrix(tb)) => {
                for (ca, cb) in ta.iter().zip(tb.iter()) {
                    assert_eq!(
                        ca.map(f64::to_bits),
                        cb.map(f64::to_bits),
                        "{label}: matrix of `{uid}`"
                    );
                }
            }
            (a, b) => panic!("{label}: canonicity variant of `{uid}` differs: {a:?} vs {b:?}"),
        }
        assert_eq!(ra.origin, rb.origin, "{label}: origin of `{uid}`");
    }
}

#[test]
fn bevy_parity_frame_doc_restore_matches_uninterrupted_bevy() {
    // ── Bevy: uninterrupted ──
    let (mut app, vehicle) = build_bevy_app();
    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_trans(app.world(), vehicle);

    // ── Runner: snapshot at SNAPSHOT_STEP, restore, continue ──
    let mut producer = build_runner_sim();
    producer.step_n(SNAPSHOT_STEP).expect("producer step_n");
    let json = producer.export_frame_document().to_json_string();
    let doc = FrameDocument::from_json_str(&json).expect("parse document");

    let mut restored = build_runner_sim();
    restored.apply_frame_document(&doc);
    restored
        .step_n(NUM_STEPS - SNAPSHOT_STEP)
        .expect("restored step_n");

    let sim_state = astrodyn::typed_bridge::trans_typed_to_raw(&restored.body(0).trans);
    assert_trans_eq(
        "Bevy uninterrupted vs runner snapshot/restore (SH 4x4 + RNP)",
        &bevy_state,
        &sim_state,
    );
}

#[test]
fn bevy_parity_frame_doc_export_eq_runner_export() {
    // Step both backends to the same instant and export from each: the
    // documents must agree uid-keyed and bit-for-bit. This is the test
    // the FrameEpochC stamping design must satisfy — the heterogeneous
    // per-frame epochs (registration-time for static frames, per-step
    // for model-driven / body frames) must match across backends.
    let mut sim = build_runner_sim();
    sim.step_n(SNAPSHOT_STEP).expect("runner step_n");
    let runner_doc = sim.export_frame_document();

    let (mut app, _vehicle) = build_bevy_app();
    step_bevy(&mut app, SNAPSHOT_STEP);
    let bevy_doc = astrodyn_bevy::frame_doc::export_frame_document(app.world_mut());

    assert_docs_eq("bevy export vs runner export", &bevy_doc, &runner_doc);
}

#[test]
fn bevy_parity_frame_doc_export_eq_runner_export_ephemeris() {
    // The SH+RNP scenario above has no DE-ephemeris source, which left
    // two hand-mirrored, mirror-critical paths unpinned cross-backend:
    // the `Origin::Derived { model: "DE4xx:{target}/{observer}" }`
    // string (each backend formats its own) and the per-step epoch
    // re-stamp that is gated on the source being ephemeris-driven.
    // Earth (central, static) + DE421 Moon pins both: the Moon's frame
    // record must come out ephemeris-Derived with a per-step epoch,
    // bit-equal across backends, while static Earth keeps its
    // registration-time epoch.
    const EPH_BODY_LABEL: &str = "bevy-parity-frame-doc-eph";
    let initial_moon_pos = moon_initial_pos();
    let mu_moon = astrodyn::MOON.shape.mu;
    let moon_source = || GravitySource {
        mu: mu_moon,
        model: GravityModel::PointMass,
    };

    // ── Runner ──
    let (mut sim, earth_idx) = new_sim_earth(DT);
    let moon_idx = sim.add_source(
        "Moon",
        GravitySourceEntry::new(
            moon_source(),
            astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(initial_moon_pos),
            None,
        ),
    );
    sim.set_source_ephemeris(moon_idx, EphemerisBody::Moon, EphemerisBody::Earth);
    sim.moon_source = Some(moon_idx);
    sim.ephemeris = Some(Ephemeris::from_bsp(&bsp_path()).expect("load DE421"));
    let mut moon_ctrl = GravityControl::new_spherical(moon_idx, GravityGradient::Skip);
    moon_ctrl.differential = true;
    sim.add_body(VehicleConfig {
        trans: iss_trans(),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(earth_idx, GravityGradient::Skip),
                moon_ctrl,
            ],
        },
        ..VehicleConfig::named(EPH_BODY_LABEL)
    });
    sim.validate().unwrap();
    sim.step_n(SNAPSHOT_STEP).expect("runner step_n");
    let runner_doc = sim.export_frame_document();

    // ── Bevy ──
    let mut app = new_bevy_app(DT);
    app.insert_resource(astrodyn_bevy::EphemerisR(
        Ephemeris::from_bsp(&bsp_path()).expect("load DE421"),
    ));
    let planet = spawn_earth_source(&mut app);
    let moon = app
        .world_mut()
        .spawn((
            astrodyn_bevy::FrameUidC(FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>()),
            Name::new("Moon"),
            MoonMarker,
            GravitySourceC(moon_source()),
            TranslationalStateC::<astrodyn::Earth>::from_untyped(TranslationalState {
                position: initial_moon_pos,
                velocity: DVec3::ZERO,
            }),
            SourceInertialPositionC(astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(
                initial_moon_pos,
            )),
            EphemerisBodyC {
                target: EphemerisBody::Moon,
                observer: EphemerisBody::Earth,
            },
        ))
        .id();
    let mut moon_ctrl = GravityControl::new_spherical(moon, GravityGradient::Skip);
    moon_ctrl.differential = true;
    app.world_mut().spawn((
        astrodyn_bevy::FrameUidC(astrodyn::named_body_frame_uid(EPH_BODY_LABEL)),
        TranslationalStateC::<astrodyn::Earth>::from(iss_trans()),
        DynamicsConfigC(astrodyn::DynamicsConfig {
            translational_dynamics: true,
            rotational_dynamics: false,
            three_dof: true,
        }),
        GravityControlsC(GravityControls {
            controls: vec![
                GravityControl::new_spherical(planet, GravityGradient::Skip),
                moon_ctrl,
            ],
        }),
    ));
    step_bevy(&mut app, SNAPSHOT_STEP);
    let bevy_doc = astrodyn_bevy::frame_doc::export_frame_document(app.world_mut());

    // Guard the pin itself: the Moon record must actually be
    // ephemeris-Derived (equality alone would also pass if both
    // backends misclassified it the same static way).
    let moon_uid = FrameUid::of::<astrodyn::PlanetInertial<astrodyn::Moon>>().to_string();
    let (_, moon_rec) = &by_uid(&runner_doc)[&moon_uid];
    assert!(
        matches!(&moon_rec.origin, Origin::Derived { model } if model == "DE4xx:Moon/Earth"),
        "Moon record must be ephemeris-derived, got {:?}",
        moon_rec.origin
    );

    assert_docs_eq(
        "bevy export vs runner export (DE421 Moon)",
        &bevy_doc,
        &runner_doc,
    );
}

#[test]
fn bevy_parity_frame_doc_apply_continue_eq_uninterrupted() {
    // Bevy export mid-run → fresh App → apply → continue ≡ the
    // uninterrupted Bevy run, bit-exact.
    let (mut app_a, vehicle_a) = build_bevy_app();
    step_bevy(&mut app_a, NUM_STEPS);
    let uninterrupted = read_trans(app_a.world(), vehicle_a);

    let (mut producer_app, _) = build_bevy_app();
    step_bevy(&mut producer_app, SNAPSHOT_STEP);
    let json =
        astrodyn_bevy::frame_doc::export_frame_document(producer_app.world_mut()).to_json_string();
    let doc = FrameDocument::from_json_str(&json).expect("parse document");

    let (mut restored, vehicle_b) = build_bevy_app();
    // Registration must run before apply (sequencing precondition).
    restored.update();
    astrodyn_bevy::frame_doc::apply_frame_document::<astrodyn::Earth>(restored.world_mut(), &doc);
    step_bevy(&mut restored, NUM_STEPS - SNAPSHOT_STEP);
    let continued = read_trans(restored.world(), vehicle_b);

    assert_trans_eq(
        "Bevy snapshot/restore vs uninterrupted Bevy",
        &continued,
        &uninterrupted,
    );
}

#[test]
fn bevy_parity_frame_doc_cross_restore_both_directions() {
    // The interchangeability statement: a document from either backend
    // restores into the other and continues bit-identically to the
    // uninterrupted other backend.
    // Reference: uninterrupted runs of both backends.
    let mut sim_ref = build_runner_sim();
    sim_ref.step_n(NUM_STEPS).expect("runner reference");
    let runner_ref = astrodyn::typed_bridge::trans_typed_to_raw(&sim_ref.body(0).trans);
    let (mut app_ref, vehicle_ref) = build_bevy_app();
    step_bevy(&mut app_ref, NUM_STEPS);
    let bevy_ref = read_trans(app_ref.world(), vehicle_ref);

    // Direction 1: runner-export → bevy-apply → continue in Bevy.
    let mut producer = build_runner_sim();
    producer.step_n(SNAPSHOT_STEP).expect("runner producer");
    let runner_doc = producer.export_frame_document();
    let (mut bevy_restored, vehicle) = build_bevy_app();
    bevy_restored.update();
    astrodyn_bevy::frame_doc::apply_frame_document::<astrodyn::Earth>(
        bevy_restored.world_mut(),
        &runner_doc,
    );
    step_bevy(&mut bevy_restored, NUM_STEPS - SNAPSHOT_STEP);
    assert_trans_eq(
        "runner-export → bevy-apply vs uninterrupted Bevy",
        &read_trans(bevy_restored.world(), vehicle),
        &bevy_ref,
    );

    // Direction 2: bevy-export → runner-apply → continue in the runner.
    let (mut bevy_producer, _) = build_bevy_app();
    step_bevy(&mut bevy_producer, SNAPSHOT_STEP);
    let bevy_doc = astrodyn_bevy::frame_doc::export_frame_document(bevy_producer.world_mut());
    let mut runner_restored = build_runner_sim();
    runner_restored.apply_frame_document(&bevy_doc);
    runner_restored
        .step_n(NUM_STEPS - SNAPSHOT_STEP)
        .expect("runner restored");
    assert_trans_eq(
        "bevy-export → runner-apply vs uninterrupted runner",
        &astrodyn::typed_bridge::trans_typed_to_raw(&runner_restored.body(0).trans),
        &runner_ref,
    );
}

// ──────────────────── Fail-loud guarantees (Bevy half) ────────────────────
// Plain-named (non-`bevy_parity_`) tests: they pin the document layer's
// panic paths, not cross-backend parity, and run in the unit/Tier-2 CI
// bucket. The runner half lives in
// `astrodyn_verif_jeod/tests/tier3_frame_doc.rs`.

/// Export refuses to *silently omit* an entity that carries frame state
/// but no identity — an entity that bypassed registration must be a loud
/// diagnostic, not a missing record (issue #664 review).
#[test]
#[should_panic(expected = "bypassed identity stamping")]
fn export_panics_on_identityless_frame_state_entity() {
    let (mut app, _vehicle) = build_bevy_app();
    step_bevy(&mut app, 1);
    // Frame state spawned around registration: no FrameUidC ever stamped.
    app.world_mut().spawn(astrodyn_bevy::FrameTransC {
        position: glam::DVec3::ZERO,
        velocity: glam::DVec3::ZERO,
    });
    let _ = astrodyn_bevy::frame_doc::export_frame_document(app.world_mut());
}

/// Retired pfix frames are the one *sanctioned* identityless frame-state
/// entity: export excludes them from the snapshot without tripping the
/// fail-loud sweep.
#[test]
fn export_excludes_retired_pfix_frame() {
    let (mut app, _vehicle) = build_bevy_app();
    step_bevy(&mut app, 1);
    let pfix_uid = FrameUid::of::<astrodyn::PlanetFixed<astrodyn::Earth>>().to_string();
    let doc = astrodyn_bevy::frame_doc::export_frame_document(app.world_mut());
    assert!(
        by_uid(&doc).contains_key(&pfix_uid),
        "while rotating, the pfix frame is part of the snapshot"
    );

    // Toggle the rotation model off: the pfix frame retires (identity
    // stripped, entity kept alive for reuse).
    let planet = {
        let mut q = app
            .world_mut()
            .query_filtered::<Entity, With<GravitySourceC>>();
        q.iter(app.world()).next().expect("one gravity source")
    };
    app.world_mut()
        .entity_mut(planet)
        .insert(astrodyn_bevy::RotationModelC(astrodyn::RotationModel::None));
    step_bevy(&mut app, 2);

    let doc = astrodyn_bevy::frame_doc::export_frame_document(app.world_mut());
    assert!(
        !by_uid(&doc).contains_key(&pfix_uid),
        "a retired pfix frame is excluded from the snapshot, not exported \
         as a phantom record"
    );
}

/// Restore targets a freshly built App: a pre-advanced clock must refuse
/// the document instead of silently double-advancing time.
#[test]
#[should_panic(expected = "restore targets a freshly built App")]
fn bevy_apply_panics_when_clock_not_zero() {
    let (mut producer, _) = build_bevy_app();
    step_bevy(&mut producer, 1);
    let doc = astrodyn_bevy::frame_doc::export_frame_document(producer.world_mut());

    let (mut app, _) = build_bevy_app();
    step_bevy(&mut app, 1); // clock no longer at zero
    astrodyn_bevy::frame_doc::apply_frame_document::<astrodyn::Earth>(app.world_mut(), &doc);
}

/// Derived time scales are functions of the epoch: a document produced
/// under a different `tai_tjt_at_epoch` must be refused, never silently
/// reinterpreted.
#[test]
#[should_panic(expected = "time-epoch mismatch")]
fn bevy_apply_panics_on_epoch_mismatch() {
    let (mut producer, _) = build_bevy_app();
    step_bevy(&mut producer, 1);
    let mut doc = astrodyn_bevy::frame_doc::export_frame_document(producer.world_mut());
    doc.header.tai_tjt_at_epoch += 1.0;

    let (mut app, _) = build_bevy_app();
    app.update();
    astrodyn_bevy::frame_doc::apply_frame_document::<astrodyn::Earth>(app.world_mut(), &doc);
}

/// Apply never reparents: a record whose declared parent disagrees with
/// the rebuilt hierarchy is a loud topology inconsistency (RF.02).
#[test]
#[should_panic(expected = "topology mismatch")]
fn bevy_apply_panics_on_topology_mismatch() {
    let (mut producer, _) = build_bevy_app();
    step_bevy(&mut producer, 1);
    let mut doc = astrodyn_bevy::frame_doc::export_frame_document(producer.world_mut());

    // Redeclare the body's parent as Earth.pfix (its real parent is the
    // root): the rebuilt App's hierarchy must win, loudly.
    let body_uid = astrodyn::named_body_frame_uid(BODY_LABEL);
    let pfix_uid = FrameUid::of::<astrodyn::PlanetFixed<astrodyn::Earth>>();
    let pfix_idx = doc
        .uids
        .iter()
        .position(|u| *u == pfix_uid)
        .expect("pfix identity in the uid table");
    let pfix_idx = u32::try_from(pfix_idx).expect("uid table fits u32");
    for rec in &mut doc.records {
        if doc.uids[rec.uid_index as usize] == body_uid {
            rec.parent = Some(pfix_idx);
        }
    }

    let (mut app, _) = build_bevy_app();
    app.update();
    astrodyn_bevy::frame_doc::apply_frame_document::<astrodyn::Earth>(app.world_mut(), &doc);
}

/// A record whose identity is unknown to the rebuilt App is a loud
/// population mismatch, never a skipped record.
#[test]
#[should_panic(expected = "no such frame")]
fn bevy_apply_panics_on_unknown_identity() {
    let (mut producer, _) = build_bevy_app();
    step_bevy(&mut producer, 1);
    let mut doc = astrodyn_bevy::frame_doc::export_frame_document(producer.world_mut());

    let body_uid = astrodyn::named_body_frame_uid(BODY_LABEL);
    let body_idx = doc
        .uids
        .iter()
        .position(|u| *u == body_uid)
        .expect("body identity in the uid table");
    doc.uids[body_idx] = astrodyn::named_body_frame_uid("no-such-body");

    let (mut app, _) = build_bevy_app();
    app.update();
    astrodyn_bevy::frame_doc::apply_frame_document::<astrodyn::Earth>(app.world_mut(), &doc);
}
