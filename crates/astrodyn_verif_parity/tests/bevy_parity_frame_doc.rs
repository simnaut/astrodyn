//! Bevy-vs-Simulation parity for the frame-document restore path
//! (issue #663).
//!
//! The Bevy adapter has no document IO until PR-5 (#664), so this wrapper
//! pins the property the document layer must not break: a runner that
//! serializes mid-run, rebuilds, applies the document, and continues must
//! land **bit-identical** to the *uninterrupted Bevy run*. Combined with
//! the existing uninterrupted runner ↔ Bevy parity suite, this proves the
//! snapshot/restore cycle is invisible to cross-backend parity.
//!
//! Scenario mirrors `bevy_parity_highfidelity_sh4x4_rnp` (GGM02C 4×4
//! spherical harmonics + EarthRNP), whose gravity consumes the
//! matrix-canonical pfix rotation each step — a single-ULP error in the
//! restored rotation diverges the trajectory.

mod common;

use astrodyn::{
    GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource,
    GravitySourceEntry, VehicleConfig,
};
use astrodyn_bevy::{
    DynamicsConfigC, GravityControlsC, GravitySourceC, IntegrationDtR, PlanetFixedRotationC,
    SourceInertialPositionC, TranslationalStateC,
};
use astrodyn_runner::RotationModel;
use bevy::prelude::*;
use glam::DMat3;

use common::*;

/// Steps before the runner's snapshot; the remainder of [`NUM_STEPS`]
/// runs after the restore.
const SNAPSHOT_STEP: usize = 50;

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
        ..VehicleConfig::named("bevy-parity-frame-doc")
    });
    sim.validate().unwrap();
    sim
}

#[test]
fn bevy_parity_frame_doc_restore_matches_uninterrupted_bevy() {
    // ── Bevy: uninterrupted ──
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(Time::<Fixed>::from_seconds(DT));
    app.insert_resource(IntegrationDtR(DT));
    app.add_plugins(astrodyn_bevy::AstrodynPlugin);

    let planet = app
        .world_mut()
        .spawn((
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
    step_bevy(&mut app, NUM_STEPS);
    let bevy_state = read_trans(app.world(), vehicle);

    // ── Runner: snapshot at SNAPSHOT_STEP, restore, continue ──
    let mut producer = build_runner_sim();
    producer.step_n(SNAPSHOT_STEP).expect("producer step_n");
    let json = producer.export_frame_document().to_json_string();
    let doc = astrodyn::frame_doc::FrameDocument::from_json_str(&json).expect("parse document");

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
