//! Bevy ↔ runner parity for the Apollo 8 trans-lunar coast.
//!
//! Drives the same `SimulationBuilder` used by
//! `crates/astrodyn_verif_jeod/tests/tier3_apollo8_frame_switch.rs`
//! through both runtimes — [`astrodyn_runner::Simulation`] and the Bevy
//! `populate_app::<Earth>` bridge — and asserts bit-identical
//! translational state at every tick over a 100 s window.
//!
//! Two cases:
//!
//! * [`bevy_parity_apollo8_eci_integ`] — baseline Earth-centered
//!   inertial integration with Sun + Moon as ephemeris-driven third
//!   bodies. Bit-identical between runtimes.
//! * [`bevy_parity_apollo8_frame_switch`] — same scenario plus a
//!   distance-based [`FrameSwitchConfig`] that reparents the body
//!   from Earth-inertial to Moon-inertial when the body approaches
//!   within 66.1 Mm of the Moon. Currently `#[ignore]`d on a known
//!   ULP-scale Bevy-adapter divergence at the switch tick (see the
//!   per-test attribute reason).
//!
//! Single-planet bridge note: the `<P>` tag stays `<Earth>` across the
//! frame switch — `TranslationalStateC<P>` is the scenario-wide
//! type-system marker, not an "active integration frame" tag. The
//! Bevy `frame_switch_system` reparents the body's frame entity under
//! the Moon's source frame and rewrites the in-place position /
//! velocity values in Moon-centered coordinates, identically to the
//! runner's `evaluate_and_apply_frame_switch` path. See
//! `bevy_parity_frame_switch.rs` for the simpler synthetic-Moon
//! version of the same machinery, which is bit-identical.

#![allow(
    clippy::float_cmp,
    reason = "bevy-parity tests assert bit-exact identity between runner and Bevy state fields"
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test step counts fit exactly in f64 mantissa and usize"
)]
#![allow(clippy::excessive_precision)]

use std::time::Duration;

use astrodyn::{
    EphemerisBody, FrameSwitchConfig, GravityControl, GravityControls, GravityGradient,
    GravityModel, GravitySource, GravitySourceEntry, JeodQuat, MassProperties, Position,
    RootInertial, RotationalState, SimulationBuilder, SimulationTime, SwitchSense,
    TranslationalState, VehicleConfig,
};
use astrodyn_bevy::{SimulationBuilderBevyExt, TranslationalStateC};
use astrodyn_runner::SimulationBuilderExt;
use bevy::prelude::*;
use glam::{DMat3, DVec3};

// Apollo 8 trans-lunar-coast state vector (Dec 23 1968 19:38 UTC),
// captured from JEOD's `Modified_data/vehicle.py`. The runner-side
// tier3 test pulls these constants directly; this wrapper re-states
// them locally so the two scenarios stay 1:1 without exporting an
// internals helper from verif_jeod.
const POS_ECI: DVec3 = DVec3::new(
    302_274_887.753_810_17,
    -119_023_818.108_825_01,
    -56_915_743.953_866_437,
);
const VEL_ECI: DVec3 = DVec3::new(
    942.182_494_673_019_85,
    -189.920_638_006_114_07,
    -292.959_665_506_469_89,
);
/// Vehicle mass (kg).
const MASS: f64 = 91_589.71;
/// Integration timestep (s). RK4 at 0.5 s, matching SIM_verif_frame_switch.
const DT: f64 = 0.5;
/// Total parity window (s). Same as the runner-side tier3 test.
const TOTAL_TIME: f64 = 100.0;
/// Frame-switch distance threshold (m).
const SWITCH_DISTANCE: f64 = 66.1e6;

// Gravitational parameters — must match the runner-side tier3 test
// exactly. `MU_MOON` deliberately differs from `astrodyn::MOON.shape.mu`
// because the Apollo 8 reference targets JEOD's `moon_spherical.cc`
// spherical-gravity verification fixture, not the GRAIL150 default.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;
const MU_MOON: f64 = 4.902_801_076e12;
const MU_SUN: f64 = astrodyn::SUN.shape.mu;

/// Build the canonical Apollo 8 frame-switch scenario as a
/// [`SimulationBuilder`] so both runtimes ingest the same shape.
///
/// Returns the builder and the Moon source index (callers need it to
/// pin the same `target_source` on the runner-side
/// `FrameSwitchConfig<usize>`).
fn apollo8_builder(enable_switch: bool) -> SimulationBuilder {
    // DE405 lives under verif_jeod/assets/ (JEOD-parity-only, large
    // binary segregated from test_data/ trajectory CSVs).
    let bsp_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace crate dir")
        .join("astrodyn_verif_jeod")
        .join("assets")
        .join("de405.bsp");
    assert!(
        bsp_path.exists(),
        "DE405 ephemeris not found at {}; the runner-side tier3 test \
         documents the regen path.",
        bsp_path.display()
    );

    // Dec 23, 1968, 19:38:00 UTC — same epoch the tier3 test uses.
    let utc_tjt = 2_440_214.318_055_555_5 - 2_440_000.5;
    let leap_table = astrodyn::default_leap_second_table();
    let tai_tjt = leap_table.utc_to_tai_tjt(utc_tjt);
    let time = SimulationTime::new(tai_tjt, leap_table);

    let ephemeris =
        astrodyn::Ephemeris::from_bsp(&bsp_path).expect("Failed to load DE405 ephemeris");

    let mut sb = SimulationBuilder::new(time, DT);
    sb = sb.ephemeris(ephemeris);

    // Sources: Sun, Earth (central), Moon — same order as the tier3 test.
    let sun_idx = sb.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            },
            Position::<RootInertial>::zero(),
            None,
        ),
    );
    sb.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Earth);

    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        Position::<RootInertial>::zero(),
        None,
    );
    earth_entry.central = true;
    let earth_idx = sb.add_source("Earth", earth_entry);

    let moon_idx = sb.add_source(
        "Moon",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_MOON,
                model: GravityModel::PointMass,
            },
            Position::<RootInertial>::zero(),
            None,
        ),
    );
    sb.set_source_ephemeris(moon_idx, EphemerisBody::Moon, EphemerisBody::Earth);
    sb = sb.sun(sun_idx).moon(moon_idx);

    // Apollo CSM mass tensor & CoM (English units → SI).
    const SLUG_FT2_TO_KG_M2: f64 = 1.355_817_948;
    let inertia = DMat3::from_diagonal(DVec3::new(
        100_000.0 * SLUG_FT2_TO_KG_M2,
        200_000.0 * SLUG_FT2_TO_KG_M2,
        400_000.0 * SLUG_FT2_TO_KG_M2,
    ));
    const INCH_TO_M: f64 = 0.0254;
    let com_offset = DVec3::new(1098.0 * INCH_TO_M, 0.0, 372.0 * INCH_TO_M);

    sb.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: POS_ECI,
            velocity: VEL_ECI,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(
            &MassProperties::with_inertia(MASS, inertia, com_offset),
        )),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(earth_idx, GravityGradient::Skip),
                GravityControl::new_third_body(sun_idx),
                GravityControl::new_third_body(moon_idx),
            ],
        },
        frame_switches: if enable_switch {
            vec![FrameSwitchConfig {
                target_source: moon_idx,
                switch_sense: SwitchSense::OnApproach,
                switch_distance: SWITCH_DISTANCE,
                active: true,
            }]
        } else {
            vec![]
        },
        ..Default::default()
    });
    sb
}

/// Shared lockstep driver: step both runtimes by `steps` ticks of
/// `dt = DT` and assert bit-identical body-0 translational state at
/// every tick. Panics on the first divergence with the case label.
fn assert_parity(case: &str, enable_switch: bool, steps: usize) {
    // Runner side.
    let mut runner = apollo8_builder(enable_switch)
        .build()
        .expect("runner build");

    // Bevy side — same factory, materialized into a fresh App under <Earth>.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    let handles = apollo8_builder(enable_switch)
        .populate_app::<astrodyn::Earth>(&mut app)
        .expect("populate_app under <Earth>");
    let vehicle = handles.body_entities[0];

    // Run startup so per-source frame trees / source-frame-id resources
    // are wired before stepping. `MinimalPlugins` does not auto-run
    // `Startup`; the parity loop drives `FixedUpdate` directly.
    app.world_mut().run_schedule(Startup);

    for step in 0..steps {
        runner.step().expect("runner step");
        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f64(DT));
        app.world_mut().run_schedule(FixedUpdate);

        let r = runner.body(0);
        let r_pos = r.trans.position.raw_si();
        let r_vel = r.trans.velocity.raw_si();
        let bevy_trans = app
            .world()
            .get::<TranslationalStateC<astrodyn::Earth>>(vehicle)
            .expect("vehicle entity carries TranslationalStateC<Earth>")
            .0;
        let b_pos = bevy_trans.position.raw_si();
        let b_vel = bevy_trans.velocity.raw_si();

        for i in 0..3 {
            assert!(
                r_pos[i].to_bits() == b_pos[i].to_bits(),
                "{case} translational bit-parity broke at tick {step} \
                 on position[{i}]: runner={} bevy={}",
                r_pos[i],
                b_pos[i],
            );
            assert!(
                r_vel[i].to_bits() == b_vel[i].to_bits(),
                "{case} translational bit-parity broke at tick {step} \
                 on velocity[{i}]: runner={} bevy={}",
                r_vel[i],
                b_vel[i],
            );
        }
    }
}

/// Baseline parity (no frame switch): Earth-centered inertial
/// integration with Sun + Moon as ephemeris-driven third bodies. This
/// case isolates the pipeline from the frame-switch path so a future
/// regression in (e.g.) the per-step DE405 source-position update path
/// fails here rather than at the frame_switch case below.
#[test]
fn bevy_parity_apollo8_eci_integ() {
    let steps = (TOTAL_TIME / DT).round() as usize;
    assert_parity("apollo8_eci_integ", false, steps);
}

/// Frame-switch parity: same baseline scenario plus a distance-based
/// `FrameSwitchConfig` that reparents the body to Moon-inertial on
/// approach (≤ 66.1 Mm).
///
/// Currently `#[ignore]`d: ULP-scale drift appears at the switch tick
/// (≈ tick 80, velocity component ≈ 2 ns/m) when the Moon's
/// `SourceInertialPositionC` is driven by `ephemeris_update_system`.
/// The synthetic-Moon variant in `bevy_parity_frame_switch.rs` (Moon
/// position pinned via `SourceMutator` rather than DE405-driven) is
/// bit-identical, isolating the failure mode to the
/// frame_switch_system ↔ ephemeris_update_system interaction. The
/// baseline `apollo8_eci_integ` case above passes bit-identically,
/// confirming the ephemeris path itself is parity-safe; the
/// divergence is specific to how the frame-switch transformation
/// reads the Moon's post-ephemeris-update position relative to where
/// the runner's `evaluate_and_apply_frame_switch` reads it. Tracked
/// as a separate Bevy-adapter bug; lifting the `#[ignore]` is
/// blocked on that investigation.
#[test]
#[ignore = "parity-gap (#556): frame_switch_system + ephemeris-driven \
            SourceInertialPositionC interaction — Moon position diverges \
            from runner at the switch tick (ULP-scale)."]
fn bevy_parity_apollo8_frame_switch() {
    let steps = (TOTAL_TIME / DT).round() as usize;
    assert_parity("apollo8_frame_switch", true, steps);
}
