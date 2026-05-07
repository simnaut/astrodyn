//! Tier 3: SIM_Apollo trajectory cross-validation through 9 detaches + 2 attaches.
//!
//! Reproduces JEOD's `sims/SIM_Apollo/SET_test/RUN_test` 12-second
//! initialization-only sim and cross-validates `cm_dyn`'s `core_body`
//! trajectory against the reference CSV. The sim has 11 scheduled
//! `add_read` events at integer seconds — 9 detaches and 2 attaches.
//! The full event sequence is applied to our mass tree (so the pipeline
//! exercises all 11 events end-to-end) via `Simulation::detach_subtree`
//! and `Simulation::attach_subtree_aligned`. `attach_subtree_aligned`
//! ports JEOD's `DynBody::attach_child` momentum-conservation algorithm
//! into [`astrodyn_dynamics::attach::combine_states_at_attach`], with full
//! struct↔body-frame distinctions per
//! `MassProperties::t_parent_this` (set per body from
//! `Modified_data/mass/*.py:pt_orientation` — `yaw_180` for CM/LES/DM/LM,
//! identity for SM/S1/S2/S3).
//!
//! Trajectory diffs are asserted through the full 12 s sim — all 11
//! attach/detach events execute and the CSM `core_body` trajectory is
//! compared against JEOD's recorded reference at every 0.1 s sample.
//! Residuals are at numerical-precision limits everywhere: ≲ 7 µm
//! position, ≲ 3 µm/s velocity, ≲ 4 µrad attitude, ≲ 14 µrad/s ang_vel.
//! That level of agreement holds across both the t=6 SM→CM attach
//! (which matches JEOD's logged composite-body angular velocity of
//! −1.7207 rad/s exactly) and the t=9 AttLmCm2 / t=10 DetLm3 sequence
//! that previously produced "larger rotation drift" before the
//! `composite_body`-integration refactor (commit `bd279c2`) and the
//! `step_ballistic` quaternion-multiply-order fix (routed through
//! `BodyAttitude::advance_under_body_rate` after issue #252).
//!
//! ### JEOD source-defect note
//!
//! `sims/SIM_Apollo/SET_test/RUN_test/input.py` calls
//! `set_vehicle_grav_controls()` only on `les_dyn` and never on
//! `cm_dyn` (the integration root after launch_stack assembly). As
//! shipped, JEOD's recorded trajectory is therefore essentially
//! gravity-free. The Docker reference-regen wrapper
//! (`trick/generate_references.sh:run_apollo_group`) injects the
//! missing `set_vehicle_grav_controls(cm_dyn)` + `set_vehicle_sv_at_earth(cm_dyn, earth)`
//! calls before the sim runs, restoring the 8x8 GGM05C + Moon/Sun
//! gravity that the per-vehicle data files (`Modified_data/vehicle/grav_controls.py`,
//! `Modified_data/vehicle/sv_at_earth.py`) clearly intend. This test
//! therefore validates against the *intended* JEOD configuration
//! rather than the as-shipped (broken) one.
//!
//! ### Scope
//!
//! - Initial state: from `apollo_trajectory.csv` row 0 (= JEOD's
//!   `cm_dyn.dyn_body.core_body.state` after launch_stack assembly,
//!   in Earth.inertial). Equivalent to the LEO LVLH-aligned state
//!   from `Modified_data/state/sv_leo_lvlh.py` shifted by the
//!   structure-to-composite offset.
//! - Epoch: 1969-07-16 13:44:00 UTC (Apollo 11 launch date), with
//!   `Leap_Second.dat` overridden to `TAI-UTC = 4.2 s` (per
//!   `Modified_data/date_n_time/UTC_16Jul1969.py`) and
//!   `UT1-TAI = 0.0115221 - 4.2 s`.
//! - Physics: 8x8 GGM05C Earth (RNP rotation), spherical Moon, spherical
//!   Sun, RK4 at `dt = 0.02 s` (DYNAMICS = 50 Hz from `S_define:72`).
//! - Mass tree: full 8-body Apollo stack (S1, S2, S3, LES, CM, SM, LM,
//!   DM) assembled via launch_stack, then 11 attach/detach events at
//!   `t = 1..11 s` per `RUN_test/input.py`.
//!
//! Mass-tree composite property validation at each phase is covered by
//! `crates/astrodyn_dynamics/tests/tier3_apollo_mass_tree.rs`; this test
//! complements it by exercising the full `Simulation::step()` pipeline
//! end-to-end through the same event sequence.

use astrodyn::{
    AtmosphereConfig, AtmosphereModel, GravityControl, GravityControls, GravityModel,
    GravitySource, MetAtmosphere, RotationalState, SimulationBuilder, SimulationTime,
    TranslationalState, EARTH,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_atmosphere::met::GeoIndexType;
use astrodyn_dynamics::{MassBodyId, MassProperties, MassTree};
use astrodyn_math::JeodQuat;
use astrodyn_runner::{RotationModel, Simulation, SimulationBuilderExt};
use astrodyn_verif_jeod::apollo_truth::{
    load_apollo_attach_truth, nearest_truth_at, ApolloTruthError, ApolloTruthRow,
};
use astrodyn_verif_jeod::crossval::{CrossvalReport, StateLog};
use glam::{DMat3, DVec3};
use std::path::PathBuf;

// ── JEOD source constants ────────────────────────────────────────────
//
// The body's initial state comes from CSV row 0 rather than a hardcoded
// constant: JEOD's `Modified_data/state/sv_leo_lvlh.py` sets the
// composite_body state at t=0, but the snippet logs core_body (which
// differs by the structure→composite offset times a rotation). Reading
// CSV row 0 keeps the test self-consistent with whatever frame the
// snippet is logging.

/// `S_define:72` — `#define DYNAMICS 0.02`.
const DT: f64 = 0.02;
/// `RUN_test/input.py:350` — `exec_set_terminate_time(12.0)`.
const SIM_DURATION_S: f64 = 12.0;

/// Trajectory comparison window: full 12 s sim. Asserts every 0.1 s
/// sample through all 11 attach/detach events (5 stage detaches, the
/// t=6 SM→CM attach, the t=7 LM detach, t=8 DM detach, t=9 LM
/// re-attach, t=10 LM detach, t=11 SM detach). See the test header
/// for the residual budget.
const TRAJECTORY_VALIDATION_END_S: f64 = 12.0;

/// `Modified_data/Earth/params.py` — Earth rotation rate.
const OMEGA_EARTH: f64 = 7.292_115_146_706_388e-5;

/// `Modified_data/vehicle/sv_at_earth.py` — earth gravity 8x8.
const GRAV_DEGREE: usize = 8;
/// `Modified_data/vehicle/sv_at_earth.py` — earth gravity 8x8.
const GRAV_ORDER: usize = 8;

// ── Unit conversions for JEOD English-unit mass data ─────────────────

const LB_TO_KG: f64 = 0.453_592_37;
const FT_TO_M: f64 = 0.3048;
const LB_FT2_TO_KG_M2: f64 = LB_TO_KG * FT_TO_M * FT_TO_M;

// ── Helpers ──────────────────────────────────────────────────────────

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data")
}

fn yaw_180() -> DMat3 {
    DMat3::from_cols(
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, -1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    )
}

/// Apollo per-body mass properties from `Modified_data/mass/*.py`.
/// `mass_lb` in pounds, `cm_x_ft` in feet (Y/Z = 0), inertia in lb·ft².
/// `t_struct_to_body` per JEOD `pt_orientation` — `yaw_180` for the CM,
/// LES, DM, and Ascent module (each declares `eigen_angle = 180°` about
/// Z); identity for SM, S1, S2, S3.
fn apollo_mass(
    mass_lb: f64,
    cm_x_ft: f64,
    ixx: f64,
    iyy: f64,
    izz: f64,
    t_struct_to_body: DMat3,
) -> MassProperties {
    MassProperties::with_inertia(
        mass_lb * LB_TO_KG,
        DMat3::from_diagonal(DVec3::new(
            ixx * LB_FT2_TO_KG_M2,
            iyy * LB_FT2_TO_KG_M2,
            izz * LB_FT2_TO_KG_M2,
        )),
        DVec3::new(cm_x_ft * FT_TO_M, 0.0, 0.0),
    )
    .with_t_parent_this(t_struct_to_body)
}

/// Per-body baseline definitions and named attachment points, ported from
/// `Modified_data/mass/*.py` and `Modified_data/attach/*.py`. Shared with
/// `crates/astrodyn_dynamics/tests/tier3_apollo_mass_tree.rs` (kept inline here
/// to avoid pulling astrodyn_dynamics tests into the runner crate's dep graph).
struct BodyIds {
    cm: MassBodyId,
    sm: MassBodyId,
    lm: MassBodyId,
    dm: MassBodyId,
    s3: MassBodyId,
    s2: MassBodyId,
    s1: MassBodyId,
    les: MassBodyId,
}

/// Apply the seven `Modified_data/attach/launch_stack.py` attachments
/// to a freshly built tree.
fn assemble_launch_stack(tree: &mut MassTree, ids: &BodyIds) {
    tree.attach_aligned(
        ids.dm,
        "Ascent Module interface",
        ids.lm,
        "Descent Module interface",
    );
    tree.attach_aligned(ids.sm, "CM interface", ids.cm, "SM interface");
    tree.attach_aligned(ids.s3, "LEM/SM/CM interface", ids.sm, "Stage 3 interface");
    tree.attach_aligned(ids.lm, "Stage 3 interface", ids.s3, "LEM/SM/CM interface");
    tree.attach_aligned(ids.s2, "Stage 3 interface", ids.s3, "Stage 2 interface");
    tree.attach_aligned(ids.s1, "Stage 2 interface", ids.s2, "Stage 1 interface");
    tree.attach_aligned(ids.les, "CM interface", ids.cm, "CM docking port");
}

/// `Modified_data/date_n_time/UTC_16Jul1969.py` — 1969-07-16 13:44:00 UTC,
/// leap_sec_override = 4.2 s, tai_to_ut1_override = 0.0115221 - 4.2.
fn apollo_time() -> SimulationTime {
    // JD(1969-07-16 0h UT) = 2440418.5 → TJT = 418.0 (TJT = JD - 2440000.5).
    // 13h44m = 49440 s = 0.572222... days.
    let utc_tjt = 418.0 + (13.0 * 3600.0 + 44.0 * 60.0) / 86_400.0;
    // SIM_Apollo overrides TAI-UTC to 4.2 s instead of the historical
    // value (which differs at this epoch). Hand-roll the conversion so
    // we don't rely on the leap-second table for this date.
    let tai_tjt = utc_tjt + 4.2 / 86_400.0;
    let mut time = SimulationTime::new(tai_tjt, astrodyn::default_leap_second_table());
    // tai_to_ut1_override_val = 0.0115221 - 4.2 = UT1-TAI offset.
    time.set_ut1_tai_offset(0.011_522_1 - 4.2);
    time
}

/// Apollo CSV reference state at one logged timestamp.
struct ApolloRef {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    quaternion: JeodQuat,
    ang_vel_body: DVec3,
}

fn load_apollo_reference() -> Vec<ApolloRef> {
    let csv_path = test_data_dir().join("apollo_trajectory.csv");
    assert!(
        csv_path.exists(),
        "apollo_trajectory.csv missing at {}. Generate with: cargo xtask regenerate-tier3",
        csv_path.display()
    );
    let content = std::fs::read_to_string(&csv_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", csv_path.display()));

    // Column layout (per APOLLO_SNIPPET in trick/generate_references.sh):
    //   0 time
    //   1 pos[0], 2 vel[0], 3 pos[1], 4 vel[1], 5 pos[2], 6 vel[2]
    //   7 q.scalar, 8-10 q.vec[0..2], 11-13 ang_vel[0..2]
    let mut out = Vec::new();
    // Parse positionally and panic on any column-count or parse error
    // rather than silently skipping rows: this is verification data,
    // and a corrupted reference trajectory should fail loudly, not
    // shift column indices and produce subtly-wrong test results.
    for (row_idx, line) in content.lines().skip(1).enumerate() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        // CSV row index in the source file (1-indexed; +2 to account
        // for skipping the header).
        let csv_row = row_idx + 2;
        assert_eq!(
            fields.len(),
            14,
            "{}:{csv_row} apollo_trajectory.csv: expected 14 columns, got {}: {line:?}",
            csv_path.display(),
            fields.len(),
        );
        let parse = |col: usize| -> f64 {
            fields[col].parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "{}:{csv_row} apollo_trajectory.csv: failed to parse column {col} ({:?}): {e}",
                    csv_path.display(),
                    fields[col],
                )
            })
        };
        out.push(ApolloRef {
            time: parse(0),
            position: DVec3::new(parse(1), parse(3), parse(5)),
            velocity: DVec3::new(parse(2), parse(4), parse(6)),
            // JEOD scalar-first [q0,q1,q2,q3] — store with same convention.
            quaternion: JeodQuat::new(parse(7), parse(8), parse(9), parse(10)),
            ang_vel_body: DVec3::new(parse(11), parse(12), parse(13)),
        });
    }
    out
}

// ── Mass-tree event schedule (RUN_test/input.py:230..345) ────────────

#[derive(Debug, Clone, Copy)]
enum Event {
    DetachS1,
    DetachS2,
    DetachLes,
    DetachS3,
    DetachLm,    // also: print "LEM_Sep" + "Apollo"
    AttachLmCm,  // attach lm under cm via "LM docking port" / "CM docking port"
    DetachLm2,   // also: print "LM_Descent" + "Lunar_Orbit"
    DetachDm,    // print "LM_Ascent"
    AttachLmCm2, // attach lm again under cm — print "Lunar_Rendezvous"
    DetachLm3,   // print "Return"
    DetachSm,    // print "Entry" + "Final"
}

const EVENTS: &[(f64, Event)] = &[
    (1.0, Event::DetachS1),
    (2.0, Event::DetachS2),
    (3.0, Event::DetachLes),
    (4.0, Event::DetachS3),
    (5.0, Event::DetachLm),
    (6.0, Event::AttachLmCm),
    (7.0, Event::DetachLm2),
    (8.0, Event::DetachDm),
    (9.0, Event::AttachLmCm2),
    (10.0, Event::DetachLm3),
    (11.0, Event::DetachSm),
];

fn apply_event(sim: &mut Simulation, body_idx: usize, ids: &BodyIds, event: Event) {
    match event {
        Event::DetachS1 => sim.detach_subtree(body_idx, ids.s1),
        Event::DetachS2 => sim.detach_subtree(body_idx, ids.s2),
        Event::DetachLes => sim.detach_subtree(body_idx, ids.les),
        Event::DetachS3 => sim.detach_subtree(body_idx, ids.s3),
        Event::DetachLm | Event::DetachLm2 | Event::DetachLm3 => {
            sim.detach_subtree(body_idx, ids.lm)
        }
        Event::DetachDm => sim.detach_subtree(body_idx, ids.dm),
        Event::DetachSm => sim.detach_subtree(body_idx, ids.sm),
        Event::AttachLmCm | Event::AttachLmCm2 => sim.attach_subtree_aligned(
            body_idx,
            ids.lm,
            "LM docking port",
            ids.cm,
            "CM docking port",
        ),
    }
}

// ── Test ─────────────────────────────────────────────────────────────

fn build_apollo_sim() -> (Simulation, usize, BodyIds) {
    // Earth: 8x8 GGM05C non-spherical, with the Earth-RNP rotation model so
    // the planet-fixed frame updates each step (matches JEOD's
    // `earth_GGM05C_MET_RNP.sm`).
    let earth_grav = astrodyn_gravity::fixtures::load_ggm05c();
    let mu_moon = astrodyn_gravity::fixtures::load_moon_grail150_mu();
    let mu_sun = astrodyn_gravity::fixtures::load_sun_spherical_mu();

    // Note: SIM_Apollo's Modified_data/Earth/params.py overrides Earth mu
    // to the historic 3.98600436e14 value, but `set_vehicle_grav_controls`
    // doesn't propagate that override into the body's Earth grav control —
    // the coefficient set's own mu is what matters. We use the GGM05C
    // fixture's mu directly to stay consistent with the harmonics.

    let mut sb = SimulationBuilder::new(apollo_time(), DT);

    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: earth_grav.mu,
                model: GravityModel::SphericalHarmonics(Box::new(earth_grav)),
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: Some(DMat3::IDENTITY),
            delta_c20: 0.0,
            rotation_model: RotationModel::EarthRNP,
            tidal_config: None,
            planet_omega: OMEGA_EARTH,
            central: true,
        },
    );
    let moon = sb.add_source(
        "Moon",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_moon,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );
    let sun = sb.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_sun,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );
    sb.set_source_ephemeris(
        moon,
        astrodyn::EphemerisBody::Moon,
        astrodyn::EphemerisBody::Earth,
    );
    sb.set_source_ephemeris(
        sun,
        astrodyn::EphemerisBody::Sun,
        astrodyn::EphemerisBody::Earth,
    );

    // Mean solar activity per `Modified_data/Earth/soflx_mean.py`.
    sb = sb.atmosphere(
        AtmosphereConfig {
            model: AtmosphereModel::Met(MetAtmosphere {
                f10: 128.8,
                f10b: 128.8,
                geo_index: 15.7,
                geo_index_type: GeoIndexType::Ap,
            }),
            r_eq: EARTH.shape.r_eq,
            r_pol: EARTH.shape.r_pol,
            planet_omega: OMEGA_EARTH,
        },
        earth,
    );

    // The body starts with CM-only mass; launch_stack assembly below
    // augments it to the full-stack composite via `add_body_to_tree` +
    // `sync_body_mass_from_tree`.
    let cm_only_mass = apollo_mass(12_807.0, 8.7, 157_372.0, 64_624.0, 64_624.0, yaw_180());

    // Initial attitude (sv_leo_lvlh.py): LVLH-aligned with Yaw_Pitch_Roll
    // = [0, 0, 0]. For a circular LEO, LVLH-aligned bodies have angular
    // velocity = -orbit-rate about body Y (matches CSV row 0:
    // ang_vel_this[1] = -1.134e-3 rad/s). The CSV row 0 quaternion is
    // the JEOD-computed scalar-first orientation of the LVLH frame
    // relative to Earth.inertial at the epoch.
    let csv = load_apollo_reference();
    assert!(!csv.is_empty(), "apollo_trajectory.csv is empty");
    let row0 = &csv[0];

    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: row0.position,
            velocity: row0.velocity,
        },
        rot: Some(RotationalState {
            quaternion: row0.quaternion,
            ang_vel_body: row0.ang_vel_body,
        }),
        mass: Some(cm_only_mass),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_nonspherical(earth, GRAV_DEGREE, GRAV_ORDER, false),
                GravityControl::new_third_body(moon),
                GravityControl::new_third_body(sun),
            ],
        },
        ..Default::default()
    });

    // astrodyn_runner uses the BSP for Moon/Sun ephemeris evaluation each
    // step. Phase 1 SIM_Apollo runs are 12 s, well within DE405/DE421
    // coverage.
    let bsp_path = astrodyn::ephemeris_assets::de421_path();
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris missing at {}",
        bsp_path.display()
    );
    let ephemeris =
        astrodyn::Ephemeris::from_bsp(&bsp_path).expect("failed to load DE421 ephemeris");
    sb = sb.ephemeris(ephemeris);

    let mut sim = sb.build().expect("apollo simulation must validate");

    // Register cm in the simulation's mass tree, then add the other 7
    // bodies and attachment points directly on the tree (they are
    // tree-only — never integrated as separate bodies, since after
    // launch_stack assembly only the root cm is integrated).
    let cm_id = sim.add_body_to_tree(0, "cm");
    let tree = sim.mass_tree.as_mut().expect("mass tree was just created");

    // Add the 7 non-cm bodies and their attachment points.
    // Per `Modified_data/mass/*.py`: SM, S1, S2, S3 use identity
    // struct→body rotation; LM (Ascent), DM, LES use yaw_180.
    let sm = tree.add_body(
        "sm".into(),
        apollo_mass(
            54_064.0,
            12.3,
            1_107_231.0,
            1_235_227.0,
            1_235_227.0,
            DMat3::IDENTITY,
        ),
    );
    let lm = tree.add_body(
        "lm".into(),
        apollo_mass(10_582.0, 5.45, 259_259.0, 155_822.0, 155_822.0, yaw_180()),
    );
    let dm = tree.add_body(
        "dm".into(),
        apollo_mass(25_640.0, 5.0, 628_180.0, 367_506.0, 367_506.0, yaw_180()),
    );
    let s3 = tree.add_body(
        "s3".into(),
        apollo_mass(
            274_171.0,
            30.65,
            16_138_048.0,
            29_532_558.0,
            29_532_558.0,
            DMat3::IDENTITY,
        ),
    );
    let s2 = tree.add_body(
        "s2".into(),
        apollo_mass(
            1_083_480.0,
            40.75,
            147_488_715.0,
            223_676_545.0,
            223_676_545.0,
            DMat3::IDENTITY,
        ),
    );
    let s1 = tree.add_body(
        "s1".into(),
        apollo_mass(
            5_031_023.0,
            69.0,
            684_848_006.0,
            2_338_482_378.0,
            2_338_482_378.0,
            DMat3::IDENTITY,
        ),
    );
    let les = tree.add_body(
        "les".into(),
        apollo_mass(9_200.0, 16.25, 5_566.0, 205_231.0, 205_231.0, yaw_180()),
    );

    // CM points
    tree.add_mass_point(
        cm_id,
        "SM interface",
        DVec3::new(11.6 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    tree.add_mass_point(
        cm_id,
        "CM docking port",
        DVec3::new(4.0 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );
    // SM points
    tree.add_mass_point(
        sm,
        "Stage 3 interface",
        DVec3::new(-20.9 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );
    tree.add_mass_point(
        sm,
        "CM interface",
        DVec3::new(24.6 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    // LM points
    tree.add_mass_point(
        lm,
        "LM docking port",
        DVec3::new(10.9 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    tree.add_mass_point(lm, "Descent Module interface", DVec3::ZERO, yaw_180());
    tree.add_mass_point(
        lm,
        "Stage 3 interface",
        DVec3::new(-10.0 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );
    // DM points
    tree.add_mass_point(dm, "Ascent Module interface", DVec3::ZERO, DMat3::IDENTITY);
    tree.add_mass_point(
        dm,
        "Stage 3 interface",
        DVec3::new(-10.0 * FT_TO_M, 0.0, 0.0),
        yaw_180(),
    );
    // S3 points
    tree.add_mass_point(s3, "Stage 2 interface", DVec3::ZERO, yaw_180());
    tree.add_mass_point(
        s3,
        "LEM/SM/CM interface",
        DVec3::new(61.3 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    // S2 points
    tree.add_mass_point(s2, "Stage 1 interface", DVec3::ZERO, yaw_180());
    tree.add_mass_point(
        s2,
        "Stage 3 interface",
        DVec3::new(81.5 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    // S1 points
    tree.add_mass_point(
        s1,
        "Stage 2 interface",
        DVec3::new(138.0 * FT_TO_M, 0.0, 0.0),
        DMat3::IDENTITY,
    );
    // LES points
    tree.add_mass_point(les, "CM interface", DVec3::ZERO, yaw_180());

    let ids = BodyIds {
        cm: cm_id,
        sm,
        lm,
        dm,
        s3,
        s2,
        s1,
        les,
    };
    assemble_launch_stack(tree, &ids);

    // Sync cm body's mass from the now-assembled tree composite.
    sim.sync_body_mass_from_tree(0);

    // SimulationBuilder set body.trans = row0 (JEOD's logged core_body
    // at t=0, after launch_stack assembled the full Apollo stack).
    // Our integration convention is body.trans = composite_body
    // inertial state; convert by subtracting the kinematic offset
    // through the now-fully-assembled mass tree.
    sim.convert_body_trans_core_to_composite(0);

    (sim, 0, ids)
}

// non-recipe: SIM_Apollo's launch-stack topology, JEOD English-unit
// per-body mass data, and 11-event detach/attach schedule are
// unique to this verification sim and not currently captured in any
// `astrodyn::recipes::scenarios::*` recipe. The JEOD-input.py defect
// (missing `set_vehicle_grav_controls(cm_dyn)` call) is patched at
// reference-CSV-regeneration time inside `trick/generate_references.sh`,
// not via any production-side workaround.
#[test]
fn tier3_sim_apollo_trajectory() {
    let csv = load_apollo_reference();
    assert!(
        (csv.last().unwrap().time - SIM_DURATION_S).abs() < 0.05,
        "apollo_trajectory.csv last row t={} disagrees with SIM_Apollo terminate_time={SIM_DURATION_S}",
        csv.last().unwrap().time
    );

    let (mut sim, body_idx, ids) = build_apollo_sim();

    // Walk the simulation in 0.1-second log windows, applying the
    // mass-tree event at each integer-second boundary just before the
    // logging step that crosses it (matching Trick's add_read semantics:
    // the event fires at the start of the cycle that begins at t=N,
    // before the data record at t=N is written).
    let mut event_iter = EVENTS.iter().peekable();
    let mut our_log = Vec::with_capacity(csv.len());
    let mut ref_log = Vec::with_capacity(csv.len());
    let mut current_t = 0.0_f64;

    // Skip CSV row 0 (initial state — no integration yet).
    for reference in csv.iter().skip(1) {
        // Apply events in JEOD's order: Trick's `trick.add_read(t, ...)`
        // job fires at the END of the cycle ending at t — after the
        // integrator has advanced state to t. So step up to event_t
        // (current_t == event_t), THEN apply the event. Verified
        // empirically: at t=4 DetachS3, JEOD's lm.vel JUMPS by +0.110 m/s
        // between the t=3.999 sample (= state at end of cycle [3.96,
        // 3.98]) and the t=4.000 sample (= state at end of cycle [3.98,
        // 4.0]). That kick equals one ordinary integration step on cm
        // cascaded through the dyn-tree to lm via JEOD's propagate_state
        // — i.e., the cycle [3.98, 4.0] integrator ran with the PRE-
        // detach mass tree (lm still in cm's tree), THEN the detach
        // fired. See `BUG_A_REPORT.md`.
        while let Some(&&(event_t, event)) = event_iter.peek() {
            if event_t <= reference.time + 1e-9 && event_t > current_t + 1e-9 {
                // Step up to and including event_t, then apply.
                while current_t + 0.5 * DT < event_t {
                    sim.step().expect("step failed");
                    current_t += DT;
                }
                apply_event(&mut sim, body_idx, &ids, event);
                event_iter.next();
            } else {
                break;
            }
        }

        // Step up to the reference timestamp.
        while current_t + DT * 0.5 < reference.time {
            sim.step().expect("step failed");
            current_t += DT;
        }

        if reference.time > TRAJECTORY_VALIDATION_END_S + 1e-6 {
            // See module docs / TRAJECTORY_VALIDATION_END_S for why
            // later samples are skipped.
            continue;
        }

        let body = sim.body(body_idx);
        // body.trans is the composite_body inertial integration state;
        // JEOD's reference CSV logs core_body, so derive it via the
        // mass tree (composite and core share body axes — only
        // position+velocity differ).
        let (core_position, core_velocity) = sim.body_core_inertial(body_idx);
        our_log.push(StateLog {
            time: reference.time,
            position: Some(core_position),
            velocity: Some(core_velocity),
            acceleration: Some(body.trans_accel),
            quaternion: body.rot.as_ref().map(|r| r.quaternion.to_glam()),
            ang_vel: body.rot.as_ref().map(|r| r.ang_vel_body),
            ang_accel: body.rot_accel,
        });
        ref_log.push(StateLog {
            time: reference.time,
            position: Some(reference.position),
            velocity: Some(reference.velocity),
            acceleration: None,
            quaternion: Some(reference.quaternion.to_glam()),
            ang_vel: Some(reference.ang_vel_body),
            ang_accel: None,
        });
    }
    assert!(!our_log.is_empty(), "trajectory log is empty");

    // Tooling-enforced cadence check: dt = 0.02 s and JEOD's CSV
    // samples at 0.1 s, so 0.1 / 0.02 = 5 — every reference row is an
    // integrator-output instant. If a future edit drifts either side
    // off the integer ratio, this fails loudly before the row loop
    // quietly compares against held off-cadence samples.
    CrossvalReport::assert_cadence_matches(&ref_log, DT, 1e-6);

    let report = CrossvalReport::compute("tier3_sim_apollo_trajectory", &our_log, &ref_log);
    report.write();

    // Tolerances per `tests/README.md` (5 % above observed max error).
    //
    // Window: full 12 s sim — every one of the 11 attach/detach events
    // is asserted end-to-end, including the t=6 SM→CM attach (whose
    // composite ang_vel matches JEOD's logged −1.7207 rad/s exactly),
    // the t=9 LM re-attach, and the t=10 LM detach. The closed-form
    // quaternion advance for detached subtrees routes through
    // `BodyAttitude::advance_under_body_rate` (issue #248 / PR #251 +
    // issue #252); fixing the multiply order on `step_ballistic`
    // removed the 1.708 mrad/s S3-attitude drift that had been
    // lever-armed up to
    // 16 mm at LM during the t=4 → t=5 free-fly. Residuals over the
    // full 12 s are now:
    //   - position:    ~7 µm / component
    //   - velocity:    ~2.5 µm/s / component
    //   - quat angle:  ~3.4 µrad
    //   - ang_vel:     ~14 µrad/s worst-component (body-Z, lever-armed
    //                  through the t=6 attach algorithm's ~4 mrad/s
    //                  body-X residue, which is sub-LSB on the input
    //                  cross-products and physically negligible).
    report.assert_position([6.90e-6, 2.50e-6, 5.27e-6]);
    report.assert_velocity([2.58e-6, 1.24e-6, 1.62e-6]);
    report.assert_quat_angle(3.59e-6);
    report.assert_ang_vel([2.29e-6, 1.19e-7, 1.46e-5]);
}

// ─── LM-state-vs-truth diagnostic ────────────────────────────────────
//
// Runs the same sim through the full 12 s and compares LM
// `composite_body` inertial state against `apollo_attach_truth.csv`
// (1 ms cadence) at every integration step plus right after each event.
// Output: stderr table highlighting the first sample to cross 1 mm,
// plus a per-step CSV under `target/tier3_crossval/` for offline
// analysis. Diagnostic only — does not assert tolerances. Marked
// `#[ignore]` because the truth CSV is gitignored and may be missing
// on a fresh clone.

const LM_DIAG_POSITION_TRIP_M: f64 = 1.0e-3;

#[derive(Clone)]
struct LmDiagSample {
    time: f64,
    /// Empty unless this row was captured immediately after an event applied.
    event_label: String,
    // ── LM (always present) ──
    err_pos: DVec3,
    err_vel: DVec3,
    err_quat_angle: f64,
    err_ang_vel: DVec3,
    our_pos: DVec3,
    truth_pos: DVec3,
    /// Raw LM ang_vel from the runner (chain-walked), expressed in body frame.
    our_ang_vel: DVec3,
    /// Raw LM ang_vel from JEOD's truth recorder, body frame.
    truth_ang_vel: DVec3,
    // ── S3 (Some only when truth CSV has s3 columns) ──
    /// `Some` when the truth row exposes `s3`; otherwise the recorder hasn't
    /// been regenerated with the s3 columns yet.
    s3_err_pos: Option<DVec3>,
    s3_err_vel: Option<DVec3>,
    s3_err_quat_angle: Option<f64>,
    s3_err_ang_vel: Option<DVec3>,
}

fn event_short_label(event: Event) -> &'static str {
    match event {
        Event::DetachS1 => "DetS1",
        Event::DetachS2 => "DetS2",
        Event::DetachLes => "DetLes",
        Event::DetachS3 => "DetS3",
        Event::DetachLm => "DetLm",
        Event::AttachLmCm => "AttLmCm",
        Event::DetachLm2 => "DetLm2",
        Event::DetachDm => "DetDm",
        Event::AttachLmCm2 => "AttLmCm2",
        Event::DetachLm3 => "DetLm3",
        Event::DetachSm => "DetSm",
    }
}

/// Walk up from CARGO_MANIFEST_DIR until we find Cargo.lock — that's the
/// workspace root. Mirrors the helper in `astrodyn_verif_jeod::crossval`.
fn workspace_target_tier3_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if dir.join("Cargo.lock").exists() {
            break;
        }
        if !dir.pop() {
            dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            break;
        }
    }
    dir.join("target").join("tier3_crossval")
}

/// Quaternion angular distance: `2 · acos(|<q1, q2>|)`. Returns the
/// smaller of the two possible rotations (q and −q represent the same
/// rotation in JEOD's left-quat convention).
fn quat_angle_between(a: JeodQuat, b: JeodQuat) -> f64 {
    let av = a.vector();
    let bv = b.vector();
    let dot = a.scalar() * b.scalar() + av.x * bv.x + av.y * bv.y + av.z * bv.z;
    2.0 * dot.abs().clamp(-1.0, 1.0).acos()
}

fn capture_lm_diag(
    sim: &Simulation,
    ids: &BodyIds,
    truth_rows: &[ApolloTruthRow],
    time: f64,
    event_label: &str,
) -> LmDiagSample {
    let our = sim.subtree_composite_inertial(ids.lm);
    let truth = nearest_truth_at(truth_rows, time);
    let truth_quat = truth.lm.quaternion;

    // S3 comparison — only meaningful when the truth recorder logged s3.
    // Even when the truth row has no s3, we still walk our own simulation
    // for s3 so the function is total; the comparison is conditioned on
    // truth.s3 being Some.
    let our_s3 = sim.subtree_composite_inertial(ids.s3);
    let s3_err_pos = truth
        .s3
        .as_ref()
        .map(|s3| our_s3.trans.position - s3.position);
    let s3_err_vel = truth
        .s3
        .as_ref()
        .map(|s3| our_s3.trans.velocity - s3.velocity);
    let s3_err_quat_angle = truth
        .s3
        .as_ref()
        .map(|s3| quat_angle_between(our_s3.rot.q_parent_this, s3.quaternion));
    let s3_err_ang_vel = truth
        .s3
        .as_ref()
        .map(|s3| our_s3.rot.ang_vel_this - s3.ang_vel_body);

    LmDiagSample {
        time,
        event_label: event_label.to_string(),
        err_pos: our.trans.position - truth.lm.position,
        err_vel: our.trans.velocity - truth.lm.velocity,
        err_quat_angle: quat_angle_between(our.rot.q_parent_this, truth_quat),
        err_ang_vel: our.rot.ang_vel_this - truth.lm.ang_vel_body,
        our_pos: our.trans.position,
        truth_pos: truth.lm.position,
        our_ang_vel: our.rot.ang_vel_this,
        truth_ang_vel: truth.lm.ang_vel_body,
        s3_err_pos,
        s3_err_vel,
        s3_err_quat_angle,
        s3_err_ang_vel,
    }
}

/// Diagnostic (ignored by default): runs the full 12 s SIM_Apollo and
/// compares LM `composite_body` inertial state against
/// `apollo_attach_truth.csv` at every integration step and right after
/// each event. Output is a stderr table flagging the first sample whose
/// position error exceeds 1 mm, plus a per-step CSV at
/// `target/tier3_crossval/apollo_lm_state_vs_truth.csv`. The truth CSV
/// is gitignored — regenerate via `cargo xtask regenerate-tier3 --force`.
///
/// Run manually:
///   `cargo nextest run -p astrodyn_runner --test tier3_sim_apollo_trajectory \
///     tier3_sim_apollo_lm_state_vs_truth --run-ignored only`
/// or
///   `cargo test -p astrodyn_runner --test tier3_sim_apollo_trajectory \
///     tier3_sim_apollo_lm_state_vs_truth -- --ignored --nocapture`
#[test]
#[ignore]
fn tier3_sim_apollo_lm_state_vs_truth() {
    let truth_rows = match load_apollo_attach_truth(env!("CARGO_MANIFEST_DIR")) {
        Ok(rows) => rows,
        Err(ApolloTruthError::Missing { path }) => panic!(
            "{} missing — regenerate via `cargo xtask regenerate-tier3 --force` \
             (the attach_truth recorder is in APOLLO_SNIPPET in trick/generate_references.sh)",
            path.display()
        ),
        Err(e) => panic!("failed to load apollo_attach_truth.csv: {e}"),
    };
    eprintln!(
        "loaded {} truth rows spanning t = {:.6} .. {:.6}",
        truth_rows.len(),
        truth_rows.first().unwrap().time,
        truth_rows.last().unwrap().time
    );

    let (mut sim, body_idx, ids) = build_apollo_sim();

    let mut event_iter = EVENTS.iter().peekable();
    let mut current_t = 0.0_f64;
    let mut samples: Vec<LmDiagSample> = Vec::new();

    samples.push(capture_lm_diag(&sim, &ids, &truth_rows, current_t, "init"));

    let n_steps = (SIM_DURATION_S / DT).round() as usize;
    for _ in 0..n_steps {
        // Apply any events whose t is at or before current_t (matches
        // the trajectory test's JEOD-order semantics).
        let mut applied = String::new();
        while let Some(&&(event_t, event)) = event_iter.peek() {
            if event_t <= current_t + 1e-9 {
                apply_event(&mut sim, body_idx, &ids, event);
                if !applied.is_empty() {
                    applied.push('+');
                }
                applied.push_str(event_short_label(event));
                event_iter.next();
            } else {
                break;
            }
        }
        if !applied.is_empty() {
            samples.push(capture_lm_diag(
                &sim,
                &ids,
                &truth_rows,
                current_t,
                &applied,
            ));
        }
        sim.step().expect("step failed");
        current_t += DT;
        samples.push(capture_lm_diag(&sim, &ids, &truth_rows, current_t, ""));
    }
    // Sweep any trailing events scheduled at current_t (none today, but
    // guard the loop for future schedule edits).
    while let Some(&&(event_t, event)) = event_iter.peek() {
        if event_t <= current_t + 1e-9 {
            apply_event(&mut sim, body_idx, &ids, event);
            samples.push(capture_lm_diag(
                &sim,
                &ids,
                &truth_rows,
                current_t,
                event_short_label(event),
            ));
            event_iter.next();
        } else {
            break;
        }
    }

    // ── stderr summary ───────────────────────────────────────────────
    let first_breach = samples
        .iter()
        .find(|s| s.err_pos.length() > LM_DIAG_POSITION_TRIP_M);

    eprintln!();
    eprintln!("==========================================================");
    eprintln!("  LM composite_body vs apollo_attach_truth.csv");
    eprintln!(
        "  position trip threshold = {:.0e} m",
        LM_DIAG_POSITION_TRIP_M
    );
    eprintln!("==========================================================");
    eprintln!(
        "  total samples: {} ({} steps + initial + post-event captures)",
        samples.len(),
        n_steps
    );
    if let Some(s) = first_breach {
        eprintln!();
        eprintln!(
            "  FIRST POSITION BREACH at t = {:.6} s (event_label: {:?})",
            s.time, s.event_label
        );
        eprintln!(
            "    err_pos = [{:>13.6e} {:>13.6e} {:>13.6e}]  |.| = {:.6e} m",
            s.err_pos.x,
            s.err_pos.y,
            s.err_pos.z,
            s.err_pos.length()
        );
        eprintln!(
            "    err_vel = [{:>13.6e} {:>13.6e} {:>13.6e}]  |.| = {:.6e} m/s",
            s.err_vel.x,
            s.err_vel.y,
            s.err_vel.z,
            s.err_vel.length()
        );
        eprintln!("    err_quat_angle = {:.6e} rad", s.err_quat_angle);
        eprintln!(
            "    err_ang_vel = [{:>13.6e} {:>13.6e} {:>13.6e}]  |.| = {:.6e} rad/s",
            s.err_ang_vel.x,
            s.err_ang_vel.y,
            s.err_ang_vel.z,
            s.err_ang_vel.length()
        );
    } else {
        eprintln!();
        eprintln!(
            "  no position breach — max |err_pos| = {:.6e} m",
            samples
                .iter()
                .map(|s| s.err_pos.length())
                .fold(0.0_f64, f64::max)
        );
    }

    // ── per-event-boundary headline (every event, regardless of trip) ─
    let any_s3 = samples.iter().any(|s| s.s3_err_pos.is_some());
    eprintln!();
    eprintln!("  ─── per-event LM error snapshots ─────────────────────");
    eprintln!(
        "  {:>10}  {:>10}  {:>13}  {:>13}  {:>13}  {:>13}",
        "t (s)", "event", "|err_pos| m", "|err_vel| m/s", "dq_ang rad", "|dω| rad/s"
    );
    for s in samples.iter().filter(|s| !s.event_label.is_empty()) {
        eprintln!(
            "  {:>10.6}  {:>10}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}",
            s.time,
            s.event_label,
            s.err_pos.length(),
            s.err_vel.length(),
            s.err_quat_angle,
            s.err_ang_vel.length()
        );
    }

    if any_s3 {
        eprintln!();
        eprintln!("  ─── per-event S3 error snapshots ─────────────────────");
        eprintln!(
            "  {:>10}  {:>10}  {:>13}  {:>13}  {:>13}  {:>13}",
            "t (s)", "event", "|err_pos| m", "|err_vel| m/s", "dq_ang rad", "|dω| rad/s"
        );
        for s in samples.iter().filter(|s| !s.event_label.is_empty()) {
            match (
                s.s3_err_pos,
                s.s3_err_vel,
                s.s3_err_quat_angle,
                s.s3_err_ang_vel,
            ) {
                (Some(ep), Some(ev), Some(eq), Some(ew)) => eprintln!(
                    "  {:>10.6}  {:>10}  {:>13.6e}  {:>13.6e}  {:>13.6e}  {:>13.6e}",
                    s.time,
                    s.event_label,
                    ep.length(),
                    ev.length(),
                    eq,
                    ew.length()
                ),
                _ => eprintln!(
                    "  {:>10.6}  {:>10}  (truth row at this time has no s3 columns)",
                    s.time, s.event_label
                ),
            }
        }
    } else {
        eprintln!();
        eprintln!(
            "  S3-vs-truth comparison skipped — truth CSV has no s3_dyn columns. \
             Regenerate via `cargo xtask regenerate-tier3 --force` after pulling \
             the recorder change in `trick/generate_references.sh`."
        );
    }

    // Sanity-check the err_ang_vel = 0 observation by dumping raw values
    // at one mid-window sample. If the bits really are equal, both rows
    // print the same numbers.
    if let Some(probe) = samples
        .iter()
        .find(|s| (s.time - 4.5).abs() < 1e-6 && s.event_label.is_empty())
    {
        eprintln!();
        eprintln!("  ─── ang_vel sanity-check at t = 4.500 ────────────────");
        eprintln!(
            "    our   ang_vel = [{:>22.16} {:>22.16} {:>22.16}]",
            probe.our_ang_vel.x, probe.our_ang_vel.y, probe.our_ang_vel.z
        );
        eprintln!(
            "    truth ang_vel = [{:>22.16} {:>22.16} {:>22.16}]",
            probe.truth_ang_vel.x, probe.truth_ang_vel.y, probe.truth_ang_vel.z
        );
        eprintln!(
            "    raw bit-diff  = [{:>+22.16e} {:>+22.16e} {:>+22.16e}]",
            probe.err_ang_vel.x, probe.err_ang_vel.y, probe.err_ang_vel.z
        );
    }
    eprintln!("==========================================================");

    // ── per-step CSV for offline analysis ────────────────────────────
    let out_dir = workspace_target_tier3_dir();
    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("create_dir_all {}: {e}", out_dir.display()));
    let out_path = out_dir.join("apollo_lm_state_vs_truth.csv");
    let mut out = String::with_capacity(samples.len() * 200);
    out.push_str(
        "time,event,err_pos_norm_m,err_pos_x,err_pos_y,err_pos_z,\
         err_vel_norm_mps,err_vel_x,err_vel_y,err_vel_z,\
         err_quat_angle_rad,err_ang_vel_norm_rps,err_ang_vel_x,err_ang_vel_y,err_ang_vel_z,\
         our_pos_x,our_pos_y,our_pos_z,truth_pos_x,truth_pos_y,truth_pos_z,\
         our_ang_vel_x,our_ang_vel_y,our_ang_vel_z,\
         truth_ang_vel_x,truth_ang_vel_y,truth_ang_vel_z,\
         s3_err_pos_norm_m,s3_err_vel_norm_mps,s3_err_quat_angle_rad,s3_err_ang_vel_norm_rps\n",
    );
    fn fmt_opt_norm(v: Option<DVec3>) -> String {
        v.map(|x| format!("{:.9e}", x.length())).unwrap_or_default()
    }
    fn fmt_opt_f64(v: Option<f64>) -> String {
        v.map(|x| format!("{:.9e}", x)).unwrap_or_default()
    }
    for s in &samples {
        out.push_str(&format!(
            "{:.6},{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},\
             {},{},{},{}\n",
            s.time,
            s.event_label,
            s.err_pos.length(),
            s.err_pos.x,
            s.err_pos.y,
            s.err_pos.z,
            s.err_vel.length(),
            s.err_vel.x,
            s.err_vel.y,
            s.err_vel.z,
            s.err_quat_angle,
            s.err_ang_vel.length(),
            s.err_ang_vel.x,
            s.err_ang_vel.y,
            s.err_ang_vel.z,
            s.our_pos.x,
            s.our_pos.y,
            s.our_pos.z,
            s.truth_pos.x,
            s.truth_pos.y,
            s.truth_pos.z,
            s.our_ang_vel.x,
            s.our_ang_vel.y,
            s.our_ang_vel.z,
            s.truth_ang_vel.x,
            s.truth_ang_vel.y,
            s.truth_ang_vel.z,
            fmt_opt_norm(s.s3_err_pos),
            fmt_opt_norm(s.s3_err_vel),
            fmt_opt_f64(s.s3_err_quat_angle),
            fmt_opt_norm(s.s3_err_ang_vel),
        ));
    }
    std::fs::write(&out_path, out).unwrap_or_else(|e| panic!("write {}: {e}", out_path.display()));
    eprintln!(
        "  per-step trace: {} ({} rows)",
        out_path.display(),
        samples.len()
    );
}
