// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: SIM_7_time_reversal — time-reversed propagation cross-validation.
//!
//! JEOD propagates forward 60,000 s then sets `scale_factor = -1.0` for another
//! 60,000 sim-seconds. Validates TAI time and trajectory position/velocity
//! during both forward and reverse phases.
//!
//! Each RUN flips `time_scale_factor` to −1.0 at the t=60,000 s reversal point.
//! Every RUN is validated **full-state** (position/velocity + TAI) against its
//! JEOD reference CSV through the `Simulation` pipeline, each wired with that
//! RUN's actual force model (read from `SET_test/RUN_*/input.py`):
//!
//! - RUN_1, RUN_10A — spherical (point-mass) Earth gravity, no perturbations.
//!   RUN_10A differs from RUN_1 only in initial attitude, which does not affect
//!   point-mass translation, so its logged trajectory is identical to RUN_1.
//! - RUN_3A (4×4), RUN_3B (8×8) — non-spherical Earth gravity (GEM-T1) with
//!   EarthRNP orientation, Sun/Moon off.
//! - RUN_4 — spherical Earth + Sun + Moon third-body gravity (DE421 ephemeris,
//!   positions refreshed each step).
//! - RUN_6A — spherical Earth + constant-density aerodynamic drag
//!   (ρ=1.4e-12, Cd=0.02, A=1 m², m=1 kg) on the highly-elliptic IC.
//! - RUN_8B — spherical Earth gravity.
//! - RUN_9D — spherical Earth + a 6-DOF rotational state (LVLH orbit-rate
//!   initial attitude) with a 10 kN structural-frame external force/torque
//!   applied over t∈[10000,20000]∪[100000,110000] s. The body-fixed thrust is
//!   re-rotated to inertial at every RK4 sub-stage from the propagated attitude
//!   (per-stage `StructuralWrench`, JEOD_INV DB.28); loads toggle at the exact
//!   window boundaries.
//!
//! Tolerances are observed-max × 1.05 per component (CLAUDE.md cross-validation
//! convention). The leap-second / UT1 setup matches `Modified_data/date_n_time/
//! 11Nov2007.py`: epoch 2007-11-20 00:00:00 UTC, TAI−UTC = 32 s,
//! UT1−TAI = −32.469 s, polar motion off.

use astrodyn_verif_jeod::tier3_csv::test_data_path;

use astrodyn::{
    AtmosphereConfig, AtmosphereModel, DragConfig, Ephemeris, EphemerisBody, ExponentialAtmosphere,
    GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource,
    GravitySourceEntry, JeodQuat, MassProperties, RotationalState, SimulationTime,
    TranslationalState, Vec3Ext, VehicleConfig, EARTH,
};
use astrodyn_runner::{RotationModel, Simulation};
use glam::{DMat3, DVec3};

/// Dynamics timestep: 0.03125 s (32 Hz) per the SIM_7_time_reversal S_define
/// `#define DYNAMICS`.
const DT_S: f64 = 0.03125;

/// UT1 − TAI offset (s) from `Modified_data/date_n_time/11Nov2007.py`
/// (`conv_tai_ut1.tai_to_ut1_override_val`). Needed by EarthRNP for GAST.
const UT1_TAI_OFFSET_S: f64 = -32.469;

fn load_mu_earth_gemt1() -> f64 {
    astrodyn::gravity_fixtures::load_gemt1().mu
}

struct ReversalRecord {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    tai_seconds: f64,
    tai_tjt: f64,
}

fn load_reversal_csv(path: &std::path::Path) -> Vec<ReversalRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SIM_7_time_reversal CSV from {}: {e}\n\
             Generate with Docker (see CLAUDE.md).",
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
            f.len() >= 9,
            "line {}: expected >=9 columns, got {}",
            i + 1,
            f.len()
        );
        let p = |idx: usize| -> f64 { f[idx].trim().parse().unwrap() };
        records.push(ReversalRecord {
            time: p(0),
            position: DVec3::new(p(1), p(3), p(5)),
            velocity: DVec3::new(p(2), p(4), p(6)),
            tai_seconds: p(7),
            tai_tjt: p(8),
        });
    }
    records
}

/// Per-component tolerances (observed-max × 1.05, CLAUDE.md convention).
struct Tols {
    pos: f64,
    vel: f64,
    tai: f64,
    roundtrip_pos: f64,
    roundtrip_vel: f64,
}

/// JEOD `set_orientation_lvlh()` with Yaw-Pitch-Roll [0, −11.6°, 0]: the LVLH
/// frame at (position, velocity) rotated −11.6° about its Y axis, as a
/// scalar-first JEOD quaternion (inertial → body left-transform).
fn lvlh_pitch_quat(position: DVec3, velocity: DVec3) -> JeodQuat {
    let lvlh = astrodyn::compute_body_lvlh_frame(position, velocity);
    // Body attitude as the JEOD left-transformation `T_inertial_body`
    // (inertial → body) = `T_lvlh_body · T_inertial_lvlh`. `lvlh.t_parent_this`
    // is `T_inertial_lvlh`. For a body pitched θ = −11.6° about the LVLH Y axis,
    // the coordinate-transform matrix LVLH→body is the passive rotation
    // `R_y(−θ) = R_y(+11.6°)`, not the active `R_y(θ)`. (Composing the active
    // form and transposing — a prior version — produced `T_body_inertial`, a
    // 54.8° attitude error invisible to every attitude-independent test; it
    // only surfaced in `tier3_sim_time_reversal_run9d`'s body-fixed thrust.)
    let t_lvlh_body = DMat3::from_rotation_y(11.6_f64.to_radians());
    let t_inertial_body = t_lvlh_body * lvlh.t_parent_this;
    let q = glam::DQuat::from_mat3(&t_inertial_body);
    JeodQuat::new(q.w, q.x, q.y, q.z)
}

/// Spherical (point-mass) central Earth source at the root origin.
fn add_spherical_earth(sim: &mut Simulation, mu: f64) -> usize {
    let mut e = GravitySourceEntry::new(
        GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        astrodyn::Position::<astrodyn::RootInertial>::zero(),
        None,
    );
    e.central = true;
    sim.add_source("Earth", e)
}

/// Non-spherical central Earth source (GEM-T1 SH) with EarthRNP orientation.
fn add_nonspherical_earth(sim: &mut Simulation, mu: f64, model: GravityModel) -> usize {
    sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource { mu, model },
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
    )
}

/// Point-mass third-body source (Sun/Moon). Position is refreshed each step by
/// the ephemeris pre-step closure; the seed is only non-zero so `validate()`
/// sees a well-formed source before the first `pre_step`.
fn add_third_body(sim: &mut Simulation, name: &str, mu: f64, seed: DVec3) -> usize {
    sim.add_source(
        name,
        GravitySourceEntry {
            source: GravitySource {
                mu,
                model: GravityModel::PointMass,
            },
            position: seed.m_at::<astrodyn::RootInertial>(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    )
}

/// Add the standard SIM_7 3-DOF vehicle (mass 1 kg, LVLH+pitch attitude) with
/// the supplied gravity controls and optional drag.
fn add_standard_body(
    sim: &mut Simulation,
    init: &ReversalRecord,
    controls: GravityControls<usize>,
    drag: Option<DragConfig>,
) -> usize {
    let quat = lvlh_pitch_quat(init.position, init.velocity);
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: init.position,
            velocity: init.velocity,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: quat,
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(
            &(MassProperties::new(1.0)),
        )),
        gravity_controls: controls,
        drag,
        ..Default::default()
    })
}

/// Generalized full-state reversal driver. Builds the simulation via `build`
/// (which adds sources, atmosphere, and the body, returning the body index),
/// then steps through the reference CSV, flipping `scale_factor` to −1.0 at the
/// reversal point. `pre_step(sim, body_idx, interval_start, interval_end)` runs
/// before each `step_until` — used to refresh ephemeris positions or toggle
/// external loads. Validates per-component position/velocity/TAI errors and the
/// round-trip return-to-initial against `tols`.
fn run_reversal_full_state(
    label: &str,
    csv_name: &str,
    set_ut1_offset: bool,
    tols: Tols,
    // Exact times (exec/CSV `time`) at which an external load toggles, so the
    // load is applied on JEOD's continuous schedule rather than the 60 s CSV
    // record grid. Empty for RUNs with no time-windowed loads.
    event_bounds: &[f64],
    build: impl FnOnce(&mut Simulation, &ReversalRecord) -> usize,
    mut pre_step: impl FnMut(&mut Simulation, usize, f64, f64),
) {
    let csv_path = test_data_path(csv_name);
    let records = load_reversal_csv(&csv_path);
    assert!(records.len() > 1, "{label}: no reference data");
    let init = &records[0];

    let leap_table = astrodyn::default_leap_second_table();
    let mut time = SimulationTime::new(init.tai_tjt, leap_table);
    if set_ut1_offset {
        time.set_ut1_tai_offset(UT1_TAI_OFFSET_S);
    }
    let mut sim = Simulation::new(time, DT_S);
    let body_idx = build(&mut sim, init);
    sim.validate().unwrap();

    let reversal_idx = records
        .windows(2)
        .position(|w| w[1].tai_seconds < w[0].tai_seconds)
        .unwrap_or_else(|| panic!("{label}: no reversal point found in CSV"));

    let (mut max_pos_err, mut max_vel_err, mut max_tai_s_err) = (0.0_f64, 0.0_f64, 0.0_f64);

    for (i, rec) in records.iter().enumerate() {
        if i == 0 {
            continue;
        }
        if i == reversal_idx + 1 && sim.time.scale_factor() > 0.0 {
            sim.time.set_scale_factor(-1.0);
        }
        // Step through any event boundaries inside (prev, rec.time] so loads
        // toggle at JEOD's exact times, not on the CSV record grid. With no
        // `event_bounds`, `stops == [rec.time]` — a single step, as before.
        let mut cur = records[i - 1].time;
        let mut stops: Vec<f64> = event_bounds
            .iter()
            .copied()
            .filter(|&b| b > cur && b < rec.time)
            .collect();
        stops.sort_by(|a, b| a.partial_cmp(b).expect("event boundary times are finite"));
        stops.push(rec.time);
        for stop in stops {
            pre_step(&mut sim, body_idx, cur, stop);
            sim.step_until(stop).expect("step_until failed");
            cur = stop;
        }

        let body = sim.body(body_idx);
        let pos_err = (body.trans.position.raw_si() - rec.position).length();
        let vel_err = (body.trans.velocity.raw_si() - rec.velocity).length();
        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);
        let elapsed_jeod = rec.tai_seconds - init.tai_seconds;
        max_tai_s_err = max_tai_s_err.max((sim.time.tai_seconds - elapsed_jeod).abs());
    }

    let fb = sim.body(body_idx);
    let roundtrip_pos = (fb.trans.position.raw_si() - init.position).length();
    let roundtrip_vel = (fb.trans.velocity.raw_si() - init.velocity).length();

    println!(
        "  {label}: {} points, pos={max_pos_err:.3e}m, vel={max_vel_err:.3e}m/s, \
         TAI={max_tai_s_err:.3e}s, rt_pos={roundtrip_pos:.3e}m, rt_vel={roundtrip_vel:.3e}m/s",
        records.len()
    );

    assert!(
        max_pos_err < tols.pos,
        "{label}: position error {max_pos_err:.4e} m exceeds {:.4e} m",
        tols.pos
    );
    assert!(
        max_vel_err < tols.vel,
        "{label}: velocity error {max_vel_err:.4e} m/s exceeds {:.4e} m/s",
        tols.vel
    );
    assert!(
        max_tai_s_err < tols.tai,
        "{label}: TAI error {max_tai_s_err:.4e} s exceeds {:.4e} s",
        tols.tai
    );
    assert!(
        roundtrip_pos < tols.roundtrip_pos,
        "{label}: round-trip position {roundtrip_pos:.4e} m exceeds {:.4e} m",
        tols.roundtrip_pos
    );
    assert!(
        roundtrip_vel < tols.roundtrip_vel,
        "{label}: round-trip velocity {roundtrip_vel:.4e} m/s exceeds {:.4e} m/s",
        tols.roundtrip_vel
    );
}

fn noop_pre_step(_: &mut Simulation, _: usize, _: f64, _: f64) {}

// ── Spherical-gravity RUNs (point-mass Earth, no perturbations) ──────────────

// non-recipe: SIM_7_time_reversal seeds from a JEOD CSV t=0 record (TAI TJT,
// position, velocity, attitude derived from LVLH+pitch). The bespoke piece is
// the negative-`time_scale_factor` flip at the reversal index — verified here
// as part of the simulation pipeline.
#[test]
fn tier3_sim_time_reversal_run1() {
    let mu = load_mu_earth_gemt1();
    run_reversal_full_state(
        "reversal_run1",
        "reversal_run1_reversal.csv",
        false,
        Tols {
            pos: 1.46e-5,
            vel: 1.72e-8,
            tai: 1e-6,
            roundtrip_pos: 2.84e-6,
            roundtrip_vel: 3.08e-9,
        },
        &[],
        |sim, init| {
            let earth = add_spherical_earth(sim, mu);
            add_standard_body(
                sim,
                init,
                GravityControls {
                    controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
                },
                None,
            )
        },
        noop_pre_step,
    );
}

// non-recipe: RUN_10A is point-mass Earth with a different initial attitude
// (Roll-Pitch-Yaw [0,85°,1°]); attitude does not affect point-mass translation,
// so its logged trajectory is identical to RUN_1 and we validate against the
// same spherical force model.
#[test]
fn tier3_sim_time_reversal_run10a() {
    let mu = load_mu_earth_gemt1();
    run_reversal_full_state(
        "reversal_run10a",
        "reversal_run10a_reversal.csv",
        false,
        Tols {
            pos: 1.46e-5,
            vel: 1.72e-8,
            tai: 1e-6,
            roundtrip_pos: 2.84e-6,
            roundtrip_vel: 3.08e-9,
        },
        &[],
        |sim, init| {
            let earth = add_spherical_earth(sim, mu);
            add_standard_body(
                sim,
                init,
                GravityControls {
                    controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
                },
                None,
            )
        },
        noop_pre_step,
    );
}

// non-recipe: RUN_8B is point-mass Earth gravity (same force model as RUN_1,
// distinct time-scale configuration).
#[test]
fn tier3_sim_time_reversal_run8b() {
    let mu = load_mu_earth_gemt1();
    run_reversal_full_state(
        "reversal_run8b",
        "reversal_run8b_reversal.csv",
        false,
        Tols {
            pos: 1.46e-5,
            vel: 1.72e-8,
            tai: 1e-6,
            roundtrip_pos: 2.84e-6,
            roundtrip_vel: 3.08e-9,
        },
        &[],
        |sim, init| {
            let earth = add_spherical_earth(sim, mu);
            add_standard_body(
                sim,
                init,
                GravityControls {
                    controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
                },
                None,
            )
        },
        noop_pre_step,
    );
}

// ── Non-spherical-gravity RUNs (GEM-T1 SH + EarthRNP, Sun/Moon off) ──────────

// non-recipe: RUN_3A is 4×4 non-spherical Earth gravity.
#[test]
fn tier3_sim_time_reversal_run3a() {
    run_reversal_full_state(
        "reversal_run3a",
        "reversal_run3a_reversal.csv",
        true,
        Tols {
            pos: 4.00e-2,
            vel: 4.44e-5,
            tai: 1e-6,
            roundtrip_pos: 3.92e-2,
            roundtrip_vel: 4.43e-5,
        },
        &[],
        |sim, init| {
            let sh = astrodyn::gravity_fixtures::load_gemt1();
            let earth =
                add_nonspherical_earth(sim, sh.mu, GravityModel::SphericalHarmonics(Box::new(sh)));
            add_standard_body(
                sim,
                init,
                GravityControls {
                    controls: vec![GravityControl::new_nonspherical(
                        earth,
                        4,
                        4,
                        GravityGradient::Skip,
                    )],
                },
                None,
            )
        },
        noop_pre_step,
    );
}

// non-recipe: RUN_3B is 8×8 non-spherical Earth gravity.
#[test]
fn tier3_sim_time_reversal_run3b() {
    run_reversal_full_state(
        "reversal_run3b",
        "reversal_run3b_reversal.csv",
        true,
        Tols {
            pos: 5.64e-2,
            vel: 6.27e-5,
            tai: 1e-6,
            roundtrip_pos: 5.59e-2,
            roundtrip_vel: 6.26e-5,
        },
        &[],
        |sim, init| {
            let sh = astrodyn::gravity_fixtures::load_gemt1();
            let earth =
                add_nonspherical_earth(sim, sh.mu, GravityModel::SphericalHarmonics(Box::new(sh)));
            add_standard_body(
                sim,
                init,
                GravityControls {
                    controls: vec![GravityControl::new_nonspherical(
                        earth,
                        8,
                        8,
                        GravityGradient::Skip,
                    )],
                },
                None,
            )
        },
        noop_pre_step,
    );
}

// ── Third-body RUN (spherical Earth + Sun + Moon, DE421 ephemeris) ───────────

// non-recipe: RUN_4 adds Sun + Moon third-body gravity. Sun/Moon positions are
// refreshed each step from DE421 at the (reversed) dynamic-time TDB.
#[test]
fn tier3_sim_time_reversal_run4() {
    let mu = load_mu_earth_gemt1();
    let mu_sun = astrodyn::gravity_fixtures::load_sun_spherical_mu();
    let mu_moon = astrodyn::gravity_fixtures::load_moon_grail150_mu();
    let bsp = astrodyn::ephemeris_assets::de421_path();
    assert!(
        bsp.exists(),
        "DE421 ephemeris not found at {} — committed under test_data/",
        bsp.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp).expect("load DE421");

    run_reversal_full_state(
        "reversal_run4",
        "reversal_run4_reversal.csv",
        false,
        Tols {
            pos: 6.53e-3,
            vel: 7.06e-6,
            tai: 1e-6,
            roundtrip_pos: 6.54e-3,
            roundtrip_vel: 3.71e-6,
        },
        &[],
        |sim, init| {
            let earth = add_spherical_earth(sim, mu);
            let sun = add_third_body(sim, "Sun", mu_sun, DVec3::new(1.5e11, 0.0, 0.0));
            let moon = add_third_body(sim, "Moon", mu_moon, DVec3::new(3.8e8, 0.0, 0.0));
            add_standard_body(
                sim,
                init,
                GravityControls {
                    controls: vec![
                        GravityControl::new_spherical(earth, GravityGradient::Skip),
                        GravityControl::new_third_body(sun),
                        GravityControl::new_third_body(moon),
                    ],
                },
                None,
            )
        },
        move |sim, _body, _start, _end| {
            // Sun/Moon positions at the current (possibly reversed) dynamic-time
            // TDB. Updated before stepping; the ≤60 s lag versus the upcoming
            // window is negligible at third-body acceleration magnitudes.
            let tdb = sim.time.tdb_julian_date();
            let (sun_pos, _) = ephemeris
                .get_earth_centered_state_typed(EphemerisBody::Sun, tdb)
                .expect("Sun ephemeris");
            let (moon_pos, _) = ephemeris
                .get_earth_centered_state_typed(EphemerisBody::Moon, tdb)
                .expect("Moon ephemeris");
            sim.set_source_position(1, sun_pos.raw_si());
            sim.set_source_position(2, moon_pos.raw_si());
        },
    );
}

// ── Drag RUN (spherical Earth + constant-density drag, elliptic IC) ──────────

// non-recipe: RUN_6A adds constant-density aerodynamic drag on the highly-
// elliptic IC (taken from the CSV t=0 row). Density is overridden to the JEOD
// constant; the co-rotating-atmosphere wind uses Earth's angular velocity.
#[test]
fn tier3_sim_time_reversal_run6a() {
    let mu = load_mu_earth_gemt1();
    run_reversal_full_state(
        "reversal_run6a",
        "reversal_run6a_reversal.csv",
        false,
        Tols {
            pos: 3.17e-3,
            vel: 3.59e-6,
            tai: 1e-6,
            roundtrip_pos: 3.05e-3,
            roundtrip_vel: 3.42e-6,
        },
        &[],
        |sim, init| {
            let earth = add_spherical_earth(sim, mu);
            // Co-rotating-atmosphere wind for relative velocity; the exponential
            // density model is present only to satisfy the atmosphere stage —
            // `constant_density` overrides its output (JEOD const_density_drag).
            sim.atmosphere = Some(AtmosphereConfig {
                model: AtmosphereModel::Exponential(ExponentialAtmosphere::default()),
                r_eq: EARTH.shape.r_eq(),
                r_pol: EARTH.shape.r_pol(),
                planet_omega: astrodyn::planet_config::EARTH.omega,
            });
            add_standard_body(
                sim,
                init,
                GravityControls {
                    controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
                },
                Some(DragConfig {
                    cd: 0.02,
                    area: 1.0,
                    constant_density: Some(1.4e-12),
                }),
            )
        },
        noop_pre_step,
    );
}

// ── 6-DOF RUN (spherical Earth + rotational state + time-windowed loads) ──────

// non-recipe: RUN_9D propagates the full rotational state (LVLH orbit-rate
// initial attitude, ISS inertia tensor) and applies a 10 kN structural-frame
// force + 10 kN·m torque over exec-time windows [10000,20000] and
// [100000,110000] s. The torque spins the body up to ω≈0.5 rad/s during each
// window, so the body-fixed thrust is re-rotated to inertial at every RK4
// sub-stage via the runner's per-stage `StructuralWrench` path (JEOD_INV DB.28).
// The loads toggle at JEOD's exact window boundaries (passed as `event_bounds`),
// not on the 60 s record grid. Forces break time-symmetry, so the round-trip
// does not return to the initial state (loose round-trip tolerance).
#[test]
fn tier3_sim_time_reversal_run9d() {
    let mu = load_mu_earth_gemt1();
    run_reversal_full_state(
        "reversal_run9d",
        "reversal_run9d_reversal.csv",
        false,
        Tols {
            pos: 1.33e-4,
            vel: 1.55e-7,
            tai: 1e-6,
            roundtrip_pos: 9.2e-6,
            roundtrip_vel: 9.1e-9,
        },
        &[10000.0, 20000.0, 100000.0, 110000.0],
        |sim, init| {
            let earth = add_spherical_earth(sim, mu);
            let quat = lvlh_pitch_quat(init.position, init.velocity);
            // Body co-rotates with the LVLH frame (JEOD lvlh_init.ang_velocity
            // = 0): the LVLH frame's inertial orbit-rate, expressed in body
            // coordinates. (R_y about Y leaves the orbit-normal Y component
            // unchanged; written via the transform for generality.)
            let lvlh = astrodyn::compute_body_lvlh_frame(init.position, init.velocity);
            let ang_vel_body =
                DMat3::from_rotation_y(11.6_f64.to_radians()).transpose() * lvlh.ang_vel_this;
            // ISS mass properties (Modified_data/mass/iss.py): off-diagonal
            // inertia tensor (about the CM, per JEOD `inertia_spec = Body`) with
            // a CM offset from the structural origin.
            let inertia = DMat3::from_cols_array_2d(&[
                [1.02e8, -6.96e6, -5.48e6],
                [-6.96e6, 0.91e8, 5.90e5],
                [-5.48e6, 5.90e5, 1.64e8],
            ]);
            let mass =
                MassProperties::with_inertia(400_000.0, inertia, DVec3::new(-3.0, -1.5, 4.0));
            sim.add_body(VehicleConfig {
                trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
                    position: init.position,
                    velocity: init.velocity,
                }),
                rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
                    &(RotationalState {
                        quaternion: quat,
                        ang_vel_body,
                    }),
                )),
                mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass))),
                gravity_controls: GravityControls {
                    controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
                },
                ..Default::default()
            })
        },
        |sim, body, interval_start, _end| {
            // JEOD `trick.add_read` toggles force_extern at exec time 10000 (on),
            // 20000 (off), 100000 (on), 110000 (off). With `event_bounds`, each
            // integration interval lies entirely inside or outside a window, so
            // the interval-start test reproduces JEOD's continuous schedule.
            let active = (10000.0..20000.0).contains(&interval_start)
                || (100000.0..110000.0).contains(&interval_start);
            let load = if active {
                DVec3::new(10000.0, 0.0, 0.0)
            } else {
                DVec3::ZERO
            };
            sim.set_body_external_force_struct(body, load);
            sim.set_body_external_torque_struct(body, load);
        },
    );
}
