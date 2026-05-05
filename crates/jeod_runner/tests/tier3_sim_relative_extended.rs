//! Tier 3: Extended relative-dynamics tests (analytical).
//!
//! These tests exercise `compute_relative_state` and
//! `compute_lvlh_relative_state` via the full `Simulation::step()` pipeline
//! under closed-form initial conditions, then verify the analytical properties
//! of the resulting trajectories:
//!
//! * `tier3_relative_two_coorbiting_vehicles` — identical circular orbits with a
//!   small along-track offset; verify the LVLH-relative separation stays bounded.
//! * `tier3_relative_hohmann_transfer_geometry` — chief in circular orbit,
//!   deputy in a coplanar ellipse intersecting the chief; verify the separation
//!   oscillates between periapsis and apoapsis bounds.
//! * `tier3_relative_same_orbit_phase_difference` — two vehicles in the same
//!   circular orbit 90° apart; verify chord length = r * sqrt(2).
//! * `tier3_relative_different_inclinations` — chief equatorial, deputy inclined
//!   by 1°; verify cross-track separation oscillates at the orbital period with
//!   amplitude a * sin(1°).
//! * `tier3_relative_round_trip_frames` — r_AB must equal -r_BA when the
//!   reference frame is inertial (no rotational state), confirming the
//!   symmetry of the relative-state operator.
//!
//! No Docker reference data required. Earth mu is read from JEOD source.

use glam::DVec3;
use jeod_runner::{RotationModel, Simulation};
use jeod_sim::{
    compute_lvlh_relative_state_typed, compute_relative_state, Earth, GravityControl,
    GravityControls, GravityModel, GravitySource, PlanetInertial, RelativeTranslation,
    SimulationTime, TranslationalState, Vec3Ext,
};
use jeod_sim::{DerivedStateConfig, GravitySourceEntry, VehicleConfig};

fn load_mu_earth() -> f64 {
    jeod_test_data::gravity_fixtures::load_ggm05c().mu
}

/// Construct a Simulation with a single point-mass Earth at the origin and
/// return `(sim, earth_source_idx)` for the caller to populate with bodies.
fn make_earth_sim(dt: f64, mu_earth: f64) -> (Simulation, usize) {
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);
    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            position: jeod_sim::Position::<jeod_sim::RootInertial>::zero(),
            velocity: jeod_sim::Velocity::<jeod_sim::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );
    (sim, earth)
}

fn add_orbital_body(sim: &mut Simulation, earth: usize, trans: TranslationalState) {
    sim.add_body(VehicleConfig {
        trans,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        derived: DerivedStateConfig {
            lvlh: true,
            ..Default::default()
        },
        ..Default::default()
    });
}

// non-recipe: bespoke geometry — two vehicles on the same circular orbit
// with a small along-track Δν offset. No recipe preset captures this pair.
#[test]
fn tier3_relative_two_coorbiting_vehicles() {
    // Two vehicles on the same 400 km circular equatorial orbit, separated by
    // a small along-track phase. In the chief's LVLH frame the deputy should
    // remain at an almost-constant along-track offset, up to truncation error
    // from the RK4 integrator at a 10 s step.
    let mu_earth = load_mu_earth();
    let dt = 10.0;
    let (mut sim, earth) = make_earth_sim(dt, mu_earth);

    let r = 6_778_137.0; // ~400 km altitude
    let v = (mu_earth / r).sqrt();

    // Chief at (r, 0, 0), velocity +y
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    );

    // Deputy 100 m "ahead" — i.e., offset by a tiny true anomaly Δν
    // so that Δs ≈ r * Δν = 100 m, Δν = 100/r rad.
    let dnu = 100.0 / r;
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(r * dnu.cos(), r * dnu.sin(), 0.0),
            velocity: DVec3::new(-v * dnu.sin(), v * dnu.cos(), 0.0),
        },
    );

    sim.validate().unwrap();

    let period = 2.0 * std::f64::consts::PI * (r * r * r / mu_earth).sqrt();
    let end_time = 3.0 * period;
    let n_steps = (end_time / dt) as usize;

    let mut max_sep = 0.0_f64;
    let mut min_sep = f64::INFINITY;
    let mut max_lvlh_x = 0.0_f64;
    let mut max_lvlh_yz = 0.0_f64;

    for step in 1..=n_steps {
        let t = step as f64 * dt;
        sim.step_until(t).expect("step_until failed");

        let chief = sim.body(0);
        let deputy = sim.body(1);
        let rel_inertial = compute_relative_state(&chief.trans, None, &deputy.trans, None);
        // `None` reference rotation: producer returns the `Inertial`
        // variant, so we read the typed root-inertial position
        // directly (the type assertion would catch any future
        // refactor that flipped the producer's branch convention).
        let RelativeTranslation::Inertial { position, .. } = rel_inertial.trans else {
            panic!("None reference rotation must yield RelativeTranslation::Inertial");
        };
        let sep = position.raw_si().length();
        max_sep = max_sep.max(sep);
        min_sep = min_sep.min(sep);

        // Earth-centered point-mass sim: the integration frame for
        // both bodies is `<PlanetInertial<Earth>>`; tag at the call
        // site to satisfy the typed entry's planet-inertial contract.
        let rel = compute_lvlh_relative_state_typed(
            chief.trans.position.m_at::<PlanetInertial<Earth>>(),
            chief.trans.velocity.m_per_s_at::<PlanetInertial<Earth>>(),
            deputy.trans.position.m_at::<PlanetInertial<Earth>>(),
            deputy.trans.velocity.m_per_s_at::<PlanetInertial<Earth>>(),
        );
        // LVLH: X ≈ along-track, Y ≈ -orbit-normal, Z ≈ -radial.
        // For co-orbiting bodies, the along-track component dominates and
        // the in-plane/out-of-plane excursions stay small.
        let rel_pos = rel.position.raw_si();
        max_lvlh_x = max_lvlh_x.max(rel_pos.x.abs());
        max_lvlh_yz = max_lvlh_yz.max((rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z).sqrt());
    }

    // RootInertial separation stays very close to the 100 m initial chord.
    assert!(
        (max_sep - 100.0).abs() < 0.01,
        "co-orbiting max inertial separation {max_sep} m drifted from 100 m"
    );
    assert!(
        (min_sep - 100.0).abs() < 0.01,
        "co-orbiting min inertial separation {min_sep} m drifted from 100 m"
    );
    // In LVLH the along-track dominates and Y/Z excursions stay tiny
    // (bounded by RK4 truncation over 3 orbits).
    assert!(
        max_lvlh_x > 99.0 && max_lvlh_x < 101.0,
        "LVLH along-track {max_lvlh_x} m not near 100 m"
    );
    assert!(
        max_lvlh_yz < 0.1,
        "LVLH out-of-track excursion {max_lvlh_yz} m exceeds 0.1 m"
    );
}

// non-recipe: chief circular + deputy ellipse with apoapsis at 1.05·r_chief;
// hand-crafted Hohmann-shape geometry not in `recipes::orbital_elements`.
#[test]
fn tier3_relative_hohmann_transfer_geometry() {
    // Chief in circular orbit at 400 km; deputy in a coplanar ellipse whose
    // periapsis coincides with the chief's orbit (same r, same direction of
    // motion). The separation |r_deputy - r_chief| must oscillate because the
    // deputy climbs to apoapsis and falls back.
    let mu_earth = load_mu_earth();
    let dt = 10.0;
    let (mut sim, earth) = make_earth_sim(dt, mu_earth);

    let r_chief = 6_778_137.0;
    let v_chief = (mu_earth / r_chief).sqrt();

    // Chief: circular
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(r_chief, 0.0, 0.0),
            velocity: DVec3::new(0.0, v_chief, 0.0),
        },
    );

    // Deputy: periapsis at (r_chief, 0, 0), same direction of motion,
    // apoapsis at r_apo = 1.05 * r_chief.
    let r_apo = 1.05 * r_chief;
    let a_d = 0.5 * (r_chief + r_apo);
    let e_d = (r_apo - r_chief) / (r_apo + r_chief);
    let v_peri = (mu_earth * (1.0 + e_d) / (a_d * (1.0 - e_d))).sqrt();
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(r_chief, 0.0, 0.0),
            velocity: DVec3::new(0.0, v_peri, 0.0),
        },
    );

    sim.validate().unwrap();

    // Initial separation: both bodies at the same point → 0.
    let init_rel = compute_relative_state(&sim.body(0).trans, None, &sim.body(1).trans, None);
    let init_sep = init_rel.trans.position_raw().length();
    assert!(
        init_sep < 1e-9,
        "Hohmann setup: initial separation {init_sep} m should be 0"
    );

    // Propagate for one deputy period so the deputy returns to its periapsis.
    let period_d = 2.0 * std::f64::consts::PI * (a_d * a_d * a_d / mu_earth).sqrt();
    let n_steps = (period_d / dt) as usize;

    let mut max_sep = 0.0_f64;
    // Track separation near apoapsis: at T_d/2, deputy is at (-r_apo, ..., 0).
    // At that time the chief has advanced by T_d/2 * ω_chief > π, so its
    // position is indeterminate by analytical inspection — but the range
    // |r_a - r_c| ≤ sep ≤ r_a + r_c must hold.

    for step in 1..=n_steps {
        let t = step as f64 * dt;
        sim.step_until(t).expect("step_until failed");

        let chief = sim.body(0);
        let deputy = sim.body(1);
        let rel = compute_relative_state(&chief.trans, None, &deputy.trans, None);
        let sep = rel.trans.position_raw().length();
        max_sep = max_sep.max(sep);
    }

    // Upper bound: r_apo + r_chief (deputy-apoapsis, chief-antipode).
    // Lower bound on achievable max separation: r_apo - r_chief when they
    // happen to be co-radial at opposite points.
    assert!(
        max_sep > 0.8 * (r_apo - r_chief),
        "Hohmann-geometry max separation {max_sep} m below expected oscillation"
    );
    assert!(
        max_sep < r_apo + r_chief + 1.0,
        "Hohmann-geometry max separation {max_sep} m exceeds geometric bound {}",
        r_apo + r_chief
    );
}

// non-recipe: two-body 90° phase-difference geometry; setup is the assertion.
#[test]
fn tier3_relative_same_orbit_phase_difference() {
    // Two vehicles in the same circular orbit, 90° apart in true anomaly.
    // At t=0 their positions are on the same circle of radius r, subtending
    // 90° at Earth, so the chord length is r * sqrt(2). Because they share
    // the orbital period, this separation is preserved forever.
    let mu_earth = load_mu_earth();
    let dt = 10.0;
    let (mut sim, earth) = make_earth_sim(dt, mu_earth);

    let r = 6_778_137.0;
    let v = (mu_earth / r).sqrt();

    // Body A: at (r, 0, 0), velocity (0, v, 0)
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    );

    // Body B: at (0, r, 0) [90° ahead], velocity (-v, 0, 0)
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(0.0, r, 0.0),
            velocity: DVec3::new(-v, 0.0, 0.0),
        },
    );

    sim.validate().unwrap();

    let expected_sep = r * std::f64::consts::SQRT_2;
    let period = 2.0 * std::f64::consts::PI * (r * r * r / mu_earth).sqrt();

    let mut max_dev = 0.0_f64;

    // Check at t=0 first (before any step)
    {
        let a = sim.body(0);
        let b = sim.body(1);
        let rel = compute_relative_state(&a.trans, None, &b.trans, None);
        let sep = rel.trans.position_raw().length();
        max_dev = max_dev.max((sep - expected_sep).abs());
    }

    // Check every dt seconds over 2 orbits
    let n_steps = (2.0 * period / dt) as usize;
    for step in 1..=n_steps {
        let t = step as f64 * dt;
        sim.step_until(t).expect("step_until failed");
        let a = sim.body(0);
        let b = sim.body(1);
        let rel = compute_relative_state(&a.trans, None, &b.trans, None);
        let sep = rel.trans.position_raw().length();
        max_dev = max_dev.max((sep - expected_sep).abs());
    }

    // RK4 on a 10 s step over 2 orbits produces sub-mm deviations.
    assert!(
        max_dev < 1e-2,
        "same-orbit 90° phase: separation drift {max_dev} m exceeds 1e-2 m"
    );
}

// non-recipe: chief equatorial vs deputy at +1° inclination — bespoke
// cross-track geometry test.
#[test]
fn tier3_relative_different_inclinations() {
    // Chief on equatorial circular orbit; deputy on the same circular orbit
    // inclined by +1° about the +X axis (i.e., node on the +X axis, AOL 0).
    // Both start at (r, 0, 0). Cross-track excursion (out-of-plane in chief's
    // frame) oscillates at the orbital period with amplitude r * sin(1°).
    let mu_earth = load_mu_earth();
    let dt = 5.0;
    let (mut sim, earth) = make_earth_sim(dt, mu_earth);

    let r = 6_778_137.0;
    let v = (mu_earth / r).sqrt();
    let inc = 1.0_f64.to_radians();

    // Chief: equatorial
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    );

    // Deputy: inclined, same initial position, velocity tipped into +Z
    // by inc. Rotation about +X by +inc sends (0, v, 0) → (0, v cos i, v sin i).
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, v * inc.cos(), v * inc.sin()),
        },
    );

    sim.validate().unwrap();

    let period = 2.0 * std::f64::consts::PI * (r * r * r / mu_earth).sqrt();
    let n_steps = (2.0 * period / dt) as usize;
    let expected_amplitude = r * inc.sin();

    let mut max_abs_z = 0.0_f64; // inertial Z separation (cross-track)
    for step in 1..=n_steps {
        let t = step as f64 * dt;
        sim.step_until(t).expect("step_until failed");

        let chief = sim.body(0);
        let deputy = sim.body(1);
        let dz = (deputy.trans.position - chief.trans.position).z.abs();
        max_abs_z = max_abs_z.max(dz);
    }

    // Amplitude should match r * sin(inc) within integration tolerance.
    let amp_err = (max_abs_z - expected_amplitude).abs();
    assert!(
        amp_err < 1.0,
        "cross-track amplitude error {amp_err} m (got {max_abs_z}, \
         expected {expected_amplitude}) exceeds 1 m"
    );
}

// non-recipe: two orbits at different radii to verify r_AB = -r_BA symmetry;
// the geometry is the test.
#[test]
fn tier3_relative_round_trip_frames() {
    // When neither body has a rotational state, the relative-state operator
    // has a clean symmetry: r_AB = -r_BA and v_AB = -v_BA. We verify this on
    // the propagated trajectory at several checkpoints.
    let mu_earth = load_mu_earth();
    let dt = 10.0;
    let (mut sim, earth) = make_earth_sim(dt, mu_earth);

    let r = 6_778_137.0;
    let v = (mu_earth / r).sqrt();

    // Body 0 at (r, 0, 0); Body 1 in a different coplanar circular orbit.
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(r, 0.0, 0.0),
            velocity: DVec3::new(0.0, v, 0.0),
        },
    );
    let r2 = r + 500_000.0;
    let v2 = (mu_earth / r2).sqrt();
    add_orbital_body(
        &mut sim,
        earth,
        TranslationalState {
            position: DVec3::new(r2, 0.0, 0.0),
            velocity: DVec3::new(0.0, v2, 0.0),
        },
    );

    sim.validate().unwrap();

    let period = 2.0 * std::f64::consts::PI * (r * r * r / mu_earth).sqrt();
    let n_steps = (period / dt) as usize;

    let mut max_sum_pos = 0.0_f64;
    let mut max_sum_vel = 0.0_f64;

    for step in 1..=n_steps {
        let t = step as f64 * dt;
        sim.step_until(t).expect("step_until failed");

        let a = sim.body(0);
        let b = sim.body(1);

        // State of A wrt B (reference = B, subject = A).
        let a_wrt_b = compute_relative_state(&b.trans, None, &a.trans, None);
        // State of B wrt A (reference = A, subject = B).
        let b_wrt_a = compute_relative_state(&a.trans, None, &b.trans, None);

        // Both call sites pass `None` for the reference rotation, so
        // the producer always lands in the `Inertial` variant. We
        // pattern-match both sides so the typed `Position<RootInertial>`
        // values can be added directly through the typed `+` operator
        // — adding a body-frame and an inertial-frame phantom would
        // be a compile error, which is exactly the symmetry guard we
        // want for this round-trip property test.
        let RelativeTranslation::Inertial {
            position: a_pos,
            velocity: a_vel,
        } = a_wrt_b.trans
        else {
            panic!("None reference rotation must yield RelativeTranslation::Inertial");
        };
        let RelativeTranslation::Inertial {
            position: b_pos,
            velocity: b_vel,
        } = b_wrt_a.trans
        else {
            panic!("None reference rotation must yield RelativeTranslation::Inertial");
        };
        let pos_sum = (a_pos.raw_si() + b_pos.raw_si()).length();
        let vel_sum = (a_vel.raw_si() + b_vel.raw_si()).length();
        max_sum_pos = max_sum_pos.max(pos_sum);
        max_sum_vel = max_sum_vel.max(vel_sum);
    }

    // The two relative states must add to zero (floating-point exact by
    // construction: both compute (p_s - p_r) with opposite role assignments).
    assert!(
        max_sum_pos < 1e-9,
        "round-trip: a_wrt_b.position + b_wrt_a.position has magnitude {max_sum_pos} m"
    );
    assert!(
        max_sum_vel < 1e-9,
        "round-trip: a_wrt_b.velocity + b_wrt_a.velocity has magnitude {max_sum_vel} m/s"
    );
}
