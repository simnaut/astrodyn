// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
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
//! No Docker reference data required. The `Simulation` construction lives
//! in the `sim_relative_extended` recipe module so the parity wrapper
//! (`bevy_parity_relative_extended.rs`) can drive the same scenarios
//! through the Bevy adapter for the `runner ↔ bevy` half of the
//! transitivity argument.

use astrodyn::{
    compute_lvlh_relative_state_typed, compute_relative_state, Earth, PlanetInertial,
    RelativeTranslation, SelfRef, Vec3Ext,
};
use astrodyn_runner::builder::SimulationBuilderExt;
use astrodyn_runner::Simulation;
use astrodyn_verif_jeod::run_verification::sim_relative_extended;
use astrodyn_verif_jeod::verification::{CsvReference, InitialConditions, VerificationCase};

/// Build the recipe's `Simulation` exactly the way the parity trait does
/// — call the scenario factory with a default `InitialConditions`, then
/// `.build()` — so the runner-side propagation here and the Bevy-side
/// propagation in `bevy_parity_relative_extended.rs` see the same
/// initial state bit-pattern.
fn build_sim(case: &VerificationCase) -> Simulation {
    (case.scenario)(&InitialConditions::default())
        .build()
        .unwrap_or_else(|e| panic!("scenario `{}` build failed: {e:?}", case.name))
}

/// Pull `(dt, num_steps)` off a recipe's [`CsvReference::SyntheticTimes`]
/// reference. Every recipe in `sim_relative_extended` uses this variant
/// because the family is analytical-only; panicking on any other
/// variant surfaces a future recipe-shape drift here rather than
/// producing a silently-truncated propagation. Returning both halves
/// of the cadence lets callers assert that the `dt` they're stepping
/// at (typically `sim.dt`) matches the cadence the recipe declared —
/// catches a future edit that updates the builder dt but forgets the
/// `SyntheticTimes` dt (or vice versa).
fn synthetic_cadence(case: &VerificationCase) -> (f64, usize) {
    match &case.reference {
        CsvReference::SyntheticTimes { dt, num_steps } => (*dt, *num_steps),
        _ => panic!("`{}`: expected SyntheticTimes reference", case.name),
    }
}

#[test]
fn tier3_relative_two_coorbiting_vehicles() {
    // Two vehicles on the same 400 km circular equatorial orbit, separated by
    // a small along-track phase. In the chief's LVLH frame the deputy should
    // remain at an almost-constant along-track offset, up to truncation error
    // from the RK4 integrator at a 10 s step.
    let case = sim_relative_extended::two_coorbiting_vehicles();
    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    let mut max_sep = 0.0_f64;
    let mut min_sep = f64::INFINITY;
    let mut max_lvlh_x = 0.0_f64;
    let mut max_lvlh_yz = 0.0_f64;

    for step in 1..=n_steps {
        let t = step as f64 * dt;
        sim.step_until(t).expect("step_until failed");

        let chief = sim.body(0);
        let deputy = sim.body(1);
        let chief_trans = astrodyn::TranslationalState {
            position: chief.trans.position.raw_si(),
            velocity: chief.trans.velocity.raw_si(),
        };
        let deputy_trans = astrodyn::TranslationalState {
            position: deputy.trans.position.raw_si(),
            velocity: deputy.trans.velocity.raw_si(),
        };
        let rel_inertial =
            compute_relative_state::<SelfRef, SelfRef>(&chief_trans, None, &deputy_trans, None);
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
        let rel = compute_lvlh_relative_state_typed::<Earth, SelfRef>(
            chief
                .trans
                .position
                .raw_si()
                .m_at::<PlanetInertial<Earth>>(),
            chief
                .trans
                .velocity
                .raw_si()
                .m_per_s_at::<PlanetInertial<Earth>>(),
            deputy
                .trans
                .position
                .raw_si()
                .m_at::<PlanetInertial<Earth>>(),
            deputy
                .trans
                .velocity
                .raw_si()
                .m_per_s_at::<PlanetInertial<Earth>>(),
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

#[test]
fn tier3_relative_hohmann_transfer_geometry() {
    // Chief in circular orbit at 400 km; deputy in a coplanar ellipse whose
    // periapsis coincides with the chief's orbit (same r, same direction of
    // motion). The separation |r_deputy - r_chief| must oscillate because the
    // deputy climbs to apoapsis and falls back.
    let case = sim_relative_extended::hohmann_transfer_geometry();
    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    // Initial separation: both bodies at the same point → 0.
    let body0 = sim.body(0);
    let body1 = sim.body(1);
    let body0_trans = astrodyn::typed_bridge::trans_typed_to_raw(&body0.trans);
    let body1_trans = astrodyn::typed_bridge::trans_typed_to_raw(&body1.trans);
    let init_rel =
        compute_relative_state::<SelfRef, SelfRef>(&body0_trans, None, &body1_trans, None);
    let init_sep = init_rel.trans.position_raw().length();
    assert!(
        init_sep < 1e-9,
        "Hohmann setup: initial separation {init_sep} m should be 0"
    );

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
        let rel = {
            let chief_t = astrodyn::TranslationalState {
                position: chief.trans.position.raw_si(),
                velocity: chief.trans.velocity.raw_si(),
            };
            let deputy_t = astrodyn::TranslationalState {
                position: deputy.trans.position.raw_si(),
                velocity: deputy.trans.velocity.raw_si(),
            };
            compute_relative_state::<SelfRef, SelfRef>(&chief_t, None, &deputy_t, None)
        };
        let sep = rel.trans.position_raw().length();
        max_sep = max_sep.max(sep);
    }

    // The recipe-encoded chief/apoapsis geometry: r_chief = 6_778_137 m,
    // r_apo = 1.05 * r_chief. Reconstruct the bounds locally so the
    // assertions match the recipe-driven initial state exactly.
    let r_chief = 6_778_137.0_f64;
    let r_apo = 1.05 * r_chief;
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

#[test]
fn tier3_relative_same_orbit_phase_difference() {
    // Two vehicles in the same circular orbit, 90° apart in true anomaly.
    // At t=0 their positions are on the same circle of radius r, subtending
    // 90° at Earth, so the chord length is r * sqrt(2). Because they share
    // the orbital period, this separation is preserved forever.
    let case = sim_relative_extended::same_orbit_phase_difference();
    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    // Recipe places both bodies on a 6_778_137 m circle 90° apart, so
    // the expected chord is r * sqrt(2). Reconstruct locally so the
    // assertion matches the recipe-encoded initial state exactly.
    let r = 6_778_137.0_f64;
    let expected_sep = r * std::f64::consts::SQRT_2;

    let mut max_dev = 0.0_f64;

    // Check at t=0 first (before any step)
    {
        let a = sim.body(0);
        let b = sim.body(1);
        let rel = {
            let a_t = astrodyn::TranslationalState {
                position: a.trans.position.raw_si(),
                velocity: a.trans.velocity.raw_si(),
            };
            let b_t = astrodyn::TranslationalState {
                position: b.trans.position.raw_si(),
                velocity: b.trans.velocity.raw_si(),
            };
            compute_relative_state::<SelfRef, SelfRef>(&a_t, None, &b_t, None)
        };
        let sep = rel.trans.position_raw().length();
        max_dev = max_dev.max((sep - expected_sep).abs());
    }

    for step in 1..=n_steps {
        let t = step as f64 * dt;
        sim.step_until(t).expect("step_until failed");
        let a = sim.body(0);
        let b = sim.body(1);
        let rel = {
            let a_t = astrodyn::TranslationalState {
                position: a.trans.position.raw_si(),
                velocity: a.trans.velocity.raw_si(),
            };
            let b_t = astrodyn::TranslationalState {
                position: b.trans.position.raw_si(),
                velocity: b.trans.velocity.raw_si(),
            };
            compute_relative_state::<SelfRef, SelfRef>(&a_t, None, &b_t, None)
        };
        let sep = rel.trans.position_raw().length();
        max_dev = max_dev.max((sep - expected_sep).abs());
    }

    // RK4 on a 10 s step over 2 orbits produces sub-mm deviations.
    assert!(
        max_dev < 1e-2,
        "same-orbit 90° phase: separation drift {max_dev} m exceeds 1e-2 m"
    );
}

#[test]
fn tier3_relative_different_inclinations() {
    // Chief on equatorial circular orbit; deputy on the same circular orbit
    // inclined by +1° about the +X axis (i.e., node on the +X axis, AOL 0).
    // Both start at (r, 0, 0). Cross-track excursion (out-of-plane in chief's
    // frame) oscillates at the orbital period with amplitude r * sin(1°).
    let case = sim_relative_extended::different_inclinations();
    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    // Recipe places both bodies at (r, 0, 0) with the deputy's velocity
    // tipped by +1° about +X. The expected cross-track amplitude is
    // r * sin(i). Reconstruct locally so the assertion matches the
    // recipe-encoded initial state exactly.
    let r = 6_778_137.0_f64;
    let inc = 1.0_f64.to_radians();
    let expected_amplitude = r * inc.sin();

    let mut max_abs_z = 0.0_f64; // inertial Z separation (cross-track)
    for step in 1..=n_steps {
        let t = step as f64 * dt;
        sim.step_until(t).expect("step_until failed");

        let chief = sim.body(0);
        let deputy = sim.body(1);
        let dz = (deputy.trans.position.raw_si() - chief.trans.position.raw_si())
            .z
            .abs();
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

#[test]
fn tier3_relative_round_trip_frames() {
    // When neither body has a rotational state, the relative-state operator
    // has a clean symmetry: r_AB = -r_BA and v_AB = -v_BA. We verify this on
    // the propagated trajectory at several checkpoints.
    let case = sim_relative_extended::round_trip_frames();
    let mut sim = build_sim(&case);
    let (dt, n_steps) = synthetic_cadence(&case);
    assert_eq!(
        dt, sim.dt,
        "`{}`: recipe SyntheticTimes dt ({dt}) and Simulation dt ({}) drifted apart",
        case.name, sim.dt
    );

    let mut max_sum_pos = 0.0_f64;
    let mut max_sum_vel = 0.0_f64;

    for step in 1..=n_steps {
        let t = step as f64 * dt;
        sim.step_until(t).expect("step_until failed");

        let a = sim.body(0);
        let b = sim.body(1);

        // State of A wrt B (reference = B, subject = A).
        let a_wrt_b = {
            let b_t = astrodyn::TranslationalState {
                position: b.trans.position.raw_si(),
                velocity: b.trans.velocity.raw_si(),
            };
            let a_t = astrodyn::TranslationalState {
                position: a.trans.position.raw_si(),
                velocity: a.trans.velocity.raw_si(),
            };
            compute_relative_state::<SelfRef, SelfRef>(&b_t, None, &a_t, None)
        };
        // State of B wrt A (reference = A, subject = B).
        let b_wrt_a = {
            let a_t = astrodyn::TranslationalState {
                position: a.trans.position.raw_si(),
                velocity: a.trans.velocity.raw_si(),
            };
            let b_t = astrodyn::TranslationalState {
                position: b.trans.position.raw_si(),
                velocity: b.trans.velocity.raw_si(),
            };
            compute_relative_state::<SelfRef, SelfRef>(&a_t, None, &b_t, None)
        };

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
