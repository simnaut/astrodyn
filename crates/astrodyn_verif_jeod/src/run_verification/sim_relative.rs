//! `VerificationCase` constructors for the SIM_Relative two-body
//! kinematic verification family.
//!
//! Two free-flying 6-DOF bodies (no gravity, no force) are propagated
//! through `Simulation::step()`; bit-identity between the runner and
//! the Bevy adapter is the parity contract. The runner-vs-JEOD
//! cross-validation lives in
//! `crates/astrodyn_verif_jeod/tests/tier3_sim_relative.rs`.
//!
//! Variants differ only in initial conditions. Each one mirrors a
//! hand-rolled test in `bevy_parity_relative.rs`:
//!
//! - [`relative_ab_rot_ab_trans`]: distinct quaternions and translational
//!   states for both bodies.
//! - [`relative_no_rot_ab_trans`]: identity rotation on both, distinct
//!   translational states.
//! - [`relative_a_rot_no_trans`]: identity translational state on both,
//!   distinct quaternions.
//! - [`lvlhrel_test0`]: 3-DOF coplanar formation with lateral offset.
//! - [`lvlhrel_test1`]: 3-DOF coplanar formation with along-track offset.
//!
//! ## CSV reference
//!
//! The recipes route through [`CsvReference::Relative`], a 57-column
//! variant the central `tier3_csv` loader knows how to parse. The
//! parity trait still consumes only the time column for cadence
//! (parity asserts bit-identity body-by-body, not against
//! JEOD-logged state), but the runner-vs-JEOD oracle now has
//! everything it needs to assert the bespoke
//! `compute_relative_state` metric directly through
//! [`ExtrasComparator::Relative`] rather than re-implementing the CSV
//! parse + per-step compare in a hand-rolled `tier3_sim_relative.rs`
//! body.

use crate::tier3_csv::{load_relative_csv, test_data_path};
use crate::verification::{
    CsvReference, ExtrasComparator, InitialConditions, Tolerances, VerificationCase,
};
use astrodyn::{
    default_leap_second_table, JeodQuat, MassProperties, RotationalState, SimulationBuilder,
    SimulationTime, TranslationalState, VehicleConfig,
};
use glam::DVec3;
use uom::si::f64::Time;
use uom::si::time::second;

/// SIM_Relative cadence: CSV records every 1 s.
const RELATIVE_DT: f64 = 1.0;

/// Build a 2-body, force-free simulation. Body 0 is the "subject";
/// body 1 is the "reference". 6-DOF flavor uses `rot: Some(_)` and a
/// unit dummy mass (rotational dynamics requires non-zero mass); 3-DOF
/// flavor passes `rot: None` so `VehicleConfig` collapses to
/// translational-only dynamics. The 3-DOF / 6-DOF distinction is
/// inferred from `rot.is_some()` by the runner and the Bevy adapter
/// (see `VehicleConfigBevyExt::spawn_bevy`).
fn build_two_body(
    trans_a: TranslationalState,
    rot_a: RotationalState,
    trans_b: TranslationalState,
    rot_b: RotationalState,
    sixdof: bool,
) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut b = SimulationBuilder::new(time, RELATIVE_DT);
    let dummy_mass = MassProperties::new(1.0);
    // allowed: typed↔raw kernel-boundary lifts at the public scenario
    // builder (named-method opt-in; see #397).
    b.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&trans_a),
        rot: if sixdof {
            Some(super::typed_helpers::rot_typed(&rot_a))
        } else {
            None
        },
        mass: if sixdof {
            Some(super::typed_helpers::mass_typed(&dummy_mass))
        } else {
            None
        },
        ..VehicleConfig::named("sim-relative-1")
    });
    b.add_body(VehicleConfig {
        trans: super::typed_helpers::trans_typed(&trans_b),
        rot: if sixdof {
            Some(super::typed_helpers::rot_typed(&rot_b))
        } else {
            None
        },
        mass: if sixdof {
            Some(super::typed_helpers::mass_typed(&dummy_mass))
        } else {
            None
        },
        ..VehicleConfig::named("sim-relative-0")
    });
    b
}

// ── 6-DOF variants ──
//
// The 6-DOF cases match the JEOD SIM_Relative reference CSV directly:
// every body's t=0 state is read from the matching CSV's first non-
// header row, so the runner-vs-JEOD oracle (`tier3_sim_relative.rs`)
// can assert `compute_relative_state` against the JEOD-logged
// relative-state columns. The factories take `_init` (which is body 0's
// t=0, populated by the central `load_reference` from CSV columns 1–6)
// and re-read the CSV to also get body B's t=0 plus quaternions /
// angular velocities for both bodies.

fn iss_like_trans_a() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    }
}

fn rot_zero() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    }
}

/// Read both bodies' t=0 state from the matching `relative_*_relative.csv`
/// fixture. Returns `(trans_a, rot_a, trans_b, rot_b)`. Panics with the
/// standard "regenerate Tier 3 reference" diagnostic if the CSV is
/// missing.
fn ics_from_csv(
    csv_name: &str,
) -> (
    TranslationalState,
    RotationalState,
    TranslationalState,
    RotationalState,
) {
    let path = test_data_path(csv_name);
    let records = load_relative_csv(&path);
    assert!(
        !records.is_empty(),
        "ics_from_csv({csv_name}): reference CSV produced 0 records"
    );
    let r = &records[0];
    let trans_a = TranslationalState {
        position: r.veh_a_pos,
        velocity: r.veh_a_vel,
    };
    let rot_a = RotationalState {
        quaternion: JeodQuat::new(
            r.veh_a_quat[0],
            r.veh_a_quat[1],
            r.veh_a_quat[2],
            r.veh_a_quat[3],
        ),
        ang_vel_body: r.veh_a_ang_vel,
    };
    let trans_b = TranslationalState {
        position: r.veh_b_pos,
        velocity: r.veh_b_vel,
    };
    let rot_b = RotationalState {
        quaternion: JeodQuat::new(
            r.veh_b_quat[0],
            r.veh_b_quat[1],
            r.veh_b_quat[2],
            r.veh_b_quat[3],
        ),
        ang_vel_body: r.veh_b_ang_vel,
    };
    (trans_a, rot_a, trans_b, rot_b)
}

fn build_ab_rot_ab_trans(_init: &InitialConditions) -> SimulationBuilder {
    let (trans_a, rot_a, trans_b, rot_b) = ics_from_csv("relative_ab_rot_ab_trans_relative.csv");
    build_two_body(trans_a, rot_a, trans_b, rot_b, true)
}

fn build_no_rot_ab_trans(_init: &InitialConditions) -> SimulationBuilder {
    let (trans_a, rot_a, trans_b, rot_b) = ics_from_csv("relative_no_rot_ab_trans_relative.csv");
    build_two_body(trans_a, rot_a, trans_b, rot_b, true)
}

fn build_a_rot_no_trans(_init: &InitialConditions) -> SimulationBuilder {
    let (trans_a, rot_a, trans_b, rot_b) = ics_from_csv("relative_a_rot_no_trans_relative.csv");
    build_two_body(trans_a, rot_a, trans_b, rot_b, true)
}

// ── 3-DOF LVLH-relative variants ──

fn build_lvlhrel(subj_offset: DVec3, subj_velocity: DVec3) -> SimulationBuilder {
    let ref_trans = iss_like_trans_a();
    let subj_trans = TranslationalState {
        position: ref_trans.position + subj_offset,
        velocity: subj_velocity,
    };
    build_two_body(ref_trans, rot_zero(), subj_trans, rot_zero(), false)
}

fn build_lvlhrel_test0(_init: &InitialConditions) -> SimulationBuilder {
    // Lateral offset (100 m radial, 100 m along-track, -50 m cross-track).
    build_lvlhrel(
        DVec3::new(100.0, 100.0, -50.0),
        DVec3::new(0.01, 7668.55, 0.005),
    )
}

fn build_lvlhrel_test1(_init: &InitialConditions) -> SimulationBuilder {
    // Coplanar formation, 1 km radial separation with slightly slower v.
    build_lvlhrel(DVec3::new(1000.0, 0.0, 0.0), DVec3::new(0.0, 7667.56, 0.0))
}

// ── VerificationCase factories ──

/// Shared "use full CSV" duration sentinel. The [`VerificationCase`]
/// contract treats `<= 0` as "consume every record"; this helper
/// keeps the factories below as table-of-cases data without each row
/// repeating the same `Time::new::<second>(0.0)` literal.
fn full_csv_duration() -> Time {
    Time::new::<second>(0.0)
}

fn zero_tolerances() -> Tolerances {
    // Parity tests don't compare against JEOD, so tolerances aren't
    // exercised by the runner-vs-JEOD path. Set every component to
    // zero so the runner-side `run_and_assert` (if accidentally
    // invoked on these recipes) opts out of every assertion via the
    // documented "all-zero skips the metric group" rule.
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: &[],
    }
}

/// Tolerance literals inherited verbatim from the bespoke
/// `tier3_sim_relative.rs` (`max_pos_err < 3.8e-5`,
/// `max_vel_err < 3.0e-6`). The metric is the magnitude of the
/// relative-state error vector, so it lands on a single per-step
/// scalar; expose it through the `extras` channel.
const RELATIVE_EXTRAS_TOL: &[(&str, f64)] = &[("rel_pos", 3.8e-5), ("rel_vel", 3.0e-6)];

fn relative_tolerances() -> Tolerances {
    Tolerances {
        position_m: [0.0; 3],
        velocity_m_s: [0.0; 3],
        quat_angle_rad: 0.0,
        ang_vel_rad_s: [0.0; 3],
        extras: RELATIVE_EXTRAS_TOL,
    }
}

/// 6-DOF: distinct quaternions and translational states for both bodies.
pub fn relative_ab_rot_ab_trans() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_relative_ab_rot_ab_trans",
        scenario: build_ab_rot_ab_trans,
        reference: CsvReference::Relative("relative_ab_rot_ab_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: relative_tolerances(),
        extras: Some(ExtrasComparator::Relative),
        pre_step: None,
    }
}

/// 6-DOF: identity rotation, distinct translational states.
pub fn relative_no_rot_ab_trans() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_relative_no_rot_ab_trans",
        scenario: build_no_rot_ab_trans,
        reference: CsvReference::Relative("relative_no_rot_ab_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: relative_tolerances(),
        extras: Some(ExtrasComparator::Relative),
        pre_step: None,
    }
}

/// 6-DOF: identity translational state, distinct rotations.
pub fn relative_a_rot_no_trans() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_relative_a_rot_no_trans",
        scenario: build_a_rot_no_trans,
        reference: CsvReference::Relative("relative_a_rot_no_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: relative_tolerances(),
        extras: Some(ExtrasComparator::Relative),
        pre_step: None,
    }
}

/// 3-DOF LVLH-relative: lateral offset.
pub fn lvlhrel_test0() -> VerificationCase {
    VerificationCase {
        name: "bevy_parity_lvlh_relative_lvlhrel_test0",
        scenario: build_lvlhrel_test0,
        reference: CsvReference::TimesOnly("relative_ab_rot_ab_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// 3-DOF LVLH-relative: coplanar along-track separation.
pub fn lvlhrel_test1() -> VerificationCase {
    VerificationCase {
        name: "bevy_parity_lvlh_relative_lvlhrel_test1",
        scenario: build_lvlhrel_test1,
        reference: CsvReference::TimesOnly("relative_ab_rot_ab_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}
