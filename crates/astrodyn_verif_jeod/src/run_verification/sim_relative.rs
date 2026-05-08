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
//! The recipes route through [`CsvReference::OrbInit`] for cadence
//! lookup only. The `relative_*_relative.csv` reference files have a
//! 57-column interleaved-state layout that the OrbInit loader cannot
//! parse correctly, but the parity trait reads only `record.time` from
//! the CSV (initial conditions are hardcoded in each scenario factory),
//! so the misparsed position/velocity columns at t=0 are never read.
//!
//! The tier3 sibling does need the full CSV layout; it uses a
//! private hand-rolled `load_relative_csv` parser, not the recipe path.

use crate::verification::{CsvReference, InitialConditions, Tolerances, VerificationCase};
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
    b.add_body(VehicleConfig {
        trans: trans_a,
        rot: if sixdof { Some(rot_a) } else { None },
        mass: if sixdof { Some(dummy_mass) } else { None },
        ..Default::default()
    });
    b.add_body(VehicleConfig {
        trans: trans_b,
        rot: if sixdof { Some(rot_b) } else { None },
        mass: if sixdof { Some(dummy_mass) } else { None },
        ..Default::default()
    });
    b
}

// ── 6-DOF variants ──

fn iss_like_trans_a() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(6_778_137.0, 0.0, 0.0),
        velocity: DVec3::new(0.0, 7668.56, 0.0),
    }
}

fn iss_like_trans_b() -> TranslationalState {
    TranslationalState {
        position: DVec3::new(6_778_237.0, 100.0, -50.0),
        velocity: DVec3::new(0.01, 7668.55, 0.005),
    }
}

fn rot_a_tumble() -> RotationalState {
    let mut q = JeodQuat::new(0.5_f64.sqrt(), 0.5, 0.0, 0.5_f64.sqrt() - 0.5);
    q.normalize();
    RotationalState {
        quaternion: q,
        ang_vel_body: DVec3::new(0.001, -0.0005, 0.001),
    }
}

fn rot_b_z_spin() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::new(0.0, 0.0, 0.001),
    }
}

fn rot_zero() -> RotationalState {
    RotationalState {
        quaternion: JeodQuat::identity(),
        ang_vel_body: DVec3::ZERO,
    }
}

fn build_ab_rot_ab_trans(_init: &InitialConditions) -> SimulationBuilder {
    build_two_body(
        iss_like_trans_a(),
        rot_a_tumble(),
        iss_like_trans_b(),
        rot_b_z_spin(),
        true,
    )
}

fn build_no_rot_ab_trans(_init: &InitialConditions) -> SimulationBuilder {
    build_two_body(
        iss_like_trans_a(),
        rot_zero(),
        iss_like_trans_b(),
        rot_zero(),
        true,
    )
}

fn build_a_rot_no_trans(_init: &InitialConditions) -> SimulationBuilder {
    // Same translational ICs for both — only rotation differs.
    let trans = iss_like_trans_a();
    build_two_body(trans, rot_a_tumble(), trans, rot_zero(), true)
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

/// 6-DOF: distinct quaternions and translational states for both bodies.
pub fn relative_ab_rot_ab_trans() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_relative_ab_rot_ab_trans",
        scenario: build_ab_rot_ab_trans,
        reference: CsvReference::OrbInit("relative_ab_rot_ab_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// 6-DOF: identity rotation, distinct translational states.
pub fn relative_no_rot_ab_trans() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_relative_no_rot_ab_trans",
        scenario: build_no_rot_ab_trans,
        reference: CsvReference::OrbInit("relative_no_rot_ab_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// 6-DOF: identity translational state, distinct rotations.
pub fn relative_a_rot_no_trans() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_relative_a_rot_no_trans",
        scenario: build_a_rot_no_trans,
        reference: CsvReference::OrbInit("relative_a_rot_no_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// 3-DOF LVLH-relative: lateral offset.
pub fn lvlhrel_test0() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_lvlhrel_test0",
        scenario: build_lvlhrel_test0,
        reference: CsvReference::OrbInit("relative_ab_rot_ab_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}

/// 3-DOF LVLH-relative: coplanar along-track separation.
pub fn lvlhrel_test1() -> VerificationCase {
    VerificationCase {
        name: "tier3_bevy_lvlhrel_test1",
        scenario: build_lvlhrel_test1,
        reference: CsvReference::OrbInit("relative_ab_rot_ab_trans_relative.csv"),
        duration: full_csv_duration(),
        tolerances: zero_tolerances(),
        extras: None,
        pre_step: None,
    }
}
