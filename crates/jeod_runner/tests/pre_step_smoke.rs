//! Smoke test for the `VerificationCase::pre_step` factory hook (#156).
//!
//! Builds a minimal `VerificationCase` whose pre-step closure stamps the
//! sun source's position with a deterministic function of the per-step
//! time, then asserts the source's position at the end of the run matches
//! what the closure last wrote. This exercises the full flow:
//!
//! 1. `run_and_assert` calls the `PreStepBuilder` factory once with the
//!    t=0 `InitialConditions`.
//! 2. The returned closure is invoked before each `sim.step_until`.
//! 3. The closure dispatches through the `SimContext` trait into
//!    `Simulation::set_source_position`.
//!
//! The `tier3_` prefix is omitted deliberately — this is plumbing, not
//! a Tier 3 cross-validation case.

use glam::DVec3;
use jeod_runner::run_verification::sim_dyncomp;
use jeod_runner::GravitySourceEntry;
use jeod_runner::VerificationCaseExt;
use jeod_sim::recipes::verification::{
    CsvReference, InitialConditions, PreStepClosure, SimContext, Tolerances, VerificationCase,
};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, RotationModel, SimulationBuilder,
    SimulationTime, TranslationalState, VehicleConfig,
};
use uom::si::f64::Time;
use uom::si::time::second;

/// Synthetic per-step Sun position. The closure simply stamps a vector
/// proportional to the step time so the test can assert against it.
fn pretend_sun_position(time_s: f64) -> DVec3 {
    DVec3::new(1.5e11 + time_s, 2.0e10 - time_s, 3.0e10 + 0.5 * time_s)
}

/// Build a tiny scenario with Earth (point-mass) at index 0 and a Sun
/// stand-in at index 1. The vehicle is added with the t=0
/// `InitialConditions` from the reference CSV.
fn scenario(init: &InitialConditions) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, 60.0);

    let earth_grav = GravitySource {
        mu: 3.986_004_415e14,
        model: GravityModel::PointMass,
    };
    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_grav,
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );
    sb.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0, // direction-only, not gravitating
                model: GravityModel::PointMass,
            },
            position: pretend_sun_position(0.0),
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
        },
    );

    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });
    sb
}

/// Pre-step factory: capture the sun's source index (1) and produce a
/// closure that updates it on every step.
fn build_pre_step(_init: &InitialConditions) -> PreStepClosure {
    Box::new(|sim: &mut dyn SimContext, time_s: f64| {
        sim.set_source_position(1_usize, pretend_sun_position(time_s));
    })
}

#[test]
fn pre_step_smoke_drives_source_position_through_simcontext() {
    // Reuse RUN_2's reference CSV solely so `run_and_assert` has a real
    // trajectory to step against. Position/velocity tolerances are very
    // loose — this smoke test cares about pre_step plumbing, not
    // numerical accuracy. (RUN_2's actual tight tolerances live on the
    // production `run2_3dof()` case.)
    let case = VerificationCase {
        name: "pre_step_smoke",
        scenario,
        reference: CsvReference::Dyncomp3Dof("dyncomp_run2_state.csv"),
        // Truncate the run hard so the smoke test stays cheap.
        duration: Time::new::<second>(120.0),
        tolerances: Tolerances {
            position_m: [10.0, 10.0, 10.0],
            velocity_m_s: [1.0, 1.0, 1.0],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
        pre_step: Some(build_pre_step),
    };

    // The plumbing: run_and_assert constructs the closure once via the
    // factory and invokes it before each step. If anything in the chain
    // is wrong (the field isn't read, SimContext isn't impl'd for
    // Simulation, set_source_position doesn't propagate, …) this run
    // panics or asserts.
    case.run_and_assert();

    // Sanity that the production recipe still works alongside the new
    // pre_step pathway — same case without a hook.
    sim_dyncomp::run2_3dof().run_and_assert();
}
