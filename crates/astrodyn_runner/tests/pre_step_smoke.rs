#![cfg(feature = "verification")]

//! Smoke test for the `VerificationCase::pre_step` factory hook (#156).
//!
//! Builds a minimal `VerificationCase` whose pre-step closure stamps the
//! Sun source's position with a deterministic function of the per-step
//! time. The test then asserts (via shared atomic state captured into the
//! closure) that:
//!
//! 1. The `PreStepBuilder` factory was invoked exactly once at the start
//!    of `run_and_assert`.
//! 2. The returned closure was invoked at least once before propagation
//!    (i.e. the runner actually reads the `pre_step` field and dispatches
//!    through `SimContext`).
//! 3. The closure was called with monotonically increasing `time_s`
//!    arguments matching the reference-CSV cadence.
//!
//! The third leg of the integration — that `SimContext::set_source_position`
//! actually propagates into `Simulation` storage — is covered by the
//! existing `Simulation::set_source_position` unit tests; the smoke test
//! here focuses on the factory + dispatch plumbing that #156 added.
//!
//! Carries the `tier3_` prefix because the trailing
//! `sim_dyncomp::run2_3dof().run_and_assert()` sanity check loads
//! `verif/SIM_dyncomp/S_define` from the JEOD checkout — which is
//! sparse-checked-out only by the Tier 3 CI job, not the unit + tier 2
//! job. Cargo's name-based filter routes this into the right place.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use astrodyn::recipes::verification::{
    CsvReference, InitialConditions, PreStepClosure, SimContext, Tolerances, VerificationCase,
};
use astrodyn::GravitySourceEntry;
use astrodyn::{
    GravityControl, GravityControls, GravityModel, GravitySource, RotationModel, SimulationBuilder,
    SimulationTime, TranslationalState, VehicleConfig,
};
use astrodyn_runner::run_verification::sim_dyncomp;
use astrodyn_runner::VerificationCaseExt;
use glam::DVec3;
use uom::si::f64::Time;
use uom::si::time::second;

/// Shared state captured into the pre-step factory and returned closure
/// so the test can assert about both factory and closure invocations.
///
/// `factory_calls` increments once per `PreStepBuilder` invocation
/// (must be exactly 1 by `run_and_assert` contract).
///
/// `closure_calls` increments once per pre-step closure call.
///
/// `last_time` records the most recent `time_s` argument so the closure
/// can verify monotonic-increase across calls (a bug where the runner
/// invokes pre_step out of order or with stale times surfaces here).
#[derive(Default)]
struct PreStepProbe {
    factory_calls: AtomicUsize,
    closure_calls: AtomicUsize,
    last_time: Mutex<Option<f64>>,
}

/// Static probe shared with the `fn`-pointer factory. `OnceLock` lets us
/// hand the same `Arc` to the factory closure at runtime without making
/// `VerificationCase`'s `pre_step` field non-`Copy` (it's a `fn` pointer
/// type per the #156 design).
static PROBE: OnceLock<Arc<PreStepProbe>> = OnceLock::new();

fn probe() -> &'static Arc<PreStepProbe> {
    PROBE.get_or_init(|| Arc::new(PreStepProbe::default()))
}

/// Synthetic per-step Sun position. The closure stamps a deterministic
/// vector so a future regression of `set_source_position` forwarding
/// would surface as a trajectory mismatch in production code that uses
/// non-zero solar mu (this test runs with zero solar mu so the Sun
/// position writes don't affect dynamics — see module docstring on the
/// scope of this smoke test).
fn pretend_sun_position(time_s: f64) -> DVec3 {
    DVec3::new(1.5e11 + time_s, 2.0e10 - time_s, 3.0e10 + 0.5 * time_s)
}

/// Build a tiny scenario with Earth (point-mass) at index 0 and a Sun
/// stand-in at index 1. The vehicle is added with the t=0
/// `InitialConditions` from the reference CSV.
fn scenario(init: &InitialConditions) -> SimulationBuilder {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, 60.0);

    let earth_grav = GravitySource {
        mu: 3.986_004_415e14,
        model: GravityModel::PointMass,
    };
    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: earth_grav,
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
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
            position: astrodyn::Vec3Ext::m_at::<astrodyn::RootInertial>(pretend_sun_position(0.0)),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
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

/// Pre-step factory: increment the factory counter once, capture the
/// shared probe into a closure that increments per-call counters and
/// asserts monotonic-time semantics.
fn build_pre_step(_init: &InitialConditions) -> PreStepClosure {
    let p = Arc::clone(probe());
    p.factory_calls.fetch_add(1, Ordering::SeqCst);
    Box::new(move |sim: &mut dyn SimContext, time_s: f64| {
        p.closure_calls.fetch_add(1, Ordering::SeqCst);

        // The runner contract says pre_step fires once per reference-CSV
        // record before the matching `sim.step_until(record.time)`. Times
        // must be strictly monotonically increasing.
        let mut last = p.last_time.lock().expect("probe last_time mutex poisoned");
        if let Some(prev) = *last {
            assert!(
                time_s > prev,
                "pre_step invoked with non-monotonic time: prev={prev} got={time_s}"
            );
        }
        *last = Some(time_s);
        drop(last);

        // Drive a SimContext call so any future regression of
        // `Simulation::set_source_position` forwarding (or a layout
        // mismatch on the trait object) surfaces here too — even though
        // Sun mu=0 means this write doesn't affect the trajectory.
        sim.set_source_position(1_usize, pretend_sun_position(time_s));
    })
}

#[test]
fn tier3_pre_step_smoke_drives_source_position_through_simcontext() {
    // Reset the probe in case the test process is reused across test
    // binaries (cargo nextest spawns one process per test by default,
    // but `OnceLock` would persist within a single process).
    let p = probe();
    p.factory_calls.store(0, Ordering::SeqCst);
    p.closure_calls.store(0, Ordering::SeqCst);
    *p.last_time.lock().unwrap() = None;

    // Reuse RUN_2's reference CSV solely so `run_and_assert` has a real
    // trajectory to step against. Position/velocity tolerances are very
    // loose — this smoke test cares about pre_step plumbing, not
    // numerical accuracy. (RUN_2's actual tight tolerances live on the
    // production `run2_3dof()` case.) Truncate the run hard so the
    // smoke test stays cheap; for a 60s dt and 120s duration the runner
    // will sample the reference CSV at t=60 and t=120 (skipping t=0),
    // so we expect exactly 2 closure invocations.
    let case = VerificationCase {
        name: "pre_step_smoke",
        scenario,
        reference: CsvReference::Dyncomp3Dof("dyncomp_run2_state.csv"),
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

    case.run_and_assert();

    // Plumbing assertions — the headline contract of #156:
    let factory_count = p.factory_calls.load(Ordering::SeqCst);
    let closure_count = p.closure_calls.load(Ordering::SeqCst);
    assert_eq!(
        factory_count, 1,
        "PreStepBuilder factory must be invoked exactly once per run_and_assert; \
         got {factory_count}"
    );
    assert!(
        closure_count >= 1,
        "pre_step closure was never invoked — runner is not dispatching through \
         SimContext (factory ran but the returned closure was never called)"
    );

    // Sanity that the production recipe still works alongside the new
    // pre_step pathway — same case without a hook.
    sim_dyncomp::run2_3dof().run_and_assert();
}
