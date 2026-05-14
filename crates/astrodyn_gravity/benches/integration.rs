//! Microbench for the RK4 6-DOF integrator.
//!
//! Two sub-benches:
//! - `trivial_accel` — `accel_fn` returns `DVec3::ZERO`. Measures the
//!   integrator plumbing alone (state copying, quaternion derivative,
//!   rotational-acceleration solve).
//! - `degree60_accel` — `accel_fn` calls `gravitation_with_scratch`
//!   for the Moon LP150Q 60×60 field. Measures the production
//!   combined cost (RK4 plumbing × 4 kernel evals).
//!
//! Diffing the two reveals whether a regression sits in the
//! integrator or in the gravity kernel.
//!
//! The realistic `accel_fn` captures `GottliebScratch` via `RefCell`
//! because `rk4_sixdof_step`'s `accel_fn` bound is `impl Fn`, not
//! `FnMut`. The interior mutability adds one `RefCell::borrow_mut` per
//! call — ~1 ns, well below the kernel cost.
//!
//! Run with criterion's quick mode:
//!
//! ```bash
//! cargo bench --bench integration -- --quick
//! ```

#![allow(missing_docs)]

use std::cell::RefCell;

use astrodyn_dynamics::integration::rk4_sixdof_step;
use astrodyn_dynamics::mass::MassProperties;
use astrodyn_dynamics::rotational::SixDofState;
use astrodyn_dynamics::state::TranslationalState;
use astrodyn_gravity::compute::gravitation_with_scratch;
use astrodyn_gravity::fixtures;
use astrodyn_gravity::gravity_source::{GravityModel, GravitySource};
use astrodyn_gravity::spherical_harmonics_calc_nonspherical::GottliebScratch;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use glam::{DMat3, DVec3};
use pprof::criterion::{Output, PProfProfiler};

fn bench_rk4_sixdof_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("rk4_sixdof_step");

    // Scenario-representative initial state — low lunar orbit position
    // and velocity. Bench measures kernel cost, not orbital fidelity;
    // specific numerics are immaterial as long as they're physical.
    let initial = SixDofState {
        trans: TranslationalState {
            position: DVec3::new(1_988_140.0, 0.0, 0.0),
            velocity: DVec3::new(0.0, 1_650.0, 0.0),
        },
        rot: Default::default(),
    };
    let mass = MassProperties::new(424.0);
    let dt = 0.03125;

    // (a) Trivial accel: measures RK4 plumbing only.
    group.bench_function("trivial_accel", |b| {
        let zero_accel = |_: &SixDofState, _: f64| -> DVec3 { DVec3::ZERO };
        let zero_torque = |_: &SixDofState| -> DVec3 { DVec3::ZERO };
        b.iter(|| {
            let s = rk4_sixdof_step(black_box(&initial), zero_accel, zero_torque, &mass, dt);
            black_box(s)
        });
    });

    // (b) Realistic accel: Moon LP150Q 60×60 gravity. The 60×60
    // truncation matches #447's tier3 Earth-Moon Clementine workload.
    let lp150q = fixtures::load_moon_lp150q();
    let moon_src = GravitySource {
        mu: lp150q.mu,
        model: GravityModel::SphericalHarmonics(Box::new(lp150q)),
    };
    let t_eye = DMat3::IDENTITY;
    // `rk4_sixdof_step`'s `accel_fn` is `impl Fn`; capture
    // `GottliebScratch` via `RefCell` to share one buffer across the 4
    // RK4 stage evaluations without bumping the closure bound to
    // `FnMut`. Borrow cost is negligible (~1 ns) vs the kernel.
    let scratch = RefCell::new(GottliebScratch::new(60));
    group.bench_function("degree60_accel", |b| {
        let accel = |s: &SixDofState, _: f64| -> DVec3 {
            let mut sc = scratch.borrow_mut();
            // Apply the inverse rotation here (matching what the
            // production caller does on every kernel call). The bench
            // measures the per-substage cost the kernel plus caller
            // pay together — same total work as the pre-hoist form.
            let kernel_out = gravitation_with_scratch(
                &moon_src,
                s.trans.position,
                &t_eye,
                60,
                60,
                false,
                false,
                0,
                0,
                &mut sc,
                0.0,
                false,
            );
            kernel_out.into_inertial(&t_eye, false).grav_accel
        };
        let zero_torque = |_: &SixDofState| -> DVec3 { DVec3::ZERO };
        b.iter(|| {
            let s = rk4_sixdof_step(black_box(&initial), accel, zero_torque, &mass, dt);
            black_box(s)
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_rk4_sixdof_step
}
criterion_main!(benches);
