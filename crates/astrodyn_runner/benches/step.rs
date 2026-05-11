//! Full-step microbench for the JEOD pipeline.
//!
//! Builds the Earth–Moon Clementine scenario via the shared
//! `astrodyn_verif_jeod::setups::earth_moon_clem` constructor (single
//! source of truth, see #447 PR-1), warms 1000 steps outside `iter()`
//! to reach steady-state caches, then measures one `sim.step()` call
//! inside `iter()`.
//!
//! Note on iter() semantics: `Simulation::step()` advances the
//! simulation time and integrates the trajectory forward by `dt`. Each
//! criterion `iter()` invocation therefore evaluates the pipeline at a
//! *different* point along the orbit, not the same point repeatedly.
//! That's intentional — it matches `tier3_perf_runner`'s steady-state
//! measurement semantics and absorbs per-step variation across the
//! orbit. Do not add a `sim.reset()` inside the closure; resetting
//! would zero `time` mid-trajectory and force the integrator to
//! re-warm every iteration.
//!
//! Cross-crate dev-dep: this bench reaches `astrodyn_verif_jeod` (a
//! verif crate, not a physics crate) so the gateway-bypass guard in
//! `scripts/check_no_bypass_deps.sh` permits it. The lib target's
//! dependency graph is unaffected — dev-deps participate only in
//! tests/benches/examples.
//!
//! Run with criterion's quick mode:
//!
//! ```bash
//! cargo bench --bench step -- --quick
//! ```

#![allow(missing_docs)]

use astrodyn_runner::SimulationBuilderExt;
use astrodyn_verif_jeod::setups::earth_moon_clem::earth_moon_clem;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

fn bench_simulation_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation_step");

    group.bench_function("earth_moon_clem_dt32hz", |b| {
        let mut sim = earth_moon_clem(0.03125, None)
            .build()
            .expect("earth_moon_clem must validate");
        // Reach steady-state caches before measurement starts.
        sim.step_n(1000).expect("warmup");

        b.iter(|| {
            sim.step().expect("step");
            black_box(&sim);
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_simulation_step
}
criterion_main!(benches);
