//! Microbench for the spherical-harmonics gravity kernel.
//!
//! Three sub-benches mirror the production workload:
//! - `d4`  — Earth GGM05C truncated at 4×4 (J2-ish).
//! - `d20` — Earth GGM05C truncated at 20×20 (intermediate fidelity).
//! - `d60` — Moon LP150Q truncated at 60×60 (Clementine workload from
//!   #447's tier3 Earth-Moon case; this is the production hot path).
//!
//! Positions are synthetic and scenario-representative (LEO altitude
//! for Earth, ~250 km lunar orbit for the Moon); the bench measures
//! kernel cost, not trajectory cost, so position values don't need to
//! match the JEOD reference state.
//!
//! Fixture loads (`load_ggm05c`, `load_moon_lp150q`), the
//! `GottliebScratch` allocation, and the identity rotation matrix all
//! live outside `iter()` — the bench measures only
//! `gravitation_with_scratch`'s arithmetic, not setup overhead.
//!
//! Run with criterion's quick mode for PR-blocking smoke:
//!
//! ```bash
//! cargo bench --bench accumulate -- --quick
//! ```
//!
//! Run a full profile to emit flamegraphs under
//! `target/criterion/accumulate/.../profile/`:
//!
//! ```bash
//! cargo bench --bench accumulate
//! ```

#![allow(missing_docs)]

use astrodyn_gravity::compute::gravitation_with_scratch;
use astrodyn_gravity::fixtures;
use astrodyn_gravity::gravity_source::{GravityModel, GravitySource};
use astrodyn_gravity::spherical_harmonics_calc_nonspherical::GottliebScratch;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use glam::{DMat3, DVec3};
use pprof::criterion::{Output, PProfProfiler};

fn bench_accumulate_gravity(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulate_gravity");

    // Fixture parses — outside iter() so the per-iteration cost is the
    // kernel call, not the binary fixture decode. Move the loaded
    // tables into the boxes directly (no clone) — the originals are
    // not used after this point.
    let ggm05c = fixtures::load_ggm05c();
    let lp150q = fixtures::load_moon_lp150q();
    let earth_mu = ggm05c.mu;
    let moon_mu = lp150q.mu;
    let earth_src = GravitySource {
        mu: earth_mu,
        model: GravityModel::SphericalHarmonics(Box::new(ggm05c)),
    };
    let moon_src = GravitySource {
        mu: moon_mu,
        model: GravityModel::SphericalHarmonics(Box::new(lp150q)),
    };

    // Scenario-representative positions in the body-fixed frame.
    // LEO altitude (~400 km) for Earth, low lunar orbit (~250 km) for
    // the Moon. The kernel cost is dominated by the degree/order
    // recursion, not the input position; specific values are
    // immaterial as long as they're physical.
    let earth_pos = DVec3::new(6_778_000.0, 0.0, 0.0);
    let moon_pos = DVec3::new(1_988_140.0, 0.0, 0.0);

    let t_eye = DMat3::IDENTITY;
    // Allocate scratch sized for the largest sub-bench so the smaller
    // ones reuse the same buffers (matches production where one
    // scratch arena lives on `Simulation`).
    let mut scratch = GottliebScratch::new(60);

    let configs: &[(&str, &GravitySource, DVec3, usize)] = &[
        ("d4", &earth_src, earth_pos, 4),
        ("d20", &earth_src, earth_pos, 20),
        ("d60", &moon_src, moon_pos, 60),
    ];

    for &(label, src, pos, deg) in configs {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(src, pos, deg),
            |b, &(src, pos, deg)| {
                b.iter(|| {
                    // Apply the inverse rotation here (matching what
                    // the production caller does on every kernel
                    // call). After hoisting, this is the work the
                    // caller can amortize across multiple substages
                    // sharing the same `t_parent_this`; including it
                    // in the bench keeps the timed region comparable
                    // to the pre-hoist combined kernel cost.
                    let kernel_out = gravitation_with_scratch(
                        black_box(src),
                        black_box(pos),
                        black_box(&t_eye),
                        deg,
                        deg,
                        /* perturbing_only */ false,
                        /* compute_gradient */ false,
                        /* gradient_degree  */ 0,
                        /* gradient_order   */ 0,
                        &mut scratch,
                        /* delta_c20        */ 0.0,
                        /* has_delta_coeffs */ false,
                    );
                    let acc = kernel_out.into_inertial(&t_eye, false);
                    black_box(acc)
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_accumulate_gravity
}
criterion_main!(benches);
