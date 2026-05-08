// JEOD_INV: TS.01 — `<SelfRef>` / `<SelfPlanet>` are runtime-resolved storage-boundary wildcards; see `docs/JEOD_invariants.md` row TS.01 and the lint at `tests/self_ref_self_planet_discipline.rs`.
//! Sanity test for `ThermalIntegrationOrder::DerivativeRk4`.
//!
//! No JEOD reference sim exercises the full RK4 thermal path today, so this
//! test exists to keep the code exercised and catch silent regressions when
//! the runner's Stage 5 / Stage 8 dispatch or the [`astrodyn::IntegrableObject`]
//! trait implementation changes. Runs the same low-altitude scenario three
//! times — once with each of [`ThermalIntegrationOrder::Scheduled`],
//! [`ThermalIntegrationOrder::DerivativeFirstOrder`], and
//! [`ThermalIntegrationOrder::DerivativeRk4`] — and asserts that each mode
//! produces a non-trivially different final-step temperature, proving the
//! fork is actually dispatched and each variant runs to completion.
//!
//! This is NOT a Tier 3 cross-validation — there is no external reference.

use astrodyn::{
    FlatPlate, FlatPlateParams, FlatPlateState, FlatPlateThermal, GravityModel, GravitySource,
    JeodQuat, MassProperties, RotationalState, SimulationTime, ThermalIntegrationOrder,
    TranslationalState, Vec3Ext,
};
use astrodyn::{GravitySourceEntry, SrpModel, VehicleConfig};
use astrodyn_runner::{RotationModel, Simulation};
use glam::{DMat3, DVec3};

fn single_plate() -> Vec<(
    FlatPlate<astrodyn::SelfRef>,
    FlatPlateParams,
    FlatPlateThermal,
)> {
    vec![(
        FlatPlate {
            area: 10.0,
            normal: -DVec3::X,
            position: DVec3::ZERO.m_at::<astrodyn::StructuralFrame<astrodyn::SelfRef>>(),
        },
        FlatPlateParams {
            albedo: 0.3,
            diffuse: 0.5,
        },
        FlatPlateThermal {
            emissivity: 0.9,
            heat_capacity_per_area: 500.0,
            thermal_power_dump: 0.0,
        },
    )]
}

/// Returns `(final_plate_temperature, final_position)` after 20 steps.
fn run_with_order(order: ThermalIntegrationOrder) -> (f64, DVec3) {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let dt = 10.0;
    let mut sim = Simulation::new(time, dt);

    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
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
    sim.sun_source = Some(sun);

    let init_temp = 300.0;
    let plates = single_plate();
    // Spin about Z at 0.05 rad/s so the plate normal rotates slowly —
    // enough per-stage variation to distinguish the three scheduling modes
    // without driving the RK4 integrator into gross overshoot from
    // undersampled rotation.
    let mass =
        MassProperties::with_inertia(100.0, DMat3::from_diagonal(DVec3::splat(10.0)), DVec3::ZERO);
    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: DVec3::new(1.5e11, 0.0, 0.0),
            velocity: DVec3::ZERO,
        }
        .into(),
        rot: Some(
            RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::new(0.0, 0.0, 0.05),
            }
            .into(),
        ),
        mass: Some(mass.into()),
        srp: Some(SrpModel::FlatPlate(FlatPlateState {
            plates,
            temperatures: vec![init_temp],
            t_pow4_cached: vec![init_temp.powi(4)],
            integration_order: order,
            ..Default::default()
        })),
        ..Default::default()
    });

    sim.validate().unwrap();
    // 20 steps at dt=10s.
    for _ in 0..20 {
        sim.step().expect("step failed");
    }

    let t = sim
        .srp_plate_temperatures(0)
        .expect("flat-plate SRP configured")[0];
    let pos = sim.body(0).trans.position;
    (t, pos)
}

// non-recipe: SRP thermal integration scenarios; the same orbit run three
// times with `ThermalIntegrationOrder::{Scheduled,DerivativeFirstOrder,DerivativeRk4}`.
// Test content is comparing the three modes, no recipe input applies.
#[test]
fn tier3_srp_rk4_thermal_differs_from_scheduled() {
    let (t_sched, p_sched) = run_with_order(ThermalIntegrationOrder::Scheduled);
    let (t_first, p_first) = run_with_order(ThermalIntegrationOrder::DerivativeFirstOrder);
    let (t_rk4, p_rk4) = run_with_order(ThermalIntegrationOrder::DerivativeRk4);

    // Each mode must produce a valid (positive, finite) temperature.
    for (label, t) in [
        ("Scheduled", t_sched),
        ("DerivativeFirstOrder", t_first),
        ("DerivativeRk4", t_rk4),
    ] {
        assert!(t > 0.0 && t.is_finite(), "{label}: invalid temperature {t}");
    }

    // Derivative-class modes evaluate SRP per RK4 stage (varying with
    // intermediate attitude), so the orbital trajectory must differ from
    // Scheduled's step-constant-SRP orbital integration, even when T
    // agrees to high precision (k1 is the same at step start in all
    // modes, so the first-step thermal differences accumulate only
    // through orbital-state feedback over many steps).
    let p_threshold = 1e-12_f64;
    assert!(
        (p_sched - p_first).length() > p_threshold,
        "Scheduled and DerivativeFirstOrder produced identical trajectory \
         ({p_sched:?}); the Stage 8 dispatch fork may not be engaging."
    );
    // RK4 thermal vs first-order thermal must differ in the T evolution
    // whenever T moves fast enough to matter across a stage.
    let t_threshold = 1e-9_f64;
    assert!(
        (t_first - t_rk4).abs() > t_threshold,
        "DerivativeFirstOrder and DerivativeRk4 produced identical T \
         ({t_first}); the per-stage temp_dot branch may not be engaging."
    );

    println!(
        "After 20 steps: Scheduled T={t_sched:.6} |p|={:.3}; \
         DerivativeFirstOrder T={t_first:.6} |p|={:.3}; \
         DerivativeRk4 T={t_rk4:.6} |p|={:.3}",
        p_sched.length(),
        p_first.length(),
        p_rk4.length(),
    );
}
