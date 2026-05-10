//! End-to-end coverage for [`AstrodynAppExt`].
//!
//! Exercises the trait's three methods on a minimal Earth + ISS-style
//! point-mass scenario:
//!
//! 1. [`AstrodynAppExt::add_astrodyn`] brings up an `App` with `Time<Fixed>`
//!    + `AstrodynPlugin` in one chained call.
//! 2. [`AstrodynAppExt::step_fixed_dt`] advances the schedule with an
//!    explicit `dt`.
//! 3. [`AstrodynAppExt::step_fixed`] reads `dt` from `Time<Fixed>` and
//!    advances. The body must move under gravity (otherwise the bring-up
//!    didn't actually wire the pipeline).
//!
//! A separate `#[should_panic]` test pins the diagnostic substring used
//! when a caller invokes `step_fixed` on an `App` that never received a
//! `Time<Fixed>` resource — the fail-loud guard at the trait surface.

use astrodyn_bevy::prelude::*;
use astrodyn_bevy::recipes::{earth, orbital_elements, vehicle};
use bevy::prelude::*;

const DT: f64 = 10.0;
const N_STEPS: usize = 10;

#[derive(Resource)]
struct VehicleEntity(Entity);

fn setup_iss(mut commands: Commands) {
    let earth_recipe = earth::point_mass();
    let earth_mu = earth_recipe.source.mu;
    let earth = commands
        .spawn((
            Name::new("Earth"),
            GravitySourceC(earth_recipe.source),
            SourceInertialPositionC::default(),
            TranslationalStateC::<Earth>::default(),
        ))
        .id();

    let cfg = VehicleBuilder::new()
        .from_orbital_elements(orbital_elements::iss(), earth_mu.m3_per_s2())
        .three_dof_point_mass(vehicle::iss_mass())
        .rk4()
        .gravity(GravityControl::new_spherical(0_usize, GravityRole::Central))
        .build();

    let vehicle_entity = cfg.spawn_bevy::<astrodyn::Earth>(&mut commands, &[earth]);
    commands.insert_resource(VehicleEntity(vehicle_entity));
}

fn read_position_norm(app: &App) -> f64 {
    let entity = app.world().resource::<VehicleEntity>().0;
    app.world()
        .get::<TranslationalStateC<Earth>>(entity)
        .expect("vehicle entity must carry TranslationalStateC<Earth>")
        .0
        .position
        .raw_si()
        .length()
}

/// `add_astrodyn` + `step_fixed_dt` + `step_fixed` chain end-to-end.
///
/// After `add_astrodyn(DT)`, the world holds `Time<Fixed>` and the
/// `AstrodynPlugin` schedule sets are configured. After `N_STEPS` of
/// `step_fixed_dt` the body has propagated, and another N steps via
/// `step_fixed` (which must read the same DT back from the resource) keeps
/// it on a bound orbit. Both halves of the run move the position vector,
/// confirming the pipeline actually ticks.
#[test]
fn app_ext_chain_runs_pipeline() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_astrodyn(DT)
        .add_systems(Startup, setup_iss);
    // Flush startup so the vehicle entity exists before any FixedUpdate
    // schedule run reads its components.
    app.update();

    let r0 = read_position_norm(&app);

    app.step_fixed_dt(N_STEPS, DT);
    let r1 = read_position_norm(&app);
    assert!(
        (r1 - r0).abs() > 0.0,
        "step_fixed_dt did not advance the body: r0 = r1 = {r0}",
    );

    // `step_fixed` reads `dt` from the resource installed by
    // `add_astrodyn`. The bonded advance + run_schedule pair must move the
    // body again.
    app.step_fixed(N_STEPS);
    let r2 = read_position_norm(&app);
    assert!(
        (r2 - r1).abs() > 0.0,
        "step_fixed did not advance the body: r1 = r2 = {r1}",
    );

    // Bound orbit sanity: both samples are within a few percent of the
    // initial radius (ISS altitude). Catches a regression that would tick
    // the schedule without the gravity stage running.
    let drift0 = (r1 - r0).abs() / r0;
    let drift1 = (r2 - r1).abs() / r1;
    assert!(
        drift0 < 0.01 && drift1 < 0.01,
        "implausible position drift over {N_STEPS}*{DT}s: r0={r0} r1={r1} r2={r2}",
    );
}

/// `step_fixed` panics with a diagnostic that names `Time<Fixed>` and the
/// fix (call `add_astrodyn` first / use `step_fixed_dt`).
#[test]
#[should_panic(expected = "Time<Fixed>` resource is missing")]
fn step_fixed_panics_without_time_fixed() {
    let mut app = App::new();
    // No `add_astrodyn`, no `insert_resource(Time::<Fixed>::...)` — the
    // resource is absent, so `step_fixed` must panic at the read site.
    app.step_fixed(1);
}
