//! Earth–Moon Clementine scenario at the JEOD `RUN_clem` epoch.
//!
//! Moon LP150Q 60×60 spherical harmonics (central) with DE421 BPC
//! libration rotation, Earth and Sun as point-mass third-body
//! perturbations with per-step DE421 ephemeris updates, cannonball SRP
//! against the Sun. Matches the `SIM_Earth_Moon RUN_clem` JEOD
//! reference configuration that
//! [`tests::tier3_simulation_earth_moon_clem`](../../tests/tier3_sim_earth_moon.rs)
//! cross-validates against.
//!
//! Callers:
//! - `crates/astrodyn_verif_jeod/tests/tier3_sim_earth_moon.rs` —
//!   feeds `Some((csv_pos, csv_vel))` from the reference CSV's t=0 row
//!   so any future JEOD refresh stays the single source of truth.
//! - `crates/astrodyn_verif_jeod/examples/earth_moon.rs` — passes
//!   `None`, falls back to the [`INIT_POS`] / [`INIT_VEL`] constants
//!   (bit-identical to the CSV first row at present).
//! - `crates/astrodyn_verif_jeod/src/bin/tier3_perf_runner.rs` —
//!   passes `None`; uses the same constants for measurement workloads.
//!
//! The function returns a fully-wired [`SimulationBuilder`]; callers
//! call `.build()` to materialize a [`astrodyn_runner::Simulation`].

use astrodyn::recipes;
use astrodyn::{
    EphemerisBody, F64Ext, GravityControl, GravityGradient, RotationModel, SimulationBuilder,
    SimulationTime, TranslationalState, VehicleBuilder,
};
use glam::DVec3;

/// JEOD `SIM_Earth_Moon RUN_clem` initial position (Moon-centered
/// inertial frame, meters), captured at the simulation epoch
/// 1994-03-01 00:00:00 UTC.
pub const INIT_POS: DVec3 = DVec3::new(1_296_944.012, -1_060_824.45, 2_522_289.146);

/// JEOD `SIM_Earth_Moon RUN_clem` initial velocity (Moon-centered
/// inertial frame, m/s), captured at the simulation epoch.
pub const INIT_VEL: DVec3 = DVec3::new(-930.578, -439.312, 862.075);

/// Cannonball SRP effective area × Cx (m²) for the Clementine vehicle
/// — matches JEOD's `Modified_data/radiation_pressure.py`.
pub const SRP_CX_AREA: f64 = 2.1432;

/// Cannonball SRP albedo for the Clementine vehicle.
pub const SRP_ALBEDO: f64 = 1.0;

/// Cannonball SRP diffuse fraction for the Clementine vehicle.
pub const SRP_DIFFUSE: f64 = 0.27;

/// Gravitational parameter of the Moon as loaded by the LP150Q fixture
/// (`load_moon_lp150q().mu`). Callers that need the central-body mu for
/// orbital-element bookkeeping (e.g. the `earth_moon` example's
/// altitude/period printer) read it from here so they pull the exact
/// numeric used inside the builder.
pub fn moon_mu() -> f64 {
    astrodyn::gravity_fixtures::load_moon_lp150q().mu
}

/// Build the Earth–Moon Clementine scenario at the JEOD `RUN_clem`
/// epoch (1994-03-01 00:00:00 UTC).
///
/// - `dt`: integration timestep, in seconds. The Tier 3 test pins it to
///   `0.03125` (32 Hz, matching JEOD's `S_define`); the example uses
///   `1.0` for a fast-running demo; the perf-runner uses the test's
///   value to measure the production workload.
/// - `initial_state`: pass `None` to use the JEOD `RUN_clem` t=0 state
///   ([`INIT_POS`] / [`INIT_VEL`]); pass `Some((pos, vel))` to override
///   from a JEOD-derived source (the Tier 3 test reads the first row
///   of the reference CSV so any future JEOD regen stays the single
///   source of truth for the initial conditions).
///
/// Returns a fully-wired [`SimulationBuilder`]; the caller materializes
/// the simulation via `.build()` ([`astrodyn_runner::SimulationBuilderExt`]
/// from `astrodyn_runner`).
///
/// # Panics
///
/// Panics if the DE421 ephemeris (with the Moon principal-axes BPC
/// kernel) fails to load, or if the Earth/Sun positions cannot be
/// queried from it. These are not recoverable runtime conditions — the
/// embedded fixtures must be present in the crate; panic surfaces the
/// failure at the point of detection per the fail-loudly invariant.
pub fn earth_moon_clem(dt: f64, initial_state: Option<(DVec3, DVec3)>) -> SimulationBuilder {
    let ephemeris = recipes::ephemeris::de421_with_moon_pa()
        .expect("DE421 + Moon BPC ephemeris must load from embedded fixtures");

    // JEOD `SIM_Earth_Moon RUN_clem` epoch: 1994-03-01 00:00:00 UTC.
    //   JD = 2449412.5; MJD = 49412.0; TJT = MJD - 40000 = 9412.0
    //   TAI-UTC = 28 s at 1994-03-01 (29th leap second added 1994-07-01)
    // Kept as an explicit literal so this setup module is self-documenting
    // for future JEOD-reference regens — it matches
    // [`astrodyn::recipes::epoch::clementine_1994`] bit-for-bit.
    let clem_tai_tjt = 9412.0 + 28.0 / 86400.0;
    let time = SimulationTime::new(clem_tai_tjt, astrodyn::default_leap_second_table());
    let epoch_tdb_jd = time.tdb_julian_date();

    // Moon LP150Q 60×60 central body with DE421 BPC libration.
    // The recipe defaults `rotation_model` to `MoonIAU` (analytic IAU
    // 2009 mean orientation) — override to `MoonDE421` so the per-step
    // ephemeris stage interpolates `t_inertial_pfix` from the BPC
    // kernel. Seeds `t_inertial_pfix` at the epoch value so the
    // initialization pass sees a consistent rotation before the first
    // step's ephemeris update.
    let mut moon_source = recipes::moon::lp150q();
    moon_source.rotation_model = RotationModel::MoonDE421;
    moon_source.t_inertial_pfix = Some(
        ephemeris
            .get_body_rotation(EphemerisBody::Moon, epoch_tdb_jd)
            .expect("Moon DE421 libration rotation at epoch"),
    );

    // Earth and Sun seed positions at t=0; the per-step ephemeris stage
    // overwrites these from DE421 once `set_source_ephemeris` is wired.
    let (earth_pos_typed, _) = ephemeris
        .get_state_typed(EphemerisBody::Earth, EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Earth–Moon state from DE421 at epoch");
    let (sun_pos_typed, _) = ephemeris
        .get_state_typed(EphemerisBody::Sun, EphemerisBody::Moon, epoch_tdb_jd)
        .expect("Sun–Moon state from DE421 at epoch");

    // Override the third-body recipe mu values with the same loaders
    // the legacy direct-API tier3 test used (`load_ggm05c().mu`,
    // `load_sun_spherical_mu()`) so the typed-builder refactor stays
    // bit-identical to the pre-#447 cross-validation baseline.
    let mut earth_source = recipes::earth::third_body(earth_pos_typed);
    earth_source.source.mu = astrodyn::gravity_fixtures::load_ggm05c().mu;
    let mut sun_source = recipes::sun::third_body(sun_pos_typed);
    sun_source.source.mu = astrodyn::gravity_fixtures::load_sun_spherical_mu();

    let mut sb = SimulationBuilder::new(time, dt);
    let moon_idx = sb.add_source("Moon", moon_source);
    let earth_idx = sb.add_source("Earth", earth_source);
    let sun_idx = sb.add_source("Sun", sun_source);
    sb.set_source_ephemeris(earth_idx, EphemerisBody::Earth, EphemerisBody::Moon);
    sb.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Moon);
    sb = sb.sun(sun_idx).ephemeris(ephemeris);

    let (pos, vel) = initial_state.unwrap_or((INIT_POS, INIT_VEL));
    let vehicle = VehicleBuilder::new()
        .with_translational(astrodyn::typed_bridge::trans_raw_to_root(
            &TranslationalState {
                position: pos,
                velocity: vel,
            },
        ))
        .three_dof_point_mass(424.0.kg())
        .rk4()
        .gravity(GravityControl::new_nonspherical(
            moon_idx,
            60,
            60,
            GravityGradient::Skip,
        ))
        .gravity(GravityControl::new_third_body(earth_idx))
        .gravity(GravityControl::new_third_body(sun_idx))
        .cannonball_srp(SRP_CX_AREA, SRP_ALBEDO, SRP_DIFFUSE)
        .build();
    sb.add_body(vehicle);
    sb
}

#[cfg(test)]
mod tests {
    use astrodyn::recipes::epoch;

    /// The explicit `SimulationTime::new(9412.0 + 28.0/86400.0, …)`
    /// literal in [`earth_moon_clem`] must agree bit-for-bit with the
    /// recipe-layer entry [`epoch::clementine_1994`]. If a future edit
    /// drifts one of them, this guard fails before any Tier 3 run.
    #[test]
    fn epoch_literal_matches_recipe_clementine_1994() {
        let literal_tai_tjt = 9412.0 + 28.0 / 86400.0;
        let recipe_tai_tjt = epoch::clementine_1994().tai_tjt_at_epoch;
        assert_eq!(literal_tai_tjt, recipe_tai_tjt);
    }
}
