//! Rosetta Earth swing-by scenario at the JEOD `RUN_rosetta` epoch.
//!
//! Earth-centric hyperbolic flyby: Earth as the central body (GGM05C
//! truncated to degree 2 / order 0 — point-mass + J2, matching JEOD's
//! `earth_grav(2, 0)`), Moon and Sun as point-mass third-body
//! perturbations with per-step DE421 ephemeris updates, cannonball SRP
//! against the Sun. Matches the `SIM_Earth_Moon RUN_rosetta` JEOD
//! reference configuration (`Modified_data/{state,mass,radiation_pressure}.py`,
//! `SET_test/RUN_rosetta/input.py`) that
//! [`tests::tier3_simulation_earth_moon_rosetta`](../../tests/tier3_sim_earth_moon.rs)
//! cross-validates against.
//!
//! The integration frame is `Earth.inertial`, so JEOD's logged
//! `composite_body.state.trans` is Earth-centered inertial — the same
//! frame this builder integrates in, so no re-centering is needed (unlike
//! the Moon-centered Clementine sibling).
//!
//! At the ~103,000 km flyby distance the J2 term and lunar/solar
//! third-body perturbations are small but non-negligible over the 4-hour
//! (15,000 s) arc; the cross-validation exercises the multi-body
//! perturbation bookkeeping on a hyperbolic trajectory — the regime a
//! downstream interplanetary-mission user is most likely to reach for.

use astrodyn::recipes::{self, epoch};
use astrodyn::{
    EphemerisBody, F64Ext, GravityControl, GravityGradient, SimulationBuilder, TranslationalState,
    VehicleBuilder,
};
use glam::DVec3;

/// JEOD `SIM_Earth_Moon RUN_rosetta` initial position (Earth-centered
/// inertial, meters) at epoch 2009-11-13 05:00:00 UTC. From
/// `Modified_data/state.py` (`[87396.6219145, 23042.6606938,
/// -48761.8708343] km`).
pub const INIT_POS: DVec3 = DVec3::new(87_396_621.914_5, 23_042_660.693_8, -48_761_870.834_3);

/// JEOD `SIM_Earth_Moon RUN_rosetta` initial velocity (Earth-centered
/// inertial, m/s) at epoch. From `state.py`
/// (`[-7.8839651, -3.2492092, 4.7952127] km/s`).
pub const INIT_VEL: DVec3 = DVec3::new(-7_883.965_1, -3_249.209_2, 4_795.212_7);

/// Vehicle mass (kg) — `Modified_data/mass.py` `set_mass("rosetta")`.
pub const MASS_KG: f64 = 3000.0;

/// Cannonball SRP effective area × Cx (m²) — `radiation_pressure.py`
/// `set_rad_pressure("rosetta")`.
pub const SRP_CX_AREA: f64 = 20.0;
/// Cannonball SRP albedo.
pub const SRP_ALBEDO: f64 = 1.0;
/// Cannonball SRP diffuse fraction.
pub const SRP_DIFFUSE: f64 = 0.27;

/// Build the Rosetta Earth swing-by scenario at the JEOD `RUN_rosetta`
/// epoch (2009-11-13 05:00:00 UTC).
///
/// - `dt`: integration timestep, in seconds (JEOD's `S_define` uses RK4;
///   the Tier 3 test pins the JEOD dynamics rate).
/// - `initial_state`: pass `None` for the JEOD t=0 state
///   ([`INIT_POS`] / [`INIT_VEL`]); the Tier 3 test passes the reference
///   CSV's first row so a future JEOD regen stays the source of truth.
///
/// Returns a fully-wired [`SimulationBuilder`]; the caller materializes
/// the simulation via `.build()`.
///
/// # Panics
///
/// Panics if the DE421 ephemeris fails to load or the Moon/Sun positions
/// cannot be queried at the epoch — non-recoverable fixture errors that
/// surface at the point of detection per the fail-loudly invariant.
pub fn earth_moon_rosetta(dt: f64, initial_state: Option<(DVec3, DVec3)>) -> SimulationBuilder {
    let ephemeris =
        recipes::ephemeris::de421().expect("DE421 ephemeris must load from embedded fixtures");

    // JEOD RUN_rosetta epoch: 2009-11-13 05:00:00 UTC (TAI-UTC = 34 s).
    let time = epoch::at_utc(2009, 11, 13, 5, 0, 0.0);
    let epoch_tdb_jd = time.tdb_julian_date();

    // Seed Moon and Sun positions relative to Earth at t=0; the per-step
    // ephemeris stage overwrites these from DE421 each step.
    let (moon_pos, _) = ephemeris
        .get_state_typed(EphemerisBody::Moon, EphemerisBody::Earth, epoch_tdb_jd)
        .expect("Moon–Earth state from DE421 at epoch");
    let (sun_pos, _) = ephemeris
        .get_state_typed(EphemerisBody::Sun, EphemerisBody::Earth, epoch_tdb_jd)
        .expect("Sun–Earth state from DE421 at epoch");

    // Earth central body: GGM05C with EarthRNP rotation (the control
    // truncates to degree 2 / order 0 below — point-mass + J2). Moon/Sun
    // point-mass third bodies; override their mu with the same loaders the
    // Clementine setup uses so the third-body bookkeeping is consistent
    // across the SIM_Earth_Moon family.
    let earth_source = recipes::earth::ggm05c();
    let mut moon_source = recipes::moon::third_body(moon_pos);
    moon_source.source.mu = astrodyn::gravity_fixtures::load_moon_lp150q().mu;
    let mut sun_source = recipes::sun::third_body(sun_pos);
    sun_source.source.mu = astrodyn::gravity_fixtures::load_sun_spherical_mu();

    let mut sb = SimulationBuilder::new(time, dt);
    let earth_idx = sb.add_source("Earth", earth_source);
    let moon_idx = sb.add_source("Moon", moon_source);
    let sun_idx = sb.add_source("Sun", sun_source);
    sb.set_source_ephemeris(moon_idx, EphemerisBody::Moon, EphemerisBody::Earth);
    sb.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Earth);
    sb = sb.sun(sun_idx).ephemeris(ephemeris);

    let (pos, vel) = initial_state.unwrap_or((INIT_POS, INIT_VEL));
    let vehicle = VehicleBuilder::new()
        .with_translational(astrodyn::typed_bridge::trans_raw_to_root(
            &TranslationalState {
                position: pos,
                velocity: vel,
            },
        ))
        .three_dof_point_mass(MASS_KG.kg())
        .rk4()
        // earth_grav(2, 0): point-mass + J2 (zonal). order 0 → axially
        // symmetric, so Earth RNP about the pole does not affect it.
        .gravity(GravityControl::new_nonspherical(
            earth_idx,
            2,
            0,
            GravityGradient::Skip,
        ))
        .gravity(GravityControl::new_third_body(moon_idx))
        .gravity(GravityControl::new_third_body(sun_idx))
        .cannonball_srp(SRP_CX_AREA, SRP_ALBEDO, SRP_DIFFUSE)
        .build();
    sb.add_body(vehicle);
    sb
}
