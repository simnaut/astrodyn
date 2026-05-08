//! Tier 3: Analytical drag verification tests.
//!
//! These tests exercise aerodynamic drag through `Simulation::step()` and verify
//! physics using analytical relationships (energy dissipation, decay rates,
//! parameter scaling). No Docker reference data is needed.
//!
//! All tests use point-mass gravity with constant atmospheric density to isolate
//! drag effects from atmosphere model complexity.

use astrodyn::recipes::helpers::energy_conservation::specific_orbital_energy;
use astrodyn::{
    AtmosphereConfig, AtmosphereModel, DragConfig, ExponentialAtmosphere, GravityControl,
    GravityControls, GravityModel, GravitySource, JeodQuat, MassProperties, RotationalState,
    SimulationTime, TranslationalState,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_runner::Simulation;
use glam::DVec3;

/// Earth rotation rate (JEOD RNPJ2000 default), sourced from
/// `astrodyn::planet_config::EARTH.omega`.
const OMEGA_EARTH: f64 = astrodyn::planet_config::EARTH.omega;

/// Earth gravitational parameter (m^3/s^2) — JEOD `earth_GGM05C.cc`.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;

/// Earth mean equatorial radius (m) — JEOD `earth.cc`.
const R_EARTH: f64 = astrodyn::EARTH.shape.r_eq;

/// Compute semi-major axis from specific energy: a = -mu / (2*E)
fn semi_major_axis_from_energy(energy: f64, mu: f64) -> f64 {
    -mu / (2.0 * energy)
}

/// Create a minimal simulation with point-mass gravity and constant-density drag.
fn make_drag_sim(
    pos: DVec3,
    vel: DVec3,
    mass: f64,
    cd: f64,
    area: f64,
    density: f64,
    dt: f64,
) -> Simulation {
    let mut sim = Simulation::new(
        SimulationTime::at_j2000(astrodyn::default_leap_second_table()),
        dt,
    );

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    // Atmosphere config is required by validation even when constant_density
    // overrides the atmospheric density. The exponential model provides a
    // placeholder; wind is zero without planet_omega.
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(ExponentialAtmosphere {
            rho_0: 1e-12,
            h_0: 400_000.0,
            scale_height: 50_000.0,
        }),
        r_eq: R_EARTH,
        r_pol: R_EARTH * (1.0 - 1.0 / 298.257_223_563),
        planet_omega: 0.0,
    });
    sim.atmosphere_planet_source = Some(earth);

    let drag_config = DragConfig {
        cd,
        area,
        constant_density: Some(density),
    };

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: pos,
            velocity: vel,
        }
        .into(),
        rot: Some(
            RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }
            .into(),
        ),
        mass: Some(MassProperties::new(mass).into()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: Some(drag_config),
        ..Default::default()
    });

    sim.validate().unwrap();
    sim
}

/// ISS-like circular orbit initial conditions at ~400 km altitude.
fn iss_circular_state() -> (DVec3, DVec3) {
    let r = R_EARTH + 400_000.0; // 400 km altitude
    let v = (MU_EARTH / r).sqrt(); // circular velocity
    (DVec3::new(r, 0.0, 0.0), DVec3::new(0.0, v, 0.0))
}

/// After N orbits with constant drag, orbital energy should decrease.
///
/// Drag removes kinetic energy from the orbit, causing the total specific
/// energy to become more negative. This is the most fundamental property
/// of atmospheric drag.
// non-recipe: all 6 tests in this file run an equatorial 400 km circular orbit
// with various Cd / area / density combinations to verify analytical drag
// laws (energy loss, altitude decay, density scaling, area scaling, no-drag
// at zero density, corotation wind). Geometry and drag parameters are the
// assertion content, not a recipe input.
#[test]
fn tier3_drag_constant_density_energy_loss() {
    let (pos, vel) = iss_circular_state();
    let mass = 1000.0;
    let cd = 2.2;
    let area = 10.0;
    let density = 1e-12; // typical at 400 km
    let dt = 10.0;

    let mut sim = make_drag_sim(pos, vel, mass, cd, area, density, dt);

    let e_initial = specific_orbital_energy(pos, vel, MU_EARTH);

    // Propagate for 1 orbit (~92 min at 400 km)
    let period = 2.0 * std::f64::consts::PI * ((R_EARTH + 400_000.0).powi(3) / MU_EARTH).sqrt();
    let n_steps = (period / dt) as usize;
    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    let e_final = specific_orbital_energy(body.trans.position, body.trans.velocity, MU_EARTH);

    println!("  Initial energy: {e_initial:.6e} J/kg");
    println!("  Final energy:   {e_final:.6e} J/kg");
    println!("  Energy change:  {:.6e} J/kg", e_final - e_initial);

    // Energy must decrease (become more negative) due to drag
    assert!(
        e_final < e_initial,
        "Drag must remove orbital energy: E_final={e_final:.6e} >= E_initial={e_initial:.6e}"
    );

    // Energy loss should be non-trivial (not just numerical noise)
    let energy_loss = e_initial - e_final;
    assert!(
        energy_loss > 1e-3,
        "Energy loss {energy_loss:.6e} J/kg is too small to be physical drag"
    );
}

/// Verify that the semi-major axis decreases due to drag.
///
/// With constant density and ballistic drag, the semi-major axis should
/// steadily decrease. The decay rate is verified against the analytical
/// circular-orbit formula for F_drag = 0.5 * rho * v^2 * Cd * A:
///   da/rev = -2*pi * rho * Cd * A * a^2 / m
/// Derivation: dE/dt = -F_drag * v = -0.5*rho*v^3*Cd*A/m; integrate over
/// one orbit of period T = 2*pi*sqrt(a^3/mu); then use da = (2*a^2/mu)*dE.
#[test]
fn tier3_drag_altitude_decay() {
    let (pos, vel) = iss_circular_state();
    let mass = 1000.0;
    let cd = 2.2;
    let area = 10.0;
    let density = 1e-12;
    let dt = 10.0;

    let mut sim = make_drag_sim(pos, vel, mass, cd, area, density, dt);

    let a_initial =
        semi_major_axis_from_energy(specific_orbital_energy(pos, vel, MU_EARTH), MU_EARTH);

    // Propagate for 2 orbits
    let period = 2.0 * std::f64::consts::PI * (a_initial.powi(3) / MU_EARTH).sqrt();
    let n_steps = (2.0 * period / dt) as usize;
    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    let a_final = semi_major_axis_from_energy(
        specific_orbital_energy(body.trans.position, body.trans.velocity, MU_EARTH),
        MU_EARTH,
    );

    let da = a_initial - a_final;
    println!("  Initial SMA: {a_initial:.3} m");
    println!("  Final SMA:   {a_final:.3} m");
    println!("  SMA decay:   {da:.3} m over 2 orbits");

    assert!(
        a_final < a_initial,
        "SMA must decrease: a_final={a_final:.3} >= a_initial={a_initial:.3}"
    );

    // Analytical circular-orbit SMA decay magnitude per revolution for the
    // ballistic drag law F_drag = 0.5 * rho * v^2 * Cd * A:
    //   |da|/rev = 2*pi * rho * Cd * A * a^2 / m
    let da_per_rev_analytical =
        2.0 * std::f64::consts::PI * density * cd * area * a_initial * a_initial / mass;
    let da_per_rev_measured = da / 2.0; // averaged over 2 orbits

    let ratio = da_per_rev_measured / da_per_rev_analytical;
    println!(
        "  Analytical da/rev: {da_per_rev_analytical:.3} m, measured: {da_per_rev_measured:.3} m, ratio: {ratio:.3}"
    );
    assert!(
        ratio > 0.95 && ratio < 1.05,
        "Measured decay rate ratio {ratio:.3} is outside [0.95, 1.05] of analytical value"
    );
}

/// Higher atmospheric density should produce faster orbital decay.
///
/// Run two identical orbits with different constant densities. The ratio
/// of energy loss should approximately match the density ratio.
#[test]
fn tier3_drag_higher_density_faster_decay() {
    let (pos, vel) = iss_circular_state();
    let mass = 1000.0;
    let cd = 2.2;
    let area = 10.0;
    let dt = 10.0;

    let e_initial = specific_orbital_energy(pos, vel, MU_EARTH);
    let period = 2.0 * std::f64::consts::PI * ((R_EARTH + 400_000.0).powi(3) / MU_EARTH).sqrt();
    let n_steps = (period / dt) as usize;

    // Case 1: low density
    let density_low = 1e-12;
    let mut sim_low = make_drag_sim(pos, vel, mass, cd, area, density_low, dt);
    sim_low.step_n(n_steps).expect("step_n failed");
    let body_low = sim_low.body(0);
    let e_loss_low = e_initial
        - specific_orbital_energy(body_low.trans.position, body_low.trans.velocity, MU_EARTH);

    // Case 2: high density (10x)
    let density_high = 1e-11;
    let mut sim_high = make_drag_sim(pos, vel, mass, cd, area, density_high, dt);
    sim_high.step_n(n_steps).expect("step_n failed");
    let body_high = sim_high.body(0);
    let e_loss_high = e_initial
        - specific_orbital_energy(body_high.trans.position, body_high.trans.velocity, MU_EARTH);

    let density_ratio = density_high / density_low;
    let loss_ratio = e_loss_high / e_loss_low;

    println!("  Density ratio: {density_ratio:.1}x");
    println!("  Energy loss ratio: {loss_ratio:.3}x");
    println!("  E_loss_low:  {e_loss_low:.6e} J/kg");
    println!("  E_loss_high: {e_loss_high:.6e} J/kg");

    // Energy loss should scale approximately linearly with density.
    // Allow factor of 2 tolerance for nonlinear effects over one orbit.
    assert!(
        loss_ratio > density_ratio * 0.5 && loss_ratio < density_ratio * 2.0,
        "Energy loss ratio {loss_ratio:.3} should be near density ratio {density_ratio:.1}"
    );
}

/// Larger Cd*A should produce faster orbital decay.
///
/// Same orbit and density, but double the cross-sectional area.
/// Energy loss should approximately double.
#[test]
fn tier3_drag_larger_area_faster_decay() {
    let (pos, vel) = iss_circular_state();
    let mass = 1000.0;
    let cd = 2.2;
    let density = 1e-12;
    let dt = 10.0;

    let e_initial = specific_orbital_energy(pos, vel, MU_EARTH);
    let period = 2.0 * std::f64::consts::PI * ((R_EARTH + 400_000.0).powi(3) / MU_EARTH).sqrt();
    let n_steps = (period / dt) as usize;

    // Case 1: small area
    let area_small = 5.0;
    let mut sim_small = make_drag_sim(pos, vel, mass, cd, area_small, density, dt);
    sim_small.step_n(n_steps).expect("step_n failed");
    let body_small = sim_small.body(0);
    let e_loss_small = e_initial
        - specific_orbital_energy(
            body_small.trans.position,
            body_small.trans.velocity,
            MU_EARTH,
        );

    // Case 2: large area (4x)
    let area_large = 20.0;
    let mut sim_large = make_drag_sim(pos, vel, mass, cd, area_large, density, dt);
    sim_large.step_n(n_steps).expect("step_n failed");
    let body_large = sim_large.body(0);
    let e_loss_large = e_initial
        - specific_orbital_energy(
            body_large.trans.position,
            body_large.trans.velocity,
            MU_EARTH,
        );

    let area_ratio = area_large / area_small;
    let loss_ratio = e_loss_large / e_loss_small;

    println!("  Area ratio: {area_ratio:.1}x");
    println!("  Energy loss ratio: {loss_ratio:.3}x");

    // Energy loss should scale approximately linearly with area.
    assert!(
        loss_ratio > area_ratio * 0.5 && loss_ratio < area_ratio * 2.0,
        "Energy loss ratio {loss_ratio:.3} should be near area ratio {area_ratio:.1}"
    );
}

/// With zero density, drag should be zero and the orbit should be conserved.
///
/// Point-mass gravity with no drag is a Kepler orbit: energy and angular
/// momentum are conserved. Any energy change indicates spurious drag.
#[test]
fn tier3_drag_no_drag_at_zero_density() {
    let (pos, vel) = iss_circular_state();
    let mass = 1000.0;
    let cd = 2.2;
    let area = 10.0;
    let density = 0.0; // zero density => no drag
    let dt = 10.0;

    let mut sim = make_drag_sim(pos, vel, mass, cd, area, density, dt);

    let e_initial = specific_orbital_energy(pos, vel, MU_EARTH);
    let h_initial = pos.cross(vel).length(); // specific angular momentum

    // Propagate for 1 orbit
    let period = 2.0 * std::f64::consts::PI * ((R_EARTH + 400_000.0).powi(3) / MU_EARTH).sqrt();
    let n_steps = (period / dt) as usize;
    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    let e_final = specific_orbital_energy(body.trans.position, body.trans.velocity, MU_EARTH);
    let h_final = body.trans.position.cross(body.trans.velocity).length();

    let de = (e_final - e_initial).abs();
    let dh = (h_final - h_initial).abs();

    println!("  Energy conservation error: {de:.6e} J/kg");
    println!("  Angular momentum conservation error: {dh:.6e} m^2/s");

    // RK4 at dt=10s conserves energy to ~1e-3 J/kg over one orbit (observed 9.5e-4)
    // and angular momentum to ~1 m^2/s (observed 0.84).
    assert!(
        de < 1e-3,
        "Energy should be conserved with zero density: |dE|={de:.6e} J/kg"
    );
    assert!(
        dh < 1.0,
        "Angular momentum should be conserved: |dH|={dh:.6e} m^2/s"
    );
}

/// Atmospheric co-rotation wind reduces effective drag for prograde orbits.
///
/// In a co-rotating atmosphere, a prograde orbit has lower relative velocity
/// (v_rel = v_orbital - v_wind) than a retrograde orbit (v_rel = v_orbital + v_wind).
/// Since drag force scales as v_rel^2, the prograde orbit should experience
/// less drag and lose less energy per orbit.
///
/// This test uses two simulations with the same orbital speed but opposite
/// directions, with atmospheric co-rotation wind enabled via the atmosphere
/// configuration and planet omega.
#[test]
fn tier3_drag_corotation_wind_effect() {
    let r = R_EARTH + 400_000.0;
    let v = (MU_EARTH / r).sqrt();
    let mass = 1000.0;
    let cd = 2.2;
    let area = 10.0;
    let density = 1e-12;
    let dt = 10.0;

    // For the co-rotation test, we use constant_density on DragConfig but
    // need the atmosphere pipeline to provide wind. The atmosphere with
    // planet_omega provides co-rotation wind = omega x r.
    //
    // However, with constant_density on DragConfig, the density is overridden
    // but wind still comes from the atmosphere. We need an atmosphere config.
    // Use ExponentialAtmosphere with a density that we'll override via
    // constant_density anyway.

    let e_initial =
        specific_orbital_energy(DVec3::new(r, 0.0, 0.0), DVec3::new(0.0, v, 0.0), MU_EARTH);

    // Prograde: velocity in +Y when position is +X (same as Earth rotation)
    let pos = DVec3::new(r, 0.0, 0.0);
    let vel_prograde = DVec3::new(0.0, v, 0.0);
    let vel_retrograde = DVec3::new(0.0, -v, 0.0);

    let period = 2.0 * std::f64::consts::PI * (r.powi(3) / MU_EARTH).sqrt();
    let n_steps = (period / dt) as usize;

    // Prograde orbit with co-rotation wind
    let mut sim_pro = make_drag_sim_with_wind(pos, vel_prograde, mass, cd, area, density, dt);
    sim_pro.step_n(n_steps).expect("step_n failed");
    let body_pro = sim_pro.body(0);
    let e_loss_pro = e_initial
        - specific_orbital_energy(body_pro.trans.position, body_pro.trans.velocity, MU_EARTH);

    // Retrograde orbit with co-rotation wind
    let mut sim_retro = make_drag_sim_with_wind(pos, vel_retrograde, mass, cd, area, density, dt);
    sim_retro.step_n(n_steps).expect("step_n failed");
    let body_retro = sim_retro.body(0);
    // For retrograde, initial energy is the same magnitude
    let e_loss_retro = e_initial
        - specific_orbital_energy(
            body_retro.trans.position,
            body_retro.trans.velocity,
            MU_EARTH,
        );

    println!("  Prograde energy loss:   {e_loss_pro:.6e} J/kg");
    println!("  Retrograde energy loss: {e_loss_retro:.6e} J/kg");

    // Both orbits must lose energy to drag — compare signed values so a
    // wrong-sign regression (energy gain) fails here rather than being
    // masked by an absolute-value comparison.
    assert!(
        e_loss_retro > e_loss_pro,
        "Retrograde orbit (dE={e_loss_retro:.6e}) should lose more energy than prograde \
         (dE={e_loss_pro:.6e}) due to atmospheric co-rotation"
    );

    // Co-rotation wind at 400 km is ~460 m/s. Orbital velocity is ~7670 m/s.
    // Prograde v_rel ~ 7210, retrograde v_rel ~ 8130.
    // Drag ratio ~ (8130/7210)^2 ~ 1.27. Allow generous tolerance.
    let ratio = e_loss_retro / e_loss_pro;
    println!("  Energy loss ratio (retro/pro): {ratio:.3}");
    assert!(
        ratio > 1.05,
        "Retrograde/prograde energy loss ratio {ratio:.3} should be > 1.05"
    );
}

/// Create a simulation with atmosphere co-rotation wind.
///
/// Uses ExponentialAtmosphere for wind computation, but constant_density
/// on DragConfig overrides the atmospheric density so that the drag force
/// is comparable between cases.
fn make_drag_sim_with_wind(
    pos: DVec3,
    vel: DVec3,
    mass: f64,
    cd: f64,
    area: f64,
    density: f64,
    dt: f64,
) -> Simulation {
    use astrodyn::{AtmosphereConfig, AtmosphereModel, ExponentialAtmosphere};

    let mut sim = Simulation::new(
        SimulationTime::at_j2000(astrodyn::default_leap_second_table()),
        dt,
    );

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: astrodyn_runner::RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    // Atmosphere with co-rotation wind: evaluate_atmosphere computes
    // wind = omega x r_inertial (see astrodyn_atmosphere::compute_corotation_wind).
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(ExponentialAtmosphere {
            rho_0: 1e-12,
            h_0: 400_000.0,
            scale_height: 50_000.0,
        }),
        r_eq: R_EARTH,
        r_pol: R_EARTH * (1.0 - 1.0 / 298.257_223_563),
        planet_omega: OMEGA_EARTH,
    });
    sim.atmosphere_planet_source = Some(earth);

    let drag_config = DragConfig {
        cd,
        area,
        constant_density: Some(density),
    };

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: pos,
            velocity: vel,
        }
        .into(),
        rot: Some(
            RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }
            .into(),
        ),
        mass: Some(MassProperties::new(mass).into()),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: Some(drag_config),
        ..Default::default()
    });

    sim.validate().unwrap();
    sim
}
