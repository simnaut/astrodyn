//! Tier 3: 6-DOF drag verification tests.
//!
//! These tests exercise aerodynamic drag with rotational dynamics through
//! `Simulation::step()`, verifying that drag interacts correctly with the
//! attitude state. All tests use analytical verification.

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, Simulation, VehicleConfig};
use jeod_sim::recipes::helpers::energy_conservation::specific_orbital_energy;
use jeod_sim::{
    AtmosphereConfig, AtmosphereModel, DragConfig, ExponentialAtmosphere, GravityControl,
    GravityControls, GravityModel, GravitySource, JeodQuat, MassProperties, RotationalState,
    SimulationTime, TranslationalState,
};

/// Earth gravitational parameter (m^3/s^2) — JEOD `earth_GGM05C.cc`.
const MU_EARTH: f64 = jeod_sim::EARTH.shape.mu;

/// Earth mean equatorial radius (m) — JEOD `earth.cc`.
const R_EARTH: f64 = jeod_sim::EARTH.shape.r_eq;

/// Create a 6-DOF simulation with point-mass gravity and constant-density drag.
#[allow(clippy::too_many_arguments)]
fn make_6dof_drag_sim(
    pos: DVec3,
    vel: DVec3,
    mass: f64,
    inertia: DMat3,
    quat: JeodQuat,
    ang_vel: DVec3,
    cd: f64,
    area: f64,
    density: f64,
    dt: f64,
) -> Simulation {
    let mut sim = Simulation::new(
        SimulationTime::at_j2000(jeod_sim::default_leap_second_table()),
        dt,
    );

    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_EARTH,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: jeod_runner::RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );

    // Atmosphere config is required by validation even when constant_density
    // overrides the atmospheric density.
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

    let mass_props = MassProperties::with_inertia(mass, inertia, DVec3::ZERO);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: pos,
            velocity: vel,
        },
        rot: Some(RotationalState {
            quaternion: quat,
            ang_vel_body: ang_vel,
        }),
        mass: Some(mass_props),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        drag: Some(drag_config),
        ..Default::default()
    });

    sim.validate().unwrap();
    sim
}

/// 6-DOF with rotation: drag should still remove orbital energy.
///
/// For the ballistic drag model (constant Cd*A), attitude does not affect
/// the drag magnitude. The force always opposes the relative velocity
/// regardless of body orientation. This test verifies that a spinning body
/// still experiences proper orbital energy dissipation.
// non-recipe: 1 t mass + ballistic Cd·A on a 400 km equatorial circular
// orbit; the geometry is bespoke and the drag config (Cd=2.2, area=10,
// constant density 1e-12) drives the assertion content.
#[test]
fn tier3_drag_with_rotation_energy_loss() {
    let r = R_EARTH + 400_000.0;
    let v = (MU_EARTH / r).sqrt();
    let pos = DVec3::new(r, 0.0, 0.0);
    let vel = DVec3::new(0.0, v, 0.0);

    let mass = 1000.0;
    let cd = 2.2;
    let area = 10.0;
    let density = 1e-12;
    let dt = 10.0;

    // Uniform sphere inertia: I = 2/5 * m * r^2 (1m radius)
    let i_val = 0.4 * mass * 1.0;
    let inertia = DMat3::from_diagonal(DVec3::splat(i_val));

    // Non-trivial initial attitude and spin
    let eigen_angle = 30.0_f64.to_radians();
    let eigen_axis = DVec3::new(1.0, 1.0, 1.0).normalize();
    let quat = JeodQuat::left_quat_from_eigen_rotation(eigen_angle, eigen_axis);
    let ang_vel = DVec3::new(0.01, -0.005, 0.003); // rad/s

    let mut sim = make_6dof_drag_sim(
        pos, vel, mass, inertia, quat, ang_vel, cd, area, density, dt,
    );

    let e_initial = specific_orbital_energy(pos, vel, MU_EARTH);

    // Propagate for 1 orbit
    let period = 2.0 * std::f64::consts::PI * (r.powi(3) / MU_EARTH).sqrt();
    let n_steps = (period / dt) as usize;
    sim.step_n(n_steps);

    let body = sim.body(0);
    let e_final = specific_orbital_energy(body.trans.position, body.trans.velocity, MU_EARTH);

    println!("  Initial energy: {e_initial:.6e} J/kg");
    println!("  Final energy:   {e_final:.6e} J/kg");
    println!("  Energy change:  {:.6e} J/kg", e_final - e_initial);

    // Energy must decrease due to drag
    assert!(
        e_final < e_initial,
        "Drag must remove orbital energy even with rotation: \
         E_final={e_final:.6e} >= E_initial={e_initial:.6e}"
    );

    // Also verify the body is still spinning (angular velocity not damped to zero
    // by ballistic drag, since ballistic drag produces no torque)
    let rot = body
        .rot
        .as_ref()
        .expect("6-DOF body should have rotational state");
    let final_ang_vel_mag = rot.ang_vel_body.length();
    let initial_ang_vel_mag = ang_vel.length();

    println!("  Initial |omega|: {initial_ang_vel_mag:.6e} rad/s");
    println!("  Final   |omega|: {final_ang_vel_mag:.6e} rad/s");

    // Ballistic drag should not produce torque, so angular velocity magnitude
    // should be approximately conserved (point-mass gravity produces no torque
    // either, since the gravity gradient flag is off).
    let ang_vel_change = (final_ang_vel_mag - initial_ang_vel_mag).abs() / initial_ang_vel_mag;
    assert!(
        ang_vel_change < 0.01,
        "Angular velocity magnitude should be conserved (no torque): \
         relative change = {ang_vel_change:.6e}"
    );
}

/// Ballistic drag should produce the same trajectory regardless of attitude.
///
/// Since ballistic drag (constant Cd*A) applies force proportional to
/// relative velocity in the inertial frame, two bodies with different
/// attitudes but same translational state should experience the same
/// translational trajectory. The force direction changes in the body frame,
/// but in the inertial frame it is always anti-velocity.
// non-recipe: same drag setup as `tier3_drag_with_rotation_energy_loss`,
// run twice with different attitudes to verify ballistic-drag invariance.
#[test]
fn tier3_drag_attitude_invariance_ballistic() {
    let r = R_EARTH + 400_000.0;
    let v = (MU_EARTH / r).sqrt();
    let pos = DVec3::new(r, 0.0, 0.0);
    let vel = DVec3::new(0.0, v, 0.0);

    let mass = 1000.0;
    let cd = 2.2;
    let area = 10.0;
    let density = 1e-12;
    let dt = 10.0;

    // Uniform sphere inertia
    let i_val = 0.4 * mass * 1.0;
    let inertia = DMat3::from_diagonal(DVec3::splat(i_val));

    // Case 1: identity attitude, no spin
    let quat1 = JeodQuat::identity();
    let ang_vel1 = DVec3::ZERO;

    // Case 2: 45-degree rotation about Z, spinning
    let quat2 =
        JeodQuat::left_quat_from_eigen_rotation(45.0_f64.to_radians(), DVec3::new(0.0, 0.0, 1.0));
    let ang_vel2 = DVec3::new(0.0, 0.0, 0.05);

    let mut sim1 = make_6dof_drag_sim(
        pos, vel, mass, inertia, quat1, ang_vel1, cd, area, density, dt,
    );
    let mut sim2 = make_6dof_drag_sim(
        pos, vel, mass, inertia, quat2, ang_vel2, cd, area, density, dt,
    );

    // Propagate for 1 orbit
    let period = 2.0 * std::f64::consts::PI * (r.powi(3) / MU_EARTH).sqrt();
    let n_steps = (period / dt) as usize;
    sim1.step_n(n_steps);
    sim2.step_n(n_steps);

    let body1 = sim1.body(0);
    let body2 = sim2.body(0);

    let pos_diff = (body1.trans.position - body2.trans.position).length();
    let vel_diff = (body1.trans.velocity - body2.trans.velocity).length();

    println!("  Position difference: {pos_diff:.6e} m");
    println!("  Velocity difference: {vel_diff:.6e} m/s");

    // For ballistic drag, the translational trajectories should be identical
    // (within numerical precision). The body frame force direction differs, but
    // when transformed back to inertial, it is the same.
    assert!(
        pos_diff < 1e-3,
        "Ballistic drag trajectory should be attitude-invariant: pos_diff={pos_diff:.6e} m"
    );
    assert!(
        vel_diff < 1e-6,
        "Ballistic drag trajectory should be attitude-invariant: vel_diff={vel_diff:.6e} m/s"
    );
}
