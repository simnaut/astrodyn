// JEOD_INV: TS.01 — `<SelfRef>` is used here at the typed↔raw kernel-boundary helpers (named-method opt-in; the implicit `From<RotationalState>` / `From<MassProperties>` bypass was removed in #397).
//! Tier 3: Analytical physics-combination tests inspired by SIM_dyncomp RUNs.
//!
//! The numbered SIM_dyncomp RUN_* scenarios already have Docker-backed
//! cross-validation tests (tier3_sim_dyncomp_run2..run10). This file adds
//! analytical verification tests for physics combinations and laws that are
//! exercised by those RUNs but which admit closed-form / conservation-law
//! verification without a JEOD reference CSV.
//!
//! JEOD scenario mapping:
//! - tier3_dyncomp_point_mass_3dof_conservation:
//!   RUN_2 family (point-mass gravity): energy + angular momentum
//!   conservation for Keplerian orbits.
//! - tier3_dyncomp_point_mass_plus_thirdbody_conservation:
//!   RUN_7A/7B analog (third-body torque effect from Sun/Moon): third bodies
//!   do not conserve orbital angular momentum, but total energy stays bounded
//!   over short spans. Uses point-mass Earth; full SH adds secular drift but
//!   is not required to exhibit third-body torques.
//! - tier3_dyncomp_drag_point_mass_monotonic_decay:
//!   RUN_6/RUN_7C/7D analog (drag with gravity): semi-major axis must trend
//!   monotonically downward under drag. Uses point-mass Earth; SH adds a
//!   secular J2 SMA correction but is not required for monotonic decay.
//! - tier3_dyncomp_6dof_rigid_body_invariance:
//!   RUN_8A (torque-free rotation in orbit): inertial angular momentum is
//!   conserved, body-frame omega varies (Euler's equations).
//! - tier3_dyncomp_external_force_impulse_response:
//!   RUN_9C/9D (external force): delta-v = F * dt / m during force window.
//! - tier3_dyncomp_external_torque_impulse_response:
//!   RUN_9A/9B/9C (external torque): delta-omega = tau * dt / I on axis.
//! - tier3_dyncomp_attitude_stability_major_axis:
//!   Related to RUN_8B (LVLH rate) attitude propagation: spin about the
//!   major principal axis is stable (intermediate-axis theorem).

use astrodyn::recipes::helpers::energy_conservation::specific_orbital_energy;
use astrodyn::{
    GravityControl, GravityControls, GravityGradient, GravityModel, GravitySource, JeodQuat,
    MassProperties, RotationalState, SimulationTime, TranslationalState,
};
use astrodyn::{GravitySourceEntry, VehicleConfig};
use astrodyn_runner::{RotationModel, Simulation};
use glam::{DMat3, DVec3};

/// Earth gravitational parameter (m^3/s^2) — JEOD `earth_GGM05C.cc`.
const MU_EARTH: f64 = astrodyn::EARTH.shape.mu;
/// Earth mean equatorial radius (m) — JEOD `earth.cc`.
const R_EARTH: f64 = astrodyn::EARTH.shape.r_eq;
/// Sun gravitational parameter (m^3/s^2) — JEOD `sun_spherical.cc`.
const MU_SUN: f64 = astrodyn::SUN.shape.mu;
/// Moon gravitational parameter (m^3/s^2) — JEOD `moon_GRAIL150.cc`.
const MU_MOON: f64 = astrodyn::MOON.shape.mu;
/// Typical Earth–Sun distance (m, ~1 AU).
const R_EARTH_SUN: f64 = 1.495_978_707e11;
/// Typical Earth–Moon distance (m).
const R_EARTH_MOON: f64 = 3.844_0e8;

/// Add a central Earth gravity source (point-mass).
fn add_earth_point_mass(sim: &mut Simulation) -> usize {
    sim.add_source(
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
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    )
}

/// Circular orbit at 400 km altitude (ISS-like) in the equatorial plane.
fn iss_circular_state() -> (DVec3, DVec3) {
    let r = R_EARTH + 400_000.0;
    let v = (MU_EARTH / r).sqrt();
    (DVec3::new(r, 0.0, 0.0), DVec3::new(0.0, v, 0.0))
}

/// Build a 3-DOF point-mass orbit simulation (pure Kepler).
fn make_kepler_sim(pos: DVec3, vel: DVec3, mass: f64, dt: f64) -> Simulation {
    let mut sim = Simulation::new(
        SimulationTime::at_j2000(astrodyn::default_leap_second_table()),
        dt,
    );
    let earth = add_earth_point_mass(&mut sim);

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: pos,
            velocity: vel,
        }),
        rot: None,
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(
            &(MassProperties::new(mass)),
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        ..Default::default()
    });

    sim.validate().unwrap();
    sim
}

// ─── Test 1: Point-mass Kepler conservation ───

/// 3-DOF point-mass orbit (RUN_2 configuration without reference data):
/// specific orbital energy and angular momentum are conserved by Kepler
/// dynamics. Any drift is numerical integrator error.
#[test]
fn tier3_dyncomp_point_mass_3dof_conservation() {
    let (pos, vel) = iss_circular_state();
    let dt = 10.0;
    let mut sim = make_kepler_sim(pos, vel, 1000.0, dt);

    let e0 = specific_orbital_energy(pos, vel, MU_EARTH);
    let h0 = pos.cross(vel);

    // Propagate for 3 orbits.
    let period = 2.0 * std::f64::consts::PI * ((R_EARTH + 400_000.0).powi(3) / MU_EARTH).sqrt();
    let n_steps = (3.0 * period / dt) as usize;
    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    let e1 = specific_orbital_energy(
        body.trans.position.raw_si(),
        body.trans.velocity.raw_si(),
        MU_EARTH,
    );
    let h1 = body
        .trans
        .position
        .raw_si()
        .cross(body.trans.velocity.raw_si());

    let de = (e1 - e0).abs() / e0.abs();
    let dh = (h1 - h0).length() / h0.length();

    println!("  3 orbits Kepler: relative dE={de:.3e}, relative dH={dh:.3e}");

    // RK4 at dt=10s: observed well below these bounds.
    assert!(de < 1.0e-7, "Kepler energy drift {de:.3e} too large");
    assert!(
        dh < 1.0e-8,
        "Kepler angular momentum drift {dh:.3e} too large"
    );
}

// ─── Test 2: Point-mass Earth + third-body produces non-conservation of h ───

/// Analytical analog of RUN_7A/7B: Sun + Moon third-body gravity exerts
/// torques about Earth, so orbital angular momentum about Earth is NOT
/// strictly conserved. Its direction drifts (nodal regression, inclination
/// wobble), yet total orbital energy stays bounded over short spans.
///
/// This test intentionally uses point-mass Earth gravity only. Full
/// spherical-harmonic Earth gravity would add secular drift, but it is not
/// required to demonstrate the third-body torque effect checked here.
#[test]
fn tier3_dyncomp_point_mass_plus_thirdbody_conservation() {
    let (pos, vel) = iss_circular_state();
    let dt = 10.0;

    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, dt);

    let earth = add_earth_point_mass(&mut sim);

    // Sun/Moon placed off the orbital (X-Y) plane so their differential
    // accelerations produce a torque on the orbit about axes in the plane
    // (nodal regression / inclination wobble).  A purely in-plane third body
    // only perturbs the energy and periapsis — it does not tilt the orbital
    // plane.
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            },
            // Sun ~23.4° out of the equator (ecliptic obliquity).
            position: astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(DVec3::new(
                R_EARTH_SUN * 0.9175,
                0.0,
                R_EARTH_SUN * 0.3977,
            )),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    let moon = sim.add_source(
        "Moon",
        GravitySourceEntry {
            source: GravitySource {
                mu: MU_MOON,
                model: GravityModel::PointMass,
            },
            // Moon ~5° off the ecliptic — put it off the XY plane too.
            position: astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(DVec3::new(
                0.0,
                R_EARTH_MOON * 0.9063,
                R_EARTH_MOON * 0.4226,
            )),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::None,
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: pos,
            velocity: vel,
        }),
        rot: None,
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(
            &(MassProperties::new(1000.0)),
        )),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(earth, GravityGradient::Skip),
                GravityControl::new_third_body(sun),
                GravityControl::new_third_body(moon),
            ],
        },
        ..Default::default()
    });

    sim.validate().unwrap();

    let e0 = specific_orbital_energy(pos, vel, MU_EARTH);
    let h0 = pos.cross(vel);

    // Integrate for one orbit.
    let period = 2.0 * std::f64::consts::PI * ((R_EARTH + 400_000.0).powi(3) / MU_EARTH).sqrt();
    let n_steps = (period / dt) as usize;
    sim.step_n(n_steps).expect("step_n failed");

    let body = sim.body(0);
    let e1 = specific_orbital_energy(
        body.trans.position.raw_si(),
        body.trans.velocity.raw_si(),
        MU_EARTH,
    );
    let h1 = body
        .trans
        .position
        .raw_si()
        .cross(body.trans.velocity.raw_si());

    // Orbital energy about Earth should remain bounded (~third-body magnitude
    // times one orbit).  Angular momentum *direction* should shift measurably.
    let relative_de = (e1 - e0).abs() / e0.abs();
    let dh_angle = {
        let cos_th = h0.dot(h1) / (h0.length() * h1.length());
        cos_th.clamp(-1.0, 1.0).acos()
    };

    println!("  1 orbit SH+3body: relative dE={relative_de:.3e}, dH_angle={dh_angle:.3e} rad");

    // Energy change from third bodies over one LEO orbit is tiny but not zero.
    assert!(
        relative_de < 1.0e-5,
        "Third-body relative energy drift {relative_de:.3e} too large"
    );

    // Verify the third-body torques had a measurable effect: the angular
    // momentum vector must have moved more than pure-Kepler numerical noise.
    assert!(
        dh_angle > 1.0e-10,
        "Third-body torque should tilt orbital plane; dH_angle={dh_angle:.3e} is pure noise"
    );
}

// ─── Test 3: drag leads to monotonic decay of SMA ───

/// Analytical analog of RUN_6/RUN_7C/7D (gravity + drag): in a point-mass
/// Earth + drag LEO orbit, the semi-major axis must trend monotonically
/// downward. We sample at orbital-period intervals (filtering out the
/// in-orbit oscillation of instantaneous position) and verify strict
/// monotonic decrease.
///
/// This test intentionally uses point-mass Earth gravity only. Full
/// spherical-harmonic Earth gravity would add secular J2 effects but is
/// not required to demonstrate monotonic SMA decay under drag.
#[test]
fn tier3_dyncomp_drag_point_mass_monotonic_decay() {
    use astrodyn::{AtmosphereConfig, AtmosphereModel, DragConfig, ExponentialAtmosphere};

    let (pos, vel) = iss_circular_state();
    let dt = 10.0;
    let mass = 1000.0;

    let mut sim = Simulation::new(
        SimulationTime::at_j2000(astrodyn::default_leap_second_table()),
        dt,
    );
    let earth = add_earth_point_mass(&mut sim);

    // Constant-density drag atmosphere so we have repeatable per-orbit decay.
    sim.atmosphere = Some(AtmosphereConfig {
        model: AtmosphereModel::Exponential(ExponentialAtmosphere {
            rho_0: 1e-11,
            h_0: 400_000.0,
            scale_height: 50_000.0,
        }),
        r_eq: R_EARTH,
        r_pol: R_EARTH * (1.0 - 1.0 / 298.257_223_563),
        planet_omega: 0.0,
    });
    sim.atmosphere_planet_source = Some(earth);

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: pos,
            velocity: vel,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(
            &(MassProperties::new(mass)),
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        drag: Some(DragConfig {
            cd: 2.2,
            area: 20.0,
            constant_density: Some(1e-11), // boosted density for clear decay
        }),
        ..Default::default()
    });

    sim.validate().unwrap();

    // Sample SMA at orbital-period intervals for several orbits.
    let period = 2.0 * std::f64::consts::PI * ((R_EARTH + 400_000.0).powi(3) / MU_EARTH).sqrt();
    let steps_per_orbit = (period / dt) as usize;

    let mut sma_samples = Vec::new();
    for _ in 0..5 {
        sim.step_n(steps_per_orbit).expect("step_n failed");
        let body = sim.body(0);
        let e = specific_orbital_energy(
            body.trans.position.raw_si(),
            body.trans.velocity.raw_si(),
            MU_EARTH,
        );
        let a = -MU_EARTH / (2.0 * e);
        sma_samples.push(a);
    }

    println!("  SMA per orbit: {:?}", sma_samples);

    // Strictly monotonic decrease.
    for window in sma_samples.windows(2) {
        assert!(
            window[1] < window[0],
            "SMA did not decrease monotonically: {} -> {}",
            window[0],
            window[1]
        );
    }

    // Total decay should be non-trivial (> 10 m over 5 orbits at this density).
    let total_decay = sma_samples[0] - sma_samples[sma_samples.len() - 1];
    assert!(
        total_decay > 10.0,
        "Total SMA decay {total_decay:.2} m is implausibly small"
    );
}

// ─── Test 4: torque-free rigid-body rotation conserves inertial H ───

/// RUN_8A physics (spherical Earth gravity + no torque + asymmetric inertia):
/// For a rigid body with no applied torque, Euler's equations give the
/// body-frame omega a time-varying path, but the *inertial-frame* angular
/// momentum vector is rigorously conserved.
#[test]
fn tier3_dyncomp_6dof_rigid_body_invariance() {
    let (pos, vel) = iss_circular_state();
    let dt = 0.5;

    // Asymmetric diagonal inertia (principal axes aligned with body frame).
    let i_x = 1000.0;
    let i_y = 2500.0;
    let i_z = 2500.0;
    let inertia = DMat3::from_cols(
        DVec3::new(i_x, 0.0, 0.0),
        DVec3::new(0.0, i_y, 0.0),
        DVec3::new(0.0, 0.0, i_z),
    );
    let mass_props = MassProperties::with_inertia(1000.0, inertia, DVec3::ZERO);

    let mut sim = Simulation::new(
        SimulationTime::at_j2000(astrodyn::default_leap_second_table()),
        dt,
    );
    let earth = add_earth_point_mass(&mut sim);

    // Initial omega tipped off the major axis to exercise all three Euler eqs.
    let omega0_body = DVec3::new(0.1, 0.02, 0.0); // rad/s
    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: pos,
            velocity: vel,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: omega0_body,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        // Gravity gradient torque OFF (RUN_8 has it off).
        compute_gravity_gradient: false,
        ..Default::default()
    });

    sim.validate().unwrap();

    // H_inertial(0) = T^T * I * omega_body
    let body = sim.body(0);
    let q0 = body.rot.as_ref().unwrap().q_inertial_body.to_jeod_quat();
    let t0 = q0.left_quat_to_transformation(); // inertial→body
    let h_body0 = inertia * omega0_body;
    let h_inertial_0 = t0.transpose() * h_body0;

    // Propagate for 60 seconds.
    sim.step_n((60.0 / dt) as usize).expect("step_n failed");

    let body = sim.body(0);
    let q1 = body.rot.as_ref().unwrap().q_inertial_body.to_jeod_quat();
    let omega1_body = body.rot.as_ref().unwrap().ang_vel_body.raw_si();
    let t1 = q1.left_quat_to_transformation();
    let h_body1 = inertia * omega1_body;
    let h_inertial_1 = t1.transpose() * h_body1;

    let dh_rel = (h_inertial_1 - h_inertial_0).length() / h_inertial_0.length();
    let mag_rel = (h_body1.length() - h_body0.length()).abs() / h_body0.length();

    println!(
        "  60s torque-free: |H_inertial| conservation {dh_rel:.3e}, |H_body| mag error {mag_rel:.3e}"
    );

    assert!(
        dh_rel < 1.0e-6,
        "RootInertial H not conserved: relative error {dh_rel:.3e}"
    );
    assert!(
        mag_rel < 1.0e-6,
        "|H| magnitude not conserved: relative error {mag_rel:.3e}"
    );

    // Sanity: with asymmetric inertia and tipped omega, body-frame omega DOES
    // change (otherwise the test would be trivial).
    let domega = (omega1_body - omega0_body).length();
    assert!(
        domega > 1.0e-4,
        "body omega did not evolve; test trivially passes: |domega|={domega:.3e}"
    );
}

// ─── Test 5: external force delta-v ───

/// RUN_9C physics (external force application): during a constant force window,
/// the body's velocity increment equals F*dt/m along the force direction.
#[test]
fn tier3_dyncomp_external_force_impulse_response() {
    let (pos, vel0) = iss_circular_state();
    let dt = 1.0;
    let mass = 1000.0;
    let mut sim = make_kepler_sim(pos, vel0, mass, dt);

    let force_inertial = DVec3::new(50.0, 0.0, 0.0); // 50 N along +X inertial
    let force_duration = 10.0;

    // Record velocity immediately before force window.
    let v_before = sim.body(0).trans.velocity.raw_si();

    // Apply force for force_duration seconds.
    sim.set_body_external_force(0, force_inertial);
    sim.step_n((force_duration / dt) as usize)
        .expect("step_n failed");
    sim.set_body_external_force(0, DVec3::ZERO);

    let v_after = sim.body(0).trans.velocity.raw_si();
    let delta_v = v_after - v_before;

    // Expected delta-v components from impulse: F*dt/m
    // The orbital motion contributes extra delta-v from gravity during the
    // window, so we subtract the no-force reference to isolate the force.
    let mut ref_sim = make_kepler_sim(pos, vel0, mass, dt);
    // Advance the reference simulation by the same elapsed time as the force
    // window started at t=0 and lasted `force_duration`.
    ref_sim
        .step_n((force_duration / dt) as usize)
        .expect("step_n failed");
    let v_reference = ref_sim.body(0).trans.velocity.raw_si();

    let force_delta_v = v_after - v_reference;
    let expected_dv = force_inertial * force_duration / mass;

    let err = (force_delta_v - expected_dv).length();
    let rel_err = err / expected_dv.length();

    println!(
        "  Impulse: measured dv={:?}, expected={:?}, rel_err={rel_err:.3e}",
        force_delta_v, expected_dv
    );

    // Generous tolerance: RK4 integration of combined force+gravity introduces
    // second-order cross-terms, but the first-order F*t/m should dominate.
    assert!(
        rel_err < 1.0e-4,
        "External-force delta-v error {rel_err:.3e} too large"
    );

    // Delta-v direction should match force direction.
    let cos_align = force_delta_v.normalize().dot(force_inertial.normalize());
    assert!(
        cos_align > 0.9999,
        "delta-v direction {force_delta_v:?} not aligned with force {force_inertial:?}: cos={cos_align}"
    );

    // Use delta_v to suppress unused-variable warning (it's informational).
    let _ = delta_v;
}

// ─── Test 6: external torque delta-omega ───

/// RUN_9A physics (external torque application): during a constant torque
/// window, the body-frame angular velocity increment about a principal axis
/// equals tau*dt/I.
#[test]
fn tier3_dyncomp_external_torque_impulse_response() {
    let (pos, vel0) = iss_circular_state();
    let dt = 1.0;
    let mass = 1000.0;

    // Diagonal inertia — apply torque along the body x-axis (principal axis).
    let i_x = 1000.0;
    let i_y = 2500.0;
    let i_z = 2500.0;
    let inertia = DMat3::from_cols(
        DVec3::new(i_x, 0.0, 0.0),
        DVec3::new(0.0, i_y, 0.0),
        DVec3::new(0.0, 0.0, i_z),
    );
    let mass_props = MassProperties::with_inertia(mass, inertia, DVec3::ZERO);

    let mut sim = Simulation::new(
        SimulationTime::at_j2000(astrodyn::default_leap_second_table()),
        dt,
    );
    let earth = add_earth_point_mass(&mut sim);

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: pos,
            velocity: vel0,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: DVec3::ZERO,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        compute_gravity_gradient: false,
        ..Default::default()
    });
    sim.validate().unwrap();

    let torque_body = DVec3::new(10.0, 0.0, 0.0); // 10 N·m about x
    let torque_duration = 10.0;

    // Apply torque window.
    sim.set_body_external_torque(0, torque_body);
    sim.step_n((torque_duration / dt) as usize)
        .expect("step_n failed");
    sim.set_body_external_torque(0, DVec3::ZERO);

    let omega_after = sim.body(0).rot.as_ref().unwrap().ang_vel_body.raw_si();
    // Expected: omega_x = tau_x * dt / I_xx (y, z remain ~zero).
    let expected_omega_x = torque_body.x * torque_duration / i_x;

    let err_x = (omega_after.x - expected_omega_x).abs();
    let rel_err = err_x / expected_omega_x.abs();
    println!(
        "  Torque impulse: omega={:?}, expected omega_x={expected_omega_x:.6}, rel_err={rel_err:.3e}",
        omega_after
    );

    assert!(
        rel_err < 1.0e-6,
        "Torque delta-omega error {rel_err:.3e} too large"
    );
    assert!(
        omega_after.y.abs() < 1.0e-8,
        "omega_y should remain zero, got {}",
        omega_after.y
    );
    assert!(
        omega_after.z.abs() < 1.0e-8,
        "omega_z should remain zero, got {}",
        omega_after.z
    );
}

// ─── Test 7: major-axis spin stability (intermediate-axis theorem) ───

/// Related to RUN_8B (rotational propagation): the intermediate-axis theorem
/// (Dzhanibekov effect) predicts stable rotation about the axis of maximum
/// inertia and unstable rotation about the intermediate axis.
///
/// Here we verify the *stable* case: a body spinning about its major
/// (largest-inertia) axis with small perpendicular perturbations must keep
/// nearly all its angular momentum along that axis.
#[test]
fn tier3_dyncomp_attitude_stability_major_axis() {
    let (pos, vel) = iss_circular_state();
    let dt = 0.1;

    // I_z is the largest principal moment (major axis).
    let i_x = 500.0;
    let i_y = 1000.0;
    let i_z = 2500.0; // major axis
    let inertia = DMat3::from_cols(
        DVec3::new(i_x, 0.0, 0.0),
        DVec3::new(0.0, i_y, 0.0),
        DVec3::new(0.0, 0.0, i_z),
    );
    let mass_props = MassProperties::with_inertia(1000.0, inertia, DVec3::ZERO);

    // Spin about z with 1% perturbation on x and y.
    let omega0 = DVec3::new(0.01, 0.01, 1.0); // rad/s

    let mut sim = Simulation::new(
        SimulationTime::at_j2000(astrodyn::default_leap_second_table()),
        dt,
    );
    let earth = add_earth_point_mass(&mut sim);

    sim.add_body(VehicleConfig {
        trans: astrodyn::typed_bridge::trans_raw_to_root(&TranslationalState {
            position: pos,
            velocity: vel,
        }),
        rot: Some(astrodyn::typed_bridge::rot_raw_to_self_ref(
            &(RotationalState {
                quaternion: JeodQuat::identity(),
                ang_vel_body: omega0,
            }),
        )),
        mass: Some(astrodyn::typed_bridge::mass_raw_to_self_ref(&(mass_props))),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        compute_gravity_gradient: false,
        ..Default::default()
    });

    sim.validate().unwrap();

    // Propagate 60 s and sample omega regularly.
    let mut max_perp = 0.0_f64;
    for _ in 0..600 {
        sim.step_n(1).expect("step_n failed");
        let omega = sim.body(0).rot.as_ref().unwrap().ang_vel_body.raw_si();
        let perp = (omega.x.powi(2) + omega.y.powi(2)).sqrt();
        if perp > max_perp {
            max_perp = perp;
        }
    }

    let omega_final = sim.body(0).rot.as_ref().unwrap().ang_vel_body.raw_si();
    println!(
        "  Major-axis spin: omega_final={:?}, max |omega_perp|={max_perp:.6} rad/s",
        omega_final
    );

    // The perpendicular components of omega oscillate but stay bounded near
    // the initial 0.01 rad/s perturbation.  For stable major-axis rotation the
    // bound should stay well below the spin rate (1 rad/s).
    assert!(
        max_perp < 0.05,
        "Major-axis spin unstable: |omega_perp| grew to {max_perp:.3}"
    );
    // Spin about z must remain dominant.
    assert!(
        omega_final.z.abs() > 0.99,
        "Z-axis spin should be preserved, got {:.4}",
        omega_final.z
    );
}
