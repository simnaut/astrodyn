//! Body initialization functions for translational state.
//!
//! Port of JEOD `DynBodyInitOrbit`, `DynBodyInitLvlh`, and NED initialization
//! from `models/dynamics/body_action/src/`.
//!
//! These functions initialize a vehicle's translational state from various
//! parameterizations: Keplerian orbital elements, LVLH-relative state, or
//! NED (North-East-Down) relative state.

use crate::state::TranslationalState;
use glam::{DMat3, DVec3};
use jeod_math::OrbitalElements;
use jeod_math::{compute_lvlh_frame, geodetic_to_cartesian, mat3_from_rows, GeodeticState};

/// Initialize translational state from Keplerian orbital elements (true anomaly).
///
/// Port of JEOD `DynBodyInitOrbit::apply()` from `dyn_body_init_orbit.cc`,
/// for the `SmaEccIncAscnodeArgperTanom` element set.
///
/// # Arguments
/// * `semi_major_axis` - Semi-major axis (m)
/// * `eccentricity` - Orbital eccentricity
/// * `inclination` - Inclination (rad)
/// * `raan` - Right ascension of ascending node (rad)
/// * `arg_periapsis` - Argument of periapsis (rad)
/// * `true_anomaly` - True anomaly (rad)
/// * `mu` - Gravitational parameter of central body (m^3/s^2)
pub fn init_from_orbital_elements(
    semi_major_axis: f64,
    eccentricity: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    true_anomaly: f64,
    mu: f64,
) -> TranslationalState {
    // JEOD dyn_body_init_orbit.cc:101-111: validate mu before use.
    assert!(
        mu > 0.0,
        "init_from_orbital_elements: mu must be positive, got {mu}"
    );
    assert!(
        semi_major_axis.is_finite(),
        "init_from_orbital_elements: semi_major_axis must be finite, got {semi_major_axis}"
    );
    assert!(
        (0.0..1.0).contains(&eccentricity),
        "init_from_orbital_elements: eccentricity must be in [0, 1), got {eccentricity}"
    );

    // Build OrbitalElements with the provided Keplerian elements.
    // Following JEOD dyn_body_init_orbit.cc: populate semiparam, angles, true_anom,
    // then call nu_to_anomalies() and to_cartesian().
    let mut oe = OrbitalElements::default();
    oe.semi_major_axis = semi_major_axis;
    oe.e_mag = eccentricity;
    oe.inclination = inclination;
    oe.long_asc_node = raan;
    oe.arg_periapsis = arg_periapsis;
    oe.semiparam = semi_major_axis * (1.0 - eccentricity * eccentricity);
    oe.true_anom = true_anomaly;
    oe.nu_to_anomalies();

    let (position, velocity) = oe
        .to_cartesian(mu)
        .expect("init_from_orbital_elements: to_cartesian failed");

    TranslationalState { position, velocity }
}

/// Initialize translational state from Keplerian orbital elements (mean anomaly).
///
/// Port of JEOD `DynBodyInitOrbit::apply()` from `dyn_body_init_orbit.cc`,
/// for the `SmaEccIncAscnodeArgperManom` element set.
///
/// Solves Kepler's equation internally to convert mean anomaly to true anomaly.
///
/// # Arguments
/// * `semi_major_axis` - Semi-major axis (m)
/// * `eccentricity` - Orbital eccentricity
/// * `inclination` - Inclination (rad)
/// * `raan` - Right ascension of ascending node (rad)
/// * `arg_periapsis` - Argument of periapsis (rad)
/// * `mean_anomaly` - Mean anomaly (rad)
/// * `mu` - Gravitational parameter of central body (m^3/s^2)
pub fn init_from_mean_anomaly(
    semi_major_axis: f64,
    eccentricity: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    mean_anomaly: f64,
    mu: f64,
) -> TranslationalState {
    // JEOD dyn_body_init_orbit.cc:101-111: validate mu before use.
    assert!(
        mu > 0.0,
        "init_from_mean_anomaly: mu must be positive, got {mu}"
    );
    assert!(
        semi_major_axis.is_finite(),
        "init_from_mean_anomaly: semi_major_axis must be finite, got {semi_major_axis}"
    );
    assert!(
        (0.0..1.0).contains(&eccentricity),
        "init_from_mean_anomaly: eccentricity must be in [0, 1), got {eccentricity}"
    );

    // Following JEOD dyn_body_init_orbit.cc lines 302-318:
    // Populate elem with semiparam, e_mag, inclination, arg_periapsis, long_asc_node,
    // set mean_anom, then call mean_anom_to_nu() to solve Kepler's equation.
    let mut oe = OrbitalElements::default();
    oe.semi_major_axis = semi_major_axis;
    oe.e_mag = eccentricity;
    oe.inclination = inclination;
    oe.long_asc_node = raan;
    oe.arg_periapsis = arg_periapsis;
    oe.semiparam = semi_major_axis * (1.0 - eccentricity * eccentricity);
    oe.mean_anom = mean_anomaly;
    oe.mean_anom_to_nu()
        .expect("init_from_mean_anomaly: Kepler solver failed");

    let (position, velocity) = oe
        .to_cartesian(mu)
        .expect("init_from_mean_anomaly: to_cartesian failed");

    TranslationalState { position, velocity }
}

/// Initialize translational state from LVLH-relative position and velocity.
///
/// Computes the LVLH frame from a reference orbit state, then transforms the
/// given LVLH-relative offsets into the inertial frame.
///
/// # Arguments
/// * `lvlh_pos` - Position relative to reference in LVLH frame (m)
/// * `lvlh_vel` - Velocity relative to reference in LVLH frame (m/s)
/// * `ref_position` - Reference orbit position in inertial frame (m)
/// * `ref_velocity` - Reference orbit velocity in inertial frame (m/s)
pub fn init_from_lvlh(
    lvlh_pos: DVec3,
    lvlh_vel: DVec3,
    ref_position: DVec3,
    ref_velocity: DVec3,
) -> TranslationalState {
    let lvlh = compute_lvlh_frame(ref_position, ref_velocity);

    // t_parent_this transforms from inertial to LVLH.
    // Its transpose transforms from LVLH to inertial.
    let t_lvlh_to_inertial = lvlh.t_parent_this.transpose();

    let position = ref_position + t_lvlh_to_inertial * lvlh_pos;
    let velocity = ref_velocity + t_lvlh_to_inertial * lvlh_vel;

    TranslationalState { position, velocity }
}

/// Initialize translational state from NED (North-East-Down) position and velocity.
///
/// Converts geodetic coordinates to PCPF Cartesian, applies NED-to-PCPF rotation
/// for velocity, rotates from PCPF to ECI, and adds the ω×r frame-rotation term
/// to account for the planet's rotation.
///
/// The `ned_velocity` is a **planet-fixed** velocity (the natural NED meaning):
/// the velocity as measured by an observer rotating with the planet. The returned
/// ECI velocity includes the contribution from the planet's rotation via
/// `v_eci = T_pcpf→eci * v_pcpf + ω_planet × r_eci`.
///
/// This matches JEOD's `DynBodyInitNedState`, which applies the frame-rotation
/// term through `RefFrameState::incr_left()` when composing the rotating PCPF
/// frame with the inertial integration frame.
///
/// # Arguments
/// * `geodetic` - Geodetic position (latitude rad, longitude rad, altitude m)
/// * `ned_velocity` - Planet-fixed velocity in NED frame (m/s)
/// * `r_eq` - Equatorial radius (m)
/// * `r_pol` - Polar radius (m)
/// * `t_eci_pcpf` - Rotation matrix from ECI to PCPF (planet-fixed) frame
/// * `omega_planet` - Planet angular velocity in ECI frame (rad/s)
pub fn init_from_ned(
    geodetic: &GeodeticState,
    ned_velocity: DVec3,
    r_eq: f64,
    r_pol: f64,
    t_eci_pcpf: &DMat3,
    omega_planet: DVec3,
) -> TranslationalState {
    // Convert geodetic to PCPF cartesian
    let pcpf_pos = geodetic_to_cartesian(geodetic, r_eq, r_pol);

    // Compute NED-to-PCPF rotation at this geodetic location.
    // t_pcpf_ned transforms vectors from PCPF to NED, so its transpose
    // transforms from NED to PCPF.
    let t_pcpf_ned = compute_ned_rotation(geodetic.latitude, geodetic.longitude);
    let pcpf_vel = t_pcpf_ned.transpose() * ned_velocity;

    // Convert PCPF to ECI.
    // t_eci_pcpf transforms from ECI to PCPF, so its transpose goes PCPF to ECI.
    let t_pcpf_to_eci = t_eci_pcpf.transpose();
    let position = t_pcpf_to_eci * pcpf_pos;

    // ECI velocity = rotated PCPF velocity + ω_planet × r_eci
    // The cross product accounts for the rotating frame contribution:
    // a point fixed in PCPF still has inertial velocity due to planet rotation.
    let velocity = t_pcpf_to_eci * pcpf_vel + omega_planet.cross(position);

    TranslationalState { position, velocity }
}

/// Compute the PCPF-to-NED transformation matrix at a given geodetic location.
///
/// The NED frame axes expressed in the PCPF frame are:
/// - North = [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]
/// - East  = [-sin(lon), cos(lon), 0]
/// - Down  = [-cos(lat)*cos(lon), -cos(lat)*sin(lon), -sin(lat)]
///
/// These vectors form the rows of the PCPF-to-NED transformation matrix.
///
/// # Arguments
/// * `lat` - Geodetic latitude (rad)
/// * `lon` - Geodetic longitude (rad)
pub fn compute_ned_rotation(lat: f64, lon: f64) -> DMat3 {
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    // Rows of the PCPF-to-NED transformation matrix
    let north = DVec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let down = DVec3::new(-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat);

    mat3_from_rows(north, east, down)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EARTH_MU: f64 = 3.986_004_415e14; // m^3/s^2
    const EARTH_R_EQ: f64 = 6_378_137.0; // WGS84 equatorial radius (m)
    const EARTH_R_POL: f64 = EARTH_R_EQ * (1.0 - 1.0 / 298.257_223_563); // JEOD: r_eq * (1 - flat_coeff)

    // =======================================================================
    // Test 1: Circular orbit from elements
    // =======================================================================

    #[test]
    fn circular_orbit_from_elements() {
        let alt = 400_000.0; // 400 km altitude
        let r = EARTH_R_EQ + alt;
        let a = r; // circular orbit: a = r
        let e = 0.0;
        let inc = 0.0; // equatorial
        let raan = 0.0;
        let argp = 0.0;
        let nu = 0.0; // at periapsis (== anywhere for circular)

        let state = init_from_orbital_elements(a, e, inc, raan, argp, nu, EARTH_MU);

        // Position magnitude should be r
        let r_mag = state.position.length();
        assert!(
            (r_mag - r).abs() < 1e-6,
            "Position magnitude: expected {}, got {}, error = {} m",
            r,
            r_mag,
            (r_mag - r).abs()
        );

        // Velocity magnitude should be sqrt(mu/r) for circular orbit
        let v_circ = (EARTH_MU / r).sqrt();
        let v_mag = state.velocity.length();
        assert!(
            (v_mag - v_circ).abs() < 1e-6,
            "Velocity magnitude: expected {}, got {}, error = {} m/s",
            v_circ,
            v_mag,
            (v_mag - v_circ).abs()
        );
    }

    // =======================================================================
    // Test 2: ISS reference state (Tier 2)
    // =======================================================================

    #[test]
    fn iss_reference_state_from_elements() {
        let root = jeod_test_data::jeod_path();
        assert!(
            root.exists(),
            "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
            root.display()
        );

        let init = jeod_test_data::orbital_init::load_orbital_init(
            &root,
            "ISS",
            "trans_Orbit_inertial_body_set01",
        );
        let expected =
            jeod_test_data::reference_state::load_reference_state(&root, "ISS", "inertial");

        // ISS set01 uses SmaEccIncAscnodeArgperTimeperi.
        // Compute mean anomaly from time_periapsis: M = n * t_peri
        let a = init.semi_major_axis;
        let n = (EARTH_MU / (a * a * a)).sqrt();
        let t_peri = init
            .time_periapsis
            .expect("ISS set01 should have time_periapsis");
        let mean_anomaly = n * t_peri;

        let state = init_from_mean_anomaly(
            init.semi_major_axis,
            init.eccentricity,
            init.inclination,
            init.ascending_node,
            init.arg_periapsis,
            mean_anomaly,
            EARTH_MU,
        );

        let pos_err = (state.position - expected.position).length();
        let vel_err = (state.velocity - expected.velocity).length();

        println!("ISS position error: {:.2} m", pos_err);
        println!("ISS velocity error: {:.6} m/s", vel_err);
        println!(
            "Computed pos: [{:.2}, {:.2}, {:.2}]",
            state.position.x, state.position.y, state.position.z
        );
        println!(
            "Expected pos: [{:.2}, {:.2}, {:.2}]",
            expected.position.x, expected.position.y, expected.position.z
        );

        // Position tolerance: 1 km (conservative for time_periapsis interpretation)
        assert!(
            pos_err < 1000.0,
            "ISS position error {:.2} m exceeds 1 km tolerance",
            pos_err
        );

        // Velocity tolerance: 1 m/s
        assert!(
            vel_err < 1.0,
            "ISS velocity error {:.6} m/s exceeds 1 m/s tolerance",
            vel_err
        );
    }

    // =======================================================================
    // Test 3: LVLH zero offset returns reference state
    // =======================================================================

    #[test]
    fn lvlh_zero_offset_returns_reference() {
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();

        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v, 0.0);

        let state = init_from_lvlh(DVec3::ZERO, DVec3::ZERO, ref_pos, ref_vel);

        let pos_err = (state.position - ref_pos).length();
        let vel_err = (state.velocity - ref_vel).length();

        assert!(
            pos_err < 1e-10,
            "LVLH zero offset position error: {} m",
            pos_err
        );
        assert!(
            vel_err < 1e-10,
            "LVLH zero offset velocity error: {} m/s",
            vel_err
        );
    }

    // =======================================================================
    // Test 4: LVLH round-trip
    // =======================================================================

    #[test]
    fn lvlh_round_trip() {
        // Reference orbit: ISS-like inclined circular orbit
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();
        let inc = 51.6_f64.to_radians();

        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v * inc.cos(), v * inc.sin());

        // Set a known LVLH offset: 100m ahead, 50m below, 20m left
        let lvlh_offset_pos = DVec3::new(100.0, 20.0, 50.0); // x=along-track, y=cross-track, z=nadir
        let lvlh_offset_vel = DVec3::new(0.1, 0.05, -0.02);

        // Initialize from LVLH
        let state = init_from_lvlh(lvlh_offset_pos, lvlh_offset_vel, ref_pos, ref_vel);

        // Now compute the LVLH frame at the reference orbit and transform back
        let lvlh = compute_lvlh_frame(ref_pos, ref_vel);
        let t = lvlh.t_parent_this;

        // Recover LVLH-relative position and velocity
        let delta_pos = state.position - ref_pos;
        let delta_vel = state.velocity - ref_vel;
        let recovered_lvlh_pos = t * delta_pos;
        let recovered_lvlh_vel = t * delta_vel;

        let pos_err = (recovered_lvlh_pos - lvlh_offset_pos).length();
        let vel_err = (recovered_lvlh_vel - lvlh_offset_vel).length();

        assert!(
            pos_err < 1e-10,
            "LVLH round-trip position error: {} m",
            pos_err
        );
        assert!(
            vel_err < 1e-10,
            "LVLH round-trip velocity error: {} m/s",
            vel_err
        );
    }

    // =======================================================================
    // Test 5: NED at equator prime meridian
    // =======================================================================

    #[test]
    fn ned_equator_prime_meridian() {
        let geodetic = GeodeticState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
        };

        // Identity ECI-to-PCPF rotation (no Earth rotation offset)
        let t_eci_pcpf = DMat3::IDENTITY;

        let state = init_from_ned(
            &geodetic,
            DVec3::ZERO, // no velocity
            EARTH_R_EQ,
            EARTH_R_POL,
            &t_eci_pcpf,
            DVec3::ZERO, // no planet rotation
        );

        // At lat=0, lon=0, alt=0, the PCPF position should be [r_eq, 0, 0]
        // With identity ECI-to-PCPF, ECI position is the same.
        assert!(
            (state.position.x - EARTH_R_EQ).abs() < 1e-6,
            "Position X: expected {}, got {}",
            EARTH_R_EQ,
            state.position.x
        );
        assert!(
            state.position.y.abs() < 1e-6,
            "Position Y: expected 0, got {}",
            state.position.y
        );
        assert!(
            state.position.z.abs() < 1e-6,
            "Position Z: expected 0, got {}",
            state.position.z
        );
    }

    // =======================================================================
    // Test 6: Elements round-trip
    // =======================================================================

    #[test]
    fn elements_round_trip() {
        // Non-trivial orbit with distinct elements
        let a = 7_000_000.0; // m
        let e = 0.01;
        let inc = 51.6_f64.to_radians();
        let raan = 30.0_f64.to_radians();
        let argp = 45.0_f64.to_radians();
        let nu = 60.0_f64.to_radians();

        // Initialize from elements
        let state = init_from_orbital_elements(a, e, inc, raan, argp, nu, EARTH_MU);

        // Convert back to orbital elements
        let oe = OrbitalElements::from_cartesian(EARTH_MU, state.position, state.velocity)
            .expect("from_cartesian failed");

        // Compare recovered elements against originals
        assert!(
            (oe.semi_major_axis - a).abs() / a < 1e-10,
            "semi_major_axis: expected {}, got {}, rel_err = {}",
            a,
            oe.semi_major_axis,
            (oe.semi_major_axis - a).abs() / a
        );
        assert!(
            (oe.e_mag - e).abs() < 1e-10,
            "eccentricity: expected {}, got {}, error = {}",
            e,
            oe.e_mag,
            (oe.e_mag - e).abs()
        );
        assert!(
            (oe.inclination - inc).abs() < 1e-10,
            "inclination: expected {}, got {}, error = {}",
            inc,
            oe.inclination,
            (oe.inclination - inc).abs()
        );
        assert!(
            (oe.long_asc_node - raan).abs() < 1e-10,
            "RAAN: expected {}, got {}, error = {}",
            raan,
            oe.long_asc_node,
            (oe.long_asc_node - raan).abs()
        );
        assert!(
            (oe.arg_periapsis - argp).abs() < 1e-10,
            "arg_periapsis: expected {}, got {}, error = {}",
            argp,
            oe.arg_periapsis,
            (oe.arg_periapsis - argp).abs()
        );
        assert!(
            (oe.true_anom - nu).abs() < 1e-10,
            "true_anomaly: expected {}, got {}, error = {}",
            nu,
            oe.true_anom,
            (oe.true_anom - nu).abs()
        );
    }

    // =======================================================================
    // Additional tests
    // =======================================================================

    #[test]
    fn mean_anomaly_agrees_with_true_anomaly_for_circular() {
        // For a circular orbit, mean anomaly == true anomaly
        let a = EARTH_R_EQ + 400_000.0;
        let e = 0.0;
        let inc = 0.0;
        let raan = 0.0;
        let argp = 0.0;
        let nu = 1.0; // radians

        let state_true = init_from_orbital_elements(a, e, inc, raan, argp, nu, EARTH_MU);
        let state_mean = init_from_mean_anomaly(a, e, inc, raan, argp, nu, EARTH_MU);

        let pos_err = (state_true.position - state_mean.position).length();
        let vel_err = (state_true.velocity - state_mean.velocity).length();

        assert!(
            pos_err < 1e-6,
            "Circular orbit: true vs mean anomaly position error = {} m",
            pos_err
        );
        assert!(
            vel_err < 1e-6,
            "Circular orbit: true vs mean anomaly velocity error = {} m/s",
            vel_err
        );
    }

    #[test]
    fn ned_rotation_orthonormal() {
        // Verify NED rotation matrix is orthonormal at several locations
        let test_cases = [
            (0.0, 0.0),             // equator, prime meridian
            (PI / 4.0, PI / 3.0),   // 45N, 60E
            (-PI / 6.0, -PI / 2.0), // 30S, 90W
            (PI / 2.0 - 0.01, 1.0), // near north pole
        ];

        for (lat, lon) in test_cases {
            let t = compute_ned_rotation(lat, lon);

            // T * T^T should be identity
            let product = t * t.transpose();
            let diff = product - DMat3::IDENTITY;
            assert!(
                diff.x_axis.length() < 1e-14,
                "NED rotation not orthonormal at lat={}, lon={}",
                lat,
                lon
            );
            assert!(diff.y_axis.length() < 1e-14);
            assert!(diff.z_axis.length() < 1e-14);

            // Determinant should be +1
            assert!(
                (t.determinant() - 1.0).abs() < 1e-14,
                "NED rotation determinant != 1 at lat={}, lon={}",
                lat,
                lon
            );
        }
    }

    #[test]
    fn ned_north_velocity_at_equator() {
        // At the equator (lat=0, lon=0), a pure North velocity in NED
        // should map to the +Z direction in PCPF (since North points toward
        // the pole at the equator).
        let geodetic = GeodeticState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
        };
        let t_eci_pcpf = DMat3::IDENTITY;

        let ned_vel = DVec3::new(1000.0, 0.0, 0.0); // 1 km/s North
        let state = init_from_ned(
            &geodetic,
            ned_vel,
            EARTH_R_EQ,
            EARTH_R_POL,
            &t_eci_pcpf,
            DVec3::ZERO,
        );

        // North at (lat=0, lon=0) in PCPF is [-sin(0)*cos(0), -sin(0)*sin(0), cos(0)] = [0, 0, 1]
        // NED-to-PCPF = T_pcpf_ned^T, where row0 of T_pcpf_ned is North = [0,0,1].
        // So column 0 of T^T = [0,0,1]. Thus NED [1000,0,0] -> PCPF [0,0,1000].
        assert!(
            state.velocity.x.abs() < 1e-6,
            "Vel X: expected 0, got {}",
            state.velocity.x
        );
        assert!(
            state.velocity.y.abs() < 1e-6,
            "Vel Y: expected 0, got {}",
            state.velocity.y
        );
        assert!(
            (state.velocity.z - 1000.0).abs() < 1e-6,
            "Vel Z: expected 1000, got {}",
            state.velocity.z
        );
    }

    #[test]
    fn ned_omega_cross_r_contribution() {
        // Verify that planet rotation adds ω×r to ECI velocity.
        // At equator (lat=0, lon=0), position is [r_eq, 0, 0] in PCPF.
        // With identity T_eci_pcpf, ECI position is the same.
        // ω = [0, 0, ω_earth], so ω × r = [0, 0, ω] × [r, 0, 0] = [0, ω*r, 0].
        let geodetic = GeodeticState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
        };
        let t_eci_pcpf = DMat3::IDENTITY;
        let omega_earth = 7.292_115_0e-5; // rad/s
        let omega = DVec3::new(0.0, 0.0, omega_earth);

        // Zero NED velocity: the only ECI velocity comes from planet rotation.
        let state = init_from_ned(
            &geodetic,
            DVec3::ZERO,
            EARTH_R_EQ,
            EARTH_R_POL,
            &t_eci_pcpf,
            omega,
        );

        // Expected: ω × r = [0, ω*r_eq, 0] ≈ [0, 465.1, 0] m/s
        let expected_vy = omega_earth * EARTH_R_EQ;
        assert!(
            state.velocity.x.abs() < 1e-6,
            "Vel X: expected 0, got {}",
            state.velocity.x
        );
        assert!(
            (state.velocity.y - expected_vy).abs() < 1e-3,
            "Vel Y: expected {:.1}, got {:.1}",
            expected_vy,
            state.velocity.y
        );
        assert!(
            state.velocity.z.abs() < 1e-6,
            "Vel Z: expected 0, got {}",
            state.velocity.z
        );
    }

    #[test]
    fn lvlh_with_inclined_orbit() {
        // Test LVLH with a non-trivial inclined orbit and non-zero offset
        let r = EARTH_R_EQ + 400_000.0;
        let v = (EARTH_MU / r).sqrt();
        let inc = 51.6_f64.to_radians();

        // Position along X-axis, velocity in the Y-Z plane (inclined orbit)
        let ref_pos = DVec3::new(r, 0.0, 0.0);
        let ref_vel = DVec3::new(0.0, v * inc.cos(), v * inc.sin());

        // Zero offset should still give reference state
        let state = init_from_lvlh(DVec3::ZERO, DVec3::ZERO, ref_pos, ref_vel);
        assert!(
            (state.position - ref_pos).length() < 1e-10,
            "Inclined LVLH zero offset position error"
        );
        assert!(
            (state.velocity - ref_vel).length() < 1e-10,
            "Inclined LVLH zero offset velocity error"
        );

        // 1 km nadir offset (Z in LVLH = toward planet center)
        let lvlh_pos = DVec3::new(0.0, 0.0, 1000.0);
        let state_nadir = init_from_lvlh(lvlh_pos, DVec3::ZERO, ref_pos, ref_vel);

        // The offset in inertial should reduce position magnitude (closer to Earth)
        let r_offset = state_nadir.position.length();
        assert!(
            r_offset < r,
            "1 km nadir offset should reduce position magnitude: {} vs {}",
            r_offset,
            r
        );
        // And the offset magnitude should be approximately 1 km
        let delta = (state_nadir.position - ref_pos).length();
        assert!(
            (delta - 1000.0).abs() < 1e-6,
            "Offset magnitude: expected 1000 m, got {} m",
            delta
        );
    }
}
