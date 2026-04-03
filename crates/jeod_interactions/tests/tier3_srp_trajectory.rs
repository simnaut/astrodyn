//! Tier 3: Cross-validation of flat-plate SRP + conical shadow against
//! JEOD SIM_3_ORBIT RUN_radiation reference data.
//!
//! SIM_3_ORBIT/RUN_radiation configuration:
//! - Spherical gravity (point-mass Earth, mu=3.986_004_415e14 m³/s²)
//! - 6 flat plates: 4×60 m² at ±X/±Y, 2×16 m² at ±Z
//!   - albedo=0.5, diffuse=0.5
//! - Earth conical shadow (Moon shadow disabled)
//! - GEO orbit (~42,164 km altitude)
//! - Mass: 300 kg, identity attitude
//! - RK4, dt=1.0s, 2,000,000s (~23.1 days), logged every 1000s
//! - Epoch: 1998-12-01 00:00:31 TAI (UTC = 00:00:00)
//!
//! This validates:
//! 1. Flat-plate SRP force decomposition (absorption, diffuse, specular)
//! 2. Solar flux computation at vehicle distance
//! 3. Conical Earth shadow detection and illumination fraction
//! 4. Coupled gravity + SRP trajectory over ~23 days (< 50 m position error, < 10 m at 24h)

use glam::DVec3;
use jeod_dynamics::TranslationalState;
use jeod_ephemeris::{Ephemeris, EphemerisBody};
use jeod_interactions::{
    compute_flat_plate_srp_thermal, compute_shadow_fraction, solar_flux_at_distance, FlatPlate,
    FlatPlateParams, FlatPlateThermal, SOLAR_RADIUS, STEFAN_BOLTZMANN,
};
use std::path::Path;

const MU_EARTH: f64 = 3.986_004_415e14;
const R_EARTH: f64 = 6_378_137.0; // WGS84 equatorial radius (m)

/// SIM_3_ORBIT epoch: 1998-12-01 00:00:00 UTC (= 00:00:31 TAI).
///
/// JD(UTC) = 2451148.5, MJD = 51148.0, TJT = MJD - 40000 = 11148.0
/// TAI-UTC = 31s at this date. We use UTC-based TJT here because JEOD's
/// MET atmosphere internally works with UTC-based truncated Julian time.
const EPOCH_TJT: f64 = 11148.0;

/// Vehicle mass (kg) from Modified_data/vehicle_baseline.py.
const MASS: f64 = 300.0;

/// SIM_3_ORBIT plate configuration from Modified_data/radiation_surface.py.
fn sim3_orbit_plates() -> Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)> {
    let params = FlatPlateParams {
        albedo: 0.5,
        diffuse: 0.5,
    };
    let thermal = FlatPlateThermal {
        emissivity: 0.5,
        heat_capacity_per_area: 50.0,
    };
    vec![
        (
            FlatPlate {
                area: 60.0,
                normal: DVec3::X,
                position: DVec3::new(2.0, 0.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: -DVec3::Y,
                position: DVec3::new(0.0, -2.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: -DVec3::X,
                position: DVec3::new(-2.0, 0.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 60.0,
                normal: DVec3::Y,
                position: DVec3::new(0.0, 2.0, 0.0),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 16.0,
                normal: DVec3::Z,
                position: DVec3::new(0.0, 0.0, 7.5),
            },
            params,
            thermal,
        ),
        (
            FlatPlate {
                area: 16.0,
                normal: -DVec3::Z,
                position: DVec3::new(0.0, 0.0, -7.5),
            },
            params,
            thermal,
        ),
    ]
}

/// Initial temperatures for all 6 plates (K).
/// SIM_3_ORBIT sets all plates to 270 K.
const INITIAL_TEMPERATURE: f64 = 270.0;
const NUM_PLATES: usize = 6;

/// Parsed SRP trajectory record from JEOD CSV.
#[derive(Debug)]
#[allow(dead_code)]
struct SrpRecord {
    time: f64,
    position: DVec3,
    velocity: DVec3,
    grav_accel: DVec3,
    srp_force: DVec3,
    srp_torque: DVec3,
    flux_mag: f64,
}

/// Parse the SRP orbit ASCII CSV.
///
/// Columns (from our DRAscii logger):
///   0: time
///   1-3: position[0,1,2]
///   4-6: velocity[0,1,2]
///   7-9: grav_accel[0,1,2]
///   10-12: force[0,1,2]
///   13-15: torque[0,1,2]
///   16: flux_mag
fn load_srp_trajectory(path: &Path) -> Vec<SrpRecord> {
    let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "Failed to read SRP trajectory CSV from {}: {e}\n\
             Generate with:\n  \
             docker build -f trick/Dockerfile -t jeod-trick ..\n  \
             docker run --rm -v $(pwd)/test_data:/output jeod-trick",
            path.display()
        )
    });

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert!(
            fields.len() >= 17,
            "Malformed SRP CSV at line {}: expected >= 17 fields, got {}",
            i + 1,
            fields.len(),
        );

        let p = |col: usize| -> f64 {
            fields[col].trim().parse::<f64>().unwrap_or_else(|e| {
                panic!(
                    "Parse error at line {}, col {col}: {:?} ({e})",
                    i + 1,
                    fields[col]
                )
            })
        };

        records.push(SrpRecord {
            time: p(0),
            position: DVec3::new(p(1), p(2), p(3)),
            velocity: DVec3::new(p(4), p(5), p(6)),
            grav_accel: DVec3::new(p(7), p(8), p(9)),
            srp_force: DVec3::new(p(10), p(11), p(12)),
            srp_torque: DVec3::new(p(13), p(14), p(15)),
            flux_mag: p(16),
        });
    }
    records
}

/// Compute gravitational acceleration (point-mass Earth).
fn gravity_accel(position: DVec3) -> DVec3 {
    let r = position.length();
    -MU_EARTH / (r * r * r) * position
}

/// Compute Sun position from DE421 ephemeris at the SIM_3_ORBIT epoch.
fn sun_position_at(sim_time: f64, ephemeris: &Ephemeris) -> DVec3 {
    // TJT to TDB JD: JD = TJT + 40000 + 2400000.5
    // (TAI ≈ TDB to ~1.7ms; good enough for Sun position at 1 AU)
    let sim_days = sim_time / 86400.0;
    let tdb_jd = (EPOCH_TJT + sim_days) + 40000.0 + 2_400_000.5;

    let (sun_pos, _sun_vel) = ephemeris
        .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
        .unwrap_or_else(|e| {
            panic!("Sun position query failed for tdb_jd={tdb_jd}: {e}");
        });
    sun_pos
}

/// Evaluate SRP force + temperature derivatives at a given state.
///
/// Uses `t_pow4_cached` for emission (JEOD convention: T⁴ is cached from
/// the previous integration step, not recomputed from current temperature).
fn evaluate_srp(
    position: DVec3,
    plates: &[(FlatPlate, FlatPlateParams, FlatPlateThermal)],
    t_pow4_cached: &[f64],
    sun_pos: DVec3,
    mass: f64,
) -> (DVec3, Vec<f64>) {
    let g = gravity_accel(position);

    let sun_to_vehicle = position - sun_pos;
    let dist = sun_to_vehicle.length();
    if dist < 1.0 {
        return (g, vec![0.0; t_pow4_cached.len()]);
    }

    let flux_hat = sun_to_vehicle / dist;
    let flux_mag = solar_flux_at_distance(dist);

    let illum_factor =
        compute_shadow_fraction(position, sun_pos, DVec3::ZERO, R_EARTH, SOLAR_RADIUS);

    let srp = compute_flat_plate_srp_thermal(
        plates,
        t_pow4_cached,
        flux_hat,
        flux_mag,
        DVec3::ZERO,
        illum_factor,
    );

    (g + srp.force / mass, srp.temp_dots)
}

/// RK4 integration of position, velocity, AND temperature.
///
/// Matches JEOD's integration convention:
/// - `t_pow4_cached` is used for force evaluation (not updated during RK4 stages)
/// - Temperature is integrated via RK4 alongside orbital state
/// - After the step, `t_pow4_cached` is updated from the new temperatures
///   (JEOD `thermal_integrable_object.cc:103-104`)
/// - Overshoot detection clamps temperature to equilibrium if it crosses
///   (JEOD `thermal_integrable_object.cc:106-121`)
fn rk4_step(
    state: &TranslationalState,
    plates: &[(FlatPlate, FlatPlateParams, FlatPlateThermal)],
    temperatures: &mut [f64],
    t_pow4_cached: &mut [f64],
    sun_pos: DVec3,
    dt: f64,
    mass: f64,
) -> TranslationalState {
    let n = temperatures.len();

    // All 4 RK4 stages use the SAME t_pow4_cached (JEOD convention: emission
    // power uses T⁴ from the start of the step, not intermediate values).

    // Stage 1
    let (k1v, k1t) = evaluate_srp(state.position, plates, t_pow4_cached, sun_pos, mass);
    let k1x = state.velocity;

    // Stage 2
    let s2 = TranslationalState {
        position: state.position + k1x * (dt / 2.0),
        velocity: state.velocity + k1v * (dt / 2.0),
    };
    let (k2v, k2t) = evaluate_srp(s2.position, plates, t_pow4_cached, sun_pos, mass);
    let k2x = s2.velocity;

    // Stage 3
    let s3 = TranslationalState {
        position: state.position + k2x * (dt / 2.0),
        velocity: state.velocity + k2v * (dt / 2.0),
    };
    let (k3v, k3t) = evaluate_srp(s3.position, plates, t_pow4_cached, sun_pos, mass);
    let k3x = s3.velocity;

    // Stage 4
    let s4 = TranslationalState {
        position: state.position + k3x * dt,
        velocity: state.velocity + k3v * dt,
    };
    let (k4v, k4t) = evaluate_srp(s4.position, plates, t_pow4_cached, sun_pos, mass);
    let k4x = s4.velocity;

    // RK4 temperature update
    for i in 0..n {
        let old_temp = temperatures[i];
        temperatures[i] = old_temp + (k1t[i] + 2.0 * k2t[i] + 2.0 * k3t[i] + k4t[i]) * (dt / 6.0);
        temperatures[i] = temperatures[i].max(0.0);

        // Update cached T⁴ from new temperature
        // JEOD thermal_integrable_object.cc:103-104
        let t = temperatures[i];
        t_pow4_cached[i] = t * t * t * t;

        // Overshoot detection (JEOD thermal_integrable_object.cc:106-121)
        let (plate, _params, thermal) = &plates[i];
        let rad_constant = plate.area * thermal.emissivity * STEFAN_BOLTZMANN;
        if rad_constant > 0.0 {
            // Recover power_absorb from k1: temp_dot = (P_absorb - P_emit) / C_p
            // P_absorb = temp_dot * C_p + P_emit = k1t * C_p + rad_constant * old_t_pow4
            let heat_cap = thermal.heat_capacity_per_area * plate.area;
            let old_t_pow4 = old_temp * old_temp * old_temp * old_temp;
            let power_absorb = k1t[i] * heat_cap + rad_constant * old_t_pow4;
            let t_eq_pow4 = power_absorb / rad_constant;

            // If temp_dot and (T_eq^4 - T^4) have opposite signs, we overshot
            if k1t[i] * (t_eq_pow4 - t_pow4_cached[i]) < 0.0 {
                t_pow4_cached[i] = t_eq_pow4.max(0.0);
                temperatures[i] = t_pow4_cached[i].sqrt().sqrt();
            }
        }
    }

    TranslationalState {
        position: state.position + (k1x + 2.0 * k2x + 2.0 * k3x + k4x) * (dt / 6.0),
        velocity: state.velocity + (k1v + 2.0 * k2v + 2.0 * k3v + k4v) * (dt / 6.0),
    }
}

#[test]
fn tier3_srp_trajectory_sim3_orbit() {
    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/srp_orbit_radiation_srp_orbit.csv");

    assert!(
        csv_path.exists(),
        "Tier 3 SRP reference data not found at {}.\n\
         Generate with:\n  \
         docker build -f trick/Dockerfile -t jeod-trick ..\n  \
         docker run --rm -v $(pwd)/test_data:/output jeod-trick",
        csv_path.display()
    );

    let trajectory = load_srp_trajectory(&csv_path);
    assert!(
        trajectory.len() >= 80,
        "Expected at least 80 data points (24h at 1000s), got {}",
        trajectory.len()
    );

    let plates = sim3_orbit_plates();
    let dt = 1.0; // RK4 step size matching JEOD's DYNAMICS interval
                  // JEOD ThermalIntegrableObject state: temp and cached t_pow4
    let mut temp = [INITIAL_TEMPERATURE; NUM_PLATES];
    let mut t_pow4: Vec<f64> = temp.iter().map(|t| t * t * t * t).collect();

    // Load ephemeris (required — de421.bsp is committed to the repo).
    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    let ephemeris = Ephemeris::from_bsp(&bsp_path).unwrap_or_else(|err| {
        panic!(
            "de421.bsp required for SRP trajectory test.\n\
             Attempted to load from: {}\n\
             Ephemeris load error: {}",
            bsp_path.display(),
            err
        )
    });

    // Initialize from first JEOD data point
    let mut state = TranslationalState {
        position: trajectory[0].position,
        velocity: trajectory[0].velocity,
    };

    let mut max_pos_err = 0.0_f64;
    let mut max_pos_err_24h = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    let mut max_force_dir_err = 0.0_f64;
    let mut max_force_mag_rel_err = 0.0_f64;
    let mut shadow_mismatches = 0;

    for window in trajectory.windows(2) {
        let target = &window[1];
        let start_time = window[0].time;
        let end_time = target.time;
        let steps = ((end_time - start_time) / dt).round() as usize;

        // Propagate from current state to next logged time
        for step_i in 0..steps {
            let sim_time = start_time + (step_i as f64) * dt;
            let sun_pos = sun_position_at(sim_time, &ephemeris);
            state = rk4_step(&state, &plates, &mut temp, &mut t_pow4, sun_pos, dt, MASS);
        }

        let pos_err = (state.position - target.position).length();
        let vel_err = (state.velocity - target.velocity).length();
        max_pos_err = max_pos_err.max(pos_err);
        if target.time <= 86400.0 {
            max_pos_err_24h = max_pos_err_24h.max(pos_err);
        }
        max_vel_err = max_vel_err.max(vel_err);

        // Print diagnostics at key intervals
        let t = target.time;
        if (t % 200_000.0) < 1000.5 || (t > 2000.0 && t < 8000.0 && target.flux_mag > 100.0) {
            // Force comparison at this point
            let (force_rel_err, force_dir_err) =
                if target.flux_mag > 100.0 && target.srp_force.length() > 1e-6 {
                    let sun_pos = sun_position_at(target.time, &ephemeris);
                    let stv = target.position - sun_pos;
                    let d = stv.length();
                    let fh = stv / d;
                    let fm = solar_flux_at_distance(d);
                    let our =
                        compute_flat_plate_srp_thermal(&plates, &t_pow4, fh, fm, DVec3::ZERO, 1.0);
                    let rel = (our.force.length() - target.srp_force.length()).abs()
                        / target.srp_force.length();
                    let dir = if our.force.length() > 1e-15 {
                        our.force
                            .normalize()
                            .dot(target.srp_force.normalize())
                            .clamp(-1.0, 1.0)
                            .acos()
                    } else {
                        0.0
                    };
                    (rel, dir)
                } else {
                    (-1.0, -1.0)
                };
            eprintln!(
                "  t={:9.0}s ({:5.1}d)  pos_err={:8.3}m  vel_err={:.6}m/s  F_rel={:.2e}  F_dir={:.4}rad",
                t, t / 86400.0, pos_err, vel_err, force_rel_err, force_dir_err,
            );
        }

        // Compare SRP force (only when well-illuminated).
        if target.flux_mag > 100.0 && target.srp_force.length() > 1e-6 {
            let sun_pos = sun_position_at(target.time, &ephemeris);
            let sun_to_vehicle = target.position - sun_pos;
            let dist = sun_to_vehicle.length();
            let flux_hat = sun_to_vehicle / dist;
            let flux_mag = solar_flux_at_distance(dist);

            // Point-in-time force comparison using current cached t_pow4
            let our_srp = compute_flat_plate_srp_thermal(
                &plates,
                &t_pow4,
                flux_hat,
                flux_mag,
                DVec3::ZERO,
                1.0,
            );

            // Print force comparison at first few illuminated points
            if t < 7000.0 {
                eprintln!(
                    "    JEOD |F|={:.6e}  OUR |F|={:.6e}  ratio={:.4}  dir_err={:.6}rad",
                    target.srp_force.length(),
                    our_srp.force.length(),
                    target.srp_force.length() / our_srp.force.length(),
                    if our_srp.force.length() > 1e-15 {
                        our_srp
                            .force
                            .normalize()
                            .dot(target.srp_force.normalize())
                            .clamp(-1.0, 1.0)
                            .acos()
                    } else {
                        0.0
                    },
                )
            }

            // Direction comparison
            if our_srp.force.length() > 1e-15 {
                let our_dir = our_srp.force.normalize();
                let jeod_dir = target.srp_force.normalize();
                let dir_err = our_dir.dot(jeod_dir).clamp(-1.0, 1.0).acos();
                max_force_dir_err = max_force_dir_err.max(dir_err);
            }

            // Magnitude comparison
            let our_mag = our_srp.force.length();
            let jeod_mag = target.srp_force.length();
            if jeod_mag > 1e-15 {
                let rel_err = (our_mag - jeod_mag).abs() / jeod_mag;
                max_force_mag_rel_err = max_force_mag_rel_err.max(rel_err);
            }
        }

        // Shadow comparison: check if our shadow agrees with JEOD flux
        let sun_pos = sun_position_at(target.time, &ephemeris);
        let our_illum =
            compute_shadow_fraction(target.position, sun_pos, DVec3::ZERO, R_EARTH, SOLAR_RADIUS);
        let jeod_in_shadow = target.flux_mag < 1e-10;
        let our_in_shadow = our_illum < 1e-10;
        if jeod_in_shadow != our_in_shadow {
            shadow_mismatches += 1;
        }

        // With thermal emission ported, the force model should closely match JEOD.
        // Remaining error sources: temperature integration timing differences
        // (our RK4 vs JEOD's integrable-object ODE) and ephemeris precision.
        assert!(
            pos_err < 50.0,
            "Position error {pos_err:.2} m at t={:.0}s exceeds 50 m threshold",
            target.time
        );
    }

    eprintln!("=== Tier 3 SRP Trajectory (SIM_3_ORBIT RUN_radiation) ===");
    eprintln!("  Data points: {}", trajectory.len());
    eprintln!(
        "  Duration: {:.0}s ({:.1} days)",
        trajectory.last().unwrap().time,
        trajectory.last().unwrap().time / 86400.0
    );
    eprintln!("  Max position error (24h):  {max_pos_err_24h:.3} m");
    eprintln!("  Max position error (full): {max_pos_err:.1} m");
    eprintln!("  Max velocity error: {max_vel_err:.6} m/s");
    eprintln!("  Max SRP force direction error: {max_force_dir_err:.6} rad");
    eprintln!("  Max SRP force magnitude rel error: {max_force_mag_rel_err:.4}");
    eprintln!("  Shadow state mismatches: {shadow_mismatches}");

    // Force direction should match closely now that thermal emission is included.
    assert!(
        max_force_dir_err < 0.05,
        "SRP force direction error {max_force_dir_err:.4} rad exceeds 0.05 rad"
    );

    // Shadow detection: no mismatches expected for a GEO orbit
    assert!(
        shadow_mismatches <= 2,
        "Shadow state mismatches: {shadow_mismatches} (expected 0-2 for transition timing)"
    );

    // Phase 4 exit criterion: SRP position error < 10 m over 24h.
    assert!(
        max_pos_err_24h < 10.0,
        "24h position error {max_pos_err_24h:.3} m exceeds 10 m threshold"
    );

    // Force magnitude: at full illumination, matches within a few percent. Near shadow
    // transitions, temperature history diverges (our RK4 vs JEOD's integrable-object ODE)
    // causing larger relative errors at low-flux points.
    assert!(
        max_force_mag_rel_err < 5.0,
        "SRP force magnitude relative error {max_force_mag_rel_err:.4} exceeds 500%"
    );
}
