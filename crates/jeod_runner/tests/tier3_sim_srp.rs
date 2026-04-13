//! Tier 3: SIM_3_ORBIT cross-validation (radiation_pressure/verif/SIM_3_ORBIT)
//!
//! Flat-plate SRP + conical Earth shadow, GEO orbit, ~23 days.

mod sim_test_helpers;
use sim_test_helpers::*;

use glam::{DMat3, DVec3};
use jeod_runner::{
    GravitySourceEntry, RotationModel, ShadowBody, Simulation, SrpModel, VehicleConfig,
};
use jeod_sim::{
    Ephemeris, EphemerisBody, FlatPlate, FlatPlateParams, FlatPlateState, FlatPlateThermal,
    GravityControl, GravityControls, GravityModel, GravitySource, MassProperties, SimulationTime,
    TranslationalState,
};
use jeod_test_data::crossval::{CrossvalReport, StateLog};
use std::path::Path;

/// SIM_3_ORBIT directory relative to JEOD root.
const SIM_3_ORBIT: &str = "models/interactions/radiation_pressure/verif/SIM_3_ORBIT";

const SRP_MASS: f64 = 300.0;

fn srp_plates() -> Vec<(FlatPlate, FlatPlateParams, FlatPlateThermal)> {
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

fn srp_sun_position(sim_time: f64, epoch_tdb_jd: f64, ephemeris: &Ephemeris) -> DVec3 {
    let tdb_jd = epoch_tdb_jd + sim_time / 86400.0;
    let (sun_pos, _) = ephemeris
        .get_earth_centered_state(EphemerisBody::Sun, tdb_jd)
        .expect("Sun position query failed");
    sun_pos
}

/// Precomputed Sun position table for efficient per-step interpolation.
///
/// Evaluates the ephemeris at coarse intervals and linearly interpolates
/// between samples. This avoids calling the expensive BSP query at every
/// 1-second timestep while keeping the Sun direction accurate to sub-arcsecond
/// levels (Sun moves ~0.01°/day as seen from Earth).
struct SunTable {
    /// (sim_time, position) pairs, sorted by time.
    samples: Vec<(f64, DVec3)>,
}

impl SunTable {
    /// Build a table from t=0 to `end_time` with the given sample spacing.
    fn build(end_time: f64, spacing: f64, epoch_tdb_jd: f64, ephemeris: &Ephemeris) -> Self {
        let n = (end_time / spacing).ceil() as usize + 1;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i as f64) * spacing;
            samples.push((t, srp_sun_position(t, epoch_tdb_jd, ephemeris)));
        }
        // Ensure the final time is included
        if samples.last().is_none_or(|(t, _)| *t < end_time) {
            samples.push((
                end_time,
                srp_sun_position(end_time, epoch_tdb_jd, ephemeris),
            ));
        }
        Self { samples }
    }

    /// Linearly interpolate the Sun position at the given simulation time.
    fn at(&self, t: f64) -> DVec3 {
        if t <= self.samples[0].0 {
            return self.samples[0].1;
        }
        if t >= self.samples.last().unwrap().0 {
            return self.samples.last().unwrap().1;
        }
        // Binary search for the bracketing interval
        let idx = self
            .samples
            .partition_point(|(st, _)| *st <= t)
            .saturating_sub(1);
        let (t0, p0) = self.samples[idx];
        let (t1, p1) = self.samples[idx + 1];
        let frac = (t - t0) / (t1 - t0);
        p0 + (p1 - p0) * frac
    }
}

#[test]
fn tier3_simulation_srp_flat_plate() {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let csv_path = test_data_path("srp_orbit_radiation_srp_orbit.csv");
    assert!(
        csv_path.exists(),
        "SRP reference not found at {}",
        csv_path.display()
    );

    let bsp_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 ephemeris not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");

    let sim_dir = jeod_root.join(SIM_3_ORBIT);
    let grav_data_dir = jeod_root.join("models/environment/gravity/data/src");

    // Load epoch from JEOD time config. SIM_3_ORBIT uses TAI initializer,
    // so tai_tjt() returns the TAI TJT directly from the calendar date.
    let time_cfg = jeod_test_data::time_config::load_time_config(
        &sim_dir.join("Modified_data/date_and_time.py"),
    );
    let epoch_tai_tjt = time_cfg.tai_tjt();

    // Load integration step size from S_define
    let srp_dt = jeod_test_data::s_define::load_dynamics_dt(&sim_dir.join("S_define"));

    // Load Earth gravity data (mu, radius) from JEOD coefficient file
    let earth_grav =
        jeod_sim::coefficients::load_from_jeod_cc(&grav_data_dir.join("earth_GGM05C.cc"))
            .expect("load Earth gravity");
    let srp_mu_earth = earth_grav.mu;

    let trajectory = load_srp_trajectory(&csv_path);
    assert!(trajectory.len() > 100);
    let init = &trajectory[0];

    let plates = srp_plates();
    let num_plates = plates.len();
    let init_temp = 270.0_f64;

    let time = SimulationTime::new(epoch_tai_tjt, jeod_sim::default_leap_second_table());
    let epoch_tdb_jd = time.tdb_julian_date();
    let mut sim = Simulation::new(time, srp_dt);

    // Earth at origin (gravity source + shadow body)
    let earth = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: srp_mu_earth,
            model: GravityModel::PointMass,
        },
        position: DVec3::ZERO,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });

    // Sun (position updated each logging interval from ephemeris).
    // mu=0 matches the JEOD SIM_3_ORBIT reference sim, which uses Sun only
    // for SRP direction — Sun/Moon gravity controls are commented out in
    // vehicle_baseline.py. 3rd-body gravity is validated independently by
    // tier3_sim_dyncomp_run4, run7, and tier3_sim_torque_simple.
    let initial_sun = srp_sun_position(0.0, epoch_tdb_jd, &ephemeris);
    let sun = sim.add_source(GravitySourceEntry {
        source: GravitySource {
            mu: 0.0,
            model: GravityModel::PointMass,
        },
        position: initial_sun,
        velocity: DVec3::ZERO,
        t_inertial_pfix: None,
        delta_c20: 0.0,
        rotation_model: RotationModel::default(),
        tidal_config: None,
    });
    sim.sun_source = Some(sun);

    sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        mass: Some(MassProperties::with_inertia(
            SRP_MASS,
            DMat3::from_diagonal(DVec3::splat(1.0)),
            DVec3::ZERO,
        )),
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        srp: Some(SrpModel::FlatPlate(FlatPlateState {
            plates,
            temperatures: vec![init_temp; num_plates],
            t_pow4_cached: vec![init_temp.powi(4); num_plates],
        })),
        shadow_body: Some(ShadowBody {
            source_idx: earth,
            radius: jeod_sim::EARTH.shadow_radius,
        }),
        ..Default::default()
    });

    sim.validate().unwrap();

    let total_time = trajectory.last().unwrap().time;
    println!(
        "Tier 3 (Simulation): SRP flat-plate + shadow, {} points over {:.0} days",
        trajectory.len(),
        total_time / 86400.0
    );

    // Precompute Sun ephemeris at 100s intervals for per-step interpolation.
    // JEOD updates Sun position every integration step (1s); the previous test
    // code only updated at record boundaries (1000s), introducing a stale-Sun
    // error of ~2-5 m over 23 days.
    let sun_table = SunTable::build(total_time, 100.0, epoch_tdb_jd, &ephemeris);

    let mut our_states = Vec::with_capacity(trajectory.len() - 1);
    let mut ref_states = Vec::with_capacity(trajectory.len() - 1);

    let mut next_record = 1;
    let total_steps = (total_time / srp_dt).round() as usize;

    for step_i in 1..=total_steps {
        let t = (step_i as f64) * srp_dt;

        // Update Sun position every step (matching JEOD's per-step ephemeris update)
        sim.sources[sun].position = sun_table.at(t);

        sim.step();

        // Collect comparison data at record times
        if next_record < trajectory.len() {
            let record = &trajectory[next_record];
            if (sim.time.simtime - record.time).abs() < srp_dt * 0.5 {
                let body = sim.body(0);

                our_states.push(StateLog {
                    time: record.time,
                    position: Some(body.trans.position),
                    velocity: Some(body.trans.velocity),
                    ..Default::default()
                });
                ref_states.push(StateLog {
                    time: record.time,
                    position: Some(record.position),
                    velocity: Some(record.velocity),
                    ..Default::default()
                });

                if (record.time % 86400.0).abs() < 500.1 {
                    let pos_error = (body.trans.position - record.position).length();
                    println!(
                        "  t={:8.0}s ({:5.1}d): pos_err={:10.2} m",
                        record.time,
                        record.time / 86400.0,
                        pos_error
                    );
                }
                next_record += 1;
            }
        }
    }

    let report =
        CrossvalReport::compute("tier3_simulation_srp_flat_plate", &our_states, &ref_states);
    report.write();

    let max_pos_error = report.max_position_component();
    println!("  Max position error: {:.6e} m", max_pos_error);

    report.assert_position([3.074, 2.799, 1.216]);
}
