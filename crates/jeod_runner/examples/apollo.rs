//! Apollo trans-lunar injection: multi-body gravity, staging, impulsive maneuver.
//!
//! Demonstrates:
//! - Earth + Moon multi-body point-mass gravity
//! - Mass tree with two-body vehicle (CSM + S-IVB stage)
//! - Impulsive TLI delta-V maneuver
//! - Stage separation via mass tree detach
//! - 3-day coast to lunar encounter
//!
//! Uses DE421 ephemeris for Moon position. No SRP or drag.
//!
//! Requires `test_data/de421.bsp` for ephemeris data.
//!
//! ```bash
//! cargo run -p jeod_runner --example apollo
//! ```

use std::path::Path;

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, Simulation, VehicleConfig};
use jeod_sim::{
    GravityControl, GravityControls, GravityModel, GravitySource, MassProperties, SimulationTime,
    TranslationalState,
};

// Gravitational parameters
const MU_EARTH: f64 = 3.986_004_418e14; // m^3/s^2
const MU_MOON: f64 = 4.902_800_066e12; // m^3/s^2
const MU_SUN: f64 = 1.327_124_400_41e20; // m^3/s^2

const R_EARTH: f64 = 6_371_000.0; // mean radius, m

// Apollo-like parking orbit: 185 km circular, 28.5° inclination (KSC latitude)
const PARKING_ALT: f64 = 185_000.0; // m
const INCLINATION_DEG: f64 = 28.5;

// Vehicle masses (approximate Apollo values)
const MASS_CSM: f64 = 28_800.0; // Command/Service Module, kg
const MASS_SIVB: f64 = 13_300.0; // S-IVB dry mass, kg (after TLI burn)

// TLI delta-V: ~3.1 km/s (prograde) applied at the right point in parking orbit
const TLI_DELTA_V: f64 = 3_130.0; // m/s

// Timing
const PARKING_ORBITS: f64 = 2.5; // coast in parking orbit before TLI
const DT: f64 = 60.0; // 1-minute timesteps
const TOTAL_DURATION: f64 = 3.0 * 86_400.0; // 3 days

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data");
    let bsp_path = data_dir.join("de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );

    let ephemeris = jeod_sim::Ephemeris::from_bsp(&bsp_path)?;
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let epoch_tdb_jd = time.tdb_julian_date();

    // Get Moon and Sun positions at epoch
    let (moon_pos, _) = ephemeris.get_state(
        jeod_sim::EphemerisBody::Moon,
        jeod_sim::EphemerisBody::Earth,
        epoch_tdb_jd,
    )?;
    let (sun_pos, _) = ephemeris.get_state(
        jeod_sim::EphemerisBody::Sun,
        jeod_sim::EphemerisBody::Earth,
        epoch_tdb_jd,
    )?;

    let mut sim = Simulation::new(time, DT);

    // Earth at origin
    let mut earth_entry = GravitySourceEntry::new(
        GravitySource {
            mu: MU_EARTH,
            model: GravityModel::PointMass,
        },
        DVec3::ZERO,
        None,
    );
    earth_entry.central = true;
    let earth = sim.add_source("Earth", earth_entry);

    // Moon with per-step ephemeris updates
    let moon = sim.add_source(
        "Moon",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_MOON,
                model: GravityModel::PointMass,
            },
            moon_pos,
            None,
        ),
    );
    sim.set_source_ephemeris(
        moon,
        jeod_sim::EphemerisBody::Moon,
        jeod_sim::EphemerisBody::Earth,
    );

    // Sun as 3rd-body (weak perturbation for TLI, but included for completeness)
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry::new(
            GravitySource {
                mu: MU_SUN,
                model: GravityModel::PointMass,
            },
            sun_pos,
            None,
        ),
    );
    sim.set_source_ephemeris(
        sun,
        jeod_sim::EphemerisBody::Sun,
        jeod_sim::EphemerisBody::Earth,
    );

    sim.ephemeris = Some(ephemeris);

    // Compute parking orbit initial state
    let r_park = R_EARTH + PARKING_ALT;
    let v_circ = (MU_EARTH / r_park).sqrt();
    let inc = INCLINATION_DEG.to_radians();

    // Start at ascending node: position along x, velocity in xy-plane tilted by inclination
    let init_pos = DVec3::new(r_park, 0.0, 0.0);
    let init_vel = DVec3::new(0.0, v_circ * inc.cos(), v_circ * inc.sin());

    // CSM core mass — S-IVB mass is added via the mass tree
    let total_mass = MASS_CSM + MASS_SIVB;
    let body_idx = sim.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init_pos,
            velocity: init_vel,
        },
        mass: Some(MassProperties::new(MASS_CSM)),
        gravity_controls: GravityControls {
            controls: vec![
                GravityControl::new_spherical(earth, false),
                GravityControl::new_third_body(moon),
                GravityControl::new_third_body(sun),
            ],
        },
        ..Default::default()
    });

    // Register in mass tree: CSM (root) + S-IVB (child)
    let csm_tree_id = sim.add_body_to_tree(body_idx, "CSM");
    // S-IVB attached at origin with identity rotation (simplified — no structural offset)
    let tree = sim.mass_tree.as_mut().unwrap();
    let sivb_tree_id = tree.add_body("S-IVB".to_string(), MassProperties::new(MASS_SIVB));
    tree.attach(sivb_tree_id, csm_tree_id, DVec3::ZERO, DMat3::IDENTITY);
    // Update body mass from tree composite
    sim.sync_body_mass_from_tree(body_idx);

    sim.validate().expect("validation failed");

    // Compute TLI time: after N parking orbits
    let parking_period = 2.0 * std::f64::consts::PI * (r_park.powi(3) / MU_EARTH).sqrt();
    let tli_time = PARKING_ORBITS * parking_period;
    let separation_time = tli_time + 600.0; // 10 minutes after TLI burn

    println!("Apollo trans-lunar injection — 3-day trajectory");
    println!(
        "  Parking orbit: {:.0} km circular, {INCLINATION_DEG}° inclination",
        PARKING_ALT / 1000.0
    );
    println!("  Vehicle: CSM ({MASS_CSM} kg) + S-IVB ({MASS_SIVB} kg)");
    println!(
        "  TLI delta-V: {TLI_DELTA_V} m/s prograde at t={:.1} h",
        tli_time / 3600.0
    );
    println!("  Stage separation at t={:.1} h", separation_time / 3600.0);
    println!("  Earth + Moon + Sun point-mass gravity | dt={DT} s");
    println!();
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>10}  {:>8}",
        "Time (h)", "Alt (km)", "Dist Moon", "Speed (m/s)", "Mass (kg)", "Phase"
    );
    println!("{}", "-".repeat(76));

    let total_steps = (TOTAL_DURATION / DT).round() as usize;
    let print_interval = 7200.0; // 2 hours
    let steps_per_print = (print_interval / DT).round() as usize;

    let mut tli_applied = false;
    let mut separated = false;

    for step in 1..=total_steps {
        let t = (step as f64) * DT;

        // Apply TLI burn (impulsive)
        if !tli_applied && t >= tli_time {
            let body = sim.body(body_idx);
            let vel_hat = body.trans.velocity.normalize();
            let new_vel = body.trans.velocity + vel_hat * TLI_DELTA_V;
            sim.set_body_velocity(body_idx, new_vel);
            tli_applied = true;
            println!(
                "{:10.1}  {:>12}  {:>12}  {:>12}  {:>10}  TLI BURN",
                t / 3600.0,
                "---",
                "---",
                "---",
                "---"
            );
        }

        // Stage separation
        if !separated && t >= separation_time {
            // Detach S-IVB: body keeps CSM mass only
            let tree = sim.mass_tree.as_mut().unwrap();
            tree.detach(sivb_tree_id);
            let csm_mass = tree.get(csm_tree_id).composite_properties.mass;
            sim.sync_body_mass_from_tree(body_idx);
            separated = true;
            println!(
                "{:10.1}  {:>12}  {:>12}  {:>12}  {:10.0}  SEPARATE",
                t / 3600.0,
                "---",
                "---",
                "---",
                csm_mass
            );
        }

        sim.step();

        if step % steps_per_print == 0 {
            let body = sim.body(body_idx);
            let pos = body.trans.position;
            let vel = body.trans.velocity;
            let moon_source_pos = sim.source_position(moon);

            let alt_km = (pos.length() - R_EARTH) / 1000.0;
            let dist_moon_km = (pos - moon_source_pos).length() / 1000.0;
            let speed = vel.length();
            let mass = if separated { MASS_CSM } else { total_mass };
            let hours = t / 3600.0;

            let phase = if !tli_applied {
                "PARK"
            } else if dist_moon_km < 50_000.0 {
                "LUNAR"
            } else {
                "COAST"
            };

            println!(
                "{hours:10.1}  {alt_km:12.1}  {dist_moon_km:12.1}  {speed:12.1}  {mass:10.0}  {phase}"
            );
        }
    }

    let final_body = sim.body(body_idx);
    let final_moon_dist = (final_body.trans.position - sim.source_position(moon)).length();
    println!();
    println!(
        "Final distance to Moon: {:.0} km after {:.1} days",
        final_moon_dist / 1000.0,
        TOTAL_DURATION / 86400.0
    );

    Ok(())
}
