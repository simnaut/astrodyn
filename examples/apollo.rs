//! Apollo trans-lunar injection: multi-body gravity, staging, impulsive maneuver.
//!
//! Propagates an Apollo-like spacecraft from a 185 km parking orbit through a
//! trans-lunar injection burn and a three-day coast. Demonstrates the full
//! set of long-duration coast physics that JEOD provides:
//!
//! - Earth + Moon + Sun point-mass gravity (Earth-centered integration, Moon
//!   and Sun as third bodies with DE421-driven positions each step).
//! - MassTree staging: CSM + S-IVB composite at TLI, S-IVB detached 10 minutes
//!   post-burn (`Simulation::attach` / `detach` drive the tree and the
//!   body's composite mass properties are re-synced automatically).
//! - Impulsive TLI delta-V applied prograde to the velocity vector once the
//!   parking orbit reaches the trigger time.
//! - Three-day coast under multi-body gravity so the Moon's influence
//!   rises above the perturbation floor.
//!
//! Requires `test_data/de421.bsp` (checked into the repo) for ephemeris data.
//!
//! Run with:
//! ```bash
//! cargo run --example apollo
//! ```

use std::path::Path;

use glam::{DMat3, DVec3};
use jeod_runner::{GravitySourceEntry, Simulation, VehicleConfig};
use jeod_sim::{
    EphemerisBody, GravityControl, GravityControls, GravityModel, GravitySource, MassProperties,
    SimulationTime, TranslationalState,
};

// Gravitational parameters (m^3/s^2).
const MU_EARTH: f64 = 3.986_004_418e14;
const MU_MOON: f64 = 4.902_800_066e12;
const MU_SUN: f64 = 1.327_124_400_41e20;

// Earth mean radius (m), used only for altitude display.
const R_EARTH: f64 = 6_371_000.0;

// Apollo-like parking orbit: 185 km circular, 28.5 deg inclination (KSC latitude).
const PARKING_ALT: f64 = 185_000.0;
const INCLINATION_DEG: f64 = 28.5;

// Vehicle masses (approximate Apollo values, kg).
const MASS_CSM: f64 = 28_800.0; // Command/Service Module
const MASS_SIVB: f64 = 13_300.0; // S-IVB dry mass (after TLI burn)

// TLI delta-V magnitude applied prograde (m/s). ~3.1 km/s is the nominal
// Apollo value.
const TLI_DELTA_V: f64 = 3_130.0;

// Timing.
const PARKING_ORBITS: f64 = 2.5; // coast in parking orbit before TLI
const DT: f64 = 60.0; // 1-minute timesteps
const TOTAL_DURATION: f64 = 3.0 * 86_400.0; // 3 days

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let bsp_path = data_dir.join("de421.bsp");
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}. Ensure test_data/de421.bsp is present (it is \
         committed to the repo).",
        bsp_path.display()
    );

    let ephemeris = jeod_sim::Ephemeris::from_bsp(&bsp_path)?;
    let time = SimulationTime::at_j2000(jeod_sim::default_leap_second_table());
    let epoch_tdb_jd = time.tdb_julian_date();

    // Seed Moon and Sun positions at epoch (refreshed every step via
    // set_source_ephemeris below).
    let (moon_pos, _) =
        ephemeris.get_state(EphemerisBody::Moon, EphemerisBody::Earth, epoch_tdb_jd)?;
    let (sun_pos, _) =
        ephemeris.get_state(EphemerisBody::Sun, EphemerisBody::Earth, epoch_tdb_jd)?;

    let mut sim = Simulation::new(time, DT);

    // Earth at origin — central source, point-mass gravity.
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

    // Moon as third body. Each step, its position/velocity are refreshed from
    // DE421 (Earth-relative).
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
    sim.set_source_ephemeris(moon, EphemerisBody::Moon, EphemerisBody::Earth);

    // Sun as third body. Weak but relevant over a multi-day coast.
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
    sim.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Earth);

    sim.ephemeris = Some(ephemeris);

    // Parking orbit: ascending node, prograde, inclined by INCLINATION_DEG.
    let r_park = R_EARTH + PARKING_ALT;
    let v_circ = (MU_EARTH / r_park).sqrt();
    let inc = INCLINATION_DEG.to_radians();

    let init_pos = DVec3::new(r_park, 0.0, 0.0);
    let init_vel = DVec3::new(0.0, v_circ * inc.cos(), v_circ * inc.sin());

    let total_mass = MASS_CSM + MASS_SIVB;

    // Spawn body with CSM mass; S-IVB is attached below via the mass tree and
    // syncs the composite back onto the body.
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

    // Build mass tree: CSM (root) with S-IVB attached as a child. The S-IVB is
    // attached at the same structural origin (no moment arm) to keep the
    // composite CoM at the CSM origin — a simplification, not a physical
    // choice. The attach/detach API is the point being demonstrated.
    let csm_tree_id = sim.add_body_to_tree(body_idx, "CSM");
    let tree = sim.mass_tree.as_mut().unwrap();
    let sivb_tree_id = tree.add_body("S-IVB".to_string(), MassProperties::new(MASS_SIVB));
    tree.attach(sivb_tree_id, csm_tree_id, DVec3::ZERO, DMat3::IDENTITY);
    sim.sync_body_mass_from_tree(body_idx);

    sim.validate().expect("validation failed");

    // Schedule: TLI burn after N parking orbits, separation 10 minutes later.
    let parking_period = 2.0 * std::f64::consts::PI * (r_park.powi(3) / MU_EARTH).sqrt();
    let tli_time = PARKING_ORBITS * parking_period;
    let separation_time = tli_time + 600.0;

    println!("Apollo trans-lunar injection — 3-day trajectory");
    println!(
        "  Parking orbit: {:.0} km circular, {INCLINATION_DEG} deg inclination",
        PARKING_ALT / 1000.0
    );
    println!("  Vehicle: CSM ({MASS_CSM} kg) + S-IVB ({MASS_SIVB} kg)");
    println!(
        "  TLI delta-V: {TLI_DELTA_V} m/s prograde at t={:.2} h",
        tli_time / 3600.0
    );
    println!("  Stage separation at t={:.2} h", separation_time / 3600.0);
    println!("  Earth + Moon + Sun point-mass gravity | dt={DT} s");
    println!();
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>10}  {:>8}",
        "Time (h)", "|r| (km)", "|r-moon|(km)", "|v| (m/s)", "Mass (kg)", "Phase"
    );
    println!("{}", "-".repeat(76));

    let total_steps = (TOTAL_DURATION / DT).round() as usize;
    let print_interval = 7_200.0; // 2 hours
    let steps_per_print = (print_interval / DT).round() as usize;

    let mut tli_applied = false;
    let mut separated = false;
    let mut min_moon_range_km = f64::INFINITY;
    let mut min_moon_range_hours = 0.0;

    for step in 1..=total_steps {
        let t = (step as f64) * DT;

        // Impulsive TLI burn: add TLI_DELTA_V along current velocity direction.
        if !tli_applied && t >= tli_time {
            let body = sim.body(body_idx);
            let vel_hat = body.trans.velocity.normalize();
            let new_vel = body.trans.velocity + vel_hat * TLI_DELTA_V;
            sim.set_body_velocity(body_idx, new_vel);
            tli_applied = true;
            println!(
                "{:10.2}  {:>12}  {:>12}  {:>12}  {:>10}  TLI BURN",
                t / 3600.0,
                "---",
                "---",
                "---",
                "---"
            );
        }

        // Stage separation: detach S-IVB from the CSM in the mass tree, then
        // resync the body's mass from the tree so the post-detach CSM-only
        // composite takes effect on the next step's force collection.
        if !separated && t >= separation_time {
            let tree = sim.mass_tree.as_mut().unwrap();
            tree.detach(sivb_tree_id);
            let csm_mass = tree.get(csm_tree_id).composite_properties.mass;
            sim.sync_body_mass_from_tree(body_idx);
            separated = true;
            println!(
                "{:10.2}  {:>12}  {:>12}  {:>12}  {:10.0}  SEPARATE",
                t / 3600.0,
                "---",
                "---",
                "---",
                csm_mass
            );
        }

        sim.step();

        // Track closest lunar approach throughout the coast.
        let body = sim.body(body_idx);
        let moon_source_pos = sim.source_position(moon);
        let range_moon_km = (body.trans.position - moon_source_pos).length() / 1000.0;
        if range_moon_km < min_moon_range_km {
            min_moon_range_km = range_moon_km;
            min_moon_range_hours = t / 3600.0;
        }

        if step % steps_per_print == 0 {
            let pos = body.trans.position;
            let vel = body.trans.velocity;

            let r_km = pos.length() / 1000.0;
            let speed = vel.length();
            let mass = if separated { MASS_CSM } else { total_mass };
            let hours = t / 3600.0;

            let phase = if !tli_applied {
                "PARK"
            } else if range_moon_km < 50_000.0 {
                "LUNAR"
            } else {
                "COAST"
            };

            println!(
                "{hours:10.2}  {r_km:12.1}  {range_moon_km:12.1}  {speed:12.1}  {mass:10.0}  {phase}"
            );
        }
    }

    let final_body = sim.body(body_idx);
    let final_moon_dist = (final_body.trans.position - sim.source_position(moon)).length() / 1000.0;
    let final_r_km = final_body.trans.position.length() / 1000.0;
    println!();
    println!("Final Earth range: {final_r_km:.1} km, Moon range: {final_moon_dist:.1} km");
    println!(
        "Closest lunar approach: {min_moon_range_km:.1} km at t = {min_moon_range_hours:.2} h"
    );

    Ok(())
}
