//! Clementine lunar orbit: multi-body gravity with Earth, Moon, and Sun.
//!
//! Verification-style example exercising:
//! - Moon LP150Q 60×60 spherical-harmonics gravity
//! - Earth and Sun as 3rd-body point-mass perturbations
//! - DE421 ephemeris for per-step source position updates
//! - Moon libration rotation from DE421 BPC data
//! - Cannonball solar radiation pressure (no shadow body — illumination
//!   stays at 1.0; matches the original example's behaviour)
//!
//! This example consumes
//! [`recipes::verification::reference_data`](jeod_sim::recipes::verification::reference_data),
//! which loads JEOD source files (`$JEOD_HOME`) for the
//! Moon LP150Q gravity model and `test_data/de421.bsp` /
//! `test_data/moon_pa_de421_1900-2050.bpc` for ephemeris and Moon
//! libration. Mission code that wants high-fidelity gravity / ephemeris
//! without a JEOD checkout is tracked in #144; until that lands the
//! example is verification-grade by necessity.
//!
//! ```bash
//! cargo run -p jeod_runner --example earth_moon
//! ```

use std::path::Path;

use glam::DVec3;
use jeod_runner::SimulationBuilderExt;
use jeod_sim::recipes::{epoch, sun, vehicle, verification::reference_data};
use jeod_sim::vehicle_builder::VehicleBuilder;
use jeod_sim::{Ephemeris, EphemerisBody, GravityControl, SimulationBuilder, TranslationalState};

// Initial state from JEOD SIM_Earth_Moon RUN_clem at t=0 (Moon-centered inertial).
const INIT_POS: DVec3 = DVec3::new(1_296_944.012, -1_060_824.45, 2_522_289.146);
const INIT_VEL: DVec3 = DVec3::new(-930.578, -439.312, 862.075);

const SRP_CX_AREA: f64 = 2.1432;
const SRP_ALBEDO: f64 = 1.0;
const SRP_DIFFUSE: f64 = 0.27;

const DT: f64 = 1.0;
const DURATION: f64 = 86_400.0;
const R_MOON: f64 = 1_737_400.0;

/// Parse `--steps N` from CLI args; fall back to `default` when absent.
/// Panics with a clear message on a malformed value (per fail-loudly policy).
fn parse_steps_arg(default: usize) -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--steps" {
            let val = args
                .next()
                .expect("--steps requires a value, e.g. --steps 10");
            return val
                .parse::<usize>()
                .unwrap_or_else(|err| panic!("--steps value {val:?} is not a usize: {err}"));
        }
    }
    default
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load DE421 + Moon BPC libration (committed at test_data/).
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data");
    let bsp_path = data_dir.join("de421.bsp");
    let bpc_path = data_dir.join("moon_pa_de421_1900-2050.bpc");
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );
    assert!(
        bpc_path.exists(),
        "Moon BPC not found at {}",
        bpc_path.display()
    );

    let mut ephemeris = Ephemeris::from_bsp(&bsp_path)?;
    ephemeris.load_bpc(&bpc_path)?;

    let time = epoch::clementine_1994();
    let epoch_tdb_jd = time.tdb_julian_date();

    // Moon central body with high-fidelity LP150Q SH gravity (verification
    // reference data — requires $JEOD_HOME).
    let mut moon_source = reference_data::moon_lp150q();
    moon_source.t_inertial_pfix =
        Some(ephemeris.get_body_rotation(EphemerisBody::Moon, epoch_tdb_jd)?);
    let moon_mu = moon_source.source.mu;

    // Earth and Sun as 3rd-body point-mass perturbations (positions
    // overwritten each step by the ephemeris stage).
    let (earth_pos_typed, _) =
        ephemeris.get_state_typed(EphemerisBody::Earth, EphemerisBody::Moon, epoch_tdb_jd)?;
    let earth_pos = earth_pos_typed.raw_si();
    let (sun_pos_typed, _) =
        ephemeris.get_state_typed(EphemerisBody::Sun, EphemerisBody::Moon, epoch_tdb_jd)?;
    let sun_pos = sun_pos_typed.raw_si();

    let mut sb = SimulationBuilder::new(time, DT);
    let moon = sb.add_source("Moon", moon_source);
    let earth = sb.add_source("Earth", jeod_sim::recipes::earth::third_body(earth_pos));
    let sun_idx = sb.add_source("Sun", sun::third_body(sun_pos));
    sb.set_source_ephemeris(earth, EphemerisBody::Earth, EphemerisBody::Moon);
    sb.set_source_ephemeris(sun_idx, EphemerisBody::Sun, EphemerisBody::Moon);
    sb = sb.sun(sun_idx).ephemeris(ephemeris);

    let trans = TranslationalState {
        position: INIT_POS,
        velocity: INIT_VEL,
    };
    let clementine = VehicleBuilder::new()
        .with_state(trans)
        .three_dof_point_mass(vehicle::clementine_mass())
        .rk4()
        .gravity(GravityControl::new_nonspherical(moon, 60, 60, false))
        .gravity(GravityControl::new_third_body(earth))
        .gravity(GravityControl::new_third_body(sun_idx))
        .cannonball_srp(SRP_CX_AREA, SRP_ALBEDO, SRP_DIFFUSE)
        .build();
    sb.add_body(clementine);

    let mut sim = sb.build().expect("earth_moon scenario must validate");

    let print_interval = 7_200.0;
    let total_steps = parse_steps_arg((DURATION / DT).round() as usize);
    let steps_per_print = (print_interval / DT).round() as usize;
    // Derive the printed duration from `total_steps * DT` so headings stay
    // accurate when `--steps` overrides the nominal 1-day run length.
    let elapsed_days = total_steps as f64 * DT / 86_400.0;

    println!("Clementine lunar orbit — {elapsed_days:.2}-day propagation");
    println!("  Moon: LP150Q 60×60 SH | Earth + Sun 3rd-body | Cannonball SRP");
    println!("  Integrator: RK4 at dt={DT} s");
    println!();
    println!(
        "{:>10}  {:>12}  {:>12}  {:>10}  {:>10}",
        "Time (h)", "Alt (km)", "Speed (m/s)", "Period (h)", "Ecc"
    );
    println!("{}", "-".repeat(62));

    for step in 1..=total_steps {
        sim.step().expect("step failed");
        if step % steps_per_print == 0 {
            let body = sim.body(0);
            let pos = body.trans.position;
            let vel = body.trans.velocity;
            let r = pos.length();
            let v = vel.length();
            let altitude_km = (r - R_MOON) / 1000.0;
            let hours = (step as f64) * DT / 3600.0;
            let energy = 0.5 * v * v - moon_mu / r;
            let sma = -moon_mu / (2.0 * energy);
            let period_h = 2.0 * std::f64::consts::PI * (sma.powi(3) / moon_mu).sqrt() / 3600.0;
            let h_vec = pos.cross(vel);
            let ecc_vec = vel.cross(h_vec) / moon_mu - pos / r;
            let ecc = ecc_vec.length();
            println!("{hours:10.1}  {altitude_km:12.1}  {v:12.1}  {period_h:10.2}  {ecc:10.6}");
        }
    }

    println!();
    let final_body = sim.body(0);
    let final_alt = (final_body.trans.position.length() - R_MOON) / 1000.0;
    println!("Final altitude: {final_alt:.1} km after {elapsed_days:.2} days");
    Ok(())
}
