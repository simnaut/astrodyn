//! Pipeline smoke test: Earth lighting via Simulation::step()
//!
//! Creates a Simulation with Earth+Sun+Moon (DE421 ephemeris), propagates
//! an ISS-like LEO orbit, and verifies that EarthLightingState is computed
//! at each step with physically plausible sunlit/shadow transitions.
//!
//! This is NOT a Tier 3 cross-validation test — no JEOD propagating sim
//! with earth lighting exists to compare against. It exercises the pipeline
//! end-to-end but asserts physical plausibility, not JEOD parity.

use astrodyn::VehicleConfig;
use astrodyn::{DerivedStateConfig, EarthLightingConfig, GravitySourceEntry};
use astrodyn::{
    Ephemeris, EphemerisBody, GravityControl, GravityControls, GravityGradient, GravityModel,
    GravitySource, SimulationTime,
};
use astrodyn_runner::{RotationModel, Simulation};
use glam::DVec3;

#[test]
fn pipeline_earth_lighting_smoke() {
    let bsp_path = astrodyn::ephemeris_assets::de421_path();
    assert!(
        bsp_path.exists(),
        "DE421 not found at {}",
        bsp_path.display()
    );
    let ephemeris = Ephemeris::from_bsp(&bsp_path).expect("load DE421");

    let mu_earth = astrodyn::gravity_fixtures::load_ggm05c().mu;

    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sim = Simulation::new(time, 60.0);

    // Earth
    let earth = sim.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
            marker_only: false,
        },
    );

    // Sun from DE421
    let j2000_jd = 2_451_545.0;
    let (initial_sun_typed, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Sun, j2000_jd)
        .expect("Sun position at J2000");
    let _initial_sun = initial_sun_typed.raw_si();
    let sun = sim.add_source(
        "Sun",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: initial_sun_typed,
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    sim.set_source_ephemeris(sun, EphemerisBody::Sun, EphemerisBody::Earth);
    sim.sun_source = Some(sun);

    // Moon from DE421
    let (initial_moon_typed, _) = ephemeris
        .get_earth_centered_state_typed(EphemerisBody::Moon, j2000_jd)
        .expect("Moon position at J2000");
    let _initial_moon = initial_moon_typed.raw_si();
    let moon = sim.add_source(
        "Moon",
        GravitySourceEntry {
            source: GravitySource {
                mu: 0.0,
                model: GravityModel::PointMass,
            },
            position: initial_moon_typed,
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: false,
            marker_only: false,
        },
    );
    sim.set_source_ephemeris(moon, EphemerisBody::Moon, EphemerisBody::Earth);
    sim.moon_source = Some(moon);
    sim.ephemeris = Some(ephemeris);

    // ISS-like LEO body with earth lighting enabled
    // earth_lighting_config = (earth_radius, moon_radius, sun_radius)
    sim.add_body(VehicleConfig {
        trans: astrodyn::TranslationalStateTyped::<astrodyn::RootInertial> {
            // allowed: typed↔raw kernel boundary
            position: astrodyn::Position::<astrodyn::RootInertial>::from_raw_si(DVec3::new(
                6_778_137.0,
                0.0,
                0.0,
            )),
            // allowed: typed↔raw kernel boundary
            velocity: astrodyn::Velocity::<astrodyn::RootInertial>::from_raw_si(DVec3::new(
                0.0, 7_668.558, 0.0,
            )),
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, GravityGradient::Skip)],
        },
        derived: DerivedStateConfig {
            earth_lighting: Some(EarthLightingConfig {
                earth_radius: astrodyn::EARTH.shadow_radius,
                moon_radius: 1_737_400.0,
                sun_radius: 6.96e8,
            }),
            ..Default::default()
        },
        ..VehicleConfig::named("pipeline-earth-lighting-0")
    });

    sim.validate().unwrap();

    // Propagate one orbit (~90 min = 90 steps at 60s dt)
    let num_steps = 90;
    let mut lit_count = 0;
    let mut shadow_count = 0;

    for _ in 0..num_steps {
        sim.step().expect("step failed");

        let body = sim.body(0);
        let lighting = body
            .earth_lighting
            .as_ref()
            .expect("earth_lighting should be computed after step()");

        // Check that lighting state has physical values
        assert!(
            (0.0..=1.0).contains(&lighting.sun_earth.visible),
            "sun_earth.visible={} out of [0,1]",
            lighting.sun_earth.visible
        );

        if lighting.sun_earth.visible > 0.5 {
            lit_count += 1;
        } else {
            shadow_count += 1;
        }
    }

    // ISS orbit at J2000: ~60% sunlit, ~40% eclipsed
    println!(
        "  Pipeline test: {} steps, {} sunlit, {} shadow",
        num_steps, lit_count, shadow_count
    );
    assert!(
        lit_count > 30 && shadow_count > 10,
        "Expected mix of sun/shadow for LEO orbit, got {lit_count} sunlit / {shadow_count} shadow"
    );
}
