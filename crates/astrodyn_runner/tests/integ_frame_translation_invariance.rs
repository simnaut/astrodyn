//! Frame-translation invariance: identical physics around a planet must
//! produce identical outputs whether the body integrates in the simulation
//! root frame (current sims) or in a non-root planet-inertial frame
//! attached as a child of root (the bug case for issue #255).
//!
//! This test exercises the [`IntegrationFrame`] / [`IntegOrigin`] /
//! `body.trans.to_inertial(&o)` plumbing end-to-end. It is **not** a Tier 3
//! cross-validation against JEOD reference data; it is a property test.
//! The two configurations differ only in where the integrated body sits in
//! the frame tree, not in any physical input — so every observable output
//! (gravity-driven trajectory, atmosphere density at the body, geodetic
//! altitude, solar beta angle) must match across the pair.
//!
//! ## Scope
//!
//! | Consumer | Frame requirement | Tested |
//! |----------|-------------------|--------|
//! | Gravity (Earth point-mass) | RootInertial — needs `to_inertial` | ✓ |
//! | Atmosphere (exponential, Earth) | PlanetInertial<Earth> — no shift | ✓ |
//! | Geodetic (Earth) | PlanetInertial<Earth> — no shift | ✓ |
//! | LVLH (around Earth) | PlanetInertial<Earth> — no shift | ✓ |
//! | Orbital elements (around Earth) | PlanetInertial<Earth> — no shift | ✓ |
//! | Solar beta | RootInertial — needs `to_inertial` | ✓ |
//!
//! Drag, SRP, and earth-lighting are tracked via the typed-sibling guards
//! exercised by the other unit tests; this file focuses on observable
//! frame-translation invariance of derived states.
//!
//! Tolerances reflect the expected f64 rounding from the 1.5e11 m
//! Earth-from-SSB offset arithmetic (`body + integ_origin - earth_offset`
//! at every gravity stage). Worst-case ulp at 1.5e11 is ~3e-5 m, so
//! per-gravity-eval drift is bounded around 3e-5 m and the resulting
//! per-step trajectory drift is ~1e-7 m. We use 1e-4 m / 1e-6 m/s as the
//! allowed bound (3 orders of magnitude over the per-step estimate to
//! cover N^2-style accumulation across the 30-step window). Any drift
//! above that level indicates a frame-mixing bug, not numerical noise.

use glam::DVec3;

use astrodyn::{
    AtmosphereConfig, AtmosphereModel, GravityControl, GravityControls, GravityModel,
    GravitySource, GravitySourceEntry, RootInertial, RotationModel, SimulationBuilder,
    SimulationTime, TranslationalState, Vec3Ext, VehicleConfig, EARTH,
};
use astrodyn_atmosphere::exponential::ExponentialAtmosphere;
use astrodyn_runner::{Simulation, SimulationBuilderExt};

const MU_EARTH: f64 = 3.986_004_415e14; // m^3/s^2
const MU_BARYCENTER: f64 = 0.0; // SSB-as-central placeholder: no mass at root

const ALT_M: f64 = 700_000.0; // 700 km altitude
const RADIUS_M: f64 = EARTH.shape.r_eq + ALT_M;
const SPEED_M_S: f64 = 7_504.567; // ~circular at radius — derived in test

const SSB_TO_EARTH_OFFSET: DVec3 = DVec3::new(1.5e11, 0.0, 0.0);
/// Sun-Earth offset (Earth-relative). Setup (a) places the Sun at this
/// position in root coords; setup (b) places it at
/// `SSB_TO_EARTH_OFFSET + SUN_FROM_EARTH` in root coords so that the
/// Earth-relative geometry (and therefore solar beta) is invariant.
const SUN_FROM_EARTH: DVec3 = DVec3::new(1.495_978_707e11, 0.0, 0.0);

const DT: f64 = 60.0; // 60-second steps
const N_STEPS: usize = 30; // 30 minutes

fn earth_point_mass(mu: f64) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        // Provide identity rotation so `Earth.pfix` exists for the
        // atmosphere/geodetic stages (otherwise t_inertial_pfix is None).
        t_inertial_pfix: Some(glam::DMat3::IDENTITY),
        rotation_model: RotationModel::None,
        delta_c20: 0.0,
        tidal_config: None,
        planet_omega: 0.0, // zero corotation so density is comparable across setups
        central: true,
    }
}

fn ssb_barycenter() -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu: MU_BARYCENTER,
            model: GravityModel::PointMass,
        },
        position: astrodyn::Position::<astrodyn::RootInertial>::zero(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: None,
        rotation_model: RotationModel::None,
        delta_c20: 0.0,
        tidal_config: None,
        planet_omega: 0.0,
        central: true,
    }
}

fn earth_at_offset(mu: f64) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu,
            model: GravityModel::PointMass,
        },
        position: SSB_TO_EARTH_OFFSET.m_at::<RootInertial>(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: Some(glam::DMat3::IDENTITY),
        rotation_model: RotationModel::None,
        delta_c20: 0.0,
        tidal_config: None,
        planet_omega: 0.0,
        central: false,
    }
}

/// A non-central Sun source positioned relative to root such that its
/// Earth-relative position equals [`SUN_FROM_EARTH`] in both setups.
fn sun_source(position_in_root: DVec3) -> GravitySourceEntry {
    GravitySourceEntry {
        source: GravitySource {
            mu: 0.0, // no gravitational pull from Sun in this test
            model: GravityModel::PointMass,
        },
        position: position_in_root.m_at::<RootInertial>(),
        velocity: astrodyn::Velocity::<astrodyn::RootInertial>::zero(),
        t_inertial_pfix: None,
        rotation_model: RotationModel::None,
        delta_c20: 0.0,
        tidal_config: None,
        planet_omega: 0.0,
        central: false,
    }
}

fn atmosphere_config() -> AtmosphereConfig {
    AtmosphereConfig {
        model: AtmosphereModel::Exponential(ExponentialAtmosphere::default()),
        r_eq: EARTH.shape.r_eq,
        r_pol: EARTH.shape.r_pol,
        // Disable corotation wind so density / drag depend only on
        // altitude — invariant across the two frame topologies.
        planet_omega: 0.0,
    }
}

fn body_config(integ_source: Option<usize>, gravity_source_idx: usize) -> VehicleConfig {
    let trans = TranslationalState {
        // Earth-relative ECI coords. In setup (a) the body integrates in
        // root=Earth.inertial, so this is also root-coords. In setup (b)
        // the body integrates in Earth.inertial (child of SSB), so this
        // is again Earth-coords — bit-identical initial state.
        position: DVec3::new(RADIUS_M, 0.0, 0.0),
        velocity: DVec3::new(0.0, SPEED_M_S, 0.0),
    };

    let mut cfg = VehicleConfig {
        trans: trans.into(),
        rot: None,
        mass: None,
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(gravity_source_idx, false)],
        },
        ..Default::default()
    };

    cfg.integ_source = integ_source;

    // Enable derived states sensitive to frame.
    cfg.derived.orbital_elements_source = Some(gravity_source_idx);
    cfg.derived.lvlh = true;
    cfg.derived.geodetic = Some(astrodyn::GeodeticConfig {
        source_idx: gravity_source_idx,
        r_eq: EARTH.shape.r_eq,
        r_pol: EARTH.shape.r_pol,
    });
    // Solar beta exercises the SRP/lighting structural guard:
    // `compute_body_solar_beta_typed` takes `Position<RootInertial>`, so the
    // body must be shifted via `to_inertial(&o)` to compile. The two setups
    // place the Sun such that Sun-Earth geometry is identical, so the
    // computed solar-beta angle must match across the pair.
    cfg.derived.solar_beta = true;

    // Atmosphere ON (so the named #255 bug class is exercised).
    cfg
}

/// Setup (a): single-planet Earth-rooted scenario.
/// - Root = Earth.inertial (central body).
/// - Sun at `SUN_FROM_EARTH` in root coords (= Earth-relative).
/// - Body integrates in root.
fn build_root_setup() -> Simulation {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, DT);
    let earth = sb.add_source("Earth", earth_point_mass(MU_EARTH));
    let sun = sb.add_source("Sun", sun_source(SUN_FROM_EARTH));
    sb = sb.atmosphere(atmosphere_config(), earth).sun(sun);
    sb.add_body(body_config(None, earth));
    sb.build().expect("setup (a) builds")
}

/// Setup (b): SSB-rooted with Earth as a non-central child at offset.
/// - Root = SSB.inertial (central, mu=0 barycenter).
/// - Earth at `SSB_TO_EARTH_OFFSET` in root coords.
/// - Sun at `SSB_TO_EARTH_OFFSET + SUN_FROM_EARTH` in root coords, so the
///   Sun-Earth relative position equals setup (a)'s.
/// - Body integrates in `Earth.inertial` via `integ_source = Some(earth)`.
fn build_offset_setup() -> Simulation {
    let time = SimulationTime::at_j2000(astrodyn::default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, DT);
    let _ssb = sb.add_source("SSB", ssb_barycenter());
    let earth = sb.add_source("Earth", earth_at_offset(MU_EARTH));
    let sun = sb.add_source("Sun", sun_source(SSB_TO_EARTH_OFFSET + SUN_FROM_EARTH));
    sb = sb.atmosphere(atmosphere_config(), earth).sun(sun);
    sb.add_body(body_config(Some(earth), earth));
    sb.build().expect("setup (b) builds")
}

#[test]
fn integ_frame_translation_invariance_geodetic_and_orbit() {
    let mut sim_a = build_root_setup();
    let mut sim_b = build_offset_setup();

    // Step both for the same wall time; per-step physics should match
    // bit-near-exactly because `IntegOrigin` is a pure translation and
    // every consumer either takes planet-centered (= integration-frame)
    // or applies the typed shift to root-inertial.
    for step in 0..N_STEPS {
        sim_a
            .step()
            .unwrap_or_else(|e| panic!("setup (a) step {step}: {e:?}"));
        sim_b
            .step()
            .unwrap_or_else(|e| panic!("setup (b) step {step}: {e:?}"));

        let a = sim_a.body(0);
        let b = sim_b.body(0);

        // Body translational state — both are stored in the body's
        // integration frame, which equals Earth.inertial in both setups.
        // Exact match expected (offset add then subtract is bit-clean).
        let pos_diff = (a.trans.position - b.trans.position).length();
        let vel_diff = (a.trans.velocity - b.trans.velocity).length();
        assert!(
            pos_diff < 1e-4 && vel_diff < 1e-6,
            "step {step}: position diff {pos_diff:e} m, velocity diff {vel_diff:e} m/s \
             (tolerance is 1e-4 m / 1e-6 m/s, accounting for ulp(1.5e11)~3e-5 m \
             per-gravity-eval rounding from the SSB offset arithmetic)",
        );

        // Geodetic altitude — driven by atmosphere/geodetic which expect
        // planet-centered position. Must be identical across setups.
        let geo_a = a
            .geodetic_state
            .as_ref()
            .expect("setup (a) has geodetic state");
        let geo_b = b
            .geodetic_state
            .as_ref()
            .expect("setup (b) has geodetic state");
        assert!(
            (geo_a.altitude - geo_b.altitude).abs() < 1e-4,
            "step {step}: geodetic altitude differs by {:e} m",
            (geo_a.altitude - geo_b.altitude).abs()
        );
        assert!(
            (geo_a.latitude - geo_b.latitude).abs() < 1e-10,
            "step {step}: geodetic latitude differs by {:e} rad",
            (geo_a.latitude - geo_b.latitude).abs()
        );

        // Orbital elements around Earth — must match.
        let oe_a = a
            .orbital_elements
            .as_ref()
            .expect("setup (a) has orbital elements");
        let oe_b = b
            .orbital_elements
            .as_ref()
            .expect("setup (b) has orbital elements");
        assert!(
            (oe_a.semi_major_axis - oe_b.semi_major_axis).abs() < 1e-3,
            "step {step}: SMA differs by {:e} m",
            (oe_a.semi_major_axis - oe_b.semi_major_axis).abs()
        );

        // LVLH frame — orientation must match.
        let lvlh_a = a.lvlh_frame.as_ref().expect("setup (a) has LVLH");
        let lvlh_b = b.lvlh_frame.as_ref().expect("setup (b) has LVLH");
        let dm = lvlh_a.t_parent_this - lvlh_b.t_parent_this;
        let m_diff = dm.x_axis.length() + dm.y_axis.length() + dm.z_axis.length();
        assert!(
            m_diff < 1e-9,
            "step {step}: LVLH t_parent_this differs by {m_diff:e}",
        );

        // Solar beta — exercises the SRP/lighting structural shift.
        // `compute_body_solar_beta_typed` requires `Position<RootInertial>`,
        // so the runner must call `body.trans.to_inertial(&o)` before
        // passing to it. Sun is placed such that its Earth-relative
        // position is identical in both setups, so the resulting solar
        // beta angle must match.
        let sb_a = a.solar_beta.expect("setup (a) has solar beta");
        let sb_b = b.solar_beta.expect("setup (b) has solar beta");
        assert!(
            (sb_a - sb_b).abs() < 1e-12,
            "step {step}: solar beta differs by {:e} rad ({sb_a} vs {sb_b}). \
             A non-zero diff here would indicate the integration-frame → \
             root-inertial shift is missing or inconsistent at the solar \
             beta call site (RF.10 shift site).",
            (sb_a - sb_b).abs()
        );
    }
}
