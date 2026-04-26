//! `VerificationCase` constructor for the polar-motion regression test.
//!
//! `RUN_2P` re-runs SIM_dyncomp RUN_2 with `Simulation::polar_motion`
//! enabled. With point-mass gravity (`t_inertial_pfix: None`), the
//! planet-fixed rotation is never used, so polar motion has zero
//! trajectory effect — errors must match RUN_2 exactly. This test
//! validates that enabling the feature does not break point-mass
//! propagation.

use glam::DVec3;
use jeod_sim::recipes::verification::{
    CsvReference, InitialConditions, Tolerances, VerificationCase,
};
use jeod_sim::{
    default_leap_second_table, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, RotationModel, SimulationBuilder, SimulationTime, TranslationalState,
    VehicleConfig,
};
use uom::si::f64::Time;
use uom::si::time::second;

/// Arcseconds → radians conversion factor (from JEOD polar_motion data).
const ARCSEC_TO_RAD: f64 = 4.848_136_811_095_36e-6;

/// Polar motion values from JEOD SIM_RNP_J2000_prop input.py (constant
/// over the 8-hour propagation; cumulative drift is < 0.001 arcsec).
const XP_ARCSEC: f64 = 0.06806;
const YP_ARCSEC: f64 = 0.24156;

fn build_run2p_polar_motion(init: &InitialConditions) -> SimulationBuilder {
    let jeod_root = jeod_test_data::jeod_path();
    assert!(
        jeod_root.exists(),
        "JEOD source not found at {}. Set JEOD_HOME or JEOD_PATH.",
        jeod_root.display()
    );

    let mu_earth = jeod_sim::coefficients::load_mu_from_jeod_cc(
        &jeod_root.join("models/environment/gravity/data/src/earth_GGM05C.cc"),
    )
    .expect("load Earth mu from GGM05C");

    let dt =
        jeod_test_data::s_define::load_dynamics_dt(&jeod_root.join("verif/SIM_dyncomp/S_define"));
    let time = SimulationTime::at_j2000(default_leap_second_table());

    let mut sb = SimulationBuilder::new(time, dt);
    sb = sb.polar_motion(XP_ARCSEC * ARCSEC_TO_RAD, YP_ARCSEC * ARCSEC_TO_RAD);

    let earth = sb.add_source(
        "Earth",
        GravitySourceEntry {
            source: GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            t_inertial_pfix: None,
            delta_c20: 0.0,
            rotation_model: RotationModel::default(),
            tidal_config: None,
            planet_omega: 0.0,
            central: true,
        },
    );
    sb.add_body(VehicleConfig {
        trans: TranslationalState {
            position: init.position,
            velocity: init.velocity,
        },
        gravity_controls: GravityControls {
            controls: vec![GravityControl::new_spherical(earth, false)],
        },
        ..Default::default()
    });
    sb
}

/// SIM_dyncomp RUN_2P — point-mass + polar-motion regression. Polar
/// motion is enabled but point-mass gravity ignores planet-fixed
/// rotation, so errors must match RUN_2.
pub fn run2p_polar_motion() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_run2p_polar_motion",
        scenario: build_run2p_polar_motion,
        reference: CsvReference::Dyncomp3Dof("dyncomp_run2p_state.csv"),
        duration: Time::new::<second>(28800.0),
        tolerances: Tolerances {
            position_m: [1.37e-6, 2.154e-6, 1.826e-6],
            velocity_m_s: [1.446e-9, 2.389e-9, 1.814e-9],
            quat_angle_rad: 0.0,
            ang_vel_rad_s: [0.0; 3],
            extras: &[],
        },
        extras: None,
    }
}
