//! `VerificationCase` constructors for the SIM_Planetary derived-state
//! trajectory tests.
//!
//! Five orbit regimes (LEO inclined, LEO polar, LEO eccentric, LEO
//! equatorial, GEO) exercise coordinate singularities (equatorial
//! RAAN, polar LVLH). All five share the same physics: point-mass
//! Earth gravity, J2000 epoch, dt loaded from
//! `models/dynamics/derived_state/verif/SIM_Planetary/S_define`.
//!
//! Initial conditions come from the t=0 row of the corresponding
//! reference CSV (treated as JEOD source data per CLAUDE.md), since
//! SIM_Planetary stores them inline rather than in a separate
//! Modified_data file.

use astrodyn::recipes::verification::{
    CsvReference, InitialConditions, Tolerances, VerificationCase,
};
use astrodyn::{
    default_leap_second_table, GravityControl, GravityControls, GravityModel, GravitySource,
    GravitySourceEntry, SimulationBuilder, SimulationTime, TranslationalState, VehicleConfig,
};
use uom::si::f64::Time;
use uom::si::time::second;

const SIM_PLANETARY: &str = "models/dynamics/derived_state/verif/SIM_Planetary";

fn load_mu_earth() -> f64 {
    // Earth mu from the committed GGM05C fixture (Wave 1 of #232).
    astrodyn_test_data::gravity_fixtures::load_ggm05c().mu
}

fn build_planetary(init: &InitialConditions) -> SimulationBuilder {
    let dt = astrodyn_test_data::s_define::load_dynamics_dt(
        &astrodyn_test_data::jeod_inputs::path(SIM_PLANETARY).join("S_define"),
    );
    let mu_earth = load_mu_earth();

    let time = SimulationTime::at_j2000(default_leap_second_table());
    let mut sb = SimulationBuilder::new(time, dt);

    let earth = sb.add_source("Earth", {
        let mut e = GravitySourceEntry::new(
            GravitySource {
                mu: mu_earth,
                model: GravityModel::PointMass,
            },
            astrodyn::Position::<astrodyn::RootInertial>::zero(),
            None,
        );
        e.central = true;
        e
    });
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

const PLANETARY_TOLS: Tolerances = Tolerances {
    position_m: [1.0, 1.0, 1.0],
    velocity_m_s: [0.001, 0.001, 0.001],
    quat_angle_rad: 0.0,
    ang_vel_rad_s: [0.0; 3],
    extras: &[],
};

/// LEO inclined orbit derived-state regression.
pub fn leo_inc() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_planetary_leo_inc",
        scenario: build_planetary,
        reference: CsvReference::OrbInit("planetary_leo_inc_planetary.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: PLANETARY_TOLS,
        extras: None,
        pre_step: None,
    }
}

/// LEO polar orbit derived-state regression.
pub fn leo_polar() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_planetary_leo_polar",
        scenario: build_planetary,
        reference: CsvReference::OrbInit("planetary_leo_polar_planetary.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: PLANETARY_TOLS,
        extras: None,
        pre_step: None,
    }
}

/// LEO eccentric orbit derived-state regression.
pub fn leo_ecc() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_planetary_leo_ecc",
        scenario: build_planetary,
        reference: CsvReference::OrbInit("planetary_leo_ecc_planetary.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: PLANETARY_TOLS,
        extras: None,
        pre_step: None,
    }
}

/// LEO equatorial orbit derived-state regression (RAAN singular).
pub fn leo_equ() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_planetary_leo_equ",
        scenario: build_planetary,
        reference: CsvReference::OrbInit("planetary_leo_equ_planetary.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: PLANETARY_TOLS,
        extras: None,
        pre_step: None,
    }
}

/// GEO orbit derived-state regression.
pub fn geo() -> VerificationCase {
    VerificationCase {
        name: "tier3_simulation_planetary_geo",
        scenario: build_planetary,
        reference: CsvReference::OrbInit("planetary_geo_planetary.csv"),
        duration: Time::new::<second>(86400.0),
        tolerances: PLANETARY_TOLS,
        extras: None,
        pre_step: None,
    }
}
