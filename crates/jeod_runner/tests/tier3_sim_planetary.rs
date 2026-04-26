//! Tier 3: SIM_Planetary — derived state trajectory in 5 orbit regimes.
//!
//! Validates Simulation trajectory against JEOD SIM_Planetary reference CSVs
//! across LEO inclined, LEO polar, LEO eccentric, LEO equatorial, and GEO
//! orbits. These exercise coordinate singularities (equatorial RAAN, polar
//! LVLH).

use jeod_runner::prelude::*;
use jeod_runner::run_verification::sim_planetary;

#[test]
fn tier3_simulation_planetary_leo_inc() {
    sim_planetary::leo_inc().run_and_assert();
}

#[test]
fn tier3_simulation_planetary_leo_polar() {
    sim_planetary::leo_polar().run_and_assert();
}

#[test]
fn tier3_simulation_planetary_leo_ecc() {
    sim_planetary::leo_ecc().run_and_assert();
}

#[test]
fn tier3_simulation_planetary_leo_equ() {
    sim_planetary::leo_equ().run_and_assert();
}

#[test]
fn tier3_simulation_planetary_geo() {
    sim_planetary::geo().run_and_assert();
}
