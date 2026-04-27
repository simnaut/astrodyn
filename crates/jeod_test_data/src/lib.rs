#![forbid(unsafe_code)]

pub use jeod_quantities::prelude::*;

pub mod apollo_mass_tree;
pub mod crossval;
pub mod dyncomp_csv;
pub mod euler_test;
pub mod gravity_control;
pub mod gravity_verif;
pub mod leap_second;
pub mod mass_data;
pub mod orbital_data;
pub mod orbital_init;
pub mod reference_state;
pub mod s_define;
pub mod tier3_csv;
pub mod time_config;

/// Get the JEOD root path from environment variables.
///
/// Checks `JEOD_PATH` first, then `JEOD_HOME` (standard JEOD/Trick convention).
/// Returns a path that may or may not exist — callers should check `.exists()`.
pub fn jeod_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("JEOD_PATH") {
        return std::path::PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("JEOD_HOME") {
        return std::path::PathBuf::from(p);
    }
    std::path::PathBuf::from("JEOD_PATH_or_JEOD_HOME_not_set")
}

/// Get the Trick root path from the `TRICK_HOME` environment variable.
///
/// Returns a path that may or may not exist — callers should check `.exists()`.
pub fn trick_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("TRICK_HOME") {
        return std::path::PathBuf::from(p);
    }
    std::path::PathBuf::from("TRICK_HOME_not_set")
}
