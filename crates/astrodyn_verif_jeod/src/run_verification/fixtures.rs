//! Shared fixture helpers across `sim_*` recipes.

/// μ for Earth from the GGM05C fixture, cached for the test-process
/// lifetime (the decode parses ~12 KiB just to read a single scalar).
pub fn load_mu_earth() -> f64 {
    static MU_EARTH: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *MU_EARTH.get_or_init(|| astrodyn::gravity_fixtures::load_ggm05c().mu)
}
