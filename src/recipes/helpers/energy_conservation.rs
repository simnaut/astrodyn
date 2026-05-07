//! Specific orbital energy bookkeeping for energy-conservation tests.
//!
//! Phase 8 of #101: extracted from inlined patterns in
//! `tier3_sim_integ_comparison.rs`. These tests propagate a circular
//! orbit and assert that `max |relative energy error|` against the
//! initial energy stays below an integrator-specific threshold.

use glam::DVec3;

/// Two-body specific orbital energy `E = v²/2 − μ/r` (units: m²/s²).
pub fn specific_orbital_energy(pos: DVec3, vel: DVec3, mu: f64) -> f64 {
    0.5 * vel.length_squared() - mu / pos.length()
}

/// Streaming Kepler-energy monitor.
///
/// Records the energy at the t = 0 sample and tracks
/// `max |E(t) − E(0)| / |E(0)|` across subsequent steps.
///
/// ```
/// use glam::DVec3;
/// use astrodyn::recipes::helpers::energy_conservation::KeplerEnergyMonitor;
/// let mu = 3.986_004_415e14_f64;
/// let r = DVec3::new(7_000_000.0, 0.0, 0.0);
/// let v = DVec3::new(0.0, (mu / 7_000_000.0_f64).sqrt(), 0.0);
/// let mut mon = KeplerEnergyMonitor::new(r, v, mu);
/// mon.observe(r, v);
/// assert!(mon.max_relative_error() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KeplerEnergyMonitor {
    mu: f64,
    e0: f64,
    max_rel_err: f64,
}

impl KeplerEnergyMonitor {
    /// Construct from the t = 0 state.
    pub fn new(pos0: DVec3, vel0: DVec3, mu: f64) -> Self {
        Self {
            mu,
            e0: specific_orbital_energy(pos0, vel0, mu),
            max_rel_err: 0.0,
        }
    }

    /// Record a new state and update the running max.
    pub fn observe(&mut self, pos: DVec3, vel: DVec3) {
        let e = specific_orbital_energy(pos, vel, self.mu);
        let rel = ((e - self.e0) / self.e0).abs();
        if rel > self.max_rel_err {
            self.max_rel_err = rel;
        }
    }

    /// Initial energy E(0).
    pub fn initial_energy(&self) -> f64 {
        self.e0
    }

    /// `max |ΔE / E(0)|` across all observed states.
    pub fn max_relative_error(&self) -> f64 {
        self.max_rel_err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circular_orbit_energy_matches_minus_mu_over_2a() {
        let mu = 3.986_004_415e14;
        let a = 7_000_000.0;
        let r = DVec3::new(a, 0.0, 0.0);
        let v = DVec3::new(0.0, (mu / a).sqrt(), 0.0);
        let e = specific_orbital_energy(r, v, mu);
        let expected = -mu / (2.0 * a);
        assert!((e - expected).abs() < 1e-3);
    }

    #[test]
    fn monitor_zero_for_identical_states() {
        let mu = 3.986_004_415e14_f64;
        let r = DVec3::new(7_000_000.0, 0.0, 0.0);
        let v = DVec3::new(0.0, (mu / 7_000_000.0_f64).sqrt(), 0.0);
        let mut mon = KeplerEnergyMonitor::new(r, v, mu);
        for _ in 0..10 {
            mon.observe(r, v);
        }
        assert!(mon.max_relative_error().abs() < 1e-15);
    }
}
