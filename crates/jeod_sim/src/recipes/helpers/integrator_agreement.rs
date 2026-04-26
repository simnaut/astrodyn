//! Cross-integrator agreement bookkeeping.
//!
//! Phase 8 of #101: lifted from the inlined pattern in
//! `tier3_sim_integ_comparison.rs`. Tests propagate the same scenario
//! through two integrators (e.g. RK4 vs RKF45) and assert that the
//! per-component divergence at the final time stays below a threshold.
//!
//! The actual propagation is left to the test's bespoke loop — this
//! module is just a small struct that bundles the two final states and
//! exposes their difference magnitudes.

use glam::DVec3;

/// Bundle of the two final states produced by two integrators on the
/// same scenario.
#[derive(Debug, Clone, Copy)]
pub struct IntegratorAgreement {
    /// Final position from integrator A.
    pub pos_a: DVec3,
    /// Final velocity from integrator A.
    pub vel_a: DVec3,
    /// Final position from integrator B.
    pub pos_b: DVec3,
    /// Final velocity from integrator B.
    pub vel_b: DVec3,
}

impl IntegratorAgreement {
    /// Magnitude of the position difference (m).
    pub fn position_diff(&self) -> f64 {
        (self.pos_a - self.pos_b).length()
    }

    /// Magnitude of the velocity difference (m/s).
    pub fn velocity_diff(&self) -> f64 {
        (self.vel_a - self.vel_b).length()
    }
}

/// Convenience: return `(pos_diff, vel_diff)` magnitudes.
pub fn integrator_divergence(pos_a: DVec3, vel_a: DVec3, pos_b: DVec3, vel_b: DVec3) -> (f64, f64) {
    let agr = IntegratorAgreement {
        pos_a,
        vel_a,
        pos_b,
        vel_b,
    };
    (agr.position_diff(), agr.velocity_diff())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_diff_for_identical_states() {
        let p = DVec3::new(1.0, 2.0, 3.0);
        let v = DVec3::new(4.0, 5.0, 6.0);
        let (dp, dv) = integrator_divergence(p, v, p, v);
        assert!(dp.abs() < 1e-15);
        assert!(dv.abs() < 1e-15);
    }
}
