//! Step / ramp / sinusoid input profiles for force–torque response
//! tests.
//!
//! Phase 8 of #101: bespoke step/ramp inputs are a recurring pattern
//! in the `tier3_sim_force_torque_response.rs` family. This module
//! packages the patterns; it does not introduce new physics.

use glam::DVec3;

/// Constant input applied for `t >= t_on`, zero before.
pub fn step_input(amplitude: DVec3, t_on: f64, t: f64) -> DVec3 {
    if t >= t_on {
        amplitude
    } else {
        DVec3::ZERO
    }
}

/// Linear ramp: zero before `t_on`, `amplitude` from `t_on + duration`
/// onwards, linear interpolation in between.
#[derive(Debug, Clone, Copy)]
pub struct RampInput {
    pub amplitude: DVec3,
    pub t_on: f64,
    pub duration: f64,
}

impl RampInput {
    pub fn evaluate(&self, t: f64) -> DVec3 {
        if t < self.t_on {
            DVec3::ZERO
        } else if t >= self.t_on + self.duration {
            self.amplitude
        } else {
            let alpha = (t - self.t_on) / self.duration;
            self.amplitude * alpha
        }
    }
}

/// Sinusoidal input: `amplitude * sin(2π f (t − t_on))` for
/// `t >= t_on`, zero before.
#[derive(Debug, Clone, Copy)]
pub struct SinusoidInput {
    pub amplitude: DVec3,
    pub frequency_hz: f64,
    pub t_on: f64,
}

impl SinusoidInput {
    pub fn evaluate(&self, t: f64) -> DVec3 {
        if t < self.t_on {
            return DVec3::ZERO;
        }
        let phase = 2.0 * std::f64::consts::PI * self.frequency_hz * (t - self.t_on);
        self.amplitude * phase.sin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_off_then_on() {
        let amp = DVec3::new(1.0, 2.0, 3.0);
        assert_eq!(step_input(amp, 1.0, 0.0), DVec3::ZERO);
        assert_eq!(step_input(amp, 1.0, 1.0), amp);
        assert_eq!(step_input(amp, 1.0, 5.0), amp);
    }

    #[test]
    fn ramp_endpoints_and_midpoint() {
        let r = RampInput {
            amplitude: DVec3::new(1.0, 0.0, 0.0),
            t_on: 0.0,
            duration: 2.0,
        };
        assert_eq!(r.evaluate(-1.0), DVec3::ZERO);
        assert_eq!(r.evaluate(0.0), DVec3::ZERO);
        let mid = r.evaluate(1.0);
        assert!((mid.x - 0.5).abs() < 1e-15);
        assert_eq!(r.evaluate(2.0), DVec3::new(1.0, 0.0, 0.0));
        assert_eq!(r.evaluate(3.0), DVec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn sinusoid_zero_at_phase_zero_and_pi() {
        let s = SinusoidInput {
            amplitude: DVec3::new(1.0, 0.0, 0.0),
            frequency_hz: 1.0,
            t_on: 0.0,
        };
        assert!(s.evaluate(0.0).length() < 1e-15);
        assert!(s.evaluate(0.5).length() < 1e-15);
        // Quarter cycle: amplitude.
        let q = s.evaluate(0.25);
        assert!((q.x - 1.0).abs() < 1e-12);
    }
}
