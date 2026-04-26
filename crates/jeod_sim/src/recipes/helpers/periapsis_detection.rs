//! Periapsis-passage detection via radial-velocity zero crossing.
//!
//! Used by Mercury / GR perihelion-advance tests where the bespoke
//! per-step loop captures an event each time `dr/dt` flips from
//! negative to positive (the body has passed periapsis since the
//! previous step).

use glam::DVec3;

/// A periapsis-passage event: the simulation time at the crossing
/// (linear interpolation against the radial-velocity sign change is
/// **not** performed here — callers that need sub-step precision
/// should refit), and the longitude of perihelion
/// `arg_periapsis + long_asc_node` (radians).
#[derive(Debug, Clone, Copy)]
pub struct PeriapsisEvent {
    /// Time of detection (s, simulation epoch-relative).
    pub time: f64,
    /// Longitude of perihelion (rad), invariant to nodal regression.
    pub long_perihelion: f64,
}

/// Streaming periapsis detector.
///
/// Feed each `(time, position, velocity)` sample; emits at most one
/// event per call. Returns `Some(event)` when the radial velocity
/// `r·v / |r|` transitions from negative to non-negative between the
/// previous and current samples.
#[derive(Debug, Clone, Copy)]
pub struct PeriapsisDetector {
    prev_rdot: f64,
    initialized: bool,
}

impl Default for PeriapsisDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PeriapsisDetector {
    pub fn new() -> Self {
        Self {
            prev_rdot: 0.0,
            initialized: false,
        }
    }

    /// Reset the detector so the next call seeds the previous radial
    /// velocity without emitting an event.
    pub fn reset(&mut self) {
        self.initialized = false;
        self.prev_rdot = 0.0;
    }

    /// Returns `true` if the previous-to-current sample contains a
    /// periapsis crossing (radial velocity transitions from `< 0` to
    /// `≥ 0`).
    pub fn observe(&mut self, pos: DVec3, vel: DVec3) -> bool {
        let r_dot = pos.dot(vel) / pos.length();
        let crossed = self.initialized && self.prev_rdot < 0.0 && r_dot >= 0.0;
        self.prev_rdot = r_dot;
        self.initialized = true;
        crossed
    }
}

/// Sweep an iterator of `(time, position, velocity)` samples and
/// return all detected periapsis events. The orbital elements are
/// computed at the post-crossing sample via
/// `OrbitalElements::from_cartesian_typed`, giving longitude of
/// perihelion `arg_periapsis + long_asc_node`.
///
/// Used by `tier3_sim_mercury` for both the in-memory sim trace and
/// the JEOD CSV trace (they share this loop body).
pub fn detect_periapsis_passages<I>(samples: I, mu: f64) -> Vec<PeriapsisEvent>
where
    I: IntoIterator<Item = (f64, DVec3, DVec3)>,
{
    use jeod_math::OrbitalElements;
    use jeod_quantities::aliases::{Position, Velocity};
    use jeod_quantities::ext::F64Ext;
    use jeod_quantities::frame::Inertial;

    let mu_typed = mu.m3_per_s2();
    let mut det = PeriapsisDetector::new();
    let mut events = Vec::new();
    for (t, r, v) in samples {
        if det.observe(r, v) {
            let pos_typed = Position::<Inertial>::from_raw_si(r);
            let vel_typed = Velocity::<Inertial>::from_raw_si(v);
            if let Ok(oe) = OrbitalElements::from_cartesian_typed(mu_typed, pos_typed, vel_typed) {
                events.push(PeriapsisEvent {
                    time: t,
                    long_perihelion: oe.arg_periapsis + oe.long_asc_node,
                });
            }
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_emits_on_negative_to_positive_crossing() {
        let mut det = PeriapsisDetector::new();
        // Sample 1: r·v < 0 (approaching periapsis).
        let crossed_1 = det.observe(DVec3::X, -DVec3::X);
        assert!(!crossed_1, "first sample seeds prev_rdot, no event");
        // Sample 2: r·v > 0 (post-periapsis).
        let crossed_2 = det.observe(DVec3::X, DVec3::X);
        assert!(crossed_2);
    }

    #[test]
    fn detector_quiet_when_monotonic() {
        let mut det = PeriapsisDetector::new();
        det.observe(DVec3::X, DVec3::X); // seed
        let crossed = det.observe(DVec3::X, DVec3::X);
        assert!(!crossed);
    }
}
